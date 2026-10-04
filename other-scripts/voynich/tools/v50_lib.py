"""v50 shared code: load C search output, per-path z against line-shuffled replicates, spec decoding, streams,
and a substitution-invariant language battery for unit streams."""
import os, sys, json, gzip, re, math, random, subprocess
import numpy as np
from collections import Counter

VOY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(VOY, 'data', 'v50_ckpt'); OUT = os.path.join(CK, 'out'); SETS = os.path.join(CK, 'sets')
BIN = os.path.join(CK, 'bin', 'v50_tool')
LOOPS = os.path.join(VOY, 'loops')


def load(name, kind='det'):
    r = np.fromfile(os.path.join(OUT, f'{name}.{kind}.bin'), dtype=np.float32).reshape(-1, 5)
    return {'n': r[:, 0], 'mi': r[:, 1], 'mis': r[:, 2], 'nu': r[:, 3], 'r3': r[:, 4], 'D': r[:, 1] - r[:, 2]}


_SPECS = None
def specs():
    """structured array of det path specs (unit, sel, start, fr, mode, j0, m, moves string)."""
    global _SPECS
    if _SPECS is None:
        rows = []
        for line in gzip.open(os.path.join(CK, 'det_specs.txt.gz'), 'rt'):
            rows.append(line.split()[1:])
        _SPECS = rows
    return _SPECS


def spec_cols():
    S = specs()
    unit = np.array([int(r[0][5:]) for r in S]); sel = np.array([int(r[1][4:]) for r in S])
    start = np.array([int(r[2][6:]) for r in S]); m = np.array([int(r[6][2:]) for r in S])
    return unit, sel, start, m


def spec_str(i, S=None):
    S = S or specs(); return ' '.join(S[i])


def noise_scale(Ds, n, groups):
    """per-path sd of a single D value, from the variance across replicate null sets, pooled within bins of
    (group, log2 n)."""
    V = np.var(np.stack(Ds), axis=0, ddof=1)
    b = np.clip(np.log2(np.maximum(n, 1)).astype(int), 0, 30)
    key = groups * 32 + b
    sd = np.zeros_like(V)
    for k in np.unique(key):
        ix = key == k
        sd[ix] = math.sqrt(max(np.mean(V[ix]), 1e-12))
    return sd


def zscores(x, reps, groups, minn=300):
    """z of D_x against the mean of replicates; returns z and null z (last replicate as pseudo-real vs the rest)."""
    Ds = [r['D'] for r in reps]
    sd = noise_scale(Ds, x['n'], groups)
    ref = np.mean(Ds[:-1], axis=0)
    k = len(Ds) - 1
    z = (x['D'] - ref) / (sd * math.sqrt(1 + 1 / k))
    z0 = (Ds[-1] - ref) / (sd * math.sqrt(1 + 1 / k))
    ok = (x['n'] >= minn) & (reps[0]['n'] >= minn)
    z[~ok] = np.nan; z0[~ok] = np.nan
    return z, z0, sd


def write_idx(idx, fn):
    np.asarray(idx, dtype=np.int32).tofile(fn); return fn


def run_sub(setname, idx, tag, kind='subdet'):
    fn = write_idx(idx, os.path.join(CK, f'{tag}.idx'))
    out = os.path.join(OUT, f'{setname}.{tag}.bin')
    subprocess.run([BIN, os.path.join(SETS, setname + '.txt'), out, kind, fn, '0'], check=True,
                   stderr=subprocess.DEVNULL)
    r = np.fromfile(out, dtype=np.float32).reshape(-1, 5)
    return {'n': r[:, 0], 'mi': r[:, 1], 'mis': r[:, 2], 'nu': r[:, 3], 'r3': r[:, 4], 'D': r[:, 1] - r[:, 2]}


def streams(setname, idx, kind='subdet', tag='st'):
    fn = write_idx(idx, os.path.join(CK, f'{tag}.idx'))
    p = subprocess.run([BIN, os.path.join(SETS, setname + '.txt'), 'STREAM', kind, fn, '0'], check=True,
                       capture_output=True, text=True)
    out = {}
    for line in p.stdout.splitlines():
        v = list(map(int, line.split())); out[v[0]] = v[1:]
    return out


# ---------------- battery ----------------
def H(c):
    t = sum(c.values()); return -sum(v / t * math.log2(v / t) for v in c.values() if v)


def pairs(u, lag):
    out = []
    for i in range(len(u) - lag):
        if all(x >= 0 for x in u[i:i + lag + 1]): out.append((u[i], u[i + lag]))
    return out


def battery(u):
    """substitution-invariant features of a unit stream with breaks (<0)."""
    x = [v for v in u if v >= 0]
    c1 = Counter(x); h1 = H(c1)
    p1 = Counter(pairs(u, 1)); p2 = Counter(pairs(u, 2))
    def mi(pc):
        a = Counter(); b = Counter()
        for (s, t), v in pc.items(): a[s] += v; b[t] += v
        return H(a) + H(b) - H(pc)
    def kl(cnt, eps=0.5):
        keys = set(cnt) | {(b, a) for a, b in cnt}
        tot = sum(cnt.values()) + eps * len(keys); s = 0.0
        for (a, b) in keys:
            p = (cnt.get((a, b), 0) + eps) / tot; q = (cnt.get((b, a), 0) + eps) / tot
            s += p * math.log2(p / q)
        return s
    # shuffled baseline (within the whole stream) for bias correction
    rng = random.Random(7); y = [v for v in u]; pos = [i for i, v in enumerate(y) if v >= 0]
    vals = [y[i] for i in pos]; rng.shuffle(vals)
    for i, v in zip(pos, vals): y[i] = v
    m1 = mi(p1); m2 = mi(p2); m1s = mi(Counter(pairs(y, 1))); m2s = mi(Counter(pairs(y, 2)))
    n = len(x); ic = sum(v * (v - 1) for v in c1.values()) / max(1, n * (n - 1))
    top = sorted(c1.values(), reverse=True); K = len(c1)
    return {'n': n, 'K': K, 'h1': h1, 'mi1': m1 - m1s, 'mi2': m2 - m2s, 'rmi1': (m1 - m1s) / max(h1, 1e-9),
            'rmi2': (m2 - m2s) / max(h1, 1e-9), 'arrow': kl(p1) / max(m1, 1e-9), 'ic': ic * K,
            'top3': sum(top[:3]) / max(1, n)}


FEATS = ['h1', 'rmi1', 'rmi2', 'arrow', 'ic', 'top3']


def lang_letter_stream(code, n, seg, skip=0):
    """letters of a reference language as a unit stream (letters -> ints by frequency rank), page breaks every seg."""
    src = {'la': 'plain/la.txt', 'cs': 'plain/cs.txt', 'de': 'pg22367.txt', 'it': 'pg1000.txt', 'es': 'pg2000.txt',
           'en': 'pg47342.txt', 'la2': 'pg218.txt', 'it2': 'pg45334.txt'}[code]
    t = open(os.path.join(VOY, 'data', src), encoding='utf-8', errors='replace').read().lower()
    import unicodedata
    t = unicodedata.normalize('NFD', t); t = ''.join(ch for ch in t if not unicodedata.combining(ch))
    s = re.sub(r'[^a-z]', '', t)
    s = s[len(s) // 10 + skip:]
    s = s[:n]
    rk = {l: i for i, (l, _) in enumerate(Counter(s).most_common())}
    u = []
    for i, ch in enumerate(s):
        if i and i % seg == 0: u.append(-2)
        u.append(min(rk[ch], 23))
    return u


def lang_word_stream(code, n, seg, skip=0, K=63):
    src = {'la': 'plain/la.txt', 'de': 'pg22367.txt', 'it': 'pg1000.txt', 'es': 'pg2000.txt', 'en': 'pg47342.txt'}[code]
    t = open(os.path.join(VOY, 'data', src), encoding='utf-8', errors='replace').read().lower()
    w = re.findall(r'[^\W\d_]+', t); w = w[len(w) // 10 + skip:][:n]
    rk = {x: i for i, (x, _) in enumerate(Counter(w).most_common(K))}
    u = []
    for i, x in enumerate(w):
        if i and i % seg == 0: u.append(-2)
        u.append(rk.get(x, K))
    return u
