"""v92 cycle 2 (K6): NAMES FADE.
A name of a thing is introduced at the head of its entry and then mentioned less and less (pronouns take over); a
spelling mood or a drifting habit is stationary along the page. For each random VIEW of the words (keep the first a
and last b glyph units, optional random glyph merge classes, rarity band, minimum recurrence on the page, unit of
text = page or paragraph, statistic = occurrence centroid / first occurrence / early-late ratio), the front-loading
of recurring units is compared with permutations of the unit's line order that keep the first line (and every
paragraph-start line) in place. Train folios (leaf parity 0) select, held-out folios test (once)."""
import os, sys, json, time, math, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v92_lib as L

NH = int(os.environ.get('V92_NH2', '2000'))
TOP = 20
GLY = None


def random_view(rng, glyphs):
    a = int(rng.integers(0, 5)); b = int(rng.integers(0, 5))
    if a + b < 2: a, b = 2, 1
    merge = None
    if rng.random() < 0.3:
        k = int(rng.integers(4, 12)); merge = {g: int(rng.integers(k)) for g in glyphs}
    return dict(a=a, b=b, merge=merge, lo=int(rng.choice([2, 3])), hi=int(rng.choice([5, 10, 20, 50, 200])),
                mrec=int(rng.choice([2, 2, 3])), level=str(rng.choice(['page', 'para'])),
                stat=str(rng.choice(['centroid', 'first', 'early'])))


def unitf(v):
    a, b, m = v['a'], v['b'], v['merge']
    def f(w):
        if m is not None: w = ''.join(chr(65 + m.get(c, 0)) for c in w)
        if len(w) <= a + b: return w
        return w[:a] + '.' + (w[-b:] if b else '')
    return f


def blocks(pages, level):
    """-> list of (half, [lines]) text units; each line = (is_fixed, [words]). fixed = first line or a para start."""
    out = []
    for p in pages:
        h = L.split_half(p['id'])
        if level == 'page':
            ls = [(li == 0 or l['ps'], l['w']) for li, l in enumerate(p['lines'])]
            if len(ls) >= 4: out.append((h, ls))
        else:
            cur = []
            for l in p['lines']:
                if l['ps'] and cur:
                    if len(cur) >= 4: out.append((h, cur))
                    cur = []
                cur.append((not cur, l['w']))
            if len(cur) >= 4: out.append((h, cur))
    return out


def prep(B, f, lo, hi, mrec):
    """per block: line-level unit sets; recurring units = count lo..hi in corpus, on >= mrec lines of the block."""
    cnt = Counter(f(w) for _, ls in B for _, ws in ls for w in ws)
    P = []
    for h, ls in B:
        sets = [set(f(w) for w in ws) for _, ws in ls]
        occ = defaultdict(list)
        for i, s in enumerate(sets):
            for u in s:
                if lo <= cnt[u] <= hi: occ[u].append(i)
        rec = {u: v for u, v in occ.items() if len(v) >= mrec}
        if not rec: continue
        n = len(ls)
        M = np.zeros((len(rec), n), bool)
        for j, (u, v) in enumerate(rec.items()): M[j, v] = True
        fixed = np.array([fx for fx, _ in ls])
        P.append((h, M, fixed))
    return P


def stat(M, order, kind):
    """M rows = recurring units, columns = line positions (in the given order). Front-loading, larger = earlier."""
    X = M[:, order]
    n = X.shape[1]
    pos = np.arange(n) / (n - 1)
    if kind == 'centroid':
        return float((0.5 - (X * pos).sum(1) / X.sum(1)).sum())
    if kind == 'first':
        return float((0.5 - pos[X.argmax(1)]).sum())
    h = n // 2
    return float(((X[:, :h].sum(1) - X[:, n - h:].sum(1)) / X.sum(1)).sum())


def perm_order(fixed, rng):
    idx = np.arange(len(fixed))
    free = idx[~fixed]
    o = idx.copy(); o[~fixed] = rng.permutation(free)
    return o


def score(P, half, kind, R, seed):
    rng = np.random.default_rng(seed)
    blocks_ = [(M, fx) for h, M, fx in P if h == half]
    if not blocks_: return None
    obs = sum(stat(M, np.arange(M.shape[1]), kind) for M, fx in blocks_)
    nul = np.array([sum(stat(M, perm_order(fx, rng), kind) for M, fx in blocks_) for _ in range(R)])
    sd = max(nul.std(ddof=1), 1e-6)
    nunits = sum(M.shape[0] for M, _ in blocks_)
    return dict(z=float((obs - nul.mean()) / sd), obs=obs, mu=float(nul.mean()), n=nunits, eff=float((obs - nul.mean()) / max(nunits, 1)))


def run(name):
    out_p = os.path.join(L.CK, 'c2_%s.json' % name)
    if os.path.exists(out_p): return name, 0.0
    t0 = time.time()
    pages = L.corpus(name)
    glyphs = sorted({c for p in pages for l in p['lines'] for w in l['w'] for c in w})
    Bs = {lev: blocks(pages, lev) for lev in ('page', 'para')}
    rng = np.random.default_rng(926)
    rows = []
    for hi_ in range(NH):
        v = random_view(rng, glyphs)
        P = prep(Bs[v['level']], unitf(v), v['lo'], v['hi'], v['mrec'])
        r = score(P, 0, v['stat'], 12, hi_)
        if r is None or r['n'] < 30: continue
        rows.append(dict(v=v, tr=r))
    rows.sort(key=lambda r: -r['tr']['z'])
    top = []
    for r in rows[:TOP]:
        v = r['v']
        P = prep(Bs[v['level']], unitf(v), v['lo'], v['hi'], v['mrec'])
        r['te'] = score(P, 1, v['stat'], 200, 7)
        top.append(r)
    # fixed reference view (frame, v90 definition) for every corpus, both halves
    ref = {}
    for lev in ('page', 'para'):
        P = prep(Bs[lev], L.frame, 2, 20, 2)
        for k in ('centroid', 'first', 'early'):
            ref['%s_%s' % (lev, k)] = [score(P, h, k, 200, 11 + h) for h in (0, 1)]
    res = dict(name=name, nh=len(rows), top=top, ref=ref, tr_z=[r['tr']['z'] for r in rows], secs=time.time() - t0)
    json.dump(res, open(out_p, 'w'), default=str)
    return name, res['secs']


NAMES = ['E_KONRAD', 'ZL3b', 'E_CIRCA', 'G_LX', 'E_APIC', 'IT2a', 'E_HYGIN', 'G_SEED', 'E_CULP', 'GC2a', 'G_SELF', 'G_GM',
         'ZL3b_WPS', 'E_KONRAD_WPS']

if __name__ == '__main__':
    from multiprocessing import Pool
    names = sys.argv[1:] or NAMES
    for n in names: L.corpus(n)
    with Pool(2) as P:
        for n, s in P.imap_unordered(run, names):
            print(n, '%.0fs' % s, flush=True)
