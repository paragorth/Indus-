"""v15 cycle 1: do word-length sequences carry more structure than the words imply?

For each text: length stream (clipped 1..10) in reading order. Statistics:
  mi_in    MI(L_t; L_t+1) for neighbours inside a line
  mi_x     MI(last length of line; first length of next line), same page
  h2       conditional entropy H(L_t | L_t-1, L_t-2) of the whole stream (plug-in)
  rep10    distinct length 10-grams seen twice or more
  maxrep   longest repeated length substring
  spec     max over 120 frequency bands (periods 2..64) of periodogram / null mean
  prank    is length a privileged vocabulary partition? mi_in excess (over
           within-line shuffle) of the length partition vs 60 random partitions
           of the vocabulary with the same class token masses (rank, 1 = top)
Nulls (all keep every word, so all glyph statistics): words shuffled within
line, within page, within section (line word counts kept). 40 / 20 / 20 reps.
Gap channel (ZL3b only: '.' vs ',' uncertain space): MI of consecutive gap
types in a line vs gaps shuffled within line.
Words per line: MI of consecutive line word counts inside a paragraph (middle
lines only) vs line order shuffled within page.
Texts: Voynich ZL3b (glyph units, EVA chars, strokes), IT2a, Currier A, B;
planted Latin in lengths ('len', 'pair', 'bacon2'); Caesar; Manzoni.
"""
import os, sys, json, math
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v15_lib as V

OUT = os.path.join(V.RES, 'cycle1')
os.makedirs(OUT, exist_ok=True)
CL = 10


def mi_pairs(x, y, K=CL + 1):
    if len(x) == 0:
        return 0.0
    c = np.zeros((K, K)); np.add.at(c, (x, y), 1)
    p = c / c.sum(); px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    m = p > 0
    return float((p[m] * np.log2(p[m] / (px @ py)[m])).sum())


def line_codes(lines, code):
    return [np.array([code(w) for w in L['words']], dtype=np.int64) for L in lines]


def mi_in(seqs):
    x = np.concatenate([s[:-1] for s in seqs if len(s) > 1]); y = np.concatenate([s[1:] for s in seqs if len(s) > 1])
    return mi_pairs(x, y, int(max(x.max(), y.max())) + 1)


def mi_x(seqs, lines):
    xs, ys = [], []
    for i in range(1, len(seqs)):
        if lines[i]['folio'] == lines[i - 1]['folio']:
            xs.append(seqs[i - 1][-1]); ys.append(seqs[i][0])
    return mi_pairs(np.array(xs), np.array(ys))


def h2(stream):
    s = np.asarray(stream)
    c3 = Counter(zip(s[:-2].tolist(), s[1:-1].tolist(), s[2:].tolist()))
    c2 = Counter(zip(s[:-2].tolist(), s[1:-1].tolist()))
    n = len(s) - 2
    return -sum(v / n * math.log2(v / c2[(a, b)]) for (a, b, _), v in c3.items())


def repeats(stream, k=10):
    s = np.asarray(stream)
    S = s.max() + 1
    def grams(k):
        h = np.zeros(len(s) - k + 1, dtype=object) if k > 18 else np.zeros(len(s) - k + 1, dtype=np.int64)
        for j in range(k):
            h = h * S + s[j:len(s) - k + 1 + j]
        return h
    g = grams(k); _, c = np.unique(g, return_counts=True)
    rep = int((c >= 2).sum())
    kk = k
    while True:
        g = grams(kk + 1); _, c = np.unique(g, return_counts=True)
        if (c >= 2).sum() == 0 or kk > 60:
            break
        kk += 1
    return rep, kk if rep else 0


BANDS = None


def spectrum(stream):
    x = np.asarray(stream, float); x = x - x.mean()
    N = len(x)
    P = np.abs(np.fft.rfft(x)) ** 2 / N
    f = np.fft.rfftfreq(N)
    edges = np.linspace(1 / 64, 0.5, 121)
    b = np.digitize(f, edges) - 1
    ok = (b >= 0) & (b < 120)
    return np.bincount(b[ok], weights=P[ok], minlength=120) / np.maximum(np.bincount(b[ok], minlength=120), 1)


def stats(lines, fn):
    seqs = line_codes(lines, lambda w: min(fn(w), CL))
    stream = np.concatenate(seqs)
    rep, mr = repeats(stream)
    return {'mi_in': mi_in(seqs), 'mi_x': mi_x(seqs, lines), 'h2': h2(stream), 'rep10': rep,
            'maxrep': mr}, spectrum(stream)


def random_partition_codes(lines, fn, rng, n_part):
    toks = [w for L in lines for w in L['words']]
    cnt = Counter(toks)
    lens = Counter(min(fn(w), CL) for w in toks)
    classes = sorted(lens); masses = [lens[c] for c in classes]
    parts = []
    for _ in range(n_part):
        vocab = list(cnt); rng.shuffle(vocab)
        m = {}; ci, acc = 0, 0
        for w in vocab:
            m[w] = classes[ci]; acc += cnt[w]
            if acc >= sum(masses[:ci + 1]) and ci < len(classes) - 1:
                ci += 1
        parts.append(m)
    return parts


def prank(lines, fn, rng, n_part=60, n_shuf=4):
    def excess(code):
        real = mi_in(line_codes(lines, code))
        nul = [mi_in(line_codes(V.shuffle_lines(lines, 'line', rng), code)) for _ in range(n_shuf)]
        return real - float(np.mean(nul))
    e_len = excess(lambda w: min(fn(w), CL))
    e_rand = [excess(lambda w, m=m: m[w]) for m in random_partition_codes(lines, fn, rng, n_part)]
    e_last = excess(lambda w: hash(w[-1]) % CL)
    e_first = excess(lambda w: hash(w[0]) % CL)
    return {'len_excess': e_len, 'rand_mean': float(np.mean(e_rand)), 'rand_sd': float(np.std(e_rand)),
            'rand_max': float(np.max(e_rand)), 'rank': int(1 + sum(e > e_len for e in e_rand)),
            'n_part': n_part, 'last_glyph_excess': e_last, 'first_glyph_excess': e_first}


def gap_test(lines, rng, n=200):
    def mi(ls):
        seqs = [np.array([1 if g == ',' else 0 for g in L['gaps'] if g != '-']) for L in ls]
        seqs = [s for s in seqs if len(s) > 1]
        x = np.concatenate([s[:-1] for s in seqs]); y = np.concatenate([s[1:] for s in seqs])
        return mi_pairs(x, y, 2), float(np.concatenate(seqs).mean())
    real, rate = mi(lines)
    nul = []
    for _ in range(n):
        ls = []
        for L in lines:
            g = L['gaps'][:]; rng.shuffle(g); d = dict(L); d['gaps'] = g; ls.append(d)
        nul.append(mi(ls)[0])
    return {'comma_rate': rate, 'mi': real, 'null_mean': float(np.mean(nul)), 'null_sd': float(np.std(nul)),
            'z': (real - np.mean(nul)) / (np.std(nul) + 1e-12)}


def wpl_test(lines, rng, n=200):
    def mi(ls):
        xs, ys = [], []
        for i in range(1, len(ls)):
            a, b = ls[i - 1], ls[i]
            if a['folio'] != b['folio'] or a['para_start'] or b['para_end'] or b['para_start'] or a['para_end']:
                continue
            xs.append(min(len(a['words']), 15)); ys.append(min(len(b['words']), 15))
        return mi_pairs(np.array(xs), np.array(ys), 16)
    real = mi(lines)
    pages = defaultdict(list)
    for i, L in enumerate(lines):
        pages[L['folio']].append(i)
    nul = []
    for _ in range(n):
        ls = list(lines)
        for f, idx in pages.items():
            mid = [i for i in idx if not lines[i]['para_start'] and not lines[i]['para_end']]
            perm = list(mid); rng.shuffle(perm)
            for i, j in zip(mid, perm):
                ls[i] = lines[j]
        nul.append(mi(ls))
    return {'mi': real, 'null_mean': float(np.mean(nul)), 'null_sd': float(np.std(nul)),
            'z': (real - np.mean(nul)) / (np.std(nul) + 1e-12)}


def run(job):
    name, fnname = job
    path = os.path.join(OUT, f'{name}_{fnname}.json')
    if os.path.exists(path):
        return json.load(open(path))
    rng = np.random.default_rng(abs(hash(path)) % 2**32)
    lines = TEXTS[name]()
    fn = {'g': V.glen, 'c': V.clen, 's': V.slen}[fnname]
    real, rspec = stats(lines, fn)
    res = {'text': name, 'unit': fnname, 'n_words': sum(len(L['words']) for L in lines), 'real': real, 'nulls': {}}
    for level, reps in (('line', 40), ('page', 20), ('section', 20)):
        vals = defaultdict(list); specs = []
        for _ in range(reps):
            s, sp = stats(V.shuffle_lines(lines, level, rng), fn)
            for k, v in s.items():
                vals[k].append(v)
            specs.append(sp)
        specs = np.array(specs); mu = specs.mean(0)
        null_max = [float((sp / np.mean(np.delete(specs, i, 0), 0)).max()) for i, sp in enumerate(specs)]
        r = {k: {'mean': float(np.mean(v)), 'sd': float(np.std(v)),
                 'z': float((real[k] - np.mean(v)) / (np.std(v) + 1e-12))} for k, v in vals.items()}
        rmax = float((rspec / mu).max())
        r['spec'] = {'real_max_ratio': rmax, 'period_at_max': float(1 / np.linspace(1 / 64, 0.5, 121)[int(np.argmax(rspec / mu))]),
                     'null_max_mean': float(np.mean(null_max)), 'null_max_max': float(np.max(null_max)),
                     'p': float((1 + sum(m >= rmax for m in null_max)) / (1 + len(null_max)))}
        res['nulls'][level] = r
    res['prank'] = prank(lines, fn, rng)
    if name == 'ZL3b':
        res['gaps'] = gap_test(lines, rng)
    res['wpl'] = wpl_test(lines, rng)
    json.dump(res, open(path, 'w'), indent=1)
    return res


def _zl():
    return V.load_v('ZL3b')


def _plant(s):
    def f():
        return V.plant(V.load_v('ZL3b'), V.caesar_letters(), s, np.random.default_rng(7))
    return f


TEXTS = {
    'ZL3b': _zl,
    'IT2a': lambda: V.load_v('IT2a'),
    'ZL3b_A': lambda: [L for L in _zl() if L['lang'] == 'A'],
    'ZL3b_B': lambda: [L for L in _zl() if L['lang'] == 'B'],
    'plant_len': _plant('len'),
    'plant_pair': _plant('pair'),
    'plant_bacon2': _plant('bacon2'),
    'Caesar': lambda: V.ref_lines('Latin-Caesar'),
    'Manzoni': lambda: V.ref_lines('Italian-Manzoni'),
}

JOBS = [('ZL3b', 'g'), ('ZL3b', 'c'), ('ZL3b', 's'), ('IT2a', 'g'), ('ZL3b_A', 'g'), ('ZL3b_B', 'g'),
        ('plant_len', 'g'), ('plant_pair', 'g'), ('plant_bacon2', 'g'), ('Caesar', 'c'), ('Manzoni', 'c')]

if __name__ == '__main__':
    with Pool(2) as p:
        for r in p.imap_unordered(run, JOBS):
            print(r['text'], r['unit'], 'done', flush=True)
