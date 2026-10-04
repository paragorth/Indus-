"""v23 cycle 2b: is the missing word-level arrow of the Voynich just the absence of function words?

Entropy production of word succession (KL of the lag-k pair law against its transpose, top-300
types + 'other'), lags 1-4, and the same on frequency-rank classes, against the exact-matching
random-page-direction null.  Kill-controls on the real languages:
  strip50  : the 50 most frequent word types deleted (no function words left)
  glue30   : the 30 most frequent types written joined to the following word (Hebrew-like clitics)
  shufl    : words shuffled inside each line (keeps line vocabulary, destroys order) - must give 0
Also: I_intro (cycle 1) recomputed with the first and last line of each paragraph removed.
"""
import sys, os, json, math, random
from collections import Counter
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v23_lib as L
from v23_cycle1 import build

NULL = 200


def transform(C, kind, seed=0):
    gf = L.gfreq(C)
    if kind == 'none': return C
    if kind == 'strip50':
        top = {w for w, _ in gf.most_common(50)}
        return [[[[w for w in l if w not in top] for l in pa] for pa in p] for p in C]
    if kind == 'glue30':
        top = {w for w, _ in gf.most_common(30)}
        def g(l):
            out, pre = [], ''
            for w in l:
                if w in top: pre += w
                else: out.append(pre + w); pre = ''
            if pre: out.append(pre)
            return out
        return [[[g(l) for l in pa] for pa in p] for p in C]
    if kind == 'shufl':
        r = random.Random(seed)
        return [[[r.sample(l, len(l)) for l in pa] for pa in p] for p in C]
    raise KeyError(kind)


def epr(pages, k, top, cls=None):
    c = Counter()
    for p in pages:
        for pa in p:
            t = [w for l in pa for w in l]          # within paragraph, across lines
            if cls: t = [cls.get(w, 'x') for w in t]
            else: t = [w if w in top else '<o>' for w in t]
            c.update(zip(t, t[k:]))
    return L.kl_asym(c)


def run(arg):
    name, kind = arg
    fn = os.path.join(L.CK, f'c2b_{name}_{kind}.json')
    if os.path.exists(fn): return arg
    C = transform(build(name), kind)
    gf = L.gfreq(C)
    top = {w for w, _ in gf.most_common(300)}
    ranks = {w: i for i, (w, _) in enumerate(gf.most_common())}
    cls = {w: (0 if r < 10 else 1 if r < 30 else 2 if r < 100 else 3 if r < 300 else 4 if r < 1000 else 5) for w, r in ranks.items()}
    R = [L.rev_words_only(p) for p in C]
    rng = random.Random(3)
    draws = [[rng.random() < 0.5 for _ in C] for _ in range(NULL)]
    res = {'name': name, 'kind': kind, 'ntok': L.ntok(C), 'cover300': sum(gf[w] for w in top) / sum(gf.values())}
    for lab, f in [(f'W{k}', (lambda P, k=k: epr(P, k, top))) for k in (1, 2, 3, 4)] + \
                  [(f'C{k}', (lambda P, k=k: epr(P, k, top, cls))) for k in (1, 2, 3)]:
        obs = f(C)
        null = np.array([f([R[i] if d[i] else C[i] for i in range(len(C))]) for d in draws])
        res[lab] = {'obs': obs, 'excess': obs - null.mean(), 'z': (obs - null.mean()) / (null.std() + 1e-12)}
    json.dump(res, open(fn, 'w'))
    print(name, kind, ' '.join(f"{k}:{v['excess']:.4f}/{v['z']:.1f}" for k, v in res.items() if isinstance(v, dict)), flush=True)
    return arg


def intro_trim(name):
    """I_intro_para with first and last line of each paragraph removed (layout control)"""
    C = build(name)
    def ir(seq):
        n = len(seq)
        if n < 10: return None
        first, last, cnt = {}, {}, Counter(seq)
        for i, w in enumerate(seq): first.setdefault(w, i); last[w] = i
        v = [(first[w] + last[w]) / (n - 1) - 1 for w in cnt if cnt[w] >= 2]
        return float(np.mean(v)) if v else None
    out = {}
    for lab, trim in (('full', (0, 0)), ('trim1', (1, 1)), ('dropfirst', (1, 0)), ('droplast', (0, 1))):
        a = []
        for p in C:
            vs = []
            for pa in p:
                ls = pa[trim[0]: len(pa) - trim[1]]
                x = ir([w for l in ls for w in l]); y = ir([w for l in ls[::-1] for w in l[::-1]])
                if x is not None: vs.append(x - y)
            if vs: a.append(float(np.mean(vs)))
        z, pv = L.signflip(a)
        out[lab] = {'z': float(z), 'eff': float(np.mean(a)), 'n': len(a)}
    return out


if __name__ == '__main__':
    langs = ['LA', 'ITA', 'DE', 'CS', 'HE']
    jobs = [(n, 'none') for n in ['ZL', 'IT', 'ZL19'] + langs + ['MkW_1', 'MkG_1']] + \
           [(n, k) for n in langs for k in ('strip50', 'glue30', 'shufl')] + [('ZL', 'shufl')]
    with Pool(2) as pool:
        for _ in pool.imap_unordered(run, jobs): pass
    it = {n: intro_trim(n) for n in ['ZL', 'IT', 'LA', 'ITA', 'DE', 'CS', 'HE', 'MkW_1']}
    json.dump(it, open(os.path.join(L.CK, 'c2b_intro.json'), 'w'))
    for n, v in it.items(): print('intro', n, {k: (round(x['z'], 1), round(x['eff'], 4)) for k, x in v.items()})
