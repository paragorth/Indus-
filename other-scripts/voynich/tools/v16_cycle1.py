"""v16 cycle 1: massive random search over mechanical re-spacing rules.

For every corpus: delete the spaces, draw R random rules from the grammar (before glyph set X, after set Y,
before+after, break between glyph-pair set S, every k glyphs, shift every written space by d, jitter,
drop/insert, random density; each with a minimum-unit-length filter), score each by two-part MDL on the
TRAIN lines (even lines), keep the 25 best, re-score them on the TEST lines (odd lines) with the full battery,
and compare with the written spacing on the same test lines.

Corpora: Voynich ZL3b, IT2a; nulls = ZL glyphs shuffled within line (space positions kept) and an order-2
Markov resynthesis (glyph+space chain, line-wise); negative controls = true-spaced Latin (Caesar) and Italian
(Manzoni); positive controls = the same texts with spaces deleted and re-inserted by a planted rule
(every 5 letters; break vowel->consonant with min unit 3).
"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v16_lib as L

R = int(os.environ.get('V16_R', 20000))
OUT = os.path.join(L.RESDIR, 'cycle1')
os.makedirs(OUT, exist_ok=True)


def build(name):
    v = L.voynich('ZL3b')
    if name == 'V-ZL3b': return v
    if name == 'V-IT2a': return L.voynich('IT2a')
    if name == 'NULL-glyphshuf': return L.shuffle_within_line(v, 1)
    if name == 'NULL-markov2': return L.markov2(v, 1)
    lat = L.reference('Latin-Caesar', v.n); ita = L.reference('Italian-Manzoni', v.n)
    base = lat if name.startswith('Latin') else ita
    if name in ('Latin', 'Italian'): return base
    if name.endswith('every5'):
        return L.planted(base, {'kind': 'every', 'k': 5, 'phase': 0}, name=name)
    if name.endswith('VC'):
        vow = [i for i, ch in enumerate(base.alphabet) if ch in 'aeiouy']
        con = [i for i in range(base.A) if i not in vow]
        return L.planted(base, {'kind': 'pairs', 'S': [(a, b) for a in vow for b in con], 'm': 3}, name=name)
    raise ValueError(name)


NAMES = ['V-ZL3b', 'V-IT2a', 'NULL-glyphshuf', 'NULL-markov2', 'Latin', 'Italian',
         'Latin-planted-every5', 'Latin-planted-VC', 'Italian-planted-every5', 'Italian-planted-VC']


def run(name):
    path = os.path.join(OUT, name + '.json')
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    c = build(name)
    even = np.arange(c.nlines) % 2 == 0
    tr, te = c.subset(even), c.subset(~even)
    rng = np.random.default_rng(16)
    pairs_seen = sorted(set(zip(tr.sym[:-1].tolist(), tr.sym[1:].tolist())))
    w_tr = L.mdl2(tr, tr.written)
    rec = []
    ckpt = os.path.join(OUT, name + '.partial.npz')
    for i in range(R):
        r = L.random_rule(tr, rng, pairs_seen)
        st = L.apply_rule(tr, r, rng)
        m = L.mdl2(tr, st)
        rec.append((m, r))
        if (i + 1) % 5000 == 0:
            print(name, i + 1, round(time.time() - t0), flush=True)
    vals = np.array([m for m, _ in rec])
    kinds = {}
    for m, r in rec:
        kinds.setdefault(r['kind'], []).append(m / tr.n)
    order = np.argsort(vals)[:25]
    srng = np.random.default_rng(7)
    best = []
    for j in order:
        m, r = rec[j]
        st = L.apply_rule(te, r, np.random.default_rng(99))
        sc = L.score(te, st, srng)
        best.append(dict(rule=L.describe(te, r), raw={k: v for k, v in r.items() if k != 'S'},
                         nS=len(r.get('S', [])), train_bpg=m / tr.n, test=sc))
    wtest = L.score(te, te.written, srng)
    out = dict(name=name, R=R, n=c.n, A=c.A, written_train_bpg=w_tr / tr.n, written_test=wtest,
               frac_beat_written_train=float(np.mean(vals < w_tr)),
               best_train_bpg=float(vals.min() / tr.n),
               kind_best={k: float(min(v)) for k, v in kinds.items()},
               kind_frac_beat={k: float(np.mean(np.array(v) < w_tr / tr.n)) for k, v in kinds.items()},
               best=best, secs=time.time() - t0)
    if c.truth is not None:
        out['truth_test'] = L.score(te, te.truth, srng)
    json.dump(out, open(path, 'w'), indent=1, default=float)
    print('done', name, round(time.time() - t0), flush=True)
    return out


if __name__ == '__main__':
    with Pool(2) as p:
        res = p.map(run, NAMES, chunksize=1)
