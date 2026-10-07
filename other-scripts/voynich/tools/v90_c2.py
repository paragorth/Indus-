"""v90 cycle 2: random Lullian 'slot alphabets' (each field's glyphs coarsened into k random concept
classes = homophone groups) x random wheel decompositions x 3 anchor schemes; every hypothesis is
scored for a CONTENT wheel on TRAIN folios (rare-value same-page recurrence of the wheel, minus that
of the full-resolution rest of the word); the top 20 are re-tested on TEST folios. Same pipeline on
shuffles, Markov text, concept plants (plain and homophonic) and habit plants."""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v90_lib as L
import v90_c1 as C1

NH = int(os.environ.get('V90_NH', '4000'))
TOP = 20


def build(name):
    toks, order = L.voynich_tokens('ZL3b')
    if name == 'C_HERBH':
        return L.plant(toks, order, 'concept', 'konrad_plants', seed=13, drift=0.5, unit='page', A=L.LULL18, homo=True), order
    if name == 'C_ASTRH':
        return L.plant(toks, order, 'concept', 'hyginus_astr', seed=14, drift=0.5, unit='para', A=L.LULL18, homo=True), order
    if name == 'C_HERB3':
        return L.plant(toks, order, 'concept', 'konrad_plants', seed=23, drift=0.5, unit='page', A=L.LULL18, npos=3), order
    if name == 'C_ASTR3':
        return L.plant(toks, order, 'concept', 'hyginus_astr', seed=24, drift=0.5, unit='para', A=L.LULL18, npos=3), order
    if name == 'HABIT3':
        return L.plant(toks, order, 'habit', seed=26, drift=0.5, A=L.LULL18, npos=3, hstr=0.6), order
    if name == 'HABITH':
        return L.plant(toks, order, 'habit', seed=16, drift=0.5, A=L.LULL18, homo=True), order
    return C1.build(name)


def coarse_F(enc, cm):
    return np.stack([enc.F[:, j] if cm[j] is None else np.asarray(cm[j])[enc.F[:, j]] for j in range(L.NF)], 1)


def run(name):
    t0 = time.time()
    toks, order = build(name)
    tr, te = L.split_pages(order)
    encs = [L.Enc(toks, order, sc) for sc in (0, 1, 2)]
    tri = set(order.index(p) for p in tr); tei = set(order.index(p) for p in te)
    rng = np.random.default_rng(902)
    cands = []   # (lift z train, hyp idx, wheel)
    hyps = []
    for h in range(NH):
        sc = int(rng.integers(3)); enc = encs[sc]
        d = L.random_decomp(rng)
        cm = [None if rng.random() < 0.5 else rng.integers(0, int(rng.integers(2, 9)), size=len(enc.fvocab[j])).tolist()
              for j in range(L.NF)]
        hyps.append({'h': h, 'scheme': sc, 'decomp': d, 'cmap': cm})
        Fc = coarse_F(enc, cm)
        for cols in d:
            if len(cols) < 2 or len(cols) == L.NF:
                continue
            r = L.recur_lift(enc, L._codes(Fc[:, cols]), tri)
            if r['n'] >= 50:
                cands.append((r['z'], h, tuple(cols), r['lift']))
    cands.sort(key=lambda x: -x[0])
    stage2 = []
    for lz, h, cols, lift in cands[:200]:
        x = hyps[h]; enc = encs[x['scheme']]; Fc = coarse_F(enc, x['cmap'])
        cj = L.conjunction(enc, Fc[:, list(cols)], tri, R=4, seed=h)
        stage2.append({'h': h, 'cols': list(cols), 'lz_train': lz, 'cz_train': cj['z'], 'lift_train': lift})
    stage2.sort(key=lambda x: -x['cz_train'])
    top = stage2[:TOP]
    for x in top:
        hy = hyps[x['h']]; enc = encs[hy['scheme']]; Fc = coarse_F(enc, hy['cmap'])
        cj = L.conjunction(enc, Fc[:, x['cols']], tei, R=16, seed=7 + x['h'])
        x.update({'scheme': hy['scheme'], 'decomp': hy['decomp'], 'cmap': hy['cmap'], 'test': cj,
                  'fields': [L.FIELDS[c] + ('' if hy['cmap'][c] is None else '/%d' % (max(hy['cmap'][c]) + 1)) for c in x['cols']]})
    res = {'name': name, 'n_hyp': NH, 'n_wheels': len(cands), 'top': top,
           'lz_train_dist': [c[0] for c in cands[:1000]], 'secs': time.time() - t0}
    json.dump(res, open(os.path.join(L.CK, 'c2_%s.json' % name), 'w'), default=float)
    return name, res['secs']


if __name__ == '__main__':
    names = sys.argv[1:] or ['VOY', 'SHUF0', 'MARK', 'C_HERB3', 'C_ASTR3', 'HABIT3', 'C_HERBH', 'HABITH', 'VOY_IT', 'SHUF1']
    with Pool(2) as P:
        for n, s in P.imap_unordered(run, names):
            print(n, '%.0fs' % s, flush=True)
