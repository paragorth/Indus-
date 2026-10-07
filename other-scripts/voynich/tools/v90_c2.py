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
    if name == 'HABITH':
        return L.plant(toks, order, 'habit', seed=16, drift=0.5, A=L.LULL18, homo=True), order
    return C1.build(name)


def score(enc, Fc, decomp, pages):
    best = (-99.0, None)
    det = []
    for cols in decomp:
        if len(cols) == L.NF:
            continue
        v = L.np.unique(Fc[:, cols], axis=0, return_inverse=True)[1].ravel() if len(cols) > 1 else Fc[:, cols[0]]
        r = L.recur_lift(enc, v, pages)
        comp = [c for c in range(L.NF) if c not in cols]
        rc = L.recur_lift(enc, enc.wheel_codes(comp), pages)
        s = r['z'] - max(rc['z'], 0.0) if r['n'] >= 50 else -99.0
        det.append({'cols': cols, 'lift': r['lift'], 'z': r['z'], 'n': r['n'], 'clift': rc['lift'], 'cz': rc['z'], 's': s})
        if s > best[0]:
            best = (s, cols)
    return best[0], det


def run(name):
    t0 = time.time()
    toks, order = build(name)
    tr, te = L.split_pages(order)
    encs = [L.Enc(toks, order, sc) for sc in (0, 1, 2)]
    tri = set(order.index(p) for p in tr); tei = set(order.index(p) for p in te)
    rng = np.random.default_rng(902)
    hyps = []
    for h in range(NH):
        sc = int(rng.integers(3)); enc = encs[sc]
        d = L.random_decomp(rng)
        cm = []
        for j in range(L.NF):
            nv = len(enc.fvocab[j])
            if rng.random() < 0.5:
                cm.append(None)
            else:
                cm.append(rng.integers(0, int(rng.integers(2, 9)), size=nv).tolist())
        Fc = np.stack([enc.F[:, j] if cm[j] is None else np.asarray(cm[j])[enc.F[:, j]] for j in range(L.NF)], 1)
        s, det = score(enc, Fc, d, tri)
        hyps.append({'h': h, 'scheme': sc, 'decomp': d, 'cmap': cm, 's_train': s, 'det_train': det})
    hyps.sort(key=lambda x: -x['s_train'])
    top = hyps[:TOP]
    for x in top:
        enc = encs[x['scheme']]
        Fc = np.stack([enc.F[:, j] if x['cmap'][j] is None else np.asarray(x['cmap'][j])[enc.F[:, j]] for j in range(L.NF)], 1)
        x['s_test'], x['det_test'] = score(enc, Fc, x['decomp'], tei)
    strain = [x['s_train'] for x in hyps]
    res = {'name': name, 'n_hyp': NH, 'top': top, 'train_scores': strain, 'secs': time.time() - t0}
    json.dump(res, open(os.path.join(L.CK, 'c2_%s.json' % name), 'w'), default=float)
    return name, res['secs']


if __name__ == '__main__':
    names = sys.argv[1:] or ['VOY', 'SHUF0', 'MARK', 'C_HERB', 'C_HERBH', 'C_ASTRH', 'HABIT', 'HABITH', 'VOY_IT', 'SHUF1']
    with Pool(2) as P:
        for n, s in P.imap_unordered(run, names):
            print(n, '%.0fs' % s, flush=True)
