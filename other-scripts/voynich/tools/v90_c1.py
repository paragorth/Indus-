"""v90 cycle 1: random wheel decompositions, compactness search on train folios, content-wheel
meaning test on test folios; plants (Lullian concept code in Konrad's herbal and Hyginus'
astronomy laid into Voynich page structure; pure-habit drift) and kills (shuffle within
hand+section, glyph Markov-2 per section x hand)."""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v90_lib as L

NDEC = int(os.environ.get('V90_NDEC', '1000'))   # per anchor scheme
NSURV = 20


def build(name):
    toks, order = L.voynich_tokens('ZL3b')
    if name == 'VOY':
        return toks, order
    if name == 'VOY_IT':
        return L.voynich_tokens('IT2a')
    if name.startswith('SHUF'):
        return L.shuffle_within(toks, seed=int(name[4:] or 0)), order
    if name == 'MARK':
        return L.markov_text(toks, seed=1), order
    if name == 'C_HERB':
        return L.plant(toks, order, 'concept', 'konrad_plants', seed=3, drift=0.5, unit='page', A=L.LULL18), order
    if name == 'C_HERB9':
        return L.plant(toks, order, 'concept', 'konrad_plants', seed=7, drift=0.5, unit='page', A=L.LULL), order
    if name == 'C_ASTR':
        return L.plant(toks, order, 'concept', 'hyginus_astr', seed=4, drift=0.5, unit='para', A=L.LULL18), order
    if name == 'C_HERB30':
        return L.plant(toks, order, 'concept', 'konrad_plants', seed=5, frac=0.3, drift=0.5, unit='page', A=L.LULL18), order
    if name == 'HABIT':
        return L.plant(toks, order, 'habit', seed=6, drift=0.5, A=L.LULL18), order
    if name == 'HABIT_S':
        return L.plant(toks, order, 'habit', seed=8, drift=0.5, A=L.LULL18, hstr=1.0), order
    raise ValueError(name)


def run(name):
    t0 = time.time()
    toks, order = build(name)
    tr, te = L.split_pages(order)
    rng = np.random.default_rng(90)
    res = {'name': name, 'schemes': {}}
    for scheme in (0, 1, 2):
        enc = L.Enc(toks, order, scheme)
        trm = np.isin(enc.page, [enc.page_order.index(p) for p in tr])
        fit = trm & (enc.par == 0); ev = trm & (enc.par == 1)
        seen = {}
        for _ in range(NDEC):
            d = L.random_decomp(rng)
            k = L.canon(d)
            if k in seen:
                continue
            seen[k] = L.compactness(enc, [list(g) for g in k], fit, ev)
        ranked = sorted(seen.items(), key=lambda kv: kv[1])
        # baselines: whole word as one wheel, every field its own wheel
        whole = L.compactness(enc, [list(range(L.NF))], fit, ev)
        indep = L.compactness(enc, [[j] for j in range(L.NF)], fit, ev)
        tei = set(enc.page_order.index(p) for p in te)
        tri = set(enc.page_order.index(p) for p in tr)
        surv = []
        for k, sc in ranked[:NSURV]:
            prof = L.wheel_profile(enc, [list(g) for g in k], tei, tri)
            surv.append({'decomp': [list(g) for g in k], 'bits': sc, 'prof': prof,
                         'spec_self': L.specificity(prof, 'r_self'), 'spec_page': L.specificity(prof, 'r_page'),
                         'spec_para': L.specificity(prof, 'r_para')})
        single = L.wheel_profile(enc, [[j] for j in range(L.NF)], tei, tri)
        wholep = L.wheel_profile(enc, [list(range(L.NF))], tei, tri)
        res['schemes'][scheme] = {'n_scored': len(seen), 'whole_bits': whole, 'indep_bits': indep,
                                  'best_bits': ranked[0][1], 'surv': surv, 'single': single, 'whole': wholep}
    res['secs'] = time.time() - t0
    json.dump(res, open(os.path.join(L.CK, 'c1_%s.json' % name), 'w'), default=float)
    return name, res['secs']


if __name__ == '__main__':
    names = sys.argv[1:] or ['VOY', 'SHUF0', 'MARK', 'C_HERB', 'C_ASTR', 'C_HERB30', 'C_HERB9', 'HABIT', 'HABIT_S', 'VOY_IT', 'SHUF1']
    with Pool(2) as P:
        for n, s in P.imap_unordered(run, names):
            print(n, '%.0fs' % s, flush=True)
