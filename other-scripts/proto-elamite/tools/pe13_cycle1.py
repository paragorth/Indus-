"""pe13 cycle 1: fit the shape of the scribe's memory.

For every corpus variant:
  (a) massive random kernel search: NK random kernels (flat / exp / power / step, with
      floor w, refractory rho at d=1, column factor g1, surface factor g2, two clocks),
      lam fitted on 4/5 of tablets, scored on the held-out 1/5 (5 folds).
      Reported: held-out gain of the train-selected best kernel over the flat kernel
      (bits per 1000 target tokens), and per family.
  (b) Nelder-Mead MLE of the interpretable model exp-with-floor (tau, w, rho, g1, g2)
      and of a free step-wise kernel f(d) (bins 1,2,3,4,5,6-8,9-12,13+; f(1)=1).
  (c) copy statistic (cross-tablet ordered adjacent line pairs).
Variants: PE (+ raw-line clock), 20 within-tablet line shuffles of PE, 5 LDA topic
surrogates, planted TOPIC / PRIME / PRIME_REF / RESET / COPY / MIX on PE's skeleton
(2 seeds), LINB, UR3, ARCH_LEX, ARCH_ADM (each with 3 shuffles).
usage: python3 pe13_cycle1.py [workers]
"""
import json, math, os, random, sys, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe13_common as C  # noqa
from scipy.optimize import minimize

OUT = os.path.join(C.CK, 'c1')
os.makedirs(OUT, exist_ok=True)
NK = 1200
BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 8), (9, 12), (13, 40)]


def par_from_vec(v, kind):
    if kind == 'expw':
        tau, w, rho, g1, g2 = v
        rho = min(max(rho, -4.6), 4.6)
        return {'shape': 'exp', 'tau': 0.3 + math.exp(min(tau, 6)), 'w': 1 / (1 + math.exp(-min(max(w, -30), 30))), 'rho': math.exp(rho),
                'g1': math.exp(min(max(g1, -20), 20)), 'g2': math.exp(min(max(g2, -20), 20))}
    f = np.ones(C.D)
    for (a, b), x in zip(BINS[1:], v):
        f[a - 1:b] = math.exp(x)
    return {'shape': 'free', 'f': f}


def nll(v, P, idx, pb, kind):
    K = C.kernel(par_from_vec(v, kind))
    cp = C.cache_prob(P, idx, K)
    return -C.ll_grid(pb, cp).max()


def analyse(name, tabs, clock='ent', nk=NK, seed=0):
    fn = os.path.join(OUT, name + ('' if clock == 'ent' else '_raw') + '.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    t0 = time.time()
    P = C.prep(tabs, clock)
    T = len(P['w'])
    rng = random.Random(1000 + seed)
    kernels = [{'shape': 'flat'}] + [C.random_kernel(rng) for _ in range(nk)]
    FO = C.folds(P['ntab'], 5, seed)
    tr_ll = np.zeros(len(kernels))
    te_ll = np.zeros(len(kernels))
    FD = []
    for f in range(5):
        te_t = FO[f]
        tr_t = np.setdiff1d(np.arange(P['ntab']), te_t)
        tr = np.where(np.isin(P['tab'], tr_t))[0]
        te = np.where(np.isin(P['tab'], te_t))[0]
        pb = C.base_probs(P, tr_t)
        FD.append((tr, te, pb[P['w'][tr]], pb[P['w'][te]]))
    for i, kp in enumerate(kernels):
        cp = C.cache_prob(P, None, C.kernel(kp))
        for tr, te, pbtr, pbte in FD:
            g = C.ll_grid(pbtr, cp[tr])
            li = int(np.argmax(g))
            l = C.LAMS[li]
            tr_ll[i] += g[li]
            te_ll[i] += np.log((1 - l) * pbte + l * cp[te]).sum()
    # selection by train LL (summed over folds); held-out of the selected kernel
    fam = {}
    for i, kp in enumerate(kernels):
        key = kp['shape']
        if key not in fam or tr_ll[i] > tr_ll[fam[key]]:
            fam[key] = i
    best = int(np.argmax(tr_ll))
    L2 = math.log(2)
    res = {'name': name, 'clock': clock, 'T': T, 'ntab': P['ntab'],
           'flat_te_bits': float(-te_ll[0] / T / L2),
           'gain_best_mbits': float((te_ll[best] - te_ll[0]) / T / L2 * 1000),
           'best_kernel': kernels[best],
           'fam_gain_mbits': {k: float((te_ll[i] - te_ll[0]) / T / L2 * 1000) for k, i in fam.items()},
           'fam_kernel': {k: kernels[i] for k, i in fam.items()}}
    # MLE on all data (lam profiled), interpretable model and free kernel
    idx = None
    pb = C.base_probs(P, np.arange(P['ntab']))[P['w']]
    flat_ll = C.ll_grid(pb, C.cache_prob(P, idx, C.kernel({'shape': 'flat'})))
    res['flat_lam'] = float(C.LAMS[int(np.argmax(flat_ll))])
    best_v, best_f = None, 1e18
    for x0 in ([0.4, 0.0, 0.0, 0.0, 0.0], [1.5, -1.0, -0.5, 0.0, -0.5], [3.0, 1.0, 0.3, -0.3, 0.3]):
        r = minimize(nll, x0, args=(P, idx, pb, 'expw'), method='Nelder-Mead',
                     options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-3})
        if r.fun < best_f:
            best_f, best_v = r.fun, r.x
    pe = par_from_vec(best_v, 'expw')
    res['expw'] = {k: float(v) for k, v in pe.items() if k != 'shape'}
    res['expw_gain_mbits_insample'] = float((-best_f - flat_ll.max()) / T / L2 * 1000)
    r = minimize(nll, np.zeros(len(BINS) - 1), args=(P, idx, pb, 'free'), method='Nelder-Mead',
                 options={'maxiter': 1500, 'xatol': 1e-3, 'fatol': 1e-3})
    res['free_f'] = [float(math.exp(x)) for x in r.x]
    res['free_gain_mbits_insample'] = float((-r.fun - flat_ll.max()) / T / L2 * 1000)
    res['copy'] = C.copy_stat(tabs)
    res['sec'] = time.time() - t0
    json.dump(res, open(fn, 'w'))
    print(name, clock, 'T', T, 'gain', round(res['gain_best_mbits'], 1), 'tau', round(pe['tau'], 2),
          'w', round(pe['w'], 2), 'rho', round(pe['rho'], 2), 'g2', round(pe['g2'], 2),
          'copy', round(res['copy'], 4), round(res['sec']), 's', flush=True)
    return res


def lda_surrogate(tabs, seed, K=15):
    from sklearn.decomposition import LatentDirichletAllocation
    vocab = {}
    for t in tabs:
        for l in t['lines']:
            for x in l['toks']:
                vocab.setdefault(x, len(vocab))
    X = np.zeros((len(tabs), len(vocab)))
    for i, t in enumerate(tabs):
        for l in t['lines']:
            for x in l['toks']:
                X[i, vocab[x]] += 1
    lda = LatentDirichletAllocation(K, random_state=seed, learning_method='batch', max_iter=50)
    th = lda.fit_transform(X)
    beta = lda.components_ / lda.components_.sum(1, keepdims=True)
    inv = {v: k for k, v in vocab.items()}
    rng = np.random.RandomState(seed)
    out = []
    for i, t in enumerate(tabs):
        p = th[i] @ beta
        p = p / p.sum()
        NL = []
        for l in t['lines']:
            x = dict(l)
            x['toks'] = [inv[j] for j in rng.choice(len(vocab), len(l['toks']), p=p)]
            x['mid'] = x['toks']
            NL.append(x)
        out.append({'id': t['id'], 'lines': NL})
    return out


def cap(tabs, n, seed):
    if len(tabs) <= n:
        return tabs
    return random.Random(seed).sample(tabs, n)


def jobs():
    CO = C.corpora()
    pe = CO['PE']
    from collections import Counter
    uni = Counter(x for t in pe for l in t['lines'] for x in l['toks'])
    J = [('PE', pe, 'ent'), ('PE', pe, 'raw')]
    for k in ('TOPIC', 'PRIME', 'PRIME_REF', 'RESET', 'COPY', 'MIX'):
        for s in (1, 2):
            J.append(('PL_%s_%d' % (k, s), C.gen_planted(pe, k, s, uni), 'ent'))
    for k in ('LINB', 'ARCH_LEX', 'ARCH_ADM', 'UR3'):
        tb = cap(CO[k], 1000, 5)
        J.append((k, tb, 'ent'))
        for s in range(3):
            J.append(('%s_shuf%d' % (k, s), C.shuffle_lines(tb, random.Random(50 + s)), 'ent'))
    for s in range(20):
        J.append(('PE_shuf%02d' % s, C.shuffle_lines(pe, random.Random(s)), 'ent'))
    for s in range(5):
        J.append(('PE_lda%d' % s, lda_surrogate(pe, s), 'ent'))
    return J


def run(j):
    name, tabs, clock = j
    return analyse(name, tabs, clock)


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    J = jobs()
    with Pool(nw) as p:
        R = p.map(run, J, chunksize=1)
    json.dump(R, open(os.path.join(C.CK, 'c1_all.json'), 'w'))
