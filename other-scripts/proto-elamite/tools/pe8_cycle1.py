"""pe8 cycle 1: how many forms?  Cluster tablet skeletons; calibrate on planted and null corpora.

Corpora (each a list of skeleton feature dicts, same size as the real set):
  REAL       1,119 tablets with >= 2 obverse entries (damage = missing features, marginalised)
  CLEAN      289 tablets with no damage at all
  PLANT_s    written from 6 fixed forms (seeds 1-5), 85% per-field fidelity, real damage pattern
  NULL_s     layout drawn independently per entry and per tablet field group (seeds 1-5)
  SHUF_s     real features, each column permuted across tablets (seeds 1-3)
Models: latent class model (EM, K = 1..12, 12 restarts): BIC (= two-part MDL) and 5-fold held-out log-lik;
        Dirichlet-process mixture (collapsed Gibbs, alpha ~ Gamma(1,1), 2 chains x 200 sweeps): clusters
        holding >= 1% of tablets.
Checkpoint: data/pe8_ckpt/c1_<name>.json.  At most 2 workers.
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe8_common import *

CK = os.path.join(DATA, 'pe8_ckpt')
os.makedirs(CK, exist_ok=True)
KMAX, RESTARTS, CVR = 12, 12, 5


def ari(a, b):
    from math import comb
    a = np.asarray(a); b = np.asarray(b)
    ct = Counter(zip(a, b))
    sij = sum(comb(v, 2) for v in ct.values())
    sa = sum(comb(v, 2) for v in Counter(a).values()); sb = sum(comb(v, 2) for v in Counter(b).values())
    n2 = comb(len(a), 2)
    e = sa * sb / n2
    return (sij - e) / (0.5 * (sa + sb) - e)


def job(args):
    name, rows, lab = args
    fn = os.path.join(CK, 'c1_%s.json' % name)
    if os.path.exists(fn):
        return json.load(open(fn))
    t0 = time.time()
    X, lev = encode(rows)
    card = [max(1, len(lev[k])) for k in FEATS]
    n = len(X)
    res = {'name': name, 'n': n, 'bic': {}, 'cv': {}, 'll': {}}
    best = {}
    for K in range(1, KMAX + 1):
        m = best_em(X, card, K, RESTARTS, 100 + K)
        best[K] = m
        res['ll'][K] = m['ll']
        res['bic'][K] = -2 * m['ll'] + nparams(card, K) * np.log(n)
        res['cv'][K] = cv_ll(X, card, K, CVR, 500 + K)
    kb = min(res['bic'], key=res['bic'].get)
    cvv = np.array([res['cv'][K] for K in range(1, KMAX + 1)])
    kc = int(np.argmax(cvv)) + 1
    res['K_bic'], res['K_cv'] = kb, kc
    # smallest K whose CV log-lik is within 0.01 nats/tablet of the best (parsimony)
    res['K_cv1'] = int(np.where(cvv >= cvv.max() - 0.01)[0][0]) + 1
    dps = [dp_gibbs(X, card, sweeps=200, burn=100, seed=s) for s in (1, 2)]
    res['dp_Kbig'] = [float(np.median(d['Kbig'])) for d in dps]
    res['dp_K'] = [float(np.median(d['K'])) for d in dps]
    zb = best[kb]['R'].argmax(1)
    res['sizes_bic'] = sorted(Counter(zb.tolist()).values(), reverse=True)
    if lab is not None:
        res['ari_bic'] = ari(zb, lab)
        res['ari_dp'] = [ari(d['z'], lab) for d in dps]
        res['ari_true6'] = ari(best[6]['R'].argmax(1), lab)
    res['secs'] = time.time() - t0
    json.dump(res, open(fn, 'w'))
    print(name, 'K_bic', kb, 'K_cv', kc, 'K_cv1', res['K_cv1'], 'DP', res['dp_Kbig'], 'secs %.0f' % res['secs'],
          flush=True)
    return res


def main():
    R = load_skeletons(min_ent=2, clean=False)
    Rc = load_skeletons(min_ent=2, clean=True)
    A = [r['A'] for r in R]
    jobs = [('REAL', [r['f'] for r in R], None), ('CLEAN', [r['f'] for r in Rc], None)]
    for s in range(1, 6):
        P, lab, modes = planted_corpus(A, len(A), seed=s)
        jobs.append(('PLANT_%d' % s, [features(a) for a in P], lab))
    for s in range(1, 6):
        jobs.append(('NULL_%d' % s, [features(a) for a in null_corpus(A, len(A), seed=s)], None))
    Xr, lev = encode([r['f'] for r in R])
    for s in range(1, 4):
        Y = column_shuffle(Xr, s)
        rows = [{k: (lev[k][Y[i, j]] if Y[i, j] >= 0 else None) for j, k in enumerate(FEATS)} for i in range(len(Y))]
        jobs.append(('SHUF_%d' % s, rows, None))
    with Pool(2) as p:
        out = p.map(job, jobs, chunksize=1)
    json.dump(out, open(os.path.join(DATA, 'pe8_cycle1.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
