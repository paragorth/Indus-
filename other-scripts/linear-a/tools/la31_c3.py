"""LA-31 cycle 3: (a) monthly wind fingerprint: which month's sailing times best explain Linear A
sharing (partial on km), against a site-permutation null of the MAXIMUM over 12 months;
(b) the harder leg: do pairs share according to the slower direction (max T) or the faster (min T)?
(c) Linear B comparison (KN, KH, PY, TH, MY, TI, MI; DAMOS words) with the same machinery.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la31_common import SITES, CRETE, OUT, CKPT, LB_SITES, euclid, hav  # noqa
from la31_stats import mantel, sharing_z  # noqa
from scipy.stats import rankdata

rng = np.random.default_rng(3103)
NP = int(os.environ.get('NPERM_M', 5000))


def close(A):
    return -np.log(np.where(np.isfinite(A), A, 1e6) + 0.1)


def partial_r(Z, P, C, iu):
    z = rankdata(Z[iu]); p = rankdata(P[iu]); c = rankdata(C[iu])
    X = np.c_[np.ones_like(c), c]
    rz = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]; rp = p - X @ np.linalg.lstsq(X, p, rcond=None)[0]
    return np.corrcoef(rz, rp)[0, 1]


def month_scan(Z, mats, km, nperm):
    K = Z.shape[0]; iu = np.triu_indices(K, 1)
    obs = np.array([partial_r(Z, M, km, iu) for M in mats])
    null = np.empty(nperm)
    for k in range(nperm):
        q = rng.permutation(K); Zq = Z[np.ix_(q, q)]
        null[k] = np.nanmax([partial_r(Zq, M, km, iu) for M in mats])
    return obs, (np.sum(null >= np.nanmax(obs)) + 1) / (nperm + 1)


def main():
    lines = []
    travel = json.load(open(os.path.join(OUT, 'travel.json')))
    for nm, codes in (('all29', list(SITES)), ('big12', ['HT', 'KH', 'PH', 'KN', 'ZA', 'PK', 'MA', 'TH', 'IO', 'AR', 'PE', 'SY'])):
        Zs = np.load(os.path.join(CKPT, f'c1_Z_{nm}.npy'))
        tc = travel['codes']; ix = [tc.index(c) for c in codes]
        walk = np.array(travel['walk'], float)[np.ix_(ix, ix)]
        km = close(euclid(codes))
        mats = []
        for m in range(1, 13):
            A = np.minimum(walk, np.array(travel[f'windsea_m{m:02d}'], float)[np.ix_(ix, ix)])
            mats.append(close((A + A.T) / 2))
        for li, L in enumerate(('W', 'S', 'L', 'E', 'ALL')):
            Z = Zs.sum(0) if L == 'ALL' else Zs[li]
            obs, p = month_scan(Z, mats, km, int(os.environ.get('NPERM_SCAN', 1000)))
            lines.append(f'{nm} {L} month scan (partial r given km, Jan..Dec): '
                         + ' '.join(f'{x:+.2f}' for x in obs) + f' | best month {np.nanargmax(obs) + 1}, max-null P {p:.3f}')
            print(lines[-1], flush=True)
            # (b) harder vs easier leg, sailing season
            A = np.array(travel['wind_sailing'], float)[np.ix_(ix, ix)]
            for leg, B in (('slower', np.maximum(A, A.T)), ('faster', np.minimum(A, A.T))):
                r, pp, _ = mantel(Z, close(B), nperm=NP, rng=rng, covar=km)
                lines.append(f'  {nm} {L} {leg}-leg (sailing season) partial r|km {r:+.2f} P {pp:.3f}')
            print(lines[-2]); print(lines[-1], flush=True)
    # (c) Linear B
    from la15_common import load_lb
    tlb = json.load(open(os.path.join(OUT, 'travel_lb.json')))
    codes = tlb['codes']
    lb = [d for d in load_lb() if d['site'] in codes]
    docs = [dict(id=d['id'], site=d['site'], support=d.get('support', ''), words=d['words'], signs=d['signs'],
                 logos=[], feat=np.zeros(8)) for d in lb]
    Z, obs = sharing_z(docs, codes, nperm=int(os.environ.get('NPERM_Z', 500)), strat=True, rng=rng)
    kmm = euclid_lb = np.array([[hav(LB_SITES[a][1:], LB_SITES[b][1:]) for b in codes] for a in codes])
    P = {'km': close(kmm), 'calm': close(np.array(tlb['calm'], float))}
    for s in ('sailing', 'etesian', 'winter'):
        A = np.array(tlb['wind_' + s], float); P['wind_' + s] = close((A + A.T) / 2)
    for L in ('W', 'S'):
        parts = []
        for k, A in P.items():
            r, p, _ = mantel(Z[L], A, nperm=NP, rng=rng)
            parts.append(f'{k} {r:+.2f} (P {p:.3f})')
        for k in ('calm', 'wind_sailing', 'wind_etesian'):
            r, p, _ = mantel(Z[L], P[k], nperm=NP, rng=rng, covar=P['km'])
            parts.append(f'{k}|km {r:+.2f} (P {p:.3f})')
        lines.append(f'LB 7 sites {L}: ' + '; '.join(parts)); print(lines[-1], flush=True)
    # LB directed gain
    from la31_c2 import gain, incidence
    from la31_c1 import plant  # noqa
    for L in ('W', 'S'):
        Y, size = incidence(docs, codes, L)
        Tet = np.array(tlb['wind_etesian'], float)
        g1, L1, base = gain(Y, size, Tet); g2, L2, _ = gain(Y, size, Tet.T, base)
        gk, Lk, _ = gain(Y, size, kmm / 5.0, base); gc, Lc, _ = gain(Y, size, np.array(tlb['calm'], float), base)
        rew = []
        for _ in range(int(os.environ.get('NREW', 300))):
            q = rng.permutation(len(codes)); Tq = Tet[np.ix_(q, q)]
            a, _, _ = gain(Y, size, Tq, base); b, _, _ = gain(Y, size, Tq.T, base); rew.append(a - b)
        rew = np.array(rew)
        lines.append(f'LB directed {L} ({Y.shape[0]} types): gain etesian fwd {g1:.2f}@{L1}, transposed {g2:.2f}@{L2}, '
                     f'km {gk:.2f}@{Lk}, calm {gc:.2f}@{Lc}; Delta {g1 - g2:+.2f}, rewired P {(np.sum(rew >= g1 - g2) + 1) / (len(rew) + 1):.3f}')
        print(lines[-1], flush=True)
    open(os.path.join(CKPT, 'c3_out.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
