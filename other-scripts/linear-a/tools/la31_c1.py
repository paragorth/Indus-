"""LA-31 cycle 1: does sharing between Linear A sites follow wind-driven sailing time better than
straight-line distance, calm-water rowing, walking, or site size?

Layers (per site pair, z against a support-stratified document-label permutation, 1,000 perms):
W shared word types, S shared rarer syllabic sign types (outside the 40 commonest), L shared
logogram types, E entry-structure similarity. Predictors: log km, calm-sea + walk time, wind time
(combined land + sail, median over 2014-2018 days of each season; symmetrised as the mean of the
two directions), walking (Crete only), size (product of document counts).
Controls: Mantel site-permutation null (= rewired site network, 5,000); planted words that spread
by etesian sailing time (or by km) on the real geography, recovered by the same pipeline.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la31_common import SITES, CRETE, OUT, CKPT, load_docs, euclid  # noqa
from la31_stats import doc_matrices, sharing_z, mantel, site_stats  # noqa

rng = np.random.default_rng(31)
NPERM_Z = int(os.environ.get('NPERM_Z', 1000)); NPERM_M = int(os.environ.get('NPERM_M', 5000))


def predictors(codes, travel):
    tc = travel['codes']; ix = [tc.index(c) for c in codes]
    def sub(k):
        A = np.array(travel[k], float)[np.ix_(ix, ix)]
        return A
    P = {'km': euclid(codes)}
    P['calm'] = sub('calm')
    for s in ('annual', 'sailing', 'etesian', 'spring', 'autumn', 'winter'):
        P['wind_' + s] = sub('wind_' + s)
    P['walk'] = sub('walk')
    out = {}
    for k, A in P.items():
        S = (A + A.T) / 2
        out[k] = -np.log(np.where(np.isfinite(S), S, 1e6) + 0.1)          # closeness
    return out


def run(docs, codes, travel, label, layers=('W', 'S', 'L', 'E', 'ALL'), nperm_m=NPERM_M, Zin=None):
    rows = []
    Z = Zin if Zin is not None else sharing_z(docs, codes, nperm=NPERM_Z, strat=True, rng=rng)[0]
    if 'ALL' in layers and 'ALL' not in Z:
        Z['ALL'] = Z['W'] + Z['S'] + Z['L'] + Z['E']
    P = predictors(codes, travel)
    n = np.array([sum(d['site'] == c for d in docs) for c in codes], float)
    P['size'] = np.log(np.outer(n, n))
    res = {}
    for L in layers:
        for k, A in P.items():
            if k == 'walk' and any(c not in CRETE for c in codes):
                continue
            r, p, _ = mantel(Z[L], A, nperm=nperm_m, rng=rng)
            res[(L, k)] = (r, p)
        # partial: wind_etesian and calm given km
        for k in ('calm', 'wind_etesian', 'wind_sailing', 'wind_winter'):
            r, p, _ = mantel(Z[L], P[k], nperm=nperm_m, rng=rng, covar=P['km'])
            res[(L, k + '|km')] = (r, p)
        r, p, _ = mantel(Z[L], P['wind_sailing'], nperm=nperm_m, rng=rng, covar=P['calm'])
        res[(L, 'wind_sailing|calm')] = (r, p)
    return res, Z


def fmt(res, L):
    return '; '.join(f"{k} {res[(L, k)][0]:+.2f} (P {res[(L, k)][1]:.3f})" for (l, k) in res if l == L)


def plant(docs, codes, T, L_h, rng, V=4000, s=1.1):
    """Replace every document's words by synthetic words that spread from an origin site with
    probability exp(-T[origin -> j] / L_h). Word counts per document are kept."""
    K = len(codes); ci = {c: i for i, c in enumerate(codes)}
    n = np.array([sum(d['site'] == c for d in docs) for c in codes], float)
    org = rng.choice(K, size=V, p=n ** 0.7 / (n ** 0.7).sum())
    pres = rng.random((V, K)) < np.exp(-np.where(np.isfinite(T), T, 1e6)[org] / L_h)
    pres[np.arange(V), org] = True
    pop = rng.permutation(1.0 / np.arange(1, V + 1) ** s)
    out = []
    for d in docs:
        j = ci[d['site']]; cand = np.nonzero(pres[:, j])[0]
        w = pop[cand] / pop[cand].sum()
        e = dict(d); e['words'] = [f'w{x}' for x in rng.choice(cand, size=len(d['words']), p=w)]
        out.append(e)
    return out


def main():
    travel = json.load(open(os.path.join(OUT, 'travel.json')))
    docs = load_docs()
    lines = []
    sets = {'all29': list(SITES), 'crete23': CRETE,
            'big12': ['HT', 'KH', 'PH', 'KN', 'ZA', 'PK', 'MA', 'TH', 'IO', 'AR', 'PE', 'SY']}
    allres = {}
    for nm, codes in sets.items():
        dd = [d for d in docs if d['site'] in codes]
        res, Z = run(dd, codes, travel, nm)
        allres[nm] = res
        np.save(os.path.join(CKPT, f'c1_Z_{nm}.npy'), np.array([Z[L] for L in ('W', 'S', 'L', 'E')]))
        for L in ('W', 'S', 'L', 'E', 'ALL'):
            lines.append(f'{nm} {L}: ' + fmt(res, L))
            print(lines[-1], flush=True)
    # ---- planted controls on the real geography (all 29 sites), W layer only
    codes = list(SITES)
    tc = travel['codes']; ix = [tc.index(c) for c in codes]
    Tet = np.array(travel['wind_etesian'], float)[np.ix_(ix, ix)]
    Tkm = euclid(codes) / 5.0       # km -> 'hours' at 5 km/h, same scale
    real_shared = None
    M, tags, F = doc_matrices(docs, codes)
    ci = {c: i for i, c in enumerate(codes)}
    lab = np.array([ci[d['site']] for d in docs])
    S0 = site_stats(M, tags, F, lab, len(codes))['W']; real_shared = np.triu(S0, 1).sum()
    # calibrate L so that planted cross-site sharing matches the real amount
    cal = {}
    for nmT, T in (('etesian', Tet), ('km', Tkm)):
        best = None
        for L_h in (2, 4, 8, 16, 32, 64, 128, 256, 512):
            ss = []
            for _ in range(3):
                pd_ = plant(docs, codes, T, L_h, rng)
                Mp, tp, Fp = doc_matrices(pd_, codes)
                ss.append(np.triu(site_stats(Mp, tp, Fp, lab, len(codes))['W'], 1).sum())
            if best is None or abs(np.mean(ss) - real_shared) < abs(best[1] - real_shared):
                best = (L_h, np.mean(ss))
        cal[nmT] = best
    lines.append(f'planted calibration: real cross-site shared word types {real_shared:.0f}; L etesian {cal["etesian"]}, km {cal["km"]}')
    print(lines[-1], flush=True)
    for nmT, T in (('etesian', Tet), ('km', Tkm)):
        for mult in (1, 4):
            L_h = cal[nmT][0] * mult
            rec = []
            for rep in range(int(os.environ.get('NREP', 12))):
                pd_ = plant(docs, codes, T, L_h, rng)
                Z, _ = sharing_z(pd_, codes, nperm=200, strat=True, rng=rng)
                P = predictors(codes, travel)
                r_w, p_w, _ = mantel(Z['W'], P['wind_etesian'], nperm=500, rng=rng)
                r_k, p_k, _ = mantel(Z['W'], P['km'], nperm=500, rng=rng)
                r_wp, p_wp, _ = mantel(Z['W'], P['wind_etesian'], nperm=500, rng=rng, covar=P['km'])
                rec.append((r_w, p_w, r_k, p_k, r_wp, p_wp))
            a = np.array(rec)
            lines.append(f'planted {nmT} L={L_h}h x{mult}: Mantel wind_etesian r {a[:,0].mean():+.2f} det {np.mean(a[:,1]<0.05):.2f}; '
                         f'km r {a[:,2].mean():+.2f} det {np.mean(a[:,3]<0.05):.2f}; wind|km r {a[:,4].mean():+.2f} det {np.mean(a[:,5]<0.05):.2f}; '
                         f'wind>km in {np.mean(a[:,0]>a[:,2]):.2f}')
            print(lines[-1], flush=True)
    open(os.path.join(CKPT, 'c1_out.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
