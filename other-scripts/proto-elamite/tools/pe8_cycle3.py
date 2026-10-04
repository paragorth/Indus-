"""pe8 cycle 3.  Test 2 (one office, one form?) and test 3 (are rare forms compositions of common ones?).

Forms: MAP cluster of each REAL tablet under the latent class model at K_cv (cycle 1), fitted (i) on all
skeleton features ('full') and (ii) without SYS/TOT ('nonum', so that a shared number system alone cannot make
neighbours look alike).  20 restarts, best log-lik.

Test 2.  Neighbour pairs: (a) Sb museum numbers differing by 1 (Louvre accession runs); (b) Sb runs whose
publication numbers are NOT adjacent (|d pub| > 5 or other volume: removes editorial ordering);
(c) NMI BK runs (Tehran); (d) publication-number neighbours (d = 1, same volume; editorial order);
(e) same findspot square or stratigraphic level (modern excavations).  Statistic: share of pairs with the same
form.  Nulls (10,000x): forms permuted over all tablets; forms permuted within publication volume.
Reference: the same statistic for each single skeleton feature (is the form more stable than its parts?).

Test 3.  Common forms = largest clusters covering >= 80% of tablets at K = 12 (finer partition); the rest
are rare forms.  A rare form (its modal skeleton) is 'composed' if every feature equals the mode of common form
A or of common form B for some pair; 'block-composed' if its obverse block equals A's and its reverse block
equals B's on >= all but one feature.  Same scores for rare tablets.  Null: rare units' features permuted
column-wise among rare units (1,000x).
Output: data/pe8_cycle3.json
"""
import json, os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe8_common import *

OBV = ['HDR', 'HLEN', 'SYS', 'NENT', 'ONE', 'CLS', 'CLSR', 'PRE', 'BARE', 'SUB']
REVB = ['REV', 'TOT', 'TOP', 'COL', 'SEAL']


def fit_forms(R, feats, K, restarts=20, seed=11):
    X, lev = encode([r['f'] for r in R], feats)
    card = [max(1, len(lev[k])) for k in feats]
    m = best_em(X, card, K, restarts, seed)
    return m, X, lev, card


def pair_stat(z, pairs):
    return float(np.mean([z[a] == z[b] for a, b in pairs])) if pairs else float('nan')


def stability(R, z, pairs, vols, rng, nperm=10000):
    z = np.asarray(z)
    obs = pair_stat(z, pairs)
    if not pairs:
        return None
    pa = np.array(pairs)
    g, w = [], []
    byvol = defaultdict(list)
    for i, v in enumerate(vols):
        byvol[v].append(i)
    for _ in range(nperm):
        zp = rng.permutation(z)
        g.append(np.mean(zp[pa[:, 0]] == zp[pa[:, 1]]))
        zw = z.copy()
        for v, ii in byvol.items():
            zw[ii] = z[rng.permutation(ii)]
        w.append(np.mean(zw[pa[:, 0]] == zw[pa[:, 1]]))
    g = np.array(g); w = np.array(w)
    return {'n_pairs': len(pairs), 'obs': obs, 'glob_mean': float(g.mean()), 'glob_p': float((g >= obs).mean()),
            'vol_mean': float(w.mean()), 'vol_p': float((w >= obs).mean()),
            'vol_z': float((obs - w.mean()) / (w.std() + 1e-12))}


def build_pairs(R):
    meta = [r['meta'] for r in R]
    P = {}
    def runs(prefix):
        L = sorted((m['mus_num'], i) for i, m in enumerate(meta) if m.get('mus_prefix') == prefix and m.get('mus_num'))
        return [(a[1], b[1]) for a, b in zip(L, L[1:]) if b[0] - a[0] == 1]
    P['sb_run'] = runs('Sb')
    P['sb_run_nonpub'] = [(a, b) for a, b in P['sb_run']
                          if meta[a]['pub_vol'] != meta[b]['pub_vol'] or meta[a]['pub_num'] is None
                          or meta[b]['pub_num'] is None or abs(meta[a]['pub_num'] - meta[b]['pub_num']) > 5]
    P['nmi_run'] = runs('NMI BK')
    L = sorted((m['pub_vol'], m['pub_num'], i) for i, m in enumerate(meta) if m.get('pub_num') is not None)
    P['pub_adj'] = [(a[2], b[2]) for a, b in zip(L, L[1:]) if a[0] == b[0] and b[1] - a[1] == 1]
    grp = defaultdict(list)
    for i, m in enumerate(meta):
        for k in ('findspot_square', 'stratigraphic_level'):
            if m.get(k):
                grp[(m['provenience'], k, m[k])].append(i)
    P['findspot'] = [p for ii in grp.values() for p in itertools.combinations(ii, 2)]
    return P


def sb_vs_pub(R):
    """Does the Sb accession order follow the publication order (editorial confound)?"""
    from scipy.stats import spearmanr
    out = {}
    byv = defaultdict(list)
    for r in R:
        m = r['meta']
        if m.get('mus_prefix') == 'Sb' and m.get('pub_num') is not None:
            byv[m['pub_vol']].append((m['mus_num'], m['pub_num']))
    for v, L in byv.items():
        if len(L) >= 10:
            out[v] = (len(L), float(spearmanr([a for a, b in L], [b for a, b in L])[0]))
    return out


def modes(m, lev, feats):
    return [[lev[k][int(np.argmax(m['th'][j][c]))] for j, k in enumerate(feats)] for c in range(m['K'])]


def composition(units, common, rng, feats, nperm=1000):
    """units: list of skeleton lists (rare forms or rare tablets; None = missing).  common: modal skeletons."""
    oi = [feats.index(k) for k in OBV]; ri = [feats.index(k) for k in REVB]
    def score(u):
        best_single = max(sum(1 for j, v in enumerate(u) if v is not None and v == c[j]) for c in common)
        nobs = sum(1 for v in u if v is not None)
        best_pair = 0; best_block = 0
        for a, b in itertools.permutations(range(len(common)), 2):
            ca, cb = common[a], common[b]
            best_pair = max(best_pair, sum(1 for j, v in enumerate(u) if v is not None and (v == ca[j] or v == cb[j])))
            bl = sum(1 for j in oi if u[j] is not None and u[j] == ca[j]) + \
                 sum(1 for j in ri if u[j] is not None and u[j] == cb[j])
            best_block = max(best_block, bl)
        return nobs, best_single, best_pair, best_block
    def summ(us):
        S = [score(u) for u in us]
        return {'full_pair': float(np.mean([p == n for n, s, p, b in S])),
                'miss_pair': float(np.mean([n - p for n, s, p, b in S])),
                'miss_single': float(np.mean([n - s for n, s, p, b in S])),
                'block_ok': float(np.mean([b >= n - 1 for n, s, p, b in S])),
                'gain_pair': float(np.mean([p - s for n, s, p, b in S]))}
    obs = summ(units)
    U = np.array([[v if v is not None else '<NA>' for v in u] for u in units], dtype=object)
    nulls = []
    for _ in range(nperm):
        V = U.copy()
        for j in range(V.shape[1]):
            V[:, j] = V[rng.permutation(len(V)), j]
        nulls.append(summ([[None if v == '<NA>' else v for v in row] for row in V]))
    res = {'n_units': len(units), 'obs': obs}
    for k in obs:
        arr = np.array([d[k] for d in nulls])
        res[k + '_null_mean'] = float(arr.mean())
        res[k + '_p_ge'] = float((arr >= obs[k]).mean())
        res[k + '_p_le'] = float((arr <= obs[k]).mean())
    return res


def main():
    c1 = {r['name']: r for r in json.load(open(os.path.join(DATA, 'pe8_cycle1.json')))}
    K = c1['REAL']['K_cv']
    R = load_skeletons(min_ent=2, clean=False)
    rng = np.random.default_rng(5)
    vols = [r['meta'].get('pub_vol', '?') for r in R]
    P = build_pairs(R)
    out = {'K': K, 'pairs': {k: len(v) for k, v in P.items()}, 'sb_vs_pub_spearman': sb_vs_pub(R), 'test2': {}}
    nonum = [k for k in FEATS if k not in ('SYS', 'TOT')]
    for label, feats in (('full', FEATS), ('nonum', nonum)):
        m, X, lev, card = fit_forms(R, feats, K)
        z = m['R'].argmax(1)
        out['test2'][label] = {pk: stability(R, z, pv, vols, rng, 4000) for pk, pv in P.items()}
        if label == 'full':
            out['forms'] = []
            for c in range(K):
                members = np.where(z == c)[0]
                prof = {k: {lev[k][v]: round(float(m['th'][j][c][v]), 2) for v in range(card[j])
                            if m['th'][j][c][v] >= 0.15} for j, k in enumerate(feats)}
                out['forms'].append({'form': c, 'n_tablets': int(len(members)), 'weight': float(m['pi'][c]),
                                     'profile': prof,
                                     'examples': [R[i]['id'] for i in members[:6]]})
            zfull = z
    # single-feature reference for test 2 (sb_run and pub_adj)
    out['test2']['single_feature'] = {}
    for k in FEATS:
        vals = [r['f'][k] for r in R]
        codes = {v: i for i, v in enumerate(sorted({str(v) for v in vals}))}
        zz = np.array([codes[str(v)] for v in vals])
        out['test2']['single_feature'][k] = {pk: stability(R, zz, P[pk], vols, rng, 1000) for pk in ('sb_run', 'pub_adj')}
    # test 3
    m12, X, lev, card = fit_forms(R, FEATS, 12, restarts=20, seed=12)
    z12 = m12['R'].argmax(1)
    sizes = Counter(z12.tolist())
    order = [c for c, _ in sizes.most_common()]
    cum, common = 0, []
    for c in order:
        if cum / len(R) >= 0.8:
            break
        common.append(c); cum += sizes[c]
    rare = [c for c in order if c not in common]
    M = modes(m12, lev, FEATS)
    out['test3'] = {'K': 12, 'sizes': [sizes[c] for c in order], 'n_common': len(common), 'n_rare': len(rare),
                    'common_modes': [dict(zip(FEATS, M[c])) for c in common],
                    'rare_modes': [dict(zip(FEATS, M[c])) for c in rare]}
    out['test3']['rare_forms'] = composition([M[c] for c in rare], [M[c] for c in common], rng, FEATS)
    rare_tabs = [i for i in range(len(R)) if z12[i] in rare]
    units = [[R[i]['f'][k] for k in FEATS] for i in rare_tabs]
    out['test3']['rare_tablets'] = composition(units, [M[c] for c in common], rng, FEATS, nperm=200)
    common_tabs = [i for i in range(len(R)) if z12[i] in common]
    sub = list(rng.choice(common_tabs, size=min(len(rare_tabs), len(common_tabs)), replace=False))
    out['test3']['common_tablets_ref'] = composition([[R[i]['f'][k] for k in FEATS] for i in sub],
                                                     [M[c] for c in common], rng, FEATS, nperm=200)
    json.dump(out, open(os.path.join(DATA, 'pe8_cycle3.json'), 'w'), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == '__main__':
    main()
