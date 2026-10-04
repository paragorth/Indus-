"""pe31 cycle 3: is there a closed interchangeable subset (a candidate syllabary inside the script)?
A. Per-sign interchangeability degree D(s) = mean of the sign's 3 highest IX values.
   Calibration: in Old Babylonian seal names (a mixed script: syllables + logograms), D must rank
   syllabic signs above logograms; in Ur III (mostly logographic names) the same check.
   PE: does D rank simple signs (pe22 closed set) above compounds, and pe6 name-spelling signs above
   others?  Frequency-matched label permutation.
B. Closure: transitivity of the top-1% IX graph vs degree-preserving rewiring, all corpora.
C. Held-out replication: IX from half the tablets, top-50 pairs re-scored on the other half,
   vs frequency-matched random pairs; controls and the within-string-shuffled PE as references."""
import json, os, random, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe31_lib as L
from pe31_cycle2 import IX

MINC = 6
PE6_NAME = {'M099', 'M136', 'M005', 'M246', 'M251', 'M254', 'M304', 'M262', 'M340', 'M390', '|M106+M288|',
            'M418', 'M206', 'M050', 'M379', 'M243', 'M036', 'M319', 'M328', 'M352', 'M384'}
SYL = re.compile(r"^(sz|s,|t,|[bdghklmnpqrstwyz])?[aeiu](sz|s,|t,|[bdghklmnpqrstwz])?$")


def degree(X):
    Y = X.copy(); np.fill_diagonal(Y, -np.inf)
    return np.sort(Y, 1)[:, -3:].mean(1)


def label_auc(D, lab, cnt_v, nperm=2000, seed=0):
    """AUC of D for a label, vs labels re-dealt among signs in the same frequency quintile"""
    lab = np.asarray(lab, bool); obs = L.auc(D, lab)
    lf = np.log(cnt_v); b = np.digitize(lf, np.quantile(lf, [.2, .4, .6, .8]))
    rng = np.random.default_rng(seed); null = []
    for _ in range(nperm):
        l2 = lab.copy()
        for k in np.unique(b):
            m = np.where(b == k)[0]; l2[m] = rng.permutation(lab[m])
        null.append(L.auc(D, l2))
    null = np.array(null)
    return dict(auc=round(obs, 3), null=round(float(null.mean()), 3), p=float((1 + (null >= obs).sum()) / (nperm + 1)),
                n_pos=int(lab.sum()), n=len(lab))


def transitivity(E, n):
    A = np.zeros((n, n), bool)
    for i, j in E:
        A[i, j] = A[j, i] = True
    Af = A.astype(float)
    tri = np.trace(Af @ Af @ Af)
    deg = Af.sum(1); trip = (deg * (deg - 1)).sum()
    return tri / trip if trip else 0.0


def closure(X, nrew=200, seed=0):
    n = X.shape[0]; iu = np.triu_indices(n, 1); x = X[iu]
    k = max(10, len(x) // 100)
    top = np.argsort(-x)[:k]
    E = [(iu[0][t], iu[1][t]) for t in top]
    obs = transitivity(E, n)
    rng = random.Random(seed); null = []
    for _ in range(nrew):
        e = list(E); S = set(map(tuple, map(sorted, e)))
        for _ in range(10 * len(e)):
            a, b = rng.sample(range(len(e)), 2)
            (u, v), (s, t) = e[a], e[b]
            if rng.random() < .5:
                s, t = t, s
            if len({u, v, s, t}) < 4:
                continue
            n1, n2 = tuple(sorted((u, t))), tuple(sorted((s, v)))
            if n1 in S or n2 in S:
                continue
            S -= {tuple(sorted((u, v))), tuple(sorted((s, t)))}; S |= {n1, n2}
            e[a], e[b] = n1, n2
        null.append(transitivity(e, n))
    null = np.array(null)
    nodes = sorted({i for ed in E for i in ed})
    return dict(k=k, nodes=len(nodes), trans=round(obs, 3), null=round(float(null.mean()), 3),
                ratio=round(obs / max(null.mean(), 1e-9), 2), p=float((1 + (null >= obs).sum()) / (nrew + 1))), nodes


def heldout(data, nsplit=5, K=50, seed=0):
    tabs = sorted({t for _, t in data}); out = []
    for s in range(nsplit):
        rng = random.Random(seed + s); tt = tabs[:]; rng.shuffle(tt); h = set(tt[:len(tt) // 2])
        A = [d for d in data if d[1] in h]; B = [d for d in data if d[1] not in h]
        aA, cA = L.alphabet(A, 4); aB, cB = L.alphabet(B, 4)
        com = sorted(set(aA) & set(aB))
        XA, _ = IX(A, com, cA); XB, _ = IX(B, com, cB)
        n = len(com); iu = np.triu_indices(n, 1); xa, xb = XA[iu], XB[iu]
        top = np.argsort(-xa)[:K]
        pct = np.array([(xb < v).mean() for v in xb[top]])
        out.append(dict(rho=round(float(np.corrcoef(np.argsort(np.argsort(xa)), np.argsort(np.argsort(xb)))[0, 1]), 3),
                        top_pct_B=round(float(pct.mean()), 3), top_in_B_top5pct=int((pct >= .95).sum())))
    return dict(rho=round(float(np.mean([o['rho'] for o in out])), 3), top_pct_B=round(float(np.mean([o['top_pct_B'] for o in out])), 3),
                top_in_B_top5pct=round(float(np.mean([o['top_in_B_top5pct'] for o in out])), 1), K=K, splits=out)


def main():
    res = {}
    corp = {'UR3_SEAL': L.control('UR3_SEAL'), 'OB_SEAL': L.control('OB_SEAL'), 'LINB': L.control('LINB'),
            'PE_base': L.pe_names(False), 'PE_var': L.pe_names(True)}
    corp['PE_shuffled'] = L.shuffle_within(corp['PE_base'], random.Random(9))
    a0, c0 = L.alphabet(corp['PE_base'], 1); cand = [s for s in a0 if 16 <= c0[s] <= 80]
    rng = random.Random(77); ps = set(rng.sample(cand, 10))
    corp['PE_plant_tablet'] = L.plant(corp['PE_base'], ps, 'tablet', rng)
    for nm, d in corp.items():
        alph, cnt = L.alphabet(d, MINC)
        X, _ = IX(d, alph, cnt)
        D = degree(X); cv = np.array([cnt[a] for a in alph], float)
        r = dict(n_signs=len(alph))
        if nm in ('OB_SEAL', 'UR3_SEAL'):
            lab = [bool(SYL.match(a)) and a == a.lower() for a in alph]
            r['syllabic_vs_D'] = label_auc(D, lab, cv)
        if nm in ('PE_base', 'PE_var'):
            r['simple_vs_D'] = label_auc(D, [not a.startswith('|') for a in alph], cv)
            r['pe6name_vs_D'] = label_auc(D, [L.base(a) in PE6_NAME for a in alph], cv)
            if nm == 'PE_var':
                r['variant_graph_vs_D'] = label_auc(D, ['~' in a for a in alph], cv)
            r['top_D'] = [(alph[i], int(cnt[alph[i]]), round(float(D[i]), 2)) for i in np.argsort(-D)[:20]]
        if nm == 'PE_plant_tablet':
            r['planted_vs_D'] = label_auc(D, [a.rstrip("'") in ps for a in alph], cv)
        r['closure'], nodes = closure(X)
        r['closure_nodes'] = [alph[i] for i in nodes] if nm.startswith('PE') else len(nodes)
        if nm in ('PE_base', 'PE_var'):
            r['closure_nodes_simple_share'] = round(float(np.mean([not alph[i].startswith('|') for i in nodes])), 3)
            r['all_simple_share'] = round(float(np.mean([not a.startswith('|') for a in alph])), 3)
        if nm != 'PE_var':
            r['heldout'] = heldout(d)
        res[nm] = r
        print(nm, json.dumps({k: v for k, v in r.items() if k not in ('top_D', 'closure_nodes')}), flush=True)
    json.dump(res, open(os.path.join(L.CK, 'cycle3.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
