"""pe17 cycle 2: score the FROZEN blind predictions against the hXRF provenance labels.

Inputs: data/pe17_frozen_ranking.json, data/pe17_frozen_affinity.json (hashes checked),
        data/pe17_xrf_labels.json (labels from Yeganeh et al. 2025, read only after freezing).
Tests (exact rank p-values = permutation null over the candidate pool):
  T1 ST-11 (P009157, Yahya clay at Susa): plateau-score rank among (a) all Susa tablets, (b) Susa tablets held
     in the National Museum of Iran (the pool the 35 were drawn from), (c) NMI Susa tablets in the same length bin.
  T2 ST-11 per-site affinity: is Yahya its top site; rank of (Yahya - mean other sites) among NMI Susa.
  T3 YT-01, YT-07 (Susa clay at Yahya): Susa-likeness (low out-of-group plateau score) among the 27 Yahya tablets; AUC.
  T4 YT-02, YT-06 (Malyan clay at Yahya): Malyan-likeness (low Yahya-vs-Malyan LOO score) among Yahya; AUC.
  T5 combined: Fisher combination of the four one-sided rank p-values (T1b, T2, T3, T4).
"""
import json, os, sys, math
import numpy as np
from itertools import combinations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe17_common import DATA, CKPT, sha, auc

fr = json.load(open(os.path.join(DATA, 'pe17_frozen_ranking.json')))
fa = json.load(open(os.path.join(DATA, 'pe17_frozen_affinity.json')))
for F in (fr, fa):
    h = F.pop('sha256_of_content_without_this_field')
    assert sha(F) == h, 'frozen file changed'
lab = json.load(open(os.path.join(DATA, 'pe17_xrf_labels.json')))
cat = json.load(open(os.path.join(CKPT, 'pe_cat.json')))
S = {r['id']: r for r in fr['susa']}
P = {r['id']: r for r in fr['plateau']}
A = fa['susa']
nmi = [i for i in S if cat.get(i, {}).get('collection', '').startswith('National Museum, Tehran')]
res = {}


def upper_p(x, pool):
    """share of pool with value >= x (one-sided, x counted)"""
    v = np.array(pool)
    return float((v >= x).sum() / len(v))


st = 'P009157'
sc = S[st]['plateau_score']
res['T1a_all_susa'] = dict(rank=S[st]['rank'], n=len(S), p=upper_p(sc, [r['plateau_score'] for r in S.values()]))
res['T1b_nmi_susa'] = dict(n=len(nmi), p=upper_p(sc, [S[i]['plateau_score'] for i in nmi]))
L = S[st]['lines']
same = [i for i in nmi if abs(S[i]['lines'] - L) <= 1]
res['T1c_nmi_lenmatched'] = dict(n=len(same), p=upper_p(sc, [S[i]['plateau_score'] for i in same]))
sites = ['Malyan', 'Yahya', 'Sialk', 'Sofalin']
aff = A[st]
res['T2_affinity'] = dict(aff=aff, top_site=max(aff, key=aff.get))
d = lambda i: A[i]['Yahya'] - np.mean([A[i][s] for s in sites if s != 'Yahya'])
res['T2_affinity']['p_yahya_contrast_nmi'] = upper_p(d(st), [d(i) for i in nmi])
res['T2_affinity']['p_yahya_raw_nmi'] = upper_p(aff['Yahya'], [A[i]['Yahya'] for i in nmi])

Y = [i for i in P if P[i]['site'] == 'Yahya']
susa_clay = ['P009536', 'P009537']
mal_clay = ['P009545', 'P009540']


def two_test(pos, key, sign):
    vals = {i: sign * P[i][key] for i in Y}
    pos_v = [vals[i] for i in pos]
    neg_v = [vals[i] for i in Y if i not in pos]
    a = auc(pos_v, neg_v)
    # exact null: all pairs of Yahya tablets
    null = [auc([vals[i] for i in c], [vals[j] for j in Y if j not in c]) for c in combinations(Y, len(pos))]
    return dict(auc=a, p=float(np.mean(np.array(null) >= a - 1e-12)),
                ranks={i: int(sorted(vals.values(), reverse=True).index(vals[i]) + 1) for i in pos}, n=len(Y))


res['T3_susa_clay_at_yahya'] = two_test(susa_clay, 'plateau_score_out_of_group', -1)
res['T4_malyan_clay_at_yahya'] = two_test(mal_clay, 'yahya_vs_malyan_loo', -1)
ps = [res['T1b_nmi_susa']['p'], res['T2_affinity']['p_yahya_contrast_nmi'],
      res['T3_susa_clay_at_yahya']['p'], res['T4_malyan_clay_at_yahya']['p']]
X2 = -2 * sum(math.log(p) for p in ps)
from scipy.stats import chi2
res['T5_fisher'] = dict(ps=ps, chi2=X2, df=8, p=float(chi2.sf(X2, 8)))
json.dump(res, open(os.path.join(DATA, 'pe17_cycle2.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
