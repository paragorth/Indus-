"""LA-26 cycle 2: the reverse direction. Do words found at several sites lie on the ring
network's paths? Which words travel along rings?

(1) Word-level ring-path statistic. For every word type found at >= 2 node sites, its site set S.
    F = sum over words of ring-linked pairs inside S / all pairs inside S.
    Nulls: (a) document-label permutation within support type (size + support control, 2,000);
           (b) curveball rewiring of the seal x site graph (5,000), observed words fixed;
           (c) distance-only: each word's extra sites redrawn with prob ~ size x exp(-d/lambda),
               lambda fitted to the observed mean pair distance (2,000).
(2) Per-word score: P(S as ring-connected | rewired graphs); words ranked.
(3) Same seal -> same inscribed sign? The stone seal S68 (HT x3, ZA x1) nodules carry RO on
    HT Wa 1117, 1118 and ZA Wa 36. Base rate from all single-sign nodule / sealing inscriptions.
(4) Calibration (Linear B analogue of the null; no LB cross-site seal data exist in DAMOS):
    LB sites as nodes, random 'ring' graphs of the LA graph's density; P must be uniform.
"""
import json, itertools, time
import numpy as np
from la26_common import *

rng = np.random.default_rng(2602)
OUT = os.path.join(CK, 'c2.json')
res = {}


def site_sets(docs, key='words'):
    S = collections.defaultdict(set)
    for d in docs:
        for w in d[key]:
            S[w].add(d['site'])
    return {w: s for w, s in S.items() if len(s) >= 2}


def F_stat(sets, Rset):
    num = den = 0
    for w, s in sets.items():
        for a, b in itertools.combinations(sorted(s), 2):
            den += 1; num += (a, b) in Rset
    return num / den if den else np.nan, num, den


def rset_from(inc, nodes):
    R = ring_matrix(inc, nodes)
    return {(a, b) for i, a in enumerate(nodes) for j, b in enumerate(nodes) if R[i, j] > 0} | \
           {(b, a) for i, a in enumerate(nodes) for j, b in enumerate(nodes) if R[i, j] > 0}


for nset_name, nodes in [('A', NODES_A), ('B', NODES_B)]:
    for admin in (False, True):
        docs = load_docs(nodes, admin_only=admin)
        sets = site_sets(docs)
        for ver in ('core', 'strict', 'broad'):
            inc = seal_incidence(ver)
            Rset = rset_from(inc, nodes)
            f0, num, den = F_stat(sets, Rset)
            # (a) document permutation
            labels = [d['site'] for d in docs]; sups = [d['support'] for d in docs]
            fa = []
            for _ in range(2000):
                lab = perm_within_support(labels, sups, rng)
                d2 = [dict(d, site=l) for d, l in zip(docs, lab)]
                fa.append(F_stat(site_sets(d2), Rset)[0])
            fa = np.array(fa)
            # (b) rewiring
            fb = np.array([F_stat(sets, rset_from(curveball_inc(inc, None, rng, 60), nodes))[0] for _ in range(5000)])
            # (c) distance-only gravity redraw
            size = collections.Counter(d['site'] for d in docs)
            Dm = dist_matrix(nodes); idx = {s: i for i, s in enumerate(nodes)}
            obs_d = np.mean([Dm[idx[a], idx[b]] for s in sets.values() for a, b in itertools.combinations(s, 2)])
            best = None
            for lam in (25, 50, 75, 100, 150, 200, 300, 500, 1e9):
                ds = []
                for _ in range(200):
                    for w, s in sets.items():
                        home = max(s, key=lambda x: size[x])
                        others = [x for x in nodes if x != home]
                        p = np.array([size[x] * math.exp(-Dm[idx[home], idx[x]] / lam) for x in others]); p /= p.sum()
                        pick = rng.choice(len(others), size=min(len(s) - 1, len(others)), replace=False, p=p)
                        ds += [Dm[idx[home], idx[others[k]]] for k in pick]
                md = np.mean(ds)
                if best is None or abs(md - obs_d) < abs(best[1] - obs_d):
                    best = (lam, md)
            lam = best[0]
            fc = []
            for _ in range(2000):
                s2 = {}
                for w, s in sets.items():
                    home = max(s, key=lambda x: size[x])
                    others = [x for x in nodes if x != home]
                    p = np.array([size[x] * math.exp(-Dm[idx[home], idx[x]] / lam) for x in others]); p /= p.sum()
                    pick = rng.choice(len(others), size=min(len(s) - 1, len(others)), replace=False, p=p)
                    s2[w] = {home} | {others[k] for k in pick}
                fc.append(F_stat(s2, Rset)[0])
            fc = np.array(fc)
            k = f'{nset_name}|{"adm" if admin else "all"}|{ver}'
            res[k] = dict(F=f0, ring_pairs=num, pairs=den, n_words=len(sets),
                          docperm_mean=float(fa.mean()), p_docperm=float((fa >= f0).mean()),
                          rewire_mean=float(fb.mean()), p_rewire=float((fb >= f0).mean()),
                          dist_lambda=lam, dist_mean=float(fc.mean()), p_dist=float((fc >= f0).mean()))
            print(k, res[k], flush=True)
json.dump(res, open(OUT, 'w'), indent=1)

# ------------------------------------------------------------- (2) per-word ranking
nodes = NODES_A
docs = load_docs(nodes)
sets = site_sets(docs)
inc = seal_incidence('core')
R0 = rset_from(inc, nodes)
rew = [rset_from(curveball_inc(inc, None, rng, 60), nodes) for _ in range(5000)]
size = collections.Counter(d['site'] for d in docs)
words = []
for w, s in sets.items():
    pr = [(a, b) for a, b in itertools.combinations(sorted(s), 2)]
    f = sum(p in R0 for p in pr) / len(pr)
    pnull = np.mean([sum(p in r for p in pr) / len(pr) >= f for r in rew])
    words.append(dict(word=w, sites=sorted(s), frac_ring=f, p_rewire=float(pnull)))
words.sort(key=lambda x: (x['p_rewire'], -x['frac_ring']))
res['words_A'] = words
print('multi-site words (A):', len(words))
for x in words:
    print(x)

# ------------------------------------------------------------- (3) same seal -> same sign
C = json.load(open(os.path.join(DATA, 'corpus.json')))
nod = [d for d in C if d['support'] in ('Nodule', 'Sealing') and len(d['words']) == 1 and '-' not in d['words'][0]]
cnt = collections.Counter(d['words'][0] for d in nod)
pRO = cnt['RO'] / sum(cnt.values())
# observed: ZA nodule RO and >= 2 of 3 HT nodules RO; P under independent draws at base rate
from math import comb
p_ht = sum(comb(3, k) * pRO ** k * (1 - pRO) ** (3 - k) for k in (2, 3))
# 'any sign' version: probability that the 4 nodules of one seal share a sign this much, any sign
q = np.array(list(cnt.values())) / sum(cnt.values())
p_any = float(sum(qq * sum(comb(3, k) * qq ** k * (1 - qq) ** (3 - k) for k in (2, 3)) for qq in q))
res['same_seal_sign'] = dict(n_single_sign_nodules=sum(cnt.values()), top=cnt.most_common(10), p_RO=pRO,
                             p_obs_RO=pRO * p_ht, p_any_sign=p_any)
print(res['same_seal_sign'])

# ------------------------------------------------------------- (4) LB calibration of the null
sys_path = os.path.join(HERE)
import la15_common
lb = la15_common.load_lb()
lbsites = [s for s, c in collections.Counter(d['site'] for d in lb).most_common() if c >= 20][:8]
lbd = [dict(d, support='tablet') for d in lb if d['site'] in lbsites]
sets_lb = site_sets(lbd)
nE = int(round(len(R0) / 2 / (len(NODES_A) * (len(NODES_A) - 1) / 2) * len(lbsites) * (len(lbsites) - 1) / 2))
pv = []
for g in range(200):
    allp = list(itertools.combinations(lbsites, 2))
    sel = rng.choice(len(allp), size=nE, replace=False)
    Rs = {allp[k] for k in sel} | {(b, a) for a, b in (allp[k] for k in sel)}
    f0 = F_stat(sets_lb, Rs)[0]
    # document-permutation null (cheap: 200)
    labels = [d['site'] for d in lbd]
    fa = []
    for _ in range(200):
        lab = list(rng.permutation(labels))
        fa.append(F_stat(site_sets([dict(d, site=l) for d, l in zip(lbd, lab)]), Rs)[0])
    pv.append(float((np.array(fa) >= f0).mean()))
pv = np.array(pv)
res['lb_calibration'] = dict(sites=lbsites, n_edges=nE, frac_p_le_05=float((pv <= 0.05).mean()),
                             p_quantiles=[float(x) for x in np.quantile(pv, [0.1, 0.25, 0.5, 0.75, 0.9])])
print(res['lb_calibration'])
json.dump(res, open(OUT, 'w'), indent=1)
