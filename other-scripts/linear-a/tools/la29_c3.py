#!/usr/bin/env python3
"""la29 cycle 3: massive random guessing with held-out documents, and a cross-script transfer.

(a) Commodity: each site's documents are split at random into halves A and B (NSPLIT splits).
    10^4 random land indices (random signed mixtures of 1-4 standardized land variables) are scored
    LOSO on A; the top 1 % are re-scored on B. The same selection is run on 10^4 Gaussian random
    fields. If the land carries information that arbitrary smooth fields do not, land survivors must
    beat field survivors on B.
(b) Signs: the same for every sign with >= 15 tokens (best index on A, its |r| on B), land vs fields.
(c) Transfer (outside Linear A): the land model fitted on Linear A sites (GRA, VIN, OLE, OLIV, LIV)
    predicts the Linear B archives' mix at KN, PY, TH, MY; scored against the fields.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la29_common import *

rng = np.random.default_rng(293)
NIDX = int(os.environ.get('NIDX', 10000))
NSPLIT = int(os.environ.get('NSPLIT', 10))
out_lines = []


def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out_lines.append(s)


land = land_table()
corr = json.load(open(os.path.join(CK, 'crete_corr.json')))
LS = np.array(sorted({max(3.0, v['efold_km']) for v in corr.values()} | {10.0, 20.0, 40.0}))

# ---------------- document-level commodity sets
docs = C.defaultdict(list)
for d in corpus():
    s = LA_NAME.get(d['site'])
    if s is None or s in ('THE', 'KEA', 'MI'): continue
    cs = {cat_la(t['v']) for t in d['tokens'] if t['t'] == 'logo'}
    if cs: docs[s].append(cs)
SITES_ = [s for s in docs if len(docs[s]) >= 4]
SITES_.sort(key=lambda s: -len(docs[s]))
cats = [c for c in CATS if sum(c in ds for s in SITES_ for ds in docs[s]) >= 3]
X, keys = land_matrix(SITES_, land)
Xs = (X - X.mean(0)) / X.std(0)
S = len(SITES_)
log('sites', {s: len(docs[s]) for s in SITES_}, 'cats', cats)


def counts(sel):
    return np.array([[sum(c in ds for ds in sel[s]) for c in cats] for s in SITES_], float)


def rand_indices(n):
    Z = np.zeros((n, S)); desc = []
    for i in range(n):
        k = rng.integers(1, 5); j = rng.choice(len(keys), k, replace=False); w = rng.normal(size=k)
        Z[i] = Xs[:, j] @ w; desc.append([(keys[a], round(float(b), 2)) for a, b in zip(j, w)])
    return Z, desc


IDX, desc = rand_indices(NIDX)
FLD = grf(SITES_, LS, NIDX, rng)
resA = []
for sp in range(NSPLIT):
    A, B = {}, {}
    for s in SITES_:
        p = rng.permutation(len(docs[s])); h = len(p) // 2
        A[s] = [docs[s][i] for i in p[:h]]; B[s] = [docs[s][i] for i in p[h:]]
    YA, YB = counts(A), counts(B)
    row = []
    for Z in (IDX, FLD):
        ska, _ = loso_skill_batch(YA, Z, iters=100)
        top = np.argsort(-ska)[:NIDX // 100]
        skb, _ = loso_skill_batch(YB, Z[top], iters=100)
        row.append((ska[top].mean(), skb.mean(), top))
    resA.append((row[0][0], row[0][1], row[1][0], row[1][1]))
    log(f' split {sp}: land top1% A {row[0][0]:+.4f} -> B {row[0][1]:+.4f} | fields top1% A {row[1][0]:+.4f} -> B {row[1][1]:+.4f}')
    if sp == 0:
        log('   best land index on A (split 0):', desc[int(row[0][2][0])])
resA = np.array(resA)
d = resA[:, 1] - resA[:, 3]
log(f'(a) commodity held-out: land survivors B {resA[:, 1].mean():+.4f} vs field survivors B {resA[:, 3].mean():+.4f}; '
    f'land > fields in {(d > 0).sum()}/{NSPLIT} splits; mean diff {d.mean():+.4f} (sd {d.std():.4f})')

# ---------------- (b) signs, document halves
sdocs = C.defaultdict(list)
for dd in corpus():
    s = LA_NAME.get(dd['site'])
    if s is None or s in ('KEA', 'MI'): continue
    if os.environ.get('ADMIN') == '1' and dd['support'] not in ADMIN_SUPPORTS: continue
    cc = C.Counter()
    for t in dd['tokens']:
        if t['t'] == 'word':
            for g in t['s']: cc[g] += 1
        elif t['t'] == 'logo':
            c = cat_la(t['v']); cc['L:' + (c if c != 'OTHER' else t['v'].split('+')[0].strip("*[]'"))] += 1
    if cc: sdocs[s].append(cc)
SS = [s for s in sdocs if sum(sum(c.values()) for c in sdocs[s]) >= 30]
SS.sort(key=lambda s: -len(sdocs[s]))
Xg, kg = land_matrix(SS, land); Xgs = (Xg - Xg.mean(0)) / Xg.std(0)
tot_all = C.Counter()
for s in SS:
    for c in sdocs[s]: tot_all.update(c)
signs = [g for g, n in tot_all.items() if n >= 15]
IDXg = np.zeros((NIDX, len(SS)))
for i in range(NIDX):
    k = rng.integers(1, 5); j = rng.choice(len(kg), k, replace=False)
    IDXg[i] = Xgs[:, j] @ rng.normal(size=k)
FLDg = grf(SS, LS, NIDX, rng)


def wcorr(l, w, Z):
    W = w / w.sum(1, keepdims=True)
    lc = l - (W * l).sum(1, keepdims=True)
    sl = np.sqrt((W * lc ** 2).sum(1))
    Zm = W @ Z.T; Z2 = W @ (Z ** 2).T
    sz = np.sqrt(np.clip(Z2 - Zm ** 2, 1e-12, None))
    return (W * lc) @ Z.T / (sl[:, None] * sz)


def prof(half):
    K = np.array([[sum(c[g] for c in half[s]) for s in SS] for g in signs], float)
    tot = np.array([sum(sum(c.values()) for c in half[s]) for s in SS], float)
    p = (K + 0.5) / (tot[None] + 1); return np.log(p / (1 - p)), (tot[None] + 1) * p * (1 - p)


resB = []
for sp in range(NSPLIT):
    A, B = {}, {}
    for s in SS:
        p = rng.permutation(len(sdocs[s])); h = len(p) // 2
        A[s] = [sdocs[s][i] for i in p[:h]]; B[s] = [sdocs[s][i] for i in p[h:]]
    (lA, wA), (lB, wB) = prof(A), prof(B)
    out = []
    for Z in (IDXg, FLDg):
        rA = wcorr(lA, wA, Z)
        jb = np.abs(rA).argmax(1)
        rB = wcorr(lB, wB, Z[jb])[np.arange(len(signs)), np.arange(len(signs))]
        sgn = np.sign(rA[np.arange(len(signs)), jb])
        out.append(sgn * rB)                     # same-direction replication
    resB.append(out)
resB = np.array(resB)                            # [split, 2, G]
land_rep = resB[:, 0].mean(0); fld_rep = resB[:, 1].mean(0)
log(f'(b) signs ({len(signs)}, sites {len(SS)}): held-out same-sign r of the best A index: land {land_rep.mean():+.3f} vs fields {fld_rep.mean():+.3f}; '
    f'signs where land replicates better than fields: {(land_rep > fld_rep).sum()}/{len(signs)}')
o = np.argsort(-(land_rep - fld_rep))
for g in o[:8]:
    log(f'    {signs[g]:10s} n {tot_all[signs[g]]:4d}  land rep {land_rep[g]:+.2f}  field rep {fld_rep[g]:+.2f}')

# ---------------- (c) transfer to Linear B
lbc, lbcats = lb_commodity()
shared = ['GRA', 'VIN', 'OLE', 'OLIV', 'LIV']
YLA = counts(docs)[:, [cats.index(c) for c in shared if c in cats]]
sh = [c for c in shared if c in cats]
LB = ['KN', 'PYL', 'THB', 'MYC']
YLB = np.array([[lbc[s][c] for c in sh] for s in LB], float)
allS = SITES_ + [s for s in LB if s not in SITES_]
Xall, kall = land_matrix(allS, land)


def transfer(xla, xlb):
    mu, sd = xla.mean(), xla.std() + 1e-12
    a0, _ = fit_softmax(YLA, None)
    a, b = fit_softmax(YLA, (xla - mu) / sd)
    g = []
    for i in range(len(LB)):
        q1 = predict(a, b, np.clip((xlb[i] - mu) / sd, -3, 3)); q0 = predict(a0, 0 * a0, 0)
        g.append((YLB[i] * (np.log(q1) - np.log(q0))).sum() / YLB[i].sum())
    return np.mean(g)


ila = [allS.index(s) for s in SITES_]; ilb = [allS.index(s) for s in LB]
tr = np.array([transfer(Xall[ila, j], Xall[ilb, j]) for j in range(len(kall))])
F2 = grf(allS, np.array([10.0, 20.0, 40.0, 80.0, 160.0]), 3000, rng)
tn = np.array([transfer(f[ila], f[ilb]) for f in F2])
o = np.argsort(-tr)
log(f'(c) LA-land-model -> LB archives ({sh}): best {kall[o[0]]} {tr[o[0]]:+.4f}; variables beating field 95% '
    f'({np.quantile(tn, .95):+.4f}): {(tr > np.quantile(tn, .95)).mean():.2f}; median variable {np.median(tr):+.4f} vs field median {np.median(tn):+.4f}')
log('    top:', [(kall[j], round(float(tr[j]), 3)) for j in o[:5]])
open(os.path.join(CK, 'c3.out'), 'w').write('\n'.join(out_lines) + '\n')
