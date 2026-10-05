"""pe41: per-sign differential votes, real PE vs the same sign in 3 token-shuffled PE corpora (frequency kept, structure destroyed).
Also split-half: is the differential reproduced in two seed-parity halves of the populations?"""
import sys, os, json, gzip
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe41_lib import LABELS, DATA
from pe41_analyze import load, votes
CK = os.path.join(DATA, 'pe41_ckpt')
runs = load(sys.argv[2:]); meta = json.load(open(os.path.join(CK, 'targets_meta.json')))
NL = len(LABELS)
def diff(sub, real='PE', shufs=('SHUF0', 'SHUF1', 'SHUF2')):
    sr = meta[real]['signs']; V = votes(sub, real, len(sr))
    out = {}
    for i, s in enumerate(sr):
        n = V[i].sum()
        if n < 30: continue
        Vs = np.zeros(NL); ns = 0
        for t in shufs:
            ss = meta[t]['signs']
            if s in ss:
                Vt = votes(sub, t, len(ss))[ss.index(s)] if False else None
        out[s] = (V[i], n)
    return out, V
# precompute vote matrices once
def allv(sub):
    M = {}
    for t in ['PE', 'SHUF0', 'SHUF1', 'SHUF2', 'PC', 'PEA', 'PEB']:
        M[t] = votes(sub, t, len(meta[t]['signs']))
    return M
def zdiff(M, real, shufs):
    sr = meta[real]['signs']; res = {}
    for i, s in enumerate(sr):
        a = M[real][i]; n = a.sum()
        b = np.zeros(NL)
        for t in shufs:
            ss = meta[t]['signs']
            if s in ss: b += M[t][ss.index(s)]
        m = b.sum()
        if n < 30 or m < 60: continue
        pa = a / n; pb = b / m; pp = (a + b) / (n + m)
        z = (pa - pb) / np.sqrt(pp * (1 - pp) * (1 / n + 1 / m) + 1e-12)
        res[s] = (z, pa, pb, n, m)
    return res
par = np.array([r['seed'] % 2 for r in runs])
H = [allv([r for r, k in zip(runs, par) if k == h]) for h in (0, 1)]
Mall = allv(runs)
R0 = zdiff(H[0], 'PE', ['SHUF0', 'SHUF1', 'SHUF2']); R1 = zdiff(H[1], 'PE', ['SHUF0', 'SHUF1', 'SHUF2'])
# control of the control: SHUF0 vs SHUF1+SHUF2 (should give ~nothing)
C0 = zdiff(H[0], 'SHUF0', ['SHUF1', 'SHUF2']); C1 = zdiff(H[1], 'SHUF0', ['SHUF1', 'SHUF2'])
def stable(A, B, zmin=3):
    out = []
    for s in A:
        if s in B:
            zz = np.minimum(A[s][0], B[s][0]); l = int(np.argmax(zz))
            if zz[l] >= zmin: out.append((s, LABELS[l], float(zz[l])))
    return out
sp = stable(R0, R1); sc = stable(C0, C1)
print('PE vs shuffled, stable (both halves z>=3):', len(sp), 'of', len(set(R0) & set(R1)))
print('SHUF0 vs other shuffles (control):', len(sc), 'of', len(set(C0) & set(C1)))
Rall = zdiff(Mall, 'PE', ['SHUF0', 'SHUF1', 'SHUF2'])
rows = []
for s, l, z in sorted(sp, key=lambda x: -x[2]):
    zA, pa, pb, n, m = Rall[s]; li = LABELS.index(l)
    rows.append(dict(sign=s, label=l, zmin_halves=z, share_pe=float(pa[li]), share_shuf=float(pb[li]), n=int(n)))
    print(s, l, round(z, 1), 'PE share %.2f vs shuffled %.2f' % (pa[li], pb[li]), int(n))
print('control signs:', sc[:10])
json.dump(dict(n_stable=len(sp), n_eval=len(set(R0) & set(R1)), n_ctrl=len(sc), n_ctrl_eval=len(set(C0) & set(C1)), rows=rows, ctrl=sc), open(sys.argv[1], 'w'), indent=1)
