"""v73 cycle-1 report: composite null-calibrated score per rule, discovery-chosen top rule, held-out test,
plant recovery, false-positive targets."""
import os, sys, json, glob, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v73_lib as L, v72_lib as V
import v73_c1 as C1

NULLSET = C1.NULLSET; GEN = [k for k in NULLSET if k != 'LSHUF']
PRIM = [0, 1, 2]   # PI, XP, VS


def load(t):
    R = np.load(os.path.join(L.CK, 'S_%s__REAL.npy' % t))
    N = np.stack([np.load(os.path.join(L.CK, 'S_%s__%s.npy' % (t, k))) for k in NULLSET])
    return R, N


def zscores(R, N):
    mu = N.mean(0); sd = N.std(0)
    floor = np.median(sd, axis=0, keepdims=True)
    return (R - mu) / np.maximum(sd, floor)


def analyse(t, bank, names, C=None, truth=None, top=5):
    R, N = load(t)
    Z = zscores(R, N)                      # rules x half x stat
    comp = Z[:, :, PRIM].sum(2)            # rules x half
    order = np.argsort(-comp[:, 0])
    best = order[0]
    gi = [NULLSET.index(k) for k in GEN]; li = NULLSET.index('LSHUF')
    beat_gen = (R[best, 1, PRIM] > N[gi, best, 1][:, PRIM].max(0))
    beat_ls = (R[best, 1, PRIM] > N[li, best, 1, PRIM])
    rep = float(np.corrcoef(comp[:, 0], comp[:, 1])[0, 1])
    res = dict(target=t, best=names[best], Zdisc=float(comp[best, 0]), Zhold=float(comp[best, 1]),
               Dhold=R[best, 1].round(4).tolist(), gen_max_hold=N[gi, best, 1].max(0).round(4).tolist(),
               beat_gen=beat_gen.tolist(), beat_lshuf=beat_ls.tolist(), rank_r=rep,
               top=[(names[i], round(float(comp[i, 0]), 2), round(float(comp[i, 1]), 2)) for i in order[:top]],
               max_hold_any=float(comp[:, 1].max()))
    if C is not None and truth is not None:
        tidx = truth_index(C, truth)
        sel = C.select(bank[best])
        ok = (sel == tidx)
        res['recovery'] = float(ok.mean())
        # best achievable in bank
        res['recovery_bank_max'] = float(max((C.select(bank[i]) == tidx).mean() for i in order[:200]))
    return res


def truth_index(C, truth):
    # lines are numbered in page order; truth is (page, line-in-page, pos)
    lid = {}
    k = 0
    for pi, p in enumerate(C.pages):
        for li, l in enumerate(p['lines']):
            if l['w']: lid[(pi, li)] = k; k += 1
    t = np.full(C.nl, -2)
    for pi, li, j in truth:
        if (pi, li) in lid: t[lid[(pi, li)]] = C.line_start[lid[(pi, li)]] + j
    return t


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'ZL3b'
    T = C1.targets(src)
    bank, names = L.make_bank(C1.NR)
    out = {}
    for t, P in T.items():
        if not os.path.exists(os.path.join(L.CK, 'S_%s__LSHUF.npy' % t)): continue
        truth = L.jload('truth_%s.json' % t)
        C = L.Corpus(P, t) if truth else None
        r = analyse(t, bank, names, C, truth)
        out[t] = r
        print(json.dumps(r)[:900], flush=True)
    L.jsave('c1_report_%s.json' % src, out)
