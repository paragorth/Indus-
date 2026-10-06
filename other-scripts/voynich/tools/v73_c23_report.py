"""v73 cycle-2/3 report (same composite and gates as cycle 1).  usage: v73_c23_report.py 2|3"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v73_lib as L, v72_lib as V
from v73_c1_report import zscores, truth_index, PRIM, NULLSET, GEN

cyc = sys.argv[1] if len(sys.argv) > 1 else '2'
if cyc == '2':
    import v73_c2 as M
    bank, names = L.make_bank2(M.NR); names = names + ['LEARNED(icm+logit)']
else:
    import v73_c3 as M
    TB = M.tables(); names = ['%s:%s' % (k, ''.join(map(str, t))) for k, t in TB] + ['ASCENT:' + k for k in M.KEYS]


def load(t):
    R = np.load(os.path.join(L.CK, 'S%s_%s__REAL.npy' % (cyc, t)))
    N = np.stack([np.load(os.path.join(L.CK, 'S%s_%s__%s.npy' % (cyc, t, k))) for k in NULLSET])
    return R, N


def selector(C, t, i):
    if cyc == '2':
        if i == len(bank):
            return C.select(np.load(os.path.join(L.CK, 'W2_%s__REAL.npy' % t)))
        return C.select(bank[i])
    if i >= len(TB):
        k = M.KEYS[i - len(TB)]
        return M.select_ptr(C, k, np.array(L.jload('tab3_%s__REAL__%s.json' % (t, k))))
    return M.select_ptr(C, *TB[i])


if __name__ == '__main__':
    T = M.targets(); out = {}
    gi = [NULLSET.index(k) for k in GEN]; li = NULLSET.index('LSHUF')
    for t, P in T.items():
        try: R, N = load(t)
        except FileNotFoundError: continue
        Z = zscores(R, N); comp = Z[:, :, PRIM].sum(2)
        order = np.argsort(-comp[:, 0]); b = order[0]
        nlearn = 1 if cyc == '2' else len(M.KEYS)
        lrn = list(range(len(names) - nlearn, len(names)))
        res = dict(target=t, best=names[b], Zdisc=float(comp[b, 0]), Zhold=float(comp[b, 1]),
                   beat_gen=(R[b, 1, PRIM] > N[gi, b, 1][:, PRIM].max(0)).tolist(),
                   beat_ls=(R[b, 1, PRIM] > N[li, b, 1, PRIM]).tolist(), Dhold=R[b, 1].round(4).tolist(),
                   learned=[(names[i], round(float(comp[i, 0]), 2), round(float(comp[i, 1]), 2),
                             (R[i, 1, PRIM] > N[gi, i, 1][:, PRIM].max(0)).tolist()) for i in lrn],
                   rank_r=float(np.corrcoef(comp[:, 0], comp[:, 1])[0, 1]), max_hold_any=float(comp[:, 1].max()),
                   top=[(names[i], round(float(comp[i, 0]), 2), round(float(comp[i, 1]), 2)) for i in order[:5]])
        truth = L.jload('truth_%s.json' % t)
        if truth or t.startswith('V_'):
            C = L.Corpus(P, t)
            if cyc == '2': C = L.extend(C)
            if truth:
                ti = truth_index(C, truth)
                res['recovery'] = float(np.mean(selector(C, t, b) == ti))
                res['recovery_learned'] = [float(np.mean(selector(C, t, i) == ti)) for i in lrn]
            else:
                idx = selector(C, t, b); w = [C.words[j] for j in idx]
                from collections import Counter
                res['stream_head'] = w[:40]; res['stream_top'] = Counter(w).most_common(20)
                res['stream_types'] = len(set(w)); res['stream_n'] = len(w)
        out[t] = res
        print(json.dumps(res)[:1500], flush=True)
    L.jsave('c%s_report.json' % cyc, out)
