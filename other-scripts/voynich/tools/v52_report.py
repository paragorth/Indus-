"""v52: summarise a search run (tag) -> printed table + json."""
import sys, os, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v52_lib as L

tag = sys.argv[1] if len(sys.argv) > 1 else 'c1'
S = pickle.load(open(os.path.join(L.CK, f'{tag}_summary.pkl'), 'rb'))


def schstr(s):
    return ' '.join(f"{t}:{''.join(a) if isinstance(a, tuple) else a}" if t in ('fix', 'end') else f"{t}{{{','.join(a)}}}"
                    for t, a in s['rules']) + f" N{s['N']}"


def slot_roles(r):
    out = []
    for s in range(r['K']):
        e = np.array(r['E'][s]); j = int(np.argmax(e))
        out.append(f"s{s}:{r['varlist'][j]}{e[j]:.2f}/H{r['H'][s]:.1f}")
    return ' '.join(out)


rep = {}
for n, d in S.items():
    fsA = np.array([x[1] for x in d['search']])
    top = d['top']
    fsB = np.array([t['B']['FS'] for t in top]); dep = np.array([t['B']['DEP'] for t in top])
    totB = np.array([t['B']['TOT'] for t in top])
    ntg = np.array([len(t['B']['tgt']) for t in top])
    best = max(top, key=lambda t: t['A']['FS'])
    rec = None
    C = pickle.load(open(os.path.join(L.CK, f'corpus_{n}.pkl'), 'rb'))
    if C.get('true') is not None:
        codes, _ = L.slot_codes(C, best['sch'])
        tr = C['true']
        if isinstance(tr, np.ndarray):
            fields = [tr[:, k] for k in range(tr.shape[1])]
        elif n == 'GORILA':
            fields = [np.array([x[k] for x in tr]) for k in range(3)]
        else:
            fields = [np.array([x[0] for x in tr]), np.array([x[1] for x in tr]),
                      np.array([x[3] if len(x) > 3 else '-' for x in tr]),
                      np.array(['WITH' in x for x in tr])]
        rec = [round(max(L.nmi(f, c) for c in codes), 3) for f in fields]
    rep[n] = dict(fsA_med=float(np.median(fsA)), fsA_max=float(fsA.max()), fsB_med=float(np.median(fsB)),
                  fsB_max=float(fsB.max()), dep_med=float(np.median(dep)), dep_min=float(dep.min()),
                  tot_med=float(np.median(totB)), ntgt=float(ntg.mean()), best=schstr(best['sch']),
                  bestB=round(best['B']['FS'], 3), bestDEP=round(best['B']['DEP'], 3),
                  roles=slot_roles(best['B']), tgt=best['B']['tgt'], recovery=rec)
    print(n, json.dumps(rep[n]))
json.dump(rep, open(os.path.join(L.CK, f'{tag}_report.json'), 'w'), indent=1)
