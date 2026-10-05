import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v67_lib as X
d = json.load(open(os.path.join(X.CK, sys.argv[1] if len(sys.argv) > 1 else 'c1.json')))
R = d['rules']; names = list(R[0]['res'].keys())
print('n rules', len(R))
for n in names:
    L = np.array([r['res'][n]['LANG'] for r in R]); A = np.array([r['res'][n]['ASYM_z'] for r in R])
    M = np.array([r['res'][n]['MI_z'] for r in R]); C = np.array([r['res'][n]['REC_z'] for r in R])
    rp = np.array([r['res'][n]['REP_ratio'] for r in R]); ag = np.array([r['res'][n]['agree'] for r in R])
    s = f"{n:8s} LANG med {np.median(L):6.2f} p95 {np.percentile(L,95):6.2f} max {L.max():6.2f} | ASYM med {np.median(A):5.2f} max {A.max():5.2f} | MI med {np.median(M):5.2f} max {M.max():5.2f} | REC med {np.median(C):5.2f} max {C.max():5.2f} | REPratio med {np.median(rp):4.2f} | agree {np.median(ag):.2f}"
    if 'nmi' in R[0]['res'][n]:
        nm = np.array([r['res'][n]['nmi'] for r in R]); s += f" | NMI med {np.median(nm):.3f} max {nm.max():.3f}"
        b = int(np.argmax(nm)); s += f" best={R[b]['rule']} LANG@best {L[b]:.2f}"
    print(s)
    top = np.argsort(-L)[:3]
    print('     top LANG rules:', [(R[i]['rule'], round(L[i], 1)) for i in top])
for i in range(6):
    r = R[[k for k in range(len(R)) if R[k]['i'] == i][0]]
    print('HAND', r['rule'], {n: (round(r['res'][n]['LANG'], 1), round(r['res'][n].get('nmi', 0), 3)) for n in names})
