import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v67_lib as X
R = [json.loads(l) for l in open(os.path.join(X.CK, 'c2.jsonl'))]
R.sort(key=lambda r: r['i'])
names = list(R[0]['res'].keys())
print('rules', len(R))
def arr(n, h, k): return np.array([r['res'][n][h].get(k, np.nan) for r in R])
for n in names:
    ao, ae = arr(n, 'o', 'agx'), arr(n, 'e', 'agx')
    lo, le = arr(n, 'o', 'LANG'), arr(n, 'e', 'LANG')
    rpo = arr(n, 'e', 'REP_ratio')
    ia = int(np.argmax(ao)); ib = int(np.argmax(lo))
    t10a = np.argsort(-ao)[:10]; t10b = np.argsort(-lo)[:10]
    s = (f"{n:5s} agx odd max {ao.max():.4f} med {np.median(ao):.4f} | sel-by-agx: rule {R[ia]['rule']} even agx {ae[ia]:.4f} even LANG {le[ia]:.2f} (top10 mean {le[t10a].mean():.2f}) | "
         f"sel-by-LANG: odd {lo[ib]:.2f} -> even {le[ib]:.2f} (top10 {le[t10b].mean():.2f}); corr(odd,even LANG) {np.corrcoef(lo, le)[0,1]:.2f}; even LANG all-rule mean {le.mean():.2f}; REPratio med {np.nanmedian(rpo):.2f}")
    if 'nmix' in R[0]['res'][n]['e']:
        nx = arr(n, 'e', 'nmix'); s += f" | nmix@agx {nx[ia]:.3f} max {nx.max():.3f}"
    print(s)
