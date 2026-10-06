"""v72 cycle 2 summary table."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v72_lib as L

D = {}
d = os.path.join(L.CK, 'c2')
for f in sorted(os.listdir(d)):
    r = json.load(open(os.path.join(d, f))); D[(r['name'], r['view'])] = r
for (n, v), r in D.items():
    a = r['LAD']
    print(f"{n:20s} {v:8s} types={r['types']:5d} alph={r['alphabet']:2d} len={r['mean_len']:.2f} "
          f"gen+{a['gain_gen']:.3f} page+{a['gain_page']:.3f} big+{a['gain_big']:.3f} EXC={a['excess']:.3f} "
          f"arr={[x['surv'] for x in r['ARR']]} G={[round(x['G'], 3) for x in r['GAP']]} "
          f"sec={r['SEC']['acc']:.2f}/{r['SEC']['null']:.2f} z{r['SEC']['z']:.1f} "
          f"rec={r['REC']['ratio']:.2f} z{r['REC']['z']:.1f}"
          + (f" pur={r['RECOV']['purity']:.3f}/{r['RECOV']['inv_purity']:.3f}" if 'RECOV' in r else ''))
