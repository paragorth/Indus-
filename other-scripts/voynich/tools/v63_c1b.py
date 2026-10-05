"""v63 cycle 1b: power of the simple orientation statistics INSIDE the Voynich (Voynich + planted
uncorrected slips, real Plaoul type mix, Voynich vocabulary), and a position-preserving column-shuffle null."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v63_lib as L, v63_plant as PL
from v63_c1 import metrics, fmt

zl = L.voynich_paras('ZL3b')
res = {}
for nm, S in [('V-ZL3b', zl)] + [(f'V-ZL + slips {r}%', PL.plant_slips(zl, r / 100, seed=2)) for r in (1, 3, 10)] + \
        [(f'V-ZL column-shuffle #{k}', PL.column_shuffle(zl, seed=k)) for k in range(4)]:
    m = metrics(S, with_D=False); res[nm] = m; print(fmt(nm, m), flush=True)
json.dump(res, open(os.path.join(L.CK, 'c1b_metrics.json'), 'w'), indent=1)
