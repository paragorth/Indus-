"""pe77 cycle 3: power of the name-magnitude feature (MAGSTR) at PE shape.  Same forger seeds / splits in
PE, PEPN (planted on name-like keys, >= 2 signs), PEPK (planted on any key with 2-20 uses) and PCS (PC
subsampled to the PE tablet count).  Paired differences over identical forgers."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe77_common as pc
from pe77_analyze import load, unforg

FE = ['MAGSTR', 'AR_tot', 'AR_last', 'AR_block', 'COH_pmi', 'BIND_all', 'MAG_all', 'NUM_rep', 'REP_str', 'ST_nlines']
tags = ['PE_300000', 'PEPN_300000', 'PEPK_300000', 'PCS_300000', 'PC_300000', 'PE_shuf_400000', 'PCS_shuf_400000']
D = {t: {r['k']: r for r in load(t)} for t in tags if os.path.exists(os.path.join(pc.CKPT, t + '.jsonl'))}
out = {}
for t, R in D.items():
    U, a, b = unforg(list(R.values()))
    out[t] = {'n': len(R), 'A_strong': a, 'A_weak': b,
              'U': {f: [round(U[f][0], 3), round(U[f][3], 3), None if U[f][1] is None else round(U[f][1], 3)] for f in FE if f in U}}
    print(t, out[t])
# paired: all forgers, z of MAGSTR
def z(r, f):
    v = r['f'].get(f)
    return None if v is None else v[1]
pairs = {}
for a, b in [('PEPN_300000', 'PE_300000'), ('PEPK_300000', 'PE_300000'), ('PCS_300000', 'PE_300000')]:
    if a not in D or b not in D:
        continue
    ks = sorted(set(D[a]) & set(D[b]))
    for f in ['MAGSTR', 'AR_last', 'COH_pmi']:
        d = [z(D[a][k], f) - z(D[b][k], f) for k in ks if z(D[a][k], f) is not None and z(D[b][k], f) is not None]
        d = np.array(d)
        pairs['%s-%s:%s' % (a, b, f)] = [len(d), round(float(d.mean()), 3), round(float(d.std() / np.sqrt(len(d))), 3),
                                         round(float((d > 0).mean()), 3)]
print(json.dumps(pairs, indent=0))
out['paired'] = pairs
json.dump(out, open(os.path.join(pc.CKPT, 'cycle3.json'), 'w'), indent=1)
