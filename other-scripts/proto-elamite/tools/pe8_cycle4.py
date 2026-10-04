"""pe8 cycle 4: do whole skeletons repeat like printed forms?  Coarse skeleton = (header type, reverse use, total
relation, one-sign share, class-sign share, prefix, subscript) on tablets where all are observed.  Statistics:
number of distinct coarse skeletons, share of tablets covered by the 6 most common, entropy (bits), and coupling
between the obverse block (header, one-sign, class, prefix, subscript) and the reverse block (reverse, total) as
mutual information.  Compared with 20 NULL corpora (layout drawn per entry / per field group) and 5 PLANTED 6-form
corpora.  Output: data/pe8_cycle4.json"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe8_common import *
OB = ['HDR', 'ONE', 'CLS', 'PRE', 'SUB']; RB = ['REV', 'TOT']

def H(c):
    n = sum(c.values()); return -sum(v / n * math.log2(v / n) for v in c.values())

def stats(fs):
    fs = [f for f in fs if all(f[k] is not None for k in OB + RB)]
    sk = Counter(tuple(f[k] for k in OB + RB) for f in fs)
    o = Counter(tuple(f[k] for k in OB) for f in fs); r = Counter(tuple(f[k] for k in RB) for f in fs)
    return {'n': len(fs), 'distinct': len(sk), 'distinct_per_tab': len(sk) / len(fs),
            'top6_cover': sum(v for _, v in sk.most_common(6)) / len(fs), 'H': H(sk),
            'MI_obv_rev': H(o) + H(r) - H(sk), 'top6': [[list(k), v] for k, v in sk.most_common(6)]}

R = load_skeletons(); A = [r['A'] for r in R]
out = {'REAL': stats([r['f'] for r in R]), 'NULL': [], 'PLANT': []}
for s in range(1, 21):
    out['NULL'].append(stats([features(a) for a in null_corpus(A, len(A), seed=100 + s)]))
for s in range(1, 6):
    out['PLANT'].append(stats([features(a) for a in planted_corpus(A, len(A), seed=s)[0]]))
for k in ('distinct_per_tab', 'top6_cover', 'H', 'MI_obv_rev'):
    nv = np.array([d[k] for d in out['NULL']]); pv = [round(d[k], 3) for d in out['PLANT']]
    print(k, 'REAL %.3f' % out['REAL'][k], 'NULL %.3f [%.3f-%.3f]' % (nv.mean(), nv.min(), nv.max()), 'PLANT', pv)
print('REAL n', out['REAL']['n'], 'top6', out['REAL']['top6'])
json.dump(out, open(os.path.join(DATA, 'pe8_cycle4.json'), 'w'), indent=1)
