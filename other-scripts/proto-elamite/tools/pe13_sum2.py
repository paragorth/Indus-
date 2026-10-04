"""pe13 cycle 2: per-sign gap profiles vs shuffled false-positive control and split halves."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe4_common import FINAL  # noqa
CK = os.path.join(HERE, '..', 'data', 'pe13_ckpt', 'c2')
L = lambda n: json.load(open(os.path.join(CK, n + '.json')))
PE = L('SIGN_PE')
SH = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(CK, 'SIGN_SHUF*.json')))]
A, B = L('SIGN_HALFA'), L('SIGN_HALFB')


def count(R, key, thr, sign=1):
    return sum(1 for s, r in R.items() if sign * r[key] >= thr)


for key, thr, sg, lab in (('z_near', 3, 1, 'near excess z>=3 (priming/runs)'), ('z_d1', 3, -1, 'gap-1 deficit z<=-3 (refractory)'),
                          ('z_d1', 3, 1, 'gap-1 excess z>=3'), ('z_near', 3, -1, 'near deficit z<=-3')):
    sh = [count(r, key, thr, sg) for r in SH]
    print('%-34s PE %3d | shuffled corpora mean %.2f max %d' % (lab, count(PE, key, thr, sg), np.mean(sh), max(sh)))
for k in ('PRIME', 'TOPIC', 'MIX'):
    R = L('SIGN_PL_' + k)
    print('planted %-6s near z>=3: %d  d1 z<=-3: %d  of %d' % (k, count(R, 'z_near', 3), count(R, 'z_d1', 3, -1), len(R)))
# aggregate
for nm, R in (('PE', PE), ('A', A), ('B', B)):
    o = np.array([r['obs'] for r in R.values()]).sum(0)
    e = np.array([r['exp'] for r in R.values()]).sum(0)
    print('%s pooled obs/exp gap1 %.3f gap2-3 %.3f gap4+ %.3f' % (nm, o[0] / e[0], o[1] / e[1], o[2] / e[2]))
for lab, sel in (('class signs', lambda s: s in FINAL), ('other signs', lambda s: s not in FINAL)):
    o = np.array([r['obs'] for s, r in PE.items() if sel(s)]).sum(0)
    e = np.array([r['exp'] for s, r in PE.items() if sel(s)]).sum(0)
    print('PE %-11s obs/exp gap1 %.3f gap2-3 %.3f gap4+ %.3f (pairs %d)' % (lab, o[0] / e[0], o[1] / e[1], o[2] / e[2], o.sum()))
common = [s for s in PE if A[s]['ntab'] >= 6 and B[s]['ntab'] >= 6]
for key in ('z_near', 'z_d1'):
    a = np.array([A[s][key] for s in common])
    b = np.array([B[s][key] for s in common])
    print('split-half %s: r = %.3f over %d signs' % (key, np.corrcoef(a, b)[0, 1], len(common)))
    for sg, nm in ((1, 'excess'), (-1, 'deficit')):
        both = [s for s in common if sg * A[s][key] >= 2 and sg * B[s][key] >= 2]
        print('   %s z>=2 in both halves: %s' % (nm, both))
print('\nTop PE signs by z_near:')
for s, r in sorted(PE.items(), key=lambda x: -x[1]['z_near'])[:15]:
    print('  %-14s ntab %3d obs %s exp %s z_near %.1f z_d1 %.1f r_d1 %.2f %s' % (
        s, r['ntab'], [int(x) for x in r['obs']], [round(x, 1) for x in r['exp']], r['z_near'], r['z_d1'], r['r_d1'], 'CLASS' if s in FINAL else ''))
print('\nMost refractory PE signs (z_d1):')
for s, r in sorted(PE.items(), key=lambda x: x[1]['z_d1'])[:12]:
    print('  %-14s ntab %3d obs %s exp %s z_d1 %.1f r_d1 %.2f %s' % (
        s, r['ntab'], [int(x) for x in r['obs']], [round(x, 1) for x in r['exp']], r['z_d1'], r['r_d1'], 'CLASS' if s in FINAL else ''))
