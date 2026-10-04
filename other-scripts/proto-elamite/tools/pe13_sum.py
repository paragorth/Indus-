"""pe13: summarise kernel fits in a checkpoint folder against their shuffle nulls.
usage: python3 pe13_sum.py c1|c2"""
import glob, json, os, re, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, '..', 'data', 'pe13_ckpt', sys.argv[1])
R = {}
for f in glob.glob(os.path.join(CK, '*.json')):
    r = json.load(open(f))
    if 'gain_best_mbits' in r:
        R[os.path.basename(f)[:-5]] = r


def row(k, r):
    e = r['expw']
    ff = ' '.join('%.2f' % x for x in r['free_f'])
    return '%-18s T %6d gainCV %7.1f  fam %s | tau %6.2f w %.2f rho %.2f g1 %.2f g2 %.2f | free[d2,3,4,5,6-8,9-12,13+] %s | copy %.3f' % (
        k, r['T'], r['gain_best_mbits'], {a: round(b, 1) for a, b in r['fam_gain_mbits'].items()},
        e['tau'], e['w'], e['rho'], e['g1'], e['g2'], ff, r['copy'])


groups = {}
for k in R:
    m = re.match(r'(.*)_shuf\d+$', k)
    if m:
        groups.setdefault(m.group(1), []).append(k)
for k in sorted(R):
    if '_shuf' in k:
        continue
    print(row(k, R[k]))
    if k.split('_raw')[0] in groups and not k.endswith('_raw'):
        S = [R[x] for x in groups[k]]
        for key, f in (('gainCV', lambda r: r['gain_best_mbits']), ('rho', lambda r: r['expw']['rho']),
                       ('g2', lambda r: r['expw']['g2']), ('g1', lambda r: r['expw']['g1']),
                       ('free_d2', lambda r: r['free_f'][0]), ('copy', lambda r: r['copy']),
                       ('insample_free', lambda r: r['free_gain_mbits_insample'])):
            v = np.array([f(r) for r in S])
            o = f(R[k])
            print('    null(%d shuffles) %-13s obs %8.3f  null mean %8.3f sd %7.3f max %8.3f min %8.3f  p_hi %.3f' % (
                len(v), key, o, v.mean(), v.std(), v.max(), v.min(), (1 + (v >= o).sum()) / (1 + len(v))))
