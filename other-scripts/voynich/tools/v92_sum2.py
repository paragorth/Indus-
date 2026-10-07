import json, os, numpy as np
CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'v92_ckpt')
for f in sorted(os.listdir(CK)):
    if not (f.startswith('c2_') and f.endswith('.json')): continue
    r = json.load(open(os.path.join(CK, f)))
    T = r['top']; z = [t['te']['z'] for t in T if t['te']]
    b = max(T, key=lambda t: t['te']['z'])
    v = b['v']
    ref = r['ref']
    print('%-12s nh=%d tr-z max %.1f | test z max %.1f med %.1f >=3 %d/20 | best a%d b%d m%s %s %s hi%d n=%d eff %.3f | ref page centroid %s first %s' % (
        r['name'], r['nh'], max(r['tr_z']), max(z), np.median(z), sum(x >= 3 for x in z), v['a'], v['b'], 'Y' if v['merge'] else 'N',
        v['level'], v['stat'], v['hi'], b['te']['n'], b['te']['eff'],
        [round(x['z'], 1) for x in ref['page_centroid']], [round(x['z'], 1) for x in ref['page_first']]))
