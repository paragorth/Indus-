import json, os, sys, numpy as np
CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'v92_ckpt')
for f in sorted(os.listdir(CK)):
    if not (f.startswith('c1_') and f.endswith('.json')): continue
    r = json.load(open(os.path.join(CK, f)))
    T = r['top']; z = [t['z'] for t in T]
    sd = np.array(r['score_dist'])
    b = max(T, key=lambda t: t['z'])
    print('%-9s nh=%d train-score max %.4f p99 %.4f | test z: max %.1f median %.1f n>=3 %d/20 | best %s g_te %.4f twin %.4f wls %s' % (
        r['name'], r['nh'], sd.max(), np.percentile(sd, 99), max(z), np.median(z), sum(x >= 3 for x in z),
        [b['h']['unit'], b['h']['lo'], b['h']['hi'], b['h']['atoms'], b['h']['cross']], b['g_te'], b['tw_mu'],
        None if b['g_wls'] is None else round(b['g_wls'], 4)))
