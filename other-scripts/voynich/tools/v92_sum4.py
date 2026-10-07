import json, os, numpy as np
CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'v92_ckpt')
for n in ['ZL3b', 'IT2a', 'GC2a', 'P_KONRAD', 'P_CIRCA', 'P_APIC', 'P_HYGIN', 'P_CULP', 'G_GM', 'G_GM2', 'G_LX', 'G_SEED', 'G_SELF']:
    p = os.path.join(CK, 'c4_%s.json' % n)
    if not os.path.exists(p): continue
    r = json.load(open(p)); T = r['top']
    a = [t['acc_te'] for t in T]; z = [t['z_te'] for t in T]; af = [t['acc_far'] for t in T]
    ref = r['ref']['h1']
    ur = np.array([u[1] for u in ref['units']])
    print('%-9s held-out acc median %.3f (max %.3f) z>=3 %d/20 | far median %.3f | ref frame-uni h1 acc %.3f z %.1f far %.3f | unit ranks: share<0.2 %.2f share>0.8 %.2f n=%d' % (
        n, np.median(a), max(a), sum(x >= 3 for x in z), np.median(af), ref['acc'], ref['z'], ref['acc_far'], (ur < 0.2).mean(), (ur > 0.8).mean(), len(ur)))
