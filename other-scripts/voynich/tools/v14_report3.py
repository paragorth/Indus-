import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from v14_lib import *
R = lambda d: DEV_ORDER.index(d) if d else len(DEV_ORDER)
for n in ('planted_noisy30', 'voynich_ZL3b', 'voynich_ZL3b_A', 'voynich_ZL3b_B', 'latin_verbose', 'italian_verbose'):
    r = load('c3a_' + n)
    if not r: continue
    print('noisy', n)
    for k, rec in enumerate(r['slots']):
        zmin = min(rec['z'].values()); tvmin = min(rec['tv'].values())
        bd = min(rec['z'], key=rec['z'].get)
        idd, idk = rec['ident']
        s = []
        for sg, nl in rec['null'].items():
            pz = (sum(x[0] <= zmin for x in nl) + 1) / (len(nl) + 1)
            pt = (sum(x[1] <= tvmin for x in nl) + 1) / (len(nl) + 1)
            pr = (sum(R(x[2][0]) <= R(idd) for x in nl) + 1) / (len(nl) + 1)
            s.append('s%.1f: Pz=%.2f Ptv=%.2f Pid=%.2f nullTVmed=%.3f' % (sg, pz, pt, pr, np.median([x[1] for x in nl])))
        print('  slot%d ident=%s k=%s best=%s z=%.1f TV=%.3f | %s' % (k + 1, idd, idk, bd, zmin, tvmin, ' | '.join(s)))
for n in ('voynich', 'latin_verbose', 'italian_verbose', 'planted_mixed'):
    r = load('c3c_' + n)
    if not r: continue
    for q, v in r.items():
        zb = min(v['z_full'].values())
        print('other', n, q, 'ident', v['ident'], 'bestz %.1f' % zb, 'null', dict(v['null_ident']), v['counts'][:14])
for n in ('planted_3tables', 'planted_mixed', 'voynich_ZL3b', 'latin_verbose', 'italian_verbose'):
    r = load('c3b_' + n)
    if not r: continue
    best = max(r, key=lambda c: r[c][1])
    print('lc', n, 'heldout best C=%d' % best, ' '.join('C%d:%.3f/%.3f' % (c, r[c][1], r[c][2]) for c in sorted(r)))
