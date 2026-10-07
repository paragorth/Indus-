import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v90_lib as L
def isc(p): return p['lift_n'] >= 50 and p['lift_z'] >= 3 and p['lift'] - p['clift'] >= 0.5 and p['clift_z'] < 2
for n in sys.argv[1:]:
    r = json.load(open(os.path.join(L.CK, 'c1b_%s.json' % n)))
    ns = nc = 0; best = None; F = {}
    for s, d in r['schemes'].items():
        for sv in d['surv']:
            ns += 1; cw = [p for p in sv['prof'] if isc(p)]
            nc += bool(cw)
            for p in cw: F['+'.join(p['cols'])] = F.get('+'.join(p['cols']), 0) + 1
            for p in sv['prof']:
                if p['lift_n'] >= 50:
                    sp = p['lift_z'] - max(p['clift_z'], 0)
                    if best is None or sp > best[0]: best = (round(sp, 2), '+'.join(p['cols']), round(p['lift'], 2), round(p['lift_z'], 1), round(p['clift'], 2), round(p['clift_z'], 1))
    d = r['schemes']['0']
    print(n, 'calls %d/%d' % (nc, ns), sorted(F.items(), key=lambda x: -x[1])[:3], 'best', best,
          'whole %.2f z%.1f' % (d['whole']['lift'], d['whole']['z']), 'f2l2 %.2f z%.1f comp %.2f z%.1f' % (d['f2l2']['lift'], d['f2l2']['z'], d['f2l2_comp']['lift'], d['f2l2_comp']['z']),
          'bits best %.2f indep %.2f' % (d['best_bits'], d['indep_bits']))
