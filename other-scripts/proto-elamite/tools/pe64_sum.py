#!/usr/bin/env python3
"""pe64 summary of cycle 1 files: validated rates per run (p_swap <= 0.005, kB >= 2)."""
import glob, json, os
import pe64_lib as L

for fn in sorted(glob.glob(os.path.join(L.CK, 'c1_*.json'))):
    d = json.load(open(fn))
    ok = [x for x in d['tests'] if x['p_swap'] <= 0.005 and x['kB'] >= 2]
    near = [x for x in d['tests'] if 0.005 < x['p_swap'] <= 0.05 and x['kB'] >= 2]
    print('%-28s rules %d cand %d surv %d rates %d tests %d | validated %s | p<=.05 %s | truth %s' % (
        os.path.basename(fn), d['n_rules'], d['n_cand'], d['n_surv'], d['n_rates'], d['n_tests'],
        sorted({(x['rate'], x['sel'], x['kB'], round(x['nullB_swap'], 1)) for x in ok}),
        sorted({x['rate'] for x in near}), d.get('truth')))
