#!/usr/bin/env python3
"""la83 cycle 2 summary: per battery item, on seed-noise runs and on perturbation levels M and S:
pass rate, sign-flip rate (effect direction reversed) and the statistic's 5-50-95 % range."""
import json, os, sys
import numpy as np
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'la83_ckpt')
STAT = {'B1_kuro': 'exact', 'B2a_frac_fixed': 'reversals', 'B2b_frac_learned': 'agree', 'B3_affix_all': 'pre',
        'B4_ME': 'z', 'B5_comm_order': 'agree', 'B6_sites': 'real', 'B7_lb_shared': 'z', 'M1_prefix_I': 'z',
        'P_planted_prefix': 'z'}


def load(level):
    fn = os.path.join(CK, 'perturb_%s.jsonl' % level)
    return [json.loads(l) for l in open(fn)] if os.path.exists(fn) else []


out = {}
for level in ('seed', 'M', 'S'):
    R = load(level)
    if not R: continue
    keys = [k for k in R[0] if not k.startswith('_')]
    res = {}
    for k in keys:
        rows = [r[k] for r in R if k in r]
        st = STAT.get(k, 'z')
        vals = np.array([float(x.get(st)) if x.get(st) is not None else np.nan for x in rows])
        res[k] = dict(n=len(rows), pass_rate=float(np.mean([bool(x['passed']) for x in rows])),
                      sign_flip=float(np.mean([not bool(x['sign']) for x in rows])), stat=st,
                      q=[float(np.nanpercentile(vals, q)) for q in (5, 50, 95)])
        if k == 'B3_affix_all':
            sv = np.array([x['suf'] for x in rows]); res[k]['q_suf'] = [float(np.percentile(sv, q)) for q in (5, 50, 95)]
        if k == 'B1_kuro':
            res[k]['P_q'] = [float(np.percentile([x['P'] for x in rows], q)) for q in (50, 95)]
    out[level] = res
    print('== level', level, 'runs', len(R))
    for k, v in res.items():
        print('%-18s pass %.3f  sign-flip %.3f  %s 5/50/95%% %s' % (k, v['pass_rate'], v['sign_flip'], v['stat'],
              ' '.join('%.3g' % x for x in v['q'])) + ('  suf %s' % ' '.join('%.3g' % x for x in v['q_suf']) if 'q_suf' in v else '')
              + ('  P50/95 %s' % ' '.join('%.2g' % x for x in v['P_q']) if 'P_q' in v else ''))
json.dump(out, open(os.path.join(CK, 'c2_summary.json'), 'w'), indent=1)
