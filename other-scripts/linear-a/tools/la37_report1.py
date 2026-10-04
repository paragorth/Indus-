#!/usr/bin/env python3
"""LA-37 cycle 1 summary of the calibration runs (reads data/la37_ckpt/c1.json)."""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import la37_common as K

R = json.load(open(os.path.join(K.CK, 'c1.json')))
by = collections.defaultdict(list)
for k, s, r in R:
    by[k.split('|')[0] if k.startswith('PL') else k].append((k, r))


def agg(name):
    rs = [r for _, r in by[name]]
    out = {}
    for key in ('T', 'z', 'fm'):
        out[key] = {g: round(float(np.nanmean([r[key][g] for r in rs])), 3) for g in ('doublet', 'sameC', 'sameV', 'top20rel')}
    out['nflag'] = round(float(np.mean([r['nflag'] for r in rs])), 1)
    out['flag_rel'] = round(float(np.nanmean([r['flag_rel'] for r in rs])), 3) if any(r['flag_rel'] == r['flag_rel'] for r in rs) else None
    out['base_rel'] = round(float(np.mean([r['base_rel'] for r in rs])), 3)
    out['ndoub'] = round(float(np.mean([r['ndoub'] for r in rs])), 1)
    return out


for name in ('LBfull', 'LBfull_sh', 'LBdraw', 'LBdraw_sh'):
    print(name, json.dumps(agg(name)))
print('LBfull doublets (pair, fm pct, z):', by['LBfull'][0][1]['doub_pct'])
pl = collections.defaultdict(list)
for k, r in by['PL']:
    if r:
        pl[k.split('|')[2]].append((k.split('|')[1], r))
for m, lst in pl.items():
    print(m, 'n', len(lst), 'flag', sum(r['flag'] for _, r in lst), 'fwer<=.05', sum(r['fwer'] <= 0.05 for _, r in lst),
          'q<=.1', sum(r['q'] <= 0.1 for _, r in lst), 'rank1 among X* pairs', sum(r['rank_among_Xpairs'] == 1 for _, r in lst),
          'median pct_all', round(float(np.median([r['pct_all'] for _, r in lst])), 3),
          'fm', [round(r['fm'], 2) for _, r in lst], 'nX', [r['nX'] for _, r in lst])
