"""pe46 cycle 2 summary: survivors and positional tracking per variable."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe46_lib as L

out = {}
for tag in ('PE', 'PEv'):
    D = {k: json.load(open(os.path.join(L.CK, f'c2_{tag}{k}.json'))) for k in ('', '~sig', '~tab', '~mkv')}
    rep = dict(survivors=[], track={})
    for split in (0, 1):
        nullh = [o['held']['CODE'] for k in ('~sig', '~tab', '~mkv') for o in D[k]['res'][split]]
        thr = float(np.percentile(nullh, 95))
        rep.setdefault('thr', []).append(round(thr, 3))
        rep.setdefault('held_med', []).append({k or 'real': round(float(np.median([o['held']['CODE'] for o in D[k]['res'][split]])), 3) for k in D})
        for o in D['']['res'][split]:
            if o['held']['CODE'] > thr:
                h = o['held']
                rep['survivors'].append(dict(split=split, s=o['s'], held=round(h['CODE'], 3), fit=round(o['fit']['CODE'], 3),
                                             R=h['R'], tgt=h['tgt'],
                                             best_var=[h['varlist'][int(np.argmax(e))] + ':%.3f' % max(e) for e in h['E']]))
    # positional tracking: per variable, median over top-20 of max over slots held-out E, real vs ~sig
    vl = D['']['res'][0][0]['held']['varlist']
    for j, v in enumerate(vl):
        r = [float(np.median([max(e[j] for e in o['held']['E']) for o in D[k]['res'][s]])) for k in ('', '~sig') for s in (0, 1)]
        rep['track'][v] = dict(real=[round(r[0], 4), round(r[1], 4)], sig=[round(r[2], 4), round(r[3], 4)])
    out[tag] = rep
json.dump(out, open(os.path.join(L.CK, 'c2_summary.json'), 'w'), indent=1)
for tag, rep in out.items():
    print(tag, 'thr', rep['thr'], 'held_med', rep['held_med'])
    print(' survivors', len(rep['survivors']))
    for s in rep['survivors'][:10]:
        print('  ', s)
    for v, t in rep['track'].items():
        print('  ', v, t)
