"""pe62 cycle-1 summary: held-out scores of real runs vs nulls, controls and plants."""
import os, json, glob, re
import numpy as np
import pe62_common as C

R = {}
for fn in sorted(glob.glob(os.path.join(C.CK, 'c1_*.json'))):
    d = json.load(open(fn))
    R[(d['corpus'], d['mode'])] = d


def fmt(s):
    return 'acc %.3f bacc %.3f bits %.3f' % (s['acc'], s['bacc'], s['bits'])


for corpus in ('PE', 'PC', 'U3'):
    print('==', corpus)
    for mode in ('real', 'blind'):
        d = R.get((corpus, mode))
        if d:
            s = d['ho']
            extra = (' | first-150 ens: ' + fmt(s['ho150'])) if 'ho150' in s else ''
            print('  %-6s n_hyp %4d ho n %d: %s maj %.3f%s groups %s' % (mode, d['n_hyp'], d['n_ho'], fmt(s), s['maj_acc'],
                                                                     extra, d['group_use']))
    for kind in ('lshuf', 'wshuf', 'nshuf'):
        ds = [d for (c, m), d in R.items() if c == corpus and m.startswith(kind)]
        if ds:
            b = [d['ho']['bits'] for d in ds]; a = [d['ho']['acc'] for d in ds]; ba = [d['ho']['bacc'] for d in ds]
            print('  %-6s x%d: acc %.3f-%.3f bacc %.3f-%.3f bits %.3f-%.3f (max)' % (kind, len(ds), min(a), max(a),
                                                                                min(ba), max(ba), min(b), max(b)))
    for kind in ('plantA', 'plantB'):
        for (c, m), d in sorted(R.items()):
            if c == corpus and m.startswith(kind):
                print('  %-8s plant %s rec %s' % (m, d['plant'], {k: round(v, 3) if isinstance(v, float) else v
                                                                   for k, v in d['plant_rec'].items()}))
