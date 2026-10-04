#!/usr/bin/env python3
"""X-2 cycle 2 summary: control calibration (LB <-> Ur III) and LA <-> PE read-off with shuffled nulls."""
import json, os, sys
from collections import defaultdict, Counter
import numpy as np
import x2_common as X

CK = os.path.join(X.DX, sys.argv[1] if len(sys.argv) > 1 else 'c2_worlds.jsonl')
G = json.load(open(X.GOLD_FILE))
rows = [json.loads(l) for l in open(CK)]
by = defaultdict(list)
for r in rows:
    by[r['name']].append(r)
out = {}
print({k: len(v) for k, v in by.items()})


def modal(d):
    c = Counter(d); t, n = c.most_common(1)[0]
    return t, n / sum(c.values())


for m in X.METHODS2:
    print('\n==== method', m)
    # ---------- control
    for ctl in ('ctl_lape', 'ctl_pe'):
        real = [r['res'][m]['gold_c'] for r in by[ctl] if r['res'][m]['gold_c']]
        sh = [r['res'][m]['gold_c'] for r in by[ctl + '_s'] if r['res'][m]['gold_c']]
        if not real:
            continue
        mr = np.array([x[0] for x in real]); ms = np.array([x[0] for x in sh])
        chance = np.mean([np.mean([1 / k for k in range(1, x[2] + 1)]) for x in real])
        realw = [r['res'][m]['gold_w'] for r in by[ctl] if r['res'][m]['gold_w']]
        shw = [r['res'][m]['gold_w'] for r in by[ctl + '_s'] if r['res'][m]['gold_w']]
        # stability -> precision
        pts = []
        for r in by[ctl]:
            for s, d in r['res'][m]['top'].items():
                if s in G['gold']:
                    t, st = modal(d)
                    if any(x in G['gold'][s] for x in [k for k in d]) or True:
                        pts.append((st, t in G['gold'][s], s, t))
        shst = [modal(d)[1] for r in by[ctl + '_s'] for s, d in r['res'][m]['top'].items() if s in G['gold']]
        thr = float(np.percentile(shst, 95)) if shst else 1.0
        hi = [p for p in pts if p[0] >= thr]
        o = {'mrr_real': float(mr.mean()), 'mrr_shuf': float(ms.mean()) if len(ms) else None,
             'mrr_shuf95': float(np.percentile(ms, 95)) if len(ms) else None, 'chance_mrr_approx': float(chance),
             'worlds_gt_shuf95': float(np.mean(mr > np.percentile(ms, 95))) if len(ms) else None,
             'P_mean_vs_shuf': float((1 + (ms >= mr.mean()).sum()) / (1 + len(ms))) if len(ms) else None,
             'word_mrr_real': float(np.mean([x[0] for x in realw])) if realw else None,
             'word_mrr_shuf': float(np.mean([x[0] for x in shw])) if shw else None,
             'stab_thr_shuf95': thr, 'n_pairs': len(pts), 'prec_all': float(np.mean([p[1] for p in pts])),
             'n_hi': len(hi), 'prec_hi': float(np.mean([p[1] for p in hi])) if hi else None,
             'hi_examples': Counter('%s>%s%s' % (p[2], p[3], '+' if p[1] else '-') for p in hi).most_common(15)}
        out['%s|%s' % (m, ctl)] = o
        print(ctl, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()})
    # ---------- LA <-> PE
    if 'lape' in by:
        L = by['lape'][0]['res'][m]
        res = {}
        nullmax = defaultdict(list); famw = []
        for r in by['lape_s']:
            mx = 0
            for s, d in r['res'][m]['top'].items():
                st = modal(d)[1]; nullmax[s].append(st); mx = max(mx, st)
            famw.append(mx)
        fam95 = float(np.percentile(famw, 95)) if famw else 1.0
        nshuf_hi = [sum(modal(d)[1] >= fam95 for d in r['res'][m]['top'].values()) for r in by['lape_s']]
        allnull = [x for v in nullmax.values() for x in v]
        item95 = float(np.percentile(allnull, 95)) if allnull else 1.0
        rev = {s: modal(d) for s, d in L['btop'].items()}
        for s, d in sorted(L['top'].items(), key=lambda kv: -sum(kv[1].values())):
            t, st = modal(d)
            mutual = rev.get(t, (None, 0))[0] == s
            pnull = float((1 + sum(x >= st for x in allnull)) / (1 + len(allnull)))
            res[s] = {'pe': t, 'stab': round(st, 3), 'mutual': mutual, 'rev_stab': round(rev.get(t, (None, 0))[1], 3) if mutual else None,
                      'p_item': round(pnull, 4), 'surv_fam': st >= fam95, 'top3': Counter(d).most_common(3)}
        wres = {s: modal(d) for s, d in L['wtop'].items()}
        nreal_hi = sum(v['surv_fam'] for v in res.values())
        o = {'fam95': fam95, 'item95': item95, 'n_real_hi': nreal_hi,
             'n_shuf_hi_mean': float(np.mean(nshuf_hi)) if nshuf_hi else None,
             'P_count': float((1 + sum(x >= nreal_hi for x in nshuf_hi)) / (1 + len(nshuf_hi))) if nshuf_hi else None,
             'pairs': res, 'words': {s: (t, round(st, 2)) for s, (t, st) in wres.items()}}
        out['%s|lape' % m] = o
        print('LA<->PE fam95 %.3f item95 %.3f survivors %d (shuffled mean %.2f, P %.3f)' % (
            fam95, item95, nreal_hi, o['n_shuf_hi_mean'] or -1, o['P_count'] or -1))
        for s, v in res.items():
            print('   ', s, v)
        print('   words', o['words'])
json.dump(out, open(os.path.join(X.DX, 'c2_summary.json'), 'w'), indent=1, default=str)
