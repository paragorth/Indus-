#!/usr/bin/env python3
"""LA-43: summary tables from the cycle checkpoints (data/la43_ckpt/c1.json, c2.json, c3.json)."""
import os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la43_common as M


def load(n):
    p = os.path.join(M.CK, f'c{n}.json')
    return json.load(open(p)) if os.path.exists(p) else []


def c1():
    D = load(1)
    print('== cycle 1 (single splits; bits / mask / auc: z, P) ==')
    agg = collections.defaultdict(list)
    for d in sorted(D, key=lambda d: (d['tag'], d['tier'])):
        r = d['res']
        grp = d['tag'].rsplit('_', 1)[0] if d['tag'][:3] in ('LB_', 'PL_', 'NP_') and d['tag'] != 'LB_KNPY' else None
        if grp:
            agg[(grp, d['tier'])].append(r)
            continue
        print(f"{d['tag']:8s} {d['tier']:4s} types {d['ntypes']:5d} N {d['N']:6d} " +
              ' '.join(f"{k} g{r[k]['gain']:+.3f}/n{r[k]['null_gain']:+.3f} z{r[k]['z']:+.1f} P{r[k]['p']:.3g}" for k in ('bits', 'mask', 'auc') if k in r))
    for (g, t), rs in sorted(agg.items()):
        zb = [x['bits']['z'] for x in rs]
        print(f"{g:8s} {t:4s} n{len(rs)} bits z median {np.median(zb):+.1f} range [{min(zb):+.1f},{max(zb):+.1f}] "
              f"P<.05 bits {sum(x['bits']['p'] < .05 for x in rs)} mask {sum(x['mask']['p'] < .05 for x in rs)} "
              f"auc {sum(x['auc']['p'] < .05 for x in rs if 'auc' in x)}")


def c2():
    D = load(2)
    if not D:
        return
    print('== cycle 2 (LOSO) ==')
    lbz = collections.defaultdict(list)
    for d in sorted([d for d in D if d['kind'] == 'tier'], key=lambda d: (d['tag'], d['tier'])):
        r = d['res']
        print(f"{d['tag']:5s} {d['tier']:4s} " + ' '.join(f"{k} g{v['gain']:+.3f}/n{v['null_gain']:+.3f} z{v['z']:+.1f} P{v['p']:.3g}" for k, v in r.items()))
    rows = [d for d in D if d['kind'] == 'row']
    print('-- rows (bits z, P) --')
    by = collections.defaultdict(dict)
    for d in rows:
        by[d['row']][d['tag']] = d
    for row in sorted(by):
        print(f"{row or 'V':3s} " + ' '.join(f"{t}:{x['nsign']} z{x['bits']['z']:+.1f} P{x['bits']['p']:.3g}" for t, x in sorted(by[row].items())))
    lb = [d for d in rows if d['tag'].startswith('LB')]
    if lb:
        print(f"LB true rows passing P<.05: {sum(d['bits']['p'] < .05 for d in lb)}/{len(lb)}")


def c3():
    D = load(3)
    if not D:
        return
    print('== cycle 3 (sound links) ==')
    for d in sorted(D, key=lambda d: d['tag']):
        print(f"{d['tag']} links {d['nlinks']} last {d['nlast']}")
        for t, x in d['tiers'].items():
            print('   ', t, ', '.join(f"{k} {v['obs']:.3f}/{v['null']:.3f} z{v['z']:+.1f} P{v['p']:.3g}" for k, v in x.items()))


if __name__ == '__main__':
    c1(); c2(); c3()
