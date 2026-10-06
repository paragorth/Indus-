#!/usr/bin/env python3
"""LA-70 cycle 2b: where does the v2 gain come from? Ablations through the identical v2 decoder on the
primary split + 10 splits (each gated on its own training half): KU-RO only; V2 without each new
component (KI as RED, *308 rule, D/B values in closure, staple order -> la60 order, new headings,
PA-DE, template words); la60 v1 minus the la67-killed items. STRICT/FULL/closures vs EMPTY."""
import json, os, statistics as st
from multiprocessing import Pool
from la70_common import *


def variants(tr):
    V2 = gate_v2(tr, 'V2')
    out = {'V2': V2}
    out['TOTonly'] = dict(EMPTY, name='TOT', roles={'KU-RO': 'TOT', 'PO-TO-KU-RO': 'TOT'})
    v = dict(V2, roles={w: r for w, r in V2['roles'].items() if r != 'RED'}, rules=dict(V2['rules'], ki_red=False))
    v['roles']['KI'] = 'COM'; out['-KIred'] = v
    out['-frac308'] = dict(V2, rules=dict(V2['rules'], frac308=False))
    out['-DBvalues'] = dict(V2, v1close=True)
    P = prior_reading()
    out['-stapleorder'] = dict(V2, order=P['order'])
    out['-newheads'] = dict(V2, roles={w: r for w, r in V2['roles'].items() if w not in ('*516', '*307', 'A-DU')})
    out['-PADE-TPL'] = dict(V2, roles={w: r for w, r in V2['roles'].items() if r not in ('MRK', 'TPL')})
    killed = {'KU-RE', 'U', 'A', 'DA', 'KA-NA', 'KA', 'KU', 'SI', 'I', 'OLE+U', 'OLE+MI', 'OLE+DI', 'KU-PA'}
    P2 = dict(P, roles={w: r for w, r in P['roles'].items() if w not in killed}, rules={})
    P2['order'] = {k: v for k, v in P['order'].items() if k != '*307'}
    out['V1-killed'] = P2
    P['rules'] = {}; out['V1'] = P
    return out


def run(s):
    A = admin_docs(load_la())
    tr, te = split(A, 'la60-main' if s < 0 else 'la60-c2-split%d' % s)
    r = {'split': s, 'ntest': len(te), 'EMPTY': score_v2(tr, te, EMPTY)}
    for k, R in variants(tr).items():
        r[k] = score_v2(tr, te, R)
    return r


if __name__ == '__main__':
    with Pool(2) as p:
        res = p.map(run, list(range(-1, 10)))
    json.dump(res, open(os.path.join(CK, 'c2b_ablation.json'), 'w'))
    keys = [k for k in res[0] if k not in ('split', 'ntest')]
    for k in keys:
        print('%-14s' % k, {f: round(st.mean(r[k].get(f, 0) for r in res), 2) for f in ('full', 'strict', 'close', 'tot_tested', 'agree', 'viol', 'rule_n')})
