#!/usr/bin/env python3
"""LA-72 cycle 2 summary: means over the 11 splits per version; gains of each small component."""
import json, os, statistics as st
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'la72_ckpt')
res = json.load(open(os.path.join(CK, 'c2_LA.json')))
out = {}
for ver in ('rd', 'read', 'rnd'):
    R = [r for r in res if r['ver'] == ver]
    m = lambda k, f: st.mean(r[k].get(f, 0) for r in R)
    nt = st.mean(r['ntest'] for r in R)
    o = dict(ntest=nt, nsplits=len(R))
    for k in ('V2c', 'EMPTY', 'V2la70', 'V1PRIOR', '-staple', '-frac308', 'TOTonly', '-DB', '-KIred', '-order'):
        o[k] = {f: round(m(k, f), 2) for f in ('full', 'strict', 'close', 'tot_tested', 'agree', 'viol', 'rule_n')}
    o['shuf'] = {f: round(st.mean(r['V2c_shuf'][f][0] for r in R), 2) for f in ('full', 'strict', 'close', 'agree', 'viol')}
    o['rand'] = {f: round(st.mean(r['V2c_rand'][f][0] for r in R), 2) for f in ('full', 'strict', 'close')}
    o['n_sig_strict'] = sum(1 for r in R if r['split'] >= 0 and r['V2c_shuf']['strict'][1] <= 0.05)
    o['n_sig_close'] = sum(1 for r in R if r['split'] >= 0 and r['V2c_shuf']['close'][1] <= 0.05)
    o['gain_pct'] = round(100 * (o['V2c']['strict'] - o['shuf']['strict']) / nt, 2)
    o['order_agree'] = round(o['V2c']['agree'] / max(1e-9, o['V2c']['agree'] + o['V2c']['viol']), 3)
    o['shuf_order_agree'] = round(o['shuf']['agree'] / max(1e-9, o['shuf']['agree'] + o['shuf']['viol']), 3)
    for k in ('-staple', '-frac308', '-DB', '-KIred', '-order'):
        d = [r['V2c']['strict'] - r[k].get('strict', 0) for r in R]
        o['gain_' + k] = (round(st.mean(d), 2), sum(x > 0 for x in d), sum(x < 0 for x in d))
    d = [r['V2c']['strict'] - r['V2la70'].get('strict', 0) for r in R]
    o['V2c_minus_la70'] = (round(st.mean(d), 2), sum(x > 0 for x in d), sum(x < 0 for x in d))
    o['primary'] = {k: R[0][k] for k in ('V2c', 'EMPTY', 'V2la70', 'hash')}
    o['primary_p'] = R[0]['V2c_shuf']
    o['hashes'] = sorted(set(r['hash'][:8] for r in R))
    out[ver] = o
    print(ver, json.dumps(o, indent=0))
lb = json.load(open(os.path.join(CK, 'c2_LB.json')))
L = {}
for k in ('LBMATCH', 'LBCORRUPT'):
    L[k] = dict(strict=round(st.mean(r[k]['strict'] for r in lb), 2), shuf=round(st.mean(r[k + '_shuf']['strict'][0] for r in lb), 2),
                gain_pct=round(100 * st.mean((r[k]['strict'] - r[k + '_shuf']['strict'][0]) / r['ntest'] for r in lb), 2),
                close=round(st.mean(r[k]['close'] for r in lb), 2))
L['EMPTY_strict'] = round(st.mean(r['EMPTY']['strict'] for r in lb), 2); L['ntest'] = st.mean(r['ntest'] for r in lb)
out['LB'] = L
print('LB', L)
json.dump(out, open(os.path.join(CK, 'c2_summary.json'), 'w'), indent=1)
