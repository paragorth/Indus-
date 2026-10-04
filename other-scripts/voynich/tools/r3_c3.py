#!/usr/bin/env python3
"""R-3 cycle 3 analysis: reachability after refinement and out-of-panel predictions."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3_lib import *

TGT = ['VMS', 'PE', 'LA', 'LB']
H3 = HELD3_NAMES
KEEP = ['p_ledger', 'p_num', 'text_len', 'lg_syl', 'wlen', 'lg_logo', 'secrecy', 'p_div', 'p_lmark',
        'p_docgood', 'p_bare', 'p_end', 'lg_end', 'wl_var']


def real3():
    f = os.path.join(CK, 'real_panels3.json')
    if os.path.exists(f):
        return json.load(open(f))
    out = {}
    for nm in TGT + ['UR3', 'LAT']:
        C = load_real(nm)
        out[nm], _ = panel_mean(C, 8, seed=11)
        out[nm + '_shuf0'], _ = panel_mean(shuffle_corpus(C, 0), 4, seed=21)
    json.dump(out, open(f, 'w'), indent=1)
    return out


def load(tag, need_h3=False):
    rows = []
    for l in open(os.path.join(CK, 'sims_%s.jsonl' % tag)):
        try:
            r = json.loads(l)
        except Exception:
            continue
        s = r['s']
        if need_h3 and 'purity' not in s:
            continue
        if any(not np.isfinite(s[k]) for k in FIT2_NAMES):
            continue
        rows.append(r)
    return rows


def main():
    R = real3()
    base = load('v2prior')
    X0 = np.array([[r['s'][k] for k in FIT2_NAMES] for r in base])
    med = np.median(X0, 0); mad = np.median(np.abs(X0 - med), 0) * 1.4826 + 1e-6
    pp = load('pp3', True)
    Hp = np.array([[r['s'][k] for k in H3] for r in pp])
    prior_ppi = {k: [float(np.quantile(Hp[:, j], 0.05)), float(np.quantile(Hp[:, j], 0.95))] for j, k in enumerate(H3)}
    out = {'prior_ppi': prior_ppi, 'targets': {}}
    print('prior PPI', {k: [round(a, 3) for a in v] for k, v in prior_ppi.items()})
    for T in TGT:
        ref = load('ref_' + T, True)
        X = np.array([[r['s'][k] for k in FIT2_NAMES] for r in ref])
        res = {}
        for nm in [T, T + '_shuf0'] + (['UR3', 'LAT'] if T == 'LB' else []):
            x = np.array([R[nm][k] for k in FIT2_NAMES])
            d = np.sqrt((((X - x) / mad) ** 2).sum(1))
            d0 = np.sqrt((((X0 - x) / mad) ** 2).sum(1))
            idx = np.argsort(d)[:100]
            pred = {}
            for k in H3:
                v = np.array([ref[i]['s'][k] for i in idx])
                q = np.quantile(v, [0.05, 0.5, 0.95])
                pred[k] = {'q05': float(q[0]), 'med': float(q[1]), 'q95': float(q[2]), 'real': R[nm][k],
                           'inside': bool(q[0] <= R[nm][k] <= q[2]),
                           'narrower': bool(q[2] - q[0] < prior_ppi[k][1] - prior_ppi[k][0])}
            th = {k: float(np.median([ref[i]['th'][k] for i in idx])) for k in KEEP}
            ppc = {}
            for j, k in enumerate(FIT2_NAMES):
                v = X[idx, j]; q = np.quantile(v, [0.05, 0.5, 0.95])
                ppc[k] = float((R[nm][k] - q[1]) / ((q[2] - q[0]) / 3.29 + 1e-9))
            res[nm] = {'dmin_ref': float(d[idx[0]]), 'dmin_prior': float(d0.min()), 'd100': float(d[idx[-1]]),
                       'pred': pred, 'theta_med': th, 'ppc_bad': {k: v for k, v in ppc.items() if abs(v) > 2}}
            print('%s/%-10s dmin ref %.2f (prior box %.2f) | ' % (T, nm, d[idx[0]], d0.min()) +
                  ' '.join('%s %.2f' % (k, v) for k, v in th.items()) + ' | ' +
                  ' '.join('%s %.3f[%.3f,%.3f] real %.3f %s%s' % (k, p['med'], p['q05'], p['q95'], p['real'],
                                                                 'IN' if p['inside'] else 'OUT', '' if p['narrower'] else '(wide)')
                           for k, p in pred.items()) + ' | misfit ' + ' '.join('%s %+.1f' % kv for kv in res[nm]['ppc_bad'].items()))
        out['targets'][T] = res
    json.dump(out, open(os.path.join(CK, 'c3_result.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
