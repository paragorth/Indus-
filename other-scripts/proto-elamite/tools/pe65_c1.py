#!/usr/bin/env python3
"""PE-65 cycle 1: ABC-RF on the 300k-world bank. Calibration, planted worlds, real PE (publication batches and
random batches), 20 unit-label shuffles, Ur III and Linear B controls at PE shape (5 thinning replicates)."""
import sys, os, json, pickle, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe65_common import *
from pe65_fit import train, read, targets

OUT = os.path.join(LOOPS, 'pe65_cycle1.txt')


def plant(kind, rep):
    """Planted worlds: draw a world of the right type from the prior (nuisance random), then force the layout."""
    ft = {'susa_state': 0, 'susa_capsat': 1, 'yahya_state': 0, 'peers': 2, 'herd': 5, 'null': 6, 'temple_susa': 3,
          'merch': 4}[kind]
    _, th = sim_docs(seed('pe65-pl-%s-%d' % (kind, rep)), force_type=ft)
    th = th.copy()
    if kind in ('susa_state', 'susa_capsat', 'temple_susa', 'yahya_state'):
        c = 1 if kind == 'yahya_state' else 0
        for k in range(K):
            th[2 + k] = 1 if k == c else 2
            th[2 + K + k] = -1 if k == c else c
    if kind == 'peers':   # Susa, Malyan, Yahya independent; Sialk under Susa, Sofalin under Susa
        roles = [1, 1, 1, 2, 2]; par = [-1, -1, -1, 0, 0]
        for k in range(K):
            th[2 + k] = roles[k]; th[2 + K + k] = par[k]
    S, _ = simulate(1, seed('pe65-pls-%s-%d' % (kind, rep)), force_theta=th)
    return S[0], th


def main():
    rf, cal = train('PE')
    pickle.dump(rf, open(os.path.join(CK, 'rf_c1.pkl'), 'wb'))
    res = dict(calibration=cal)
    print('calib', json.dumps({k: cal[k] for k in ('type_acc', 'centre_auc', 'parent_acc', 'susa_auc', 'inst_vs_none_acc')}), flush=True)
    # planted
    pl = {}
    for kind in ('susa_state', 'susa_capsat', 'yahya_state', 'peers', 'herd', 'null', 'temple_susa', 'merch'):
        rr = []
        for rep in range(8):
            s, th = plant(kind, rep)
            rr.append(read(rf, s))
        pl[kind] = dict(type={t: round(float(np.mean([r['type'].get(t, 0) for r in rr])), 3) for t in TYPES},
                        top=[max(r['type'], key=r['type'].get) for r in rr],
                        centre=[round(float(np.mean([r['centre'][k] for r in rr])), 3) for k in range(K)],
                        susa_state=round(float(np.mean([r['susa_state'] for r in rr])), 3))
        print('plant', kind, pl[kind], flush=True)
    res['planted'] = pl
    # real
    real = {}
    real['PE_pub'] = read(rf, stats(pe_docs_all('pub')))
    real['PE_rand'] = read(rf, stats(pe_docs_all('rand')))
    print('PE_pub', real['PE_pub'], flush=True); print('PE_rand', real['PE_rand'], flush=True)
    sh = []
    base_docs = pe_docs_all('pub')
    for i in range(20):
        sh.append(read(rf, stats(shuffle_units(base_docs, str(i)))))
    real['PE_shuf'] = dict(type={t: [round(float(np.mean([r['type'].get(t, 0) for r in sh])), 3),
                                     round(float(np.max([r['type'].get(t, 0) for r in sh])), 3)] for t in TYPES},
                           centre=[round(float(np.mean([r['centre'][k] for r in sh])), 3) for k in range(K)],
                           susa_state=[round(float(np.mean([r['susa_state'] for r in sh])), 3), round(float(np.max([r['susa_state'] for r in sh])), 3)])
    print('PE_shuf', real['PE_shuf'], flush=True)
    for nm in ('UR', 'LB'):
        rr = [read(rf, stats(control_docs(nm, rep))) for rep in range(5)]
        real[nm] = dict(type={t: round(float(np.mean([r['type'].get(t, 0) for r in rr])), 3) for t in TYPES},
                        centre=[round(float(np.mean([r['centre'][k] for r in rr])), 3) for k in range(K)],
                        dependent=[round(float(np.mean([r['dependent'][k] for r in rr])), 3) for k in range(K)],
                        susa_state=round(float(np.mean([r['susa_state'] for r in rr])), 3), reps=rr)
        print(nm, {k: v for k, v in real[nm].items() if k != 'reps'}, flush=True)
    res['real'] = real
    jdump(res, 'c1_results.json')


if __name__ == '__main__':
    main()
