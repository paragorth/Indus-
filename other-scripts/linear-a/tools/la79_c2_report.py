"""LA-79 cycle 2 report: which 1950-knowable property predicts survival into the later futures?"""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la79_common import CK
from la78_engine import spearman
COLS = ['fam', 'K', 'ins', 'loso', 'lofo', 'ent', 'bal', 'W1', 'W2', 'W3']
c = {k: i for i, k in enumerate(COLS)}
PRED = ['ins', 'loso', 'lofo', 'ent', 'bal', 'W1', 'K']


def law(R, label):
    later = (R[:, c['W2']] + R[:, c['W3']]) / 2
    surv = (R[:, c['W2']] > 0) & (R[:, c['W3']] > 0)
    out = dict(n=len(R), surv_rate=round(float(surv.mean()), 4), W1_pos=round(float((R[:, c['W1']] > 0).mean()), 3))
    out['rho'] = {p: round(spearman(R[:, c[p]], later), 3) for p in PRED}
    # decile survival by each predictor
    dec = {}
    for p in ('ins', 'loso', 'lofo', 'ent', 'W1'):
        o = np.argsort(R[:, c[p]])
        n = len(o) // 10
        dec[p] = [round(float(surv[o[k * n:(k + 1) * n]].mean()), 3) for k in (0, 4, 9)]  # bottom, middle, top decile
    out['decile_surv_bottom_mid_top'] = dec
    print(label, json.dumps(out))
    return out


def main():
    res = {}
    real = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(CK, 'c2_real_1?.npy')))])
    res['real'] = law(real, 'real')
    # within-family rho for the main predictors
    res['real_by_family'] = {}
    for fi, fam in enumerate(['FIN', 'INI', 'SEC', 'PAIR', 'LENFIN', 'ANY', 'WORD']):
        R = real[real[:, 0] == fi]
        later = (R[:, c['W2']] + R[:, c['W3']]) / 2
        res['real_by_family'][fam] = {p: round(spearman(R[:, c[p]], later), 3) for p in ('ins', 'loso', 'lofo', 'ent', 'W1')}
        res['real_by_family'][fam]['surv'] = round(float(((R[:, c['W2']] > 0) & (R[:, c['W3']] > 0)).mean()), 3)
    print('by family', json.dumps(res['real_by_family']))
    # replication across the two independent seeds
    a = np.load(os.path.join(CK, 'c2_real_10.npy')); b = np.load(os.path.join(CK, 'c2_real_11.npy'))
    res['seed_split'] = [law(a, 'seed10')['rho'], law(b, 'seed11')['rho']]
    sh = [np.load(f) for f in sorted(glob.glob(os.path.join(CK, 'c2_shuf_*.npy')))]
    res['shuf'] = [law(R, 'shuf') for R in sh]
    pl = {}
    for f in sorted(glob.glob(os.path.join(CK, 'c2_plant_*.npy'))):
        R = np.load(f)
        v = 'stable' if 'stable' in f else 'drift'
        o = law(R[1:], os.path.basename(f))
        t = R[0]
        # truth rank under each predictor and its own survival
        o['truth_pct'] = {p: round(float((R[1:, c[p]] < t[c[p]]).mean()), 3) for p in ('ins', 'loso', 'lofo', 'W1')}
        o['truth_ent_pct'] = round(float((R[1:, c['ent']] < t[c['ent']]).mean()), 3)
        o['truth_surv'] = bool(t[c['W2']] > 0 and t[c['W3']] > 0)
        pl.setdefault(v, []).append(o)
    res['plant'] = pl
    for v, L in pl.items():
        print(v, 'mean rho', {p: round(float(np.mean([o['rho'][p] for o in L])), 3) for p in PRED},
              'truth pct', {p: round(float(np.mean([o['truth_pct'][p] for o in L])), 3) for p in ('ins', 'loso', 'lofo', 'W1')},
              'truth surv', sum(o['truth_surv'] for o in L), '/', len(L))
    json.dump(res, open(os.path.join(CK, 'c2_report.json'), 'w'))


main()
