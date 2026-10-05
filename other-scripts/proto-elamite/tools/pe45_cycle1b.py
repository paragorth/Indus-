"""pe45 cycle 1b: which mechanism makes same-tablet names alike -- kinship or a tablet pool?
ABC model choice inside the bank.  Mechanisms per simulated grammar:
  POOL  p_pool > 0          (prior 0.5)
  KIN   p_mark*kin_frac > 0.25 (lineage markers shared by kin-drawn tablets)
  INH   p_inh > 0.5         (children copy a father's element)
Posterior P(mechanism) = share among the k nearest bank sims on ALL statistics (A+B).
Calibration: 400 bank sims used as pseudo-observed (leave-one-out): AUC of posterior P vs truth.
Targets: PE, PE nulls (5 each), UR3 (own bank) and UR3 null.
usage: python3 pe45_cycle1b.py -> data/pe45_ckpt/cycle1b.json"""
import os, sys, json, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe45_common import stats, encode, load_corpora, PNAMES, CK  # noqa
from pe45_abc import load_bank, scale, null_within, null_across  # noqa
from pe45_signs import auc  # noqa

J = {k: i for i, k in enumerate(PNAMES)}


def mech(P):
    return {'POOL': P[:, J['p_pool']] > 0,
            'KIN': P[:, J['p_mark']] * P[:, J['kin_frac']] > 0.25,
            'INH': P[:, J['p_inh']] > 0.5}


def post(Z, zo, M, k=300, excl=None):
    d = np.sqrt(((Z - zo) ** 2).sum(1))
    if excl is not None:
        d[excl] = np.inf
    acc = np.argsort(d)[:k]
    return {m: float(v[acc].mean()) for m, v in M.items()}


def run(which, targets, rng):
    P, S = load_bank(which)
    med, mad = scale(S)
    Z = (S - med) / mad
    M = mech(P)
    prior = {m: float(v.mean()) for m, v in M.items()}
    cal = {m: ([], []) for m in M}
    for i in rng.sample(range(len(P)), 400):
        p = post(Z, Z[i], M, excl=i)
        for m in M:
            (cal[m][0] if M[m][i] else cal[m][1]).append(p[m])
    out = {'prior': prior, 'cal_auc': {m: auc(*cal[m]) for m in M}}
    for nm, o in targets.items():
        out[nm] = [post(Z, (x - med) / mad, M) for x in o]
    return out


def main():
    D = load_corpora()
    rng = random.Random(451)
    C, _ = encode([t['names'] for t in D['PE']])
    tg = {'PE': [stats(C)], 'NULL_across': [stats(null_across(C, rng), seed=r) for r in range(5)],
          'NULL_within': [stats(null_within(C, rng), seed=r) for r in range(5)]}
    res = {'PE': run('PE', tg, rng)}
    CU, _ = encode([t['names'] for t in D['UR3']])
    tu = {'UR3': [stats(CU)], 'UR3_NULL_across': [stats(null_across(CU, rng), seed=r) for r in range(5)]}
    res['UR3'] = run('UR3', tu, rng)
    json.dump(res, open(os.path.join(CK, 'cycle1b.json'), 'w'))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
