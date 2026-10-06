#!/usr/bin/env python3
"""la66 cycle 3: factors and ratios (pe58 port).  For every frozen descriptor: within-document factor
(document x stratum mean log shift, document bootstrap), lattice-of-simple-ratios test against jittered
factors, against the factors of all other recurring features (empirical pool) and against factors from
quantity-shuffled corpora.  Calibration: Ur III per-head rations at LA size (opaque; truth gurusz:geme2 = 2),
Ur III group totals at LA size, Linear B truth descriptors at LA size, planted lattice vs planted arbitrary
factors on the LA skeleton (power).  Freezes the factor table with a sha256."""
import sys, os, json, math, random
import numpy as np
from collections import defaultdict, Counter
from la66_lib import *


def all_factors(rows, min_docs=4, B=200, seed=0):
    cnt = defaultdict(set)
    for r in rows:
        for f in r['f']:
            cnt[f].add(r['doc'])
    out = {}
    for f, ds in cnt.items():
        if len(ds) < min_docs:
            continue
        fb = factor_boot(rows, f, B=B, seed=seed)
        if fb and fb[3] >= 3:
            out[f] = fb
    return out


def pair_rate(rows, a, b):
    """direct rate a:b inside documents carrying both (entries with a and not b vs b and not a)"""
    g = defaultdict(lambda: ([], []))
    for r in rows:
        ia, ib = a in r['f'], b in r['f']
        if ia and not ib:
            g[r['doc']][0].append(math.log(r['q']))
        elif ib and not ia:
            g[r['doc']][1].append(math.log(r['q']))
    d = [np.mean(x) - np.mean(y) for x, y in g.values() if x and y]
    return (float(np.mean(d)), len(d)) if d else (None, 0)


def lattice_of(F, floor=0.15):
    logs = [v[0] for v in F.values() if abs(v[0]) >= floor]
    return lattice_test(logs), logs


def emp_test(obs_logs, pool_logs, n=20000, seed=0):
    if len(pool_logs) < len(obs_logs) or not obs_logs:
        return None
    rng = np.random.default_rng(seed)
    o = float(lat_dist(obs_logs).mean())
    dp = lat_dist(pool_logs)
    ne = np.array([dp[rng.choice(len(dp), len(obs_logs), replace=False)].mean() for _ in range(n)])
    return dict(obs=o, null=float(ne.mean()), p=float((np.sum(ne <= o) + 1) / (n + 1)))


def main():
    out = {}
    L = la_rows()
    fz = json.load(open(os.path.join(CK, 'c2_class.json')))
    desc = fz['descriptors']
    # 1 LA descriptor factors
    F = {f: factor_boot(L, f, B=1000, seed=1) for f in desc}
    F = {f: v for f, v in F.items() if v}
    out['LA_factors'] = F
    lt, logs = lattice_of(F)
    pool = all_factors(L)
    pool_logs = [v[0] for f, v in pool.items() if abs(v[0]) >= 0.15 and f not in desc]
    out['LA_lattice'] = dict(jit=lt, emp=emp_test(logs, pool_logs), n_pool=len(pool_logs))
    # factors under quantity shuffles (are descriptor factors larger than chance?)
    sh = []
    for s in range(50):
        Ls = null_q_doc(L, 5000 + s)
        sh.append([abs(factor_boot(Ls, f, B=1, seed=s)[0]) for f in F if factor_boot(Ls, f, B=1, seed=s)])
    obs_abs = float(np.mean([abs(v[0]) for v in F.values()]))
    shm = [float(np.mean(x)) for x in sh]
    out['LA_mag_vs_qdoc'] = dict(obs=obs_abs, null=float(np.mean(shm)), p=float((np.sum(np.array(shm) >= obs_abs) + 1) / 51))
    # 2 Ur III per-head ration ladder at LA size (opaque) and group totals
    res = []
    for total in (False, True):
        for s in range(10):
            U = thin(ur_ration_rows(total), len(L), s)
            O, mp = opaque(U, s)
            g, e, d = mp.get('W:gurusz'), mp.get('W:geme2'), mp.get('W:dumu')
            r_ge, n_ge = pair_rate(O, g, e) if g and e else (None, 0)
            r_ed, n_ed = pair_rate(O, e, d) if e and d else (None, 0)
            FU = all_factors(O)
            ltU, lU = lattice_of(FU)
            res.append(dict(total=total, seed=s, gurusz_geme2=r_ge, n=n_ge, geme2_dumu=r_ed, n2=n_ed,
                            lattice=ltU, nfac=len(lU)))
    out['UR_ration'] = res
    # 3 Linear B truth descriptors at LA size
    LBT = {'W:o': -1, 'W:o-pe-ro': -1, 'W:ko-wa': -1, 'W:ko-wo': -1, 'W:to-so': 1, 'W:to-so-de': 1}
    lbres = []
    for s in range(10):
        B = thin(lb_rows(), len(L), s)
        fb = {f: factor_boot(B, f, B=200, seed=s) for f in LBT}
        lbres.append({f: v for f, v in fb.items() if v})
    out['LB_truth'] = lbres
    # 4 planted lattice vs arbitrary factors on the LA skeleton (same number of features as LA descriptors)
    k = max(4, len(F))
    pw = []
    latv = [math.log(x) for x in (2, 1 / 2, 3, 1 / 3, 3 / 2, 2 / 3, 4, 1 / 4)]
    for s in range(20):
        rng = np.random.default_rng(s)
        for kind in ('lattice', 'arbitrary'):
            if kind == 'lattice':
                vals = list(rng.choice(latv, k))
            else:
                vals = list(rng.choice([-1, 1], k) * rng.uniform(0.2, 1.4, k))
            P, eff = plant(L, 300 + s, k=k, values=vals)
            FP = {f: factor_boot(P, f, B=1, seed=s) for f in eff}
            logsP = [v[0] for v in FP.values() if v and abs(v[0]) >= 0.15]
            err = [abs(FP[f][0] - eff[f]) for f in eff if FP.get(f)]
            pw.append(dict(seed=s, kind=kind, p=lattice_test(logsP, n=4000)['p'], k=len(logsP),
                           err=float(np.median(err)) if err else None))
    out['plant_power'] = pw
    json.dump(out, open(os.path.join(CK, 'c3.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
