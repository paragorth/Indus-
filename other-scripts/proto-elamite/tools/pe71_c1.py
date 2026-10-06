"""pe71 cycle 1: try to kill 'M153 compounds sit on sealed tablets' (pe70 C).

Targets: T1 |M153+X| (the pe70 lead), T2 every compound containing M153, T3 any M153 (compound or plain),
T4 plain M153 only.
Statistic: number of distinct SEALING UNITS among the family's tablets (tablets sharing a seal id count once);
also raw sealed tablets.
Nulls (5,000 each): labels (sealed flag + unit) permuted (a) within volume x size band, (b) within volume x size
band x M157-header x has-M288 (the pe70 sealed document type), (c) find-lot: circular shift along publication
number within volume.
Kill tests: held-out (tablets of every recurring seal group removed: where pe70 saw the lead), leave-one-volume-out,
decoy families (component-sign families of matched size; random unions of compound types of matched size;
all single sign types of the same size for T1 = look-elsewhere), planted calibration.
usage: pe71_c1.py -> data/pe71_ckpt/c1.json
"""
import json, os, collections
import numpy as np
from pe71_lib import load, strata, perm_index, shift_index, unit_codes, fam_stats, families, type_families, comps, CK

NP = 5000


def targets(R):
    T = {}
    T['T1 |M153+X|'] = [i for i, r in enumerate(R) if '|M153+X|' in r['toks']]
    T['T2 M153 compounds'] = [i for i, r in enumerate(R) if any(t.startswith('|') and 'M153' in comps(t) for t in r['toks'])]
    T['T3 any M153'] = [i for i, r in enumerate(R) if any('M153' in comps(t) for t in r['toks'])]
    T['T4 plain M153'] = [i for i, r in enumerate(R) if 'M153' in r['toks']]
    return T


def nulls(R, rng, n=NP):
    return {'volband': perm_index(strata(R, 'volband'), rng, n), 'doc': perm_index(strata(R, 'doc'), rng, n),
            'findlot': shift_index(R, rng, n)}


def run_set(R, rng, label, n=NP, decoys=True):
    sealed = np.array([r['sealed'] for r in R], int)
    units = unit_codes(R)
    P = nulls(R, rng, n)
    T = targets(R)
    out = dict(label=label, N=len(R), sealed=int(sealed.sum()))
    for tn, f in T.items():
        if len(f) < 2:
            continue
        out[tn] = {k: fam_stats(f, sealed, units, Pk) for k, Pk in P.items()}
        out[tn]['ids'] = [(R[i]['id'], R[i]['vol'], R[i]['unit']) for i in f if R[i]['sealed']]
        print(label, tn, {k: (v['n'], v['sealed'], v['units'], round(v['e_u'], 2), v['p_u']) for k, v in out[tn].items() if k != 'ids'}, flush=True)
    if not decoys:
        return out
    # decoys
    F = families(R)
    TF = type_families(R)
    for tn in ('T2 M153 compounds', 'T3 any M153'):
        n0 = len(T[tn])
        dec = {k: v for k, v in F.items() if k != 'M153' and 0.5 * n0 <= len(v) <= 2 * n0}
        res = {}
        for nm in ('doc', 'findlot'):
            ps = [fam_stats(v, sealed, units, P[nm])['p_u'] for v in dec.values()]
            res[nm] = dict(n_decoys=len(ps), share_le=float(np.mean([p <= out[tn][nm]['p_u'] for p in ps])) if ps else None,
                           share_p05=float(np.mean([p <= .05 for p in ps])) if ps else None)
        # random unions of compound types with matched tablet count
        ctypes = [k for k in TF if k.startswith('|') and 'M153' not in comps(k)]
        zs = []
        for _ in range(400):
            fam = set()
            for k in rng.permutation(ctypes):
                if len(fam) >= n0:
                    break
                fam |= set(TF[k])
            zs.append(fam_stats(sorted(fam), sealed, units, P['doc'][:1000])['p_u'])
        res['random_unions_doc'] = dict(n=len(zs), share_le=float(np.mean([p <= out[tn]['doc']['p_u'] for p in zs])),
                                        share_p05=float(np.mean([p <= .05 for p in zs])))
        out[tn]['decoys'] = res
        print(label, tn, 'decoys', res, flush=True)
    # look-elsewhere for T1: every sign type with the same tablet count band
    n1 = len(T['T1 |M153+X|'])
    dec = {k: v for k, v in TF.items() if k != '|M153+X|' and n1 - 3 <= len(v) <= n1 + 4}
    ps = {k: fam_stats(v, sealed, units, P['doc'])['p_u'] for k, v in dec.items()}
    p1 = out['T1 |M153+X|']['doc']['p_u']
    out['T1 |M153+X|']['lookelsewhere'] = dict(n_types=len(ps), n_le=int(sum(p <= p1 for p in ps.values())),
                                              share_p05=float(np.mean([p <= .05 for p in ps.values()])),
                                              best=sorted(ps.items(), key=lambda x: x[1])[:12])
    print(label, 'T1 look-elsewhere', out['T1 |M153+X|']['lookelsewhere'], flush=True)
    return out


def planted(R, rng, n_rep=50):
    """calibration: an invented 8-tablet sign family placed on 6 sealed tablets (distinct units) + 2 unsealed,
    tablets drawn at random among short M157/M288 tablets of the same volumes as the real T1 (worst case: the
    sealed document type itself); and placed at random (false-positive rate)."""
    sealed = np.array([r['sealed'] for r in R], int)
    units = unit_codes(R)
    P = perm_index(strata(R, 'doc'), rng, 2000)
    vols = {'MDP 06', 'MDP 26S', 'MDP 26'}
    S = [i for i, r in enumerate(R) if r['sealed'] and r['vol'] in vols]
    U = [i for i, r in enumerate(R) if not r['sealed'] and r['vol'] in vols and r['n_lines'] <= 8]
    hit6, hit4, fp = 0, 0, 0
    for _ in range(n_rep):
        byu = {}
        for i in rng.permutation(S):
            byu.setdefault(units[i], i)
        su = list(byu.values())
        f6 = list(rng.choice(su, 6, replace=False)) + list(rng.choice(U, 2, replace=False))
        f4 = list(rng.choice(su, 4, replace=False)) + list(rng.choice(U, 4, replace=False))
        fr = list(rng.choice(S + U, 8, replace=False))
        hit6 += fam_stats(f6, sealed, units, P)['p_u'] < .01
        hit4 += fam_stats(f4, sealed, units, P)['p_u'] < .05
        fp += fam_stats(fr, sealed, units, P)['p_u'] < .05
    return dict(n=n_rep, power_6of8_p01=hit6 / n_rep, power_4of8_p05=hit4 / n_rep, fp_random_p05=fp / n_rep)


def main():
    rng = np.random.default_rng(71)
    R = load()
    res = {}
    res['all'] = run_set(R, rng, 'ALL')
    # held-out: drop every tablet of a recurring seal group (pe70's discovery material)
    cu = collections.Counter(r['unit'] for r in R if r['unit'])
    keep = [r for r in R if not (r['unit'] and cu[r['unit']] >= 2)]
    res['heldout'] = run_set(keep, rng, 'HELDOUT(no recurring-seal tablets)', decoys=False)
    # one tablet per unit, random pick, 20 draws (doc null)
    draws = []
    for k in range(20):
        seen, sub = set(), []
        for i in rng.permutation(len(R)):
            u = R[i]['unit']
            if u and u in seen:
                continue
            if u:
                seen.add(u)
            sub.append(R[i])
        o = run_set(sub, rng, 'ONEPERSEAL %d' % k, n=1000, decoys=False)
        draws.append({t: o[t]['doc']['p_t'] for t in o if t.startswith('T')})
    res['oneperseal'] = dict(draws=draws, median={t: float(np.median([d[t] for d in draws if t in d])) for t in draws[0]})
    print('one per seal median p', res['oneperseal']['median'], flush=True)
    # leave one volume out
    T = targets(R)
    lv = {}
    for v in sorted({R[i]['vol'] for f in T.values() for i in f}):
        sub = [r for r in R if r['vol'] != v]
        o = run_set(sub, rng, 'LOVO -' + v, n=2000, decoys=False)
        lv[v] = {t: (o[t]['doc']['sealed'], o[t]['doc']['n'], o[t]['doc']['units'], round(o[t]['doc']['e_u'], 2), o[t]['doc']['p_u'])
                 for t in o if t.startswith('T')}
    res['lovo'] = lv
    res['planted'] = planted(R, rng)
    print('planted', res['planted'], flush=True)
    json.dump(res, open(os.path.join(CK, 'c1.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
