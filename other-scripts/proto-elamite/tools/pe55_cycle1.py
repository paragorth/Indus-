"""pe55 cycle 1: calibrate the physics ruler on Ur III (opaque), Ur III thinned to PE size, Ur III field
texts (area), and planted PE-shaped farming archives.
usage: pe55_cycle1.py ur3|thin|area|plant   -> data/pe55_ckpt/c1_<part>.json"""
import sys, json, math, time
import numpy as np
from collections import Counter
from pe55_lib import *

part = sys.argv[1]
rng = np.random.default_rng(55)
NH = 3000


def fit(D, n_hyp=NH, allow_area=True):
    R = random_search(D, rng, n_hyp, allow_area, n_rand_lab=2000)
    order = np.argsort(-R['s_best'])
    top = order[: max(1, n_hyp // 100)]
    b = order[0]
    return dict(logu=float(R['logu'][b]), loga=float(R['loga'][b]), lab=R['lab_best'][b].tolist(),
                score=float(R['s_best'][b]), top_logu=R['logu'][top].tolist(), top_loga=R['loga'][top].tolist(),
                rand_lab_q99=float(np.quantile(R['s_rand'], 0.99)), rand_lab_max=float(R['s_rand'].max()))


def heldout(D, n_split=8, n_null=8, allow_area=True):
    out = []
    for s in range(n_split):
        A = split_tabs(D, rng)
        Dtr, Dte = subset(D, A, True), subset(D, A, False)
        F = fit(Dtr, NH // 2, allow_area)
        g, n = apply_frozen(Dte, F['logu'], F['loga'], F['lab'], D['keys'])
        # null 2: random units, frozen labels
        gu = [apply_frozen(Dte, rng.uniform(*LOGU), rng.uniform(*LOGA), F['lab'], D['keys'])[0] for _ in range(200)]
        # null 3: labels permuted among keys
        gl = []
        for _ in range(200):
            lab = np.array(F['lab'])
            gl.append(apply_frozen(Dte, F['logu'], F['loga'], rng.permutation(lab).tolist(), D['keys'])[0])
        # null 1: numbers shuffled across lines (whole pipeline)
        gs = []
        for _ in range(n_null):
            Ds = shuffle_q(D, rng)
            Fs = fit(subset(Ds, A, True), NH // 4, allow_area)
            gs.append(apply_frozen(subset(Ds, A, False), Fs['logu'], Fs['loga'], Fs['lab'], D['keys'])[0])
        out.append(dict(logu=F['logu'], loga=F['loga'], gain=g, n=n, p_units=float(np.mean(np.array(gu) >= g)),
                        p_labels=float(np.mean(np.array(gl) >= g)), shuf=gs,
                        p_shuf=float((1 + sum(x >= g for x in gs)) / (1 + len(gs)))))
    return out


def summarize_labels(D, lab):
    tr = {}
    for p in D['P']:
        tr.setdefault(p['kc'], Counter())[p.get('truth') or '?'] += 1
    rows = []
    for k, a in zip(D['keys'], lab):
        t = tr.get(k, Counter()).most_common(1)
        rows.append((k, lab_name(a), t[0][0] if t else '?', int((D['k'] == D['keys'].index(k)).sum())))
    return rows


def coarse_acc(rows):
    ok = sum(n for k, a, t, n in rows if a != 'none' and coarse(a) == t)
    lab = sum(n for k, a, t, n in rows if a != 'none')
    return ok, lab


res = {'part': part}
t0 = time.time()
if part in ('ur3', 'thin'):
    P, m = opaque(ur3_pairs())
    D = build_arrays(P, 'kc', 3)
    if part == 'ur3':
        F = fit(D)
        rows = summarize_labels(D, F['lab'])
        res.update(F, rows=rows, coarse=coarse_acc(rows), n_pairs=len(D['lr']), truth_logu=0.0)
        # per-truth-class unit implied
        res['heldout'] = heldout(D, 6, 6)
    else:
        reps = []
        tabs = np.unique(D['tab'])
        for r in range(10):
            sel = set(rng.choice(tabs, 220, replace=False))
            Dt = subset(D, sel, True)
            cnt = Counter(Dt['k'].tolist())
            keys = [D['keys'][i] for i in sorted(cnt) if cnt[i] >= 3]
            Dt = build_arrays(Dt['P'], 'kc', 3, keys)
            F = fit(Dt)
            rows = summarize_labels(Dt, F['lab'])
            ho = heldout(Dt, 3, 4)
            reps.append(dict(logu=F['logu'], n=len(Dt['lr']), coarse=coarse_acc(rows), rows=rows,
                             top_logu_q=np.quantile(F['top_logu'], [0.05, 0.5, 0.95]).tolist(),
                             ho_p_shuf=[h['p_shuf'] for h in ho], ho_p_units=[h['p_units'] for h in ho],
                             ho_p_labels=[h['p_labels'] for h in ho]))
            print(r, reps[-1]['logu'], reps[-1]['coarse'], reps[-1]['ho_p_shuf'], flush=True)
        res['reps'] = reps
elif part == 'area':
    A = ur3_area_tabs()
    P0 = adjacent_pairs(A)
    U, _ = opaque(ur3_pairs())
    P = P0 + U
    P, m = opaque(P, seed=9)
    D = build_arrays(P, 'kc', 3)
    F = fit(D)
    rows = summarize_labels(D, F['lab'])
    res.update(F, rows=rows, coarse=coarse_acc(rows), truth_logu=0.0, truth_loga=math.log(0.36))
    seed_tabs = {t['id'] for t in A if t['lab'] == 'seed'}
    r = [p['q'] / p['c'] for p in P0 if p['tab'] in seed_tabs]
    res['seed_tab_ratio_q'] = np.quantile(r, [0.1, 0.5, 0.9]).tolist()
elif part == 'plant':
    P = pe_pairs()
    keys = [k for k, n in Counter(p['kc'] for p in P).items() if n >= 3 and k != '-']
    reps = []
    for r in range(12):
        tu, ta = rng.uniform(math.log(0.05), math.log(20)), rng.uniform(math.log(0.05), math.log(5))
        heads = [a for a in ACTS if not a.startswith('area')]
        truth = {k: (rng.choice(ACTS) if rng.random() < 0.8 else 'none') for k in keys}
        Q = []
        for p in P:
            p = dict(p)
            a = truth.get(p['kc'], 'none')
            if a == 'none' or rng.random() < 0.3:
                p['q'] = p['q']  # keep real (junk w.r.t. physics): none-keys and 30% of each key
            else:
                j = ACTS.index(a)
                lit = p['c'] * math.exp(rng.normal(MU[j], SD[j])) * (math.exp(ta) if ISAREA[j] else 1)
                p['q'] = max(1.0, round(lit / math.exp(tu)))
            p['truth'] = coarse(a)
            Q.append(p)
        D = build_arrays(Q, 'kc', 3)
        F = fit(D)
        rows = summarize_labels(D, F['lab'])
        exact = sum(1 for k, a in zip(D['keys'], F['lab']) if lab_name(a) == truth[k])
        ho = heldout(D, 2, 4)
        reps.append(dict(tu=tu, ta=ta, logu=F['logu'], loga=F['loga'], err_u=F['logu'] - tu,
                         err_alias=min(abs(F['logu'] - tu - math.log(f)) for f in (1, 30, 1 / 30, 360, 1 / 360, 12, 1 / 12)),
                         coarse=coarse_acc(rows), exact=exact, nkeys=len(D['keys']),
                         ho_p_shuf=[h['p_shuf'] for h in ho]))
        print(r, reps[-1], flush=True)
    res['reps'] = reps
res['secs'] = time.time() - t0
json.dump(res, open(os.path.join(CK, f'c1_{part}.json'), 'w'), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('done', part, res.get('logu'), res.get('coarse'), res['secs'])
