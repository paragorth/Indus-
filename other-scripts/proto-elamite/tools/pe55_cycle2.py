"""pe55 cycle 2: the physics ruler on Proto-Elamite, and the random-ruler test.
usage: pe55_cycle2.py pe <capset> <key> | rulers <ur3|pe>   -> data/pe55_ckpt/c2_*.json
  pe:     random search on PE count/capacity neighbours (u litres per N39C, A ha per count unit, label
          per key sign); u profile; 8 held-out splits with three nulls (numbers shuffled across lines,
          random units, labels permuted).
  rulers: the physical ruler against 150 nonsense rulers (activity centres redrawn log-uniformly over
          the same span, same widths), each fitted on half the tablets and scored on the other half."""
import sys, json, math, time
import numpy as np
from collections import Counter
import pe55_lib as L
from pe55_lib import *

mode = sys.argv[1]
rng = np.random.default_rng(552)
t0 = time.time()
AREA = os.environ.get('PE55_AREA', '1') == '1'


def fit(D, n_hyp=3000):
    R = random_search(D, rng, n_hyp, AREA, n_rand_lab=1000)
    o = np.argsort(-R['s_best'])
    b = o[0]
    return dict(logu=float(R['logu'][b]), loga=float(R['loga'][b]), lab=R['lab_best'][b].tolist(),
                score=float(R['s_best'][b]), R=R, order=o)


def profile(D, n=400):
    lu = np.linspace(*LOGU, n)
    best = np.full(n, -1e9)
    for la in np.linspace(*LOGA, 12):
        G = gains(D, lu, np.full(n, la))
        s, _ = best_labels(G, AREA)
        best = np.maximum(best, s)
    return lu, best


def heldout(D, n_split=8, n_null=6):
    out = []
    for s in range(n_split):
        A = split_tabs(D, rng)
        F = fit(subset(D, A, True), 1500)
        Dte = subset(D, A, False)
        g, n = apply_frozen(Dte, F['logu'], F['loga'], F['lab'], D['keys'])
        gu = [apply_frozen(Dte, rng.uniform(*LOGU), rng.uniform(*LOGA), F['lab'], D['keys'])[0] for _ in range(200)]
        gl = [apply_frozen(Dte, F['logu'], F['loga'], rng.permutation(F['lab']).tolist(), D['keys'])[0] for _ in range(200)]
        gs = []
        for _ in range(n_null):
            Ds = shuffle_q(D, rng)
            Fs = fit(subset(Ds, A, True), 800)
            gs.append(apply_frozen(subset(Ds, A, False), Fs['logu'], Fs['loga'], Fs['lab'], D['keys'])[0])
        out.append(dict(logu=F['logu'], loga=F['loga'], gain=g, n=n, p_units=float(np.mean(np.array(gu) >= g)),
                        p_labels=float(np.mean(np.array(gl) >= g)), shuf=gs,
                        p_shuf=float((1 + sum(x >= g for x in gs)) / (1 + len(gs))),
                        lab={D['keys'][i]: lab_name(a) for i, a in enumerate(F['lab']) if a >= 0}))
        print('split', s, round(F['logu'], 2), round(g, 1), out[-1]['p_shuf'], out[-1]['p_units'], out[-1]['p_labels'], flush=True)
    return out


if mode == 'pe':
    capset, key = sys.argv[2], sys.argv[3]
    from pe27_common import pe_tablets
    P = adjacent_pairs(pe_tablets(capset))
    D = build_arrays(P, key, 3)
    F = fit(D, 6000)
    R = F.pop('R')
    o = F.pop('order')
    top = o[:60]
    lu, prof = profile(D)
    profs = [profile(shuffle_q(D, rng), 200)[1].max() for _ in range(10)]
    res = dict(capset=capset, key=key, n_pairs=len(D['lr']), keys=D['keys'], **F,
               lab_named={D['keys'][i]: lab_name(a) for i, a in enumerate(F['lab'])},
               hits=hits(D, F['logu'], F['loga'], [lab_name(a) if a >= 0 else None for a in F['lab']]),
               top_logu=R['logu'][top].tolist(), top_loga=R['loga'][top].tolist(),
               prof_lu=lu.tolist(), prof=prof.tolist(), shuf_prof_max=profs,
               rand_lab_q99=float(np.quantile(R['s_rand'], 0.99)))
    print('fit', F['logu'], math.exp(F['logu']), F['score'], 'shuf prof max', profs, flush=True)
    res['heldout'] = heldout(D)
    fn = f'c2_pe_{capset}_{key}' + ('' if AREA else '_noarea') + '.json'
elif mode == 'rulers':
    src = sys.argv[2]
    if src == 'ur3':
        P, _ = opaque(ur3_pairs())
        key = 'kc'
    else:
        P = pe_pairs()
        key = 'kc'
    D = build_arrays(P, key, 3)
    MU0 = L.MU.copy()
    lo, hi = MU0[~L.ISAREA].min(), MU0[~L.ISAREA].max()
    splits = [split_tabs(D, rng) for _ in range(4)]

    def run_ruler(mu):
        L.MU[:] = mu
        gs, us = [], []
        for A in splits:
            F = fit(subset(D, A, True), 1200)
            g, _ = apply_frozen(subset(D, A, False), F['logu'], F['loga'], F['lab'], D['keys'])
            gs.append(g); us.append(F['logu'])
        return float(np.mean(gs)), us
    real, real_u = run_ruler(MU0)
    rr = []
    for j in range(150):
        mu = MU0.copy()
        mu[~L.ISAREA] = rng.uniform(lo, hi, (~L.ISAREA).sum())
        mu[L.ISAREA] = rng.uniform(math.log(10), math.log(10000), L.ISAREA.sum())
        rr.append(run_ruler(mu)[0])
        if j % 25 == 0:
            print(j, real, np.quantile(rr, [0.5, 0.95]), flush=True)
    L.MU[:] = MU0
    res = dict(src=src, real=real, real_u=real_u, rand=rr, rank=float(np.mean(np.array(rr) >= real)))
    fn = f'c2_rulers_{src}' + ('' if AREA else '_noarea') + '.json'
res['secs'] = time.time() - t0
json.dump(res, open(os.path.join(CK, fn), 'w'), default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('done', fn, res['secs'])
