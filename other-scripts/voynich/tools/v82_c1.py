"""v82 cycle 1: THE KILL GENERATOR. Random sweep of the mood + agreement generator family (v82_lib.gen_moodagr) fitted
to ZL3b, each scored with the frozen v81 junction-preserving test (onset slot, 3 frozen definitions, train leaf half 0,
test leaf half 1) and with the surface statistics it must match.
  stage 'sweep'  N random settings, 5 re-rolls (screen)
  stage 'climb'  local random perturbation of the best near-matching settings (maximise min-over-definitions gain
                 subject to matching), R rounds
  stage 'final'  the best settings: 3 fresh seeds, 20 re-rolls, same-section and same-page nulls, lag-2 ratio;
                 ZL-fitted and IT2a-fitted
Pre-registered kill: min over the 3 definitions of (gain - null) >= +0.030 with z >= 3 under the same-section null,
in every fresh seed, while every surface statistic is inside its band (v82_lib.bands).
"""
import os, sys, json, random, copy, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v82_lib as K, v72_lib as V
from multiprocessing import Pool

Z = I = B = None; MC = {}


def init():
    global Z, I, B
    Z = V.voynich('ZL3b'); I = V.voynich('IT2a')
    B = K.bands(K.surf_stats(Z), K.surf_stats(I))


def score(args):
    P, seed, nrep, page, which = args
    src = Z if which == 'ZL' else I
    X = K.gen_moodagr(src, P, seed, MC.setdefault(which, {}))
    c = K.Mini(X); fz = K.frozen(c, nrep=nrep)
    s = K.surf_stats(X); ok, bad = K.match(s, B)
    out = dict(P=P, seed=seed, which=which, g=[x[0] for x in fz], z=[x[1] for x in fz], stats=s, ok=ok, bad=bad)
    if page:
        fp = K.frozen(c, nrep=nrep, page=True); out['gp'] = [x[0] for x in fp]; out['zp'] = [x[1] for x in fp]
        out['lag'] = K.lag_ratio(c)
    return out


def jl(name, recs):
    with open(os.path.join(K.CK, name), 'a') as f:
        for r in recs: f.write(json.dumps(r, default=float) + '\n')


def load(name):
    p = os.path.join(K.CK, name)
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def fit(r):
    """objective: min-over-definitions gain (capped at +0.08: overshooting does not help a kill), penalised per
    out-of-band statistic and by the normalised distance outside the bands."""
    d = sum(max(0.0, abs(r['stats'][k] - B[k][0]) / B[k][1] - 1) for k in K.STAT_KEYS)
    return min(min(r['g']), 0.08) - 0.03 * len(r['bad']) - 0.01 * d


def perturb(P, rng):
    Q = copy.deepcopy(P); R = K.sample_params(rng)
    for key in rng.sample(list(R), rng.randint(1, 3)): Q[key] = R[key]
    for key in ('mood_beta', 'agr_lam', 'p_cite', 'p_mod', 'mood_rate'):
        if rng.random() < 0.3: Q[key] = Q[key] * math.exp(rng.gauss(0, 0.3))
    if Q['base'] != 'stack': Q['p_vert'] = 0.0
    if Q['base'] == 'junc': Q['p_cite'] = 0.0
    return Q


import math

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'sweep':
        N = int(sys.argv[2]); rng = random.Random(8201)
        done = len(load('c1_sweep.jsonl'))
        jobs = [(K.sample_params(rng), 100 + i, 5, False, 'ZL') for i in range(N)][done:]
        with Pool(2, initializer=init) as Pl:
            buf = []
            for i, r in enumerate(Pl.imap_unordered(score, jobs, chunksize=4)):
                buf.append(r)
                if len(buf) >= 20: jl('c1_sweep.jsonl', buf); buf = []
                if i % 100 == 0:
                    print(i + done, time.strftime('%H:%M:%S'), flush=True)
            jl('c1_sweep.jsonl', buf)
    elif stage == 'climb':
        R = int(sys.argv[2]); rng = random.Random(8202); init()
        pop = load('c1_sweep.jsonl') + load('c1_climb.jsonl')
        with Pool(2, initializer=init) as Pl:
            for rd in range(R):
                pop.sort(key=fit, reverse=True); top = pop[:12]
                jobs = [(perturb(t['P'], rng), 5000 + rd * 100 + j, 5, False, 'ZL') for j in range(4) for t in top]
                new = list(Pl.imap_unordered(score, jobs)); jl('c1_climb.jsonl', new); pop += new
                b = max(new, key=fit)
                print('round', rd, 'best new min-gain %.4f bad %s | overall best %.4f' % (min(b['g']), b['bad'], fit(max(pop, key=fit))), flush=True)
    elif stage == 'shape':
        # second climb: a generator that matches the Voynich's own gains (not only exceeds +0.03) and every band
        R = int(sys.argv[2]); rng = random.Random(8203); init()
        VZ = np.array([0.0305, 0.0491, 0.0857])          # ZL frozen values (this lib, 20 re-rolls)
        def sfit(r):
            d = sum(max(0.0, abs(r['stats'][k] - B[k][0]) / B[k][1] - 1) for k in K.STAT_KEYS)
            g = np.clip(np.array(r['g']), 1e-3, None)
            return -float(np.abs(np.log(g / VZ)).sum()) - 0.5 * len(r['bad']) - 0.2 * d
        pop = load('c1_sweep.jsonl') + load('c1_climb.jsonl') + load('c1_shape.jsonl')
        with Pool(2, initializer=init) as Pl:
            for rd in range(R):
                pop.sort(key=sfit, reverse=True); top = pop[:12]
                jobs = [(perturb(t['P'], rng), 7000 + rd * 100 + j, 5, False, 'ZL') for j in range(4) for t in top]
                new = list(Pl.imap_unordered(score, jobs)); jl('c1_shape.jsonl', new); pop += new
                b = max(pop, key=sfit)
                print('round', rd, 'best', ['%.3f' % x for x in b['g']], b['bad'], '%.3f' % sfit(b), flush=True)
    elif stage == 'final':
        pop = load('c1_sweep.jsonl') + load('c1_climb.jsonl') + load('c1_shape.jsonl')
        init()
        okp = sorted([r for r in pop if r['ok']], key=lambda r: min(r['g']), reverse=True)[:6]
        near = sorted([r for r in pop if len(r['bad']) <= 1], key=lambda r: min(r['g']), reverse=True)[:4]
        anyp = sorted(pop, key=fit, reverse=True)[:4]
        VZ = np.array([0.0305, 0.0491, 0.0857])
        shp = sorted([r for r in pop if r['ok']], key=lambda r: float(np.abs(np.log(np.clip(r['g'], 1e-3, None) / VZ)).sum()))[:4]
        okp = okp + shp
        fam = {}
        for r in sorted(pop, key=fit, reverse=True):
            fam.setdefault(r['P']['agr'], r)
        cand = []; seen = set()
        for r in okp + near + anyp + list(fam.values()):
            key = json.dumps(r['P'], sort_keys=True)
            if key not in seen: seen.add(key); cand.append(r)
        jobs = [(r['P'], s, 20, True, w) for r in cand for s in (9001, 9002, 9003) for w in ('ZL', 'IT')]
        with Pool(2, initializer=init) as Pl:
            out = list(Pl.imap(score, jobs))
        jl('c1_final.jsonl', out)
