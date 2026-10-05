"""v66 cycle 1: does a text's keyword graph align with the medieval medical concept graph?

Per text: best injective concept->keyword match score S* (annealing, many restarts) vs
 (a) the same text with keyword incidence rewired keeping page and keyword degrees (curveball),
 (b) bigram-Markov resynthesis of the text (keywords re-selected).
Held-out: fit the map on half the pages, score it on the other half vs random maps and vs maps
fitted on rewired halves. Controls (real herbals, opaque tokens + padding, leave-one-out reference
graph): true class recovery vs chance; oracle-keyword upper bound. Unrelated genres must not align.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v66_lib import *
from multiprocessing import Pool

NU = 227
R_REAL, R_NULL, NNULL, NMK = 64, 24, 10, 3


def oracle_kw(units, back, K=80):
    """one opaque token per concept class (most frequent, in >= 4 units), filled with bursty words."""
    df = Counter(w for u in units for w in set(u))
    best = {}
    for w, n in df.items():
        c = classify(back.get(w, w))
        if c and n >= 4 and (c not in best or n > df[best[c]]): best[c] = w
    kw = list(best.values())
    for w in keywords(units, K * 2):
        if len(kw) >= K: break
        if w not in kw: kw.append(w)
    return kw


def halves(units):
    return units[0::2], units[1::2]


def heldout(Cz, units, kw, seed):
    A, Bu = halves(units)
    XA = incidence(A, kw); XB = incidence(Bu, kw)
    BA, BB = zoff(phi(XA)), zoff(phi(XB))
    _, mA, _, _ = qap(Cz, BA, restarts=R_NULL, seed=seed)
    s_ho = score(Cz, BB, mA)
    rng = np.random.default_rng(seed)
    rnd = [score(Cz, BB, rng.choice(len(kw), Cz.shape[0], replace=False)) for _ in range(500)]
    nul = []
    for r in range(6):
        Bn = zoff(phi(curveball(XA, seed + 100 + r)))
        _, mn, _, _ = qap(Cz, Bn, restarts=R_NULL // 2, seed=seed + 200 + r)
        nul.append(score(Cz, BB, mn))
    # reverse direction
    _, mB, _, _ = qap(Cz, BB, restarts=R_NULL, seed=seed + 1)
    agree = float(np.mean(mA == mB))
    return {'ho': s_ho, 'ho_rand_mu': float(np.mean(rnd)), 'ho_rand_sd': float(np.std(rnd)),
            'ho_null_mu': float(np.mean(nul)), 'ho_null_sd': float(np.std(nul) + 1e-9), 'agree': agree,
            'agree_chance': 1.0 / len(kw)}


def job(args):
    name, kind, seed = args
    t0 = time.time()
    T = load_texts()
    if kind == 'voynich':
        _, units, _ = voynich_pages(name)
        back = {}
        ref = REF
    else:
        units = sample_units(T[name]['units'], NU, seed)
        units, back = opaque(units, seed)
        ref = [r for r in REF if r != name and r != TWIN.get(name)]
    Cz = zoff(concept_graph(T, ref))
    kw, X, B = keyword_graph(units)
    S, m, rsc, _ = qap(Cz, B, restarts=R_REAL, seed=seed)
    out = {'name': name, 'kind': kind, 'seed': seed, 'S': S, 'restart_spread': [float(np.percentile(rsc, q)) for q in (5, 50, 95)],
           'map': {CLASS_NAMES[i]: kw[k] for i, k in enumerate(m)}}
    nul = []
    for r in range(NNULL):
        Bn = zoff(phi(curveball(X, seed * 1000 + r)))
        nul.append(qap(Cz, Bn, restarts=R_NULL, seed=seed * 1000 + r)[0])
    out['rew_mu'], out['rew_sd'] = float(np.mean(nul)), float(np.std(nul) + 1e-9)
    out['z_rew'] = (S - out['rew_mu']) / out['rew_sd']
    mk = []
    for r in range(NMK):
        mu = markov_units(units, seed * 77 + r)
        _, _, Bm = keyword_graph(mu)
        mk.append(qap(Cz, Bm, restarts=R_NULL, seed=seed * 77 + r)[0])
    out['mk'] = mk
    out['heldout'] = heldout(Cz, units, kw, seed)
    if back:
        out['rec'] = recovery(m, kw, back)
        okw = oracle_kw(units, back)
        Xo = incidence(units, okw); Bo = zoff(phi(Xo))
        So, mo, _, _ = qap(Cz, Bo, restarts=R_REAL, seed=seed + 5)
        out['oracle'] = {'S': So, 'rec': recovery(mo, okw, back)}
        onul = []
        for r in range(5):
            Bn = zoff(phi(curveball(Xo, seed * 31 + r)))
            onul.append(qap(Cz, Bn, restarts=R_NULL, seed=seed * 31 + r)[0])
        out['oracle']['z_rew'] = (So - np.mean(onul)) / (np.std(onul) + 1e-9)
    out['sec'] = time.time() - t0
    print(name, seed, 'S %.3f z_rew %.2f mk %s' % (S, out['z_rew'], np.round(mk, 3)), out.get('rec'), out.get('oracle', {}).get('rec'), 'ho', round((out['heldout']['ho'] - out['heldout']['ho_null_mu']) / out['heldout']['ho_null_sd'], 2), flush=True)
    return out


if __name__ == '__main__':
    jobs = [('ZL3b', 'voynich', 11), ('IT2a', 'voynich', 12)]
    for s in (1, 2):
        jobs += [(n, 'control', s) for n in ['culpeper', 'konrad_plants', 'macer', 'circa_fr', 'celsus_lat', 'v21_IT']]
        jobs += [(n, 'unrelated', s) for n in UNRELATED]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(CK, 'cycle1.json'), 'w'), indent=1, default=float)
