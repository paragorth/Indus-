#!/usr/bin/env python3
"""la74 cycle 1: random vessel sets vs a fitted no-vessel model, held-out documents; controls."""
import sys, os, json, math, time
sys.path.insert(0, os.path.dirname(__file__))
from la74_common import *
from multiprocessing import Pool

NH = int(os.environ.get('NH', 3000)); NSIM = 20000

def job(args):
    seed, train, test, signs, V = args
    rng = np.random.default_rng(seed)
    nv = NoVessel(train, V)
    ltr = [nv.logp(s) for s in train]; lte = [nv.logp(s) for s in test]
    base_tr = sum(ltr); base_te = sum(lte)
    res = []
    for h in range(NH // 2):
        H = sample_H(rng)
        d = simulate(H, NSIM, rng)
        mp, sc, e = fit_mapping(d, H['k'], train, ltr, signs, rng, iters=60)
        te = ll_strings(d, mp, test, lte, e)
        res.append(dict(H=H, map=mp, e=e, gtr=(sc - base_tr) / len(train), gte=(te - base_te) / len(test)))
    return res

def refine_job(args):
    seed, cands, train, test, V, signs = args
    rng = np.random.default_rng(seed)
    nv = NoVessel(train, V)
    ltr = [nv.logp(s) for s in train]; lte = [nv.logp(s) for s in test]
    btr, bte = sum(ltr), sum(lte)
    out = []
    for c in cands:
        H = dict(c['H']); mp = list(c['map'])
        def sc(H, mp):
            d = simulate(H, NSIM, np.random.default_rng(12345))   # common random numbers
            return max((ll_strings(d, mp, train, ltr, e), e) for e in EGRID) + (d,)
        best, be, bd = sc(H, mp)
        for it in range(NREF):
            H2 = dict(H); mp2 = list(mp); r = rng.random()
            if r < 0.45:
                sz = list(H['sizes']); i = int(rng.integers(len(sz))); sz[i] *= math.exp(rng.normal(0, 0.25))
                sz = sorted([min(x, 0.99) for x in sz], reverse=True); H2['sizes'] = sz
            elif r < 0.7:
                key = ['u', 'g', 'rho'][int(rng.integers(3))]
                H2[key] = float(np.clip(H[key] + rng.normal(0, 0.15), 0.01, 0.99))
            elif r < 0.8:
                H2['cmax'] = int(np.clip(H['cmax'] + rng.choice([-1, 1]), 1, 4))
            else:
                a = int(rng.integers(H['k'])); cand = [x for x in signs if x not in mp]
                if rng.random() < 0.5 and H['k'] > 1:
                    b = int(rng.integers(H['k'])); mp2[a], mp2[b] = mp2[b], mp2[a]
                elif cand: mp2[a] = cand[int(rng.integers(len(cand)))]
            s2, e2, d2 = sc(H2, mp2)
            if s2 > best: best, be, bd, H, mp = s2, e2, d2, H2, mp2
        te = ll_strings(bd, mp, test, lte, be)
        out.append(dict(H=H, map=mp, e=be, gtr=(best - btr) / len(train), gte=(te - bte) / len(test)))
    return out

NREF = int(os.environ.get('NREF', 120)); NTOP = int(os.environ.get('NTOP', 20))

def run(name, strings, groups, seed, signs, V=None):
    V = V or VFULL
    """strings: list of tuples; groups: doc id per string. 50/50 doc split."""
    rng = np.random.default_rng(seed)
    docs = sorted(set(groups)); rng.shuffle(docs); tr = set(docs[:len(docs) // 2])
    train = [s for s, g in zip(strings, groups) if g in tr]
    test = [s for s, g in zip(strings, groups) if g not in tr]
    with Pool(2) as P:
        out = P.map(job, [(seed * 10 + w, train, test, signs, V) for w in range(2)])
    res = [r for o in out for r in o]
    res.sort(key=lambda r: -r['gtr'])
    top0 = res[:NTOP]
    with Pool(2) as P:
        out = P.map(refine_job, [(seed * 77 + w, top0[w::2], train, test, V, signs) for w in range(2)])
    top = sorted([r for o in out for r in o], key=lambda r: -r['gtr'])
    summ = dict(name=name, n_train=len(train), n_test=len(test),
                best_gtr=top[0]['gtr'], top_gte_mean=float(np.mean([r['gte'] for r in top])),
                top_gte_pos=int(sum(r['gte'] > 0 for r in top)), ntop=len(top),
                all_gte_mean=float(np.mean([r['gte'] for r in res])),
                frac_e1=float(np.mean([r['e'] == 1.0 for r in res])),
                best=dict(map=top[0]['map'], sizes=top[0]['H']['sizes'], e=top[0]['e'], H=top[0]['H'], gte=top[0]['gte']))
    json.dump(dict(summ=summ, top=top), open(os.path.join(CK, f'c1_{name}.json'), 'w'), default=float)
    print(json.dumps(summ, default=float), flush=True)
    return summ

if __name__ == '__main__':
    os.makedirs(CK, exist_ok=True)
    S = load_strings(); V = vocab([r['s'] for r in S]); globals()['VFULL'] = V
    st = [norm(r['s'], V) for r in S]; grp = [r['doc'] for r in S]
    signs = V[:12]
    which = sys.argv[1:] or ['real', 'shuf', 'nvworld', 'planted']
    rng = np.random.default_rng(74)
    for w in which:
        if w == 'real': run('real', st, grp, 1, signs)
        if w in ('real2', 'real3'): run(w, st, grp, int(w[-1]), signs)
        if w == 'shuf': run('shuf', shuffle_signs(st, rng), grp, 1, signs)
        if w == 'nvworld':
            nv = NoVessel(st, V); run('nvworld', nv_sample(nv, len(st), rng), grp, 1, signs)
        if w.startswith('planted'):
            prng = np.random.default_rng(int(w[7:] or 0) + 500)
            H = dict(k=4, sizes=[0.5, 0.25, 0.1, 0.05], u=0.3, g=0.3, rho=0.6, cmax=2) if w == 'planted' else sample_H(prng, 3, 5)
            mp = list(prng.choice(signs[:8], H['k'], replace=False))
            d = simulate(H, 200000, prng)
            pl = dist_sample(d, mp, len(st), prng)
            # contaminate 25% with NV strings (realistic)
            nv = NoVessel(st, V); mix = nv_sample(nv, len(st), prng)
            pl = [b if prng.random() < 0.25 else a for a, b in zip(pl, mix)]
            json.dump(dict(H=H, map=mp), open(os.path.join(CK, f'truth_{w}.json'), 'w'), default=float)
            run(w, pl, grp, 1, signs)
