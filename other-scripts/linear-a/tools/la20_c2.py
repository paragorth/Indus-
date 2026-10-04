#!/usr/bin/env python3
"""LA-20 cycle 2: (a) planted hierarchies in Linear A-sized data: does the pipeline find them?
(b) mixtures of K hidden orders (K=1..4) on real lists, held-out gain, with the max over K
taken in every shuffled replicate too (search-size correction); planted 2-order mixture as
positive control for the mixture model.
usage: la20_c2.py plant|mix TYPES NREP"""
import sys, json, time
from scipy.stats import kendalltau
from la20_common import *

mode, types, NREP = sys.argv[1], sys.argv[2], int(sys.argv[3])
docs = load_la()


def plant(ords, strength, frac, rng, mixture=False):
    items = sorted({x for o in ords for x in o})
    z = {x: rng.gauss(0, 1) * strength for x in items}
    out = []
    for i, o in enumerate(ords):
        if rng.random() < frac:
            sign = -1 if (mixture and i % 2) else 1
            key = {x: sign * z[x] + (-np.log(-np.log(rng.random()))) for x in o}  # Gumbel -> PL sample
            out.append(sorted(o, key=lambda x: -key[x]))
        else:
            o = list(o); rng.shuffle(o); out.append(o)
    return out, z


if mode == 'plant':
    res = {}
    for T in types:
        ords = [o for _, o in orders(docs, T)]
        null = json.load(open(os.path.join(CK, 'c1_%s.json' % T)))['nulls']['BT']
        na = np.array([x[1] / x[0] for x in null]); thr = np.quantile(na, 0.95)
        cnt = Counter(x for o in ords for x in o)
        for s in [0.25, 0.5, 1.0, 2.0, 4.0]:
            for fr in [0.25, 0.5, 1.0]:
                det, accs, taus = 0, [], []
                for r in range(NREP):
                    rng = random.Random(r * 31 + int(s * 100) + int(fr * 1000))
                    p, z = plant(ords, s, fr, rng)
                    a = cv_score(p, fit_bt, seed=0); ac = a[1] / a[0]
                    accs.append(ac); det += ac > thr
                    sc = fit_bt(p); rec = [x for x in sc if cnt[x] >= 3]
                    taus.append(kendalltau([z[x] for x in rec], [sc[x] for x in rec])[0])
                res['%s_%g_%g' % (T, s, fr)] = (det / NREP, float(np.mean(accs)), float(np.nanmean(taus)))
                print(T, 'strength', s, 'frac', fr, 'detect %.2f acc %.3f tau(recurrent) %.2f' % res['%s_%g_%g' % (T, s, fr)], 'null95 %.3f' % thr, flush=True)
        json.dump(res, open(os.path.join(CK, 'c2_plant_%s.json' % T), 'w'))

if mode == 'mix':
    KS = [1, 2, 3, 4]
    for T in types:
        ords = [o for _, o in orders(docs, T)]
        out = {'real': {}, 'null': [], 'pmix': {}}
        t0 = time.time()
        out['real'] = {K: mixture_heldout(ords, K, restarts=3) for K in KS}
        print(T, 'real', out['real'], round(time.time() - t0), flush=True)
        # positive control: planted 2-order mixture (half the lists reversed), strength 2
        for r in range(3):
            p, _ = plant(ords, 2.0, 1.0, random.Random(77 + r), mixture=True)
            out['pmix'][r] = {K: mixture_heldout(p, K, restarts=3) for K in KS}
            print(T, 'planted-mixture', out['pmix'][r], flush=True)
        for r in range(NREP):
            s = shuffle_within(ords, random.Random(5000 + r))
            out['null'].append({K: mixture_heldout(s, K, seed=r + 1, restarts=3) for K in KS})
            if r % 5 == 0:
                print(T, 'null', r, round(time.time() - t0), flush=True)
                json.dump(out, open(os.path.join(CK, 'c2_mix_%s.json' % T), 'w'))
        json.dump(out, open(os.path.join(CK, 'c2_mix_%s.json' % T), 'w'))
