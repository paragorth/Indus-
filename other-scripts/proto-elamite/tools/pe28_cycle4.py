"""pe28 cycle 4: BAYESIAN rate inference with random restarts (joint, all keys at once).
Model: every target line is either 'computed' (prob PI) from one of its windows (uniform over windows)
by y = sum r(key) x, or 'free' with the empirical frequency f(y) of its value in its system.  Each key is
absent (prob 1-RHO) or takes a value from its notation grid (RHO / |grid|).  Anchors fixed.  Gibbs sampling
over keys (states that explain no line are lumped with 'absent'), annealed from T=3 to 1, NRS restarts from
random initial rates.  Output: per key, fraction of restarts whose final sample has the same value
(stability) and the posterior frequency over the last sweeps.
Held-out: NS 70/30 splits, MAP-ish rates (value held in >= 50% of restarts) fitted on 70%, exact hits on
30% (windows fully fixed, >= 1 new key) vs (i) values re-dealt within (system, final sign) and (ii) random
grid rates, 200x.  Ur III control the same way."""
import os, sys, json, time, math
import numpy as np
from collections import Counter, defaultdict
from fractions import Fraction as Fr
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe28_common import *  # noqa

PI = float(os.environ.get('PI', 0.3))
RHO = float(os.environ.get('RHO', 0.05))
NRS = int(os.environ.get('NRS', 8))
SW = int(os.environ.get('SW', 10))
NS = int(os.environ.get('NS', 8))
NR = int(os.environ.get('NR', 200))
t0 = time.time()


def prep(E, gs=None):
    freq = defaultdict(Counter)
    for e in E:
        freq[e['sys']][e['y']] += 1
    tot = {s: sum(c.values()) for s, c in freq.items()}
    kt = defaultdict(set)
    sup = defaultdict(set)
    for i, e in enumerate(E):
        for w in e['W']:
            for k in w:
                kt[k].add(i); sup[k].add(e['t'])
    keys = [k for k in kt if len(sup[k]) >= 2]
    f = [(freq[e['sys']][e['y']] + 1) / (tot[e['sys']] + 50) for e in E]
    return keys, kt, f


def gibbs(E, A, keys, kt, f, rng, gs=None):
    R = dict(A)
    for k in keys:
        if k not in A and rng.random() < 0.3:
            g = GSET[gs or k[2]]
            R[k] = list(g)[int(rng.integers(len(g)))]

    def ll_t(i, R):
        e = E[i]
        m = 0
        for w in e['W']:
            if all(k in R for k in w) and sum(R[k] * x for k, x in w.items()) == e['y']:
                m += 1
        return math.log(PI * m / len(e['W']) + (1 - PI) * f[i])
    order = [k for k in keys if k not in A]
    for sw in range(SW):
        T = max(1.0, 3.0 - 2.0 * sw / max(1, SW - 3))
        rng.shuffle(order)
        for k in order:
            R.pop(k, None)
            g = GSET[gs or k[2]]
            cand = set()
            for i in kt[k]:
                e = E[i]
                for w in e['W']:
                    if k in w and all(kk in R for kk in w if kk != k):
                        rest = e['y'] - sum(R[kk] * x for kk, x in w.items() if kk != k)
                        if rest > 0:
                            v = rest / w[k]
                            if v in g:
                                cand.add(v)
            if not cand:
                continue
            base = sum(ll_t(i, R) for i in kt[k])
            states = [None] + sorted(cand)
            lp = [math.log(1 - RHO)]
            for v in states[1:]:
                R[k] = v
                lp.append(math.log(RHO / len(g)) + sum(ll_t(i, R) for i in kt[k]) - base)
            R.pop(k, None)
            lp = np.array(lp) / T
            p = np.exp(lp - lp.max()); p /= p.sum()
            c = int(rng.choice(len(states), p=p))
            if states[c] is not None:
                R[k] = states[c]
    return {k: v for k, v in R.items() if k not in A}


G = {}


def _run(seed):
    return gibbs(G['E'], G['A'], G['keys'], G['kt'], G['f'], np.random.default_rng(seed), G['gs'])


def fit(E, A, gs=None, seed=0):
    keys, kt, f = prep(E, gs)
    G.update(E=E, A=A, keys=keys, kt=kt, f=f, gs=gs)
    with Pool(2) as p:
        runs = p.map(_run, [seed * 1000 + r for r in range(NRS)])
    cnt = Counter((k, v) for R in runs for k, v in R.items())
    stab = {}
    for (k, v), c in cnt.items():
        if c / NRS >= 0.5:
            stab[k] = (v, c / NRS)
    return stab


def heldout(E, ntab, A, gs, label, seed):
    rng = np.random.default_rng(seed)
    real, nsc, nd, nr = 0, 0, np.zeros(NR), np.zeros(NR)
    perkey = defaultdict(lambda: [0, 0, 0.0])
    for s in range(NS):
        test = set(rng.choice(ntab, int(0.3 * ntab), replace=False).tolist())
        Etr = [e for e in E if e['t'] not in test]
        Ete = [e for e in E if e['t'] in test]
        st = fit(Etr, A, gs, seed=s + 1)
        rates = dict(A); rates.update({k: v for k, (v, c) in st.items()})
        n, h, det = predict(Ete, rates, A)
        real += h; nsc += n
        for k in st:
            perkey[k][0] += 1
        for e in Ete:
            for w in e['W']:
                if all(k in rates for k in w) and any(k not in A for k in w) and \
                        sum(rates[k] * x for k, x in w.items()) == e['y']:
                    for k in w:
                        if k not in A:
                            perkey[k][1] += 1
                    break
        for r in range(NR):
            Ed = redeal(Ete, rng)
            nd[r] += predict(Ed, rates, A)[1]
            if r < 20:
                for e in Ed:
                    for w in e['W']:
                        if all(k in rates for k in w) and any(k not in A for k in w) and \
                                sum(rates[k] * x for k, x in w.items()) == e['y']:
                            for k in w:
                                if k not in A:
                                    perkey[k][2] += 1 / 20
                            break
            fr = dict(rates)
            for k in st:
                g = G_UR if gs == 'UR' else grid_for(k[2])
                fr[k] = g[int(rng.integers(len(g)))]
            nr[r] += predict(Ete, fr, A)[1]
        print(label, 'split', s, len(st), n, h, round(time.time() - t0), flush=True)
    out = {'scored': nsc, 'hits': real, 'null_redeal': float(nd.mean()), 'p_redeal': float((1 + (nd >= real).sum()) / (1 + NR)),
           'null_rand': float(nr.mean()), 'p_rand': float((1 + (nr >= real).sum()) / (1 + NR)),
           'per_key': sorted(((keystr(k), v[0], v[1], round(v[2], 2)) for k, v in perkey.items()), key=lambda a: -a[2])[:40]}
    print(label, 'HELDOUT', out['scored'], real, out['null_redeal'], out['p_redeal'], out['null_rand'], out['p_rand'], flush=True)
    for r in out['per_key'][:12]:
        print('    ', r, flush=True)
    return out


if __name__ == '__main__':
    res = {'PI': PI, 'RHO': RHO}
    US = ur_seqs()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(US), 500, replace=False)
    EU = build([US[i] for i in idx], ur=True)
    AU = {('gurusz', 'sze-bi', 'UR'): Fr(60)}
    st = fit(EU, AU, 'UR')
    UT = [US[i] for i in idx]
    rows = []
    for k, (v, c) in sorted(st.items(), key=lambda a: -a[1][1]):
        tabs = [UT[e['t']] for e in EU if any(k in w for w in e['W'])]
        wr = [t for t in tabs if t[2]]
        ok = sum(1 for t in wr if v in [Fr(r) for r in t[2]])
        rows.append((keystr(k), str(v), c, ok, len(wr)))
    res['ur3_full'] = rows
    print('UR3 full', len(st), rows[:15], round(time.time() - t0), flush=True)
    res['ur3_heldout'] = heldout(EU, 500, AU, 'UR', 'UR3', 11)
    S = pe_seqs()
    E = build(S)
    A = anchors_pe()
    st = fit(E, A)
    res['pe_full'] = [(keystr(k), str(v), c) for k, (v, c) in sorted(st.items(), key=lambda a: -a[1][1])]
    print('PE full', len(st), res['pe_full'][:30], round(time.time() - t0), flush=True)
    json.dump(res, open(os.path.join(CK, f'cycle4_pi{PI}_rho{RHO}.json'), 'w'), indent=1, default=str)
    res['pe_heldout'] = heldout(E, len(S), A, None, 'PE', 12)
    json.dump(res, open(os.path.join(CK, f'cycle4_pi{PI}_rho{RHO}.json'), 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))
