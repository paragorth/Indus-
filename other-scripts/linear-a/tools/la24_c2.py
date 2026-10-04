#!/usr/bin/env python3
"""LA-24 cycle 2: BASELINE-FREE APPORTIONMENT FINGERPRINTS (cycle 1's MDL design had no power).

F1 remainder signature: a list holds both q and q+1 (q >= 3) - the trace of a total divided into
   equal shares and rounded (largest remainder / divisor methods give q,q,q+1 patterns).
F1r the same, and the q/q+1 cluster sums to a multiple of 5 (a round total was divided).
F2 common unit: >= 3 distinct values (values > 1), all u*w with w in 1..8 and u >= 2 (u not
   forced to be a power of 10: u in 2..50 tried; u = 5 or 10 'round-number' cases are counted
   separately as F2_round).
F3 fraction absorption, per fraction value set V: a fraction-bearing list whose entries (all but
   at most one, >= 3) equal u*w, w in 1..8, with u NOT an integer - the fractions carry the
   remainder of an exact division. Scored for CONV, LA1BIN and 3,000 random V.
Nulls for F1/F2 (2,000 each): N1 amounts shuffled across lists (keeps the corpus marginal, i.e.
the preference for round numbers), N1s shuffled within site, N3c frequency-weighted band
replacement (keeps scale and repeats). Power: 20 planted largest-remainder/ROUND lists embedded
in the corpus. Comparators: Ur III ration lists, LB lists.
"""
import json, os, random, sys, time
import numpy as np
from collections import Counter, defaultdict
from fractions import Fraction as Fr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la24_common import *

NR = int(os.environ.get('NR', 2000))
NV = int(os.environ.get('NV', 3000))


def f1(vals):
    s = Counter(int(round(v)) for v in vals if abs(v - round(v)) < 1e-9)
    hit = hitr = False
    for q in s:
        if q >= 3 and (q + 1) in s:
            hit = True
            if (q * s[q] + (q + 1) * s[q + 1]) % 5 == 0: hitr = True
    return hit, hitr


def f2(vals):
    d = sorted({v for v in vals if v > 1 and abs(v - round(v)) < 1e-9})
    if len(d) < 3: return False, False
    d = [int(round(v)) for v in d]
    for u in range(50, 1, -1):
        if all(v % u == 0 and v // u <= 8 for v in d):
            return True, u in (5, 10, 20, 25, 50)
    return False, False


def f3(vals):
    n = len(vals)
    if n < 3: return False
    x = np.array(vals)
    if np.all(np.abs(x - np.round(x)) < 1e-9): return False
    for i in range(n):
        for w in range(1, 9):
            u = x[i] / w
            if abs(u - round(u)) < 1e-9 or u <= 0: continue
            r = x / u
            ok = (np.abs(r - np.round(r)) < 1e-7) & (np.round(r) >= 1) & (np.round(r) <= 8)
            if ok.sum() >= max(3, n - 1): return True
    return False


def stats(L, nv):
    a = np.array([f1([nv(t) for t in l]) + f2([nv(t) for t in l]) for l in L])
    return a.sum(0)  # F1, F1r, F2, F2_round


def run_f12(name, lists, nv, sites=None, nr=NR):
    L = [l['amts'] for l in lists]
    obs = stats(L, nv)
    B = Baseline(L, nv); pool = band_pool(B, nv)
    rng = random.Random(5)
    res = {'name': name, 'n': len(L), 'obs': obs.tolist()}
    kinds = {'N1': lambda: null_shuffle(L, rng), 'N3c': lambda: [null_band_w(t, pool, nv, rng) for t in L]}
    if sites:
        def n1s():
            out = [None] * len(L); by = defaultdict(list)
            for i, s in enumerate(sites): by[s].append(i)
            for s, ii in by.items():
                sub = null_shuffle([L[i] for i in ii], rng)
                for i, x in zip(ii, sub): out[i] = x
            return out
        kinds['N1s'] = n1s
    for k, fn in kinds.items():
        sims = np.array([stats(fn(), nv) for _ in range(nr)])
        res[k] = {'mean': sims.mean(0).round(2).tolist(),
                  'P_upper': (((sims >= obs).sum(0) + 1) / (nr + 1)).round(4).tolist(),
                  'P_lower': (((sims <= obs).sum(0) + 1) / (nr + 1)).round(4).tolist()}
    print(json.dumps(res), flush=True)
    return res


def main():
    t0 = time.time()
    out = {}
    la = la_lists()
    nv = la_numval(CONV)
    out['LA'] = run_f12('LA', la, nv, sites=[l['site'] for l in la])
    hits = [(l['id'], [round(nv(t), 3) for t in l['amts']]) for l in la if f1([nv(t) for t in l['amts']])[0]]
    out['LA_F1_lists'] = hits
    out['LA_F2_lists'] = [(l['id'], [round(nv(t), 3) for t in l['amts']]) for l in la if f2([nv(t) for t in l['amts']])[0]]
    # power: 20 planted HAMIL/ROUND equal-ish share lists replace 20 real lists
    rng = random.Random(77)
    pl = [dict(l) for l in la]
    for i in rng.sample(range(len(la)), 20):
        n = min(max(len(la[i]['amts']), 3), 8)
        w = [rng.choice([1, 1, 1, 2]) for _ in range(n)]
        vals = apportion(rng.randint(3 * n, 30 * n), w, rng.choice(['HAMIL', 'ROUND']), 1.0)
        pl[i] = {'id': 'PL', 'site': 'PL', 'amts': [(int(v), ()) for v in vals if v > 0] or [(1, ())]}
    out['PLANT'] = run_f12('PLANT', pl, nv, nr=500)
    ur = ur_lists(max_lists=300, seed=2)
    if ur: out['UR'] = run_f12('UR', ur, lambda a: float(a[0]), nr=500)
    # LB: integers only meaningful for F1/F2 (counted units such as persons or animals)
    out['LB'] = run_f12('LB', lb_lists(), lambda a: float(a[0]), nr=500)

    # F3 fraction absorption across value sets
    fl = [l for l in la if any(a[1] for a in l['amts'])]
    def f3score(V):
        return [f3([val(t, V) for t in l['amts']]) for l in fl]
    conv = f3score(CONV); bin_ = f3score(LA1BIN)
    rngv = random.Random(31)
    Vs = [random_V(rngv) for _ in range(NV)]
    sc = np.array([sum(f3score(V)) for V in Vs])
    o = {'n_frac_lists': len(fl), 'CONV': sum(conv), 'LA1BIN': sum(bin_),
         'random_mean': float(sc.mean()), 'random_sd': float(sc.std()), 'random_max': int(sc.max()),
         'P_CONV': float(((sc >= sum(conv)).sum() + 1) / (NV + 1)),
         'P_LA1BIN': float(((sc >= sum(bin_)).sum() + 1) / (NV + 1)),
         'CONV_lists': [(fl[i]['id'], [round(val(t, CONV), 4) for t in fl[i]['amts']]) for i in range(len(fl)) if conv[i]],
         'LA1BIN_lists': [(fl[i]['id'], [round(val(t, LA1BIN), 4) for t in fl[i]['amts']]) for i in range(len(fl)) if bin_[i]]}
    # letter pinning: mean score of random sets by each letter's value
    pin = {}
    for let in ('J', 'E', 'JE', 'D', 'B', 'K', 'F', 'A', 'H', 'L2'):
        by = defaultdict(list)
        for V, s in zip(Vs, sc): by[str(V[let])].append(s)
        best = sorted(((np.mean(v), k, len(v)) for k, v in by.items()), reverse=True)[:3]
        # permutation: spread of best-group mean under shuffled scores
        vals = [str(V[let]) for V in Vs]
        sims = []
        prng = np.random.default_rng(3)
        for _ in range(300):
            ss = prng.permutation(sc); g = defaultdict(list)
            for v, s in zip(vals, ss): g[v].append(s)
            sims.append(max(np.mean(x) for x in g.values()))
        pin[let] = {'best': [(k, round(m, 3), c) for m, k, c in best],
                    'P_best': float((np.sum(np.array(sims) >= best[0][0]) + 1) / 301)}
    o['pin'] = pin
    # power for F3: 25 planted lists with fractional unit share written in CONV letters
    rp = random.Random(8); hitsP = []
    for _ in range(200):
        n = rp.randint(3, 7)
        while True:
            w = [rp.choice([1, 2, 3, 4]) for _ in range(n)]
            u = rp.choice([1, 2, 3, 5]) + rp.choice([0.25, 0.5, 0.75])
            a = [to_amount(u * x) for x in w]
            if all(z is not None for z in a) and any(z[1] for z in a): break
        hitsP.append((f3([val(t, CONV) for t in a]), np.mean([f3([val(t, V) for t in a]) for V in Vs[:200]])))
    o['planted_F3_CONV'] = float(np.mean([h[0] for h in hitsP]))
    o['planted_F3_randomV'] = float(np.mean([h[1] for h in hitsP]))
    out['F3'] = o
    out['secs'] = time.time() - t0
    print(json.dumps({k: v for k, v in o.items() if not k.endswith('_lists')}), flush=True)
    json.dump(out, open(os.path.join(CK, 'c2.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
