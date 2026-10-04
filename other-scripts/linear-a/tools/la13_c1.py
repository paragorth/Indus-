#!/usr/bin/env python3
"""LA-13 cycle 1: exhaustive single-rule search for cross-site sign correspondences, with nulls and controls.
Statistics: max independent support over all rules (family-wise, so search size is corrected by the max-stat null),
and the number of rules with support >= 3, for position-specific and pooled rules, word length >= 2 and >= 3.
Nulls: N1 site labels permuted over documents (plain, and stratified by support type); N2 per-site bigram resampled types.
Controls: planted rules in Linear A (Khania), Linear B site-sized subsamples with planted rules and unplanted,
Linear A random 2-way split. Two workers; checkpoint per stage in data/la13/c1_*.json."""
import sys, os, json, random, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la13_common import *

KEYS = ['any_max', 'any_n3', 'pos_max', 'pos_n3', 'npairs']

def stats_for(units, labels, minlen):
    T = types_by_group([(u[0], u[1], u[2], [w for w in u[3] if len(w) >= minlen]) for u in units], labels)
    S, P = summary_stats(T)
    return S, P

def job_shuffle(args):
    seed, strat, minlen = args
    rnd = random.Random(seed)
    labs = shuffle_labels(UNITS, rnd, strat)
    S, P = stats_for(UNITS, labs, minlen)
    # per-rule counts for rules present in the real data (key without site pair permutation is meaningless; keep max only)
    return S

def job_markov(args):
    seed, minlen = args
    rnd = random.Random(seed)
    T = types_by_group([(u[0], u[1], u[2], [w for w in u[3] if len(w) >= minlen]) for u in UNITS])
    S, P = summary_stats(markov_types(T, rnd))
    return S

def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]

def ckpt(name, obj):
    json.dump(obj, open(os.path.join(OUT, f'c1_{name}.json'), 'w'), indent=1, default=str)

def load(name):
    p = os.path.join(OUT, f'c1_{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None

def plant(units, site, rules, rate, rnd):
    """replace sign X by Y in words of documents at `site` (each occurrence with probability rate)."""
    out = []
    for u in units:
        if u[1] != site: out.append(u); continue
        ws = []
        for w in u[3]:
            w = list(w)
            for i, s in enumerate(w):
                for (x, y) in rules:
                    if s == x and rnd.random() < rate: w[i] = y
            ws.append(tuple(w))
        out.append((u[0], u[1], u[2], ws))
    return out

def detect(units, rules, nperm, rnd, minlen=2):
    """run the search on units; return rank / support of planted rules and the plain-shuffle max-stat threshold."""
    S, P = stats_for(units, None, minlen)
    R = rule_support(P, False)
    ranked = sorted(R.items(), key=lambda kv: -kv[1][0])
    mx = []
    for k in range(nperm):
        labs = shuffle_labels(units, rnd, False)
        s2, _ = stats_for(units, labs, minlen); mx.append(s2['any_max'])
    thr = pct(mx, 0.95)
    res = []
    for (x, y) in rules:
        best = None
        for i, (r, v) in enumerate(ranked):
            if (r[2], r[3]) == (x, y) or (r[2], r[3]) == (y, x):
                best = (i + 1, v[0], r[0], r[1]); break
        res.append(dict(rule=f'{x}->{y}', rank=best[0] if best else None, k=best[1] if best else 0,
                        sites=best[2:] if best else None))
    top = [(r, v[0]) for r, v in ranked[:5]]
    return dict(planted=res, thr95=thr, null_max_mean=sum(mx) / len(mx), real_max=S['any_max'], top5=top)

def main():
    global UNITS
    UNITS = la_units()
    out = open(os.path.join(OUT, 'c1_report.txt'), 'w')
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()
    T = types_by_group(UNITS)
    allg = collections.Counter(w for g in T for w in T[g])
    say('types per group', {g: len(v) for g, v in T.items()}, '; types at >=2 groups:', sum(1 for c in allg.values() if c >= 2))
    real = {}
    for ml in (2, 3):
        S, P = stats_for(UNITS, None, ml); real[ml] = S
        say(f'REAL minlen{ml}', S)
    # ---- nulls
    for ml in (2, 3):
        for kind in ('N1', 'N1s', 'N2'):
            name = f'{kind}_m{ml}'
            res = load(name)
            if res is None:
                with Pool(2) as pool:
                    if kind == 'N2': res = pool.map(job_markov, [(s, ml) for s in range(200)])
                    else: res = pool.map(job_shuffle, [(s, kind == 'N1s', ml) for s in range(500)])
                ckpt(name, res)
            line = []
            for k in KEYS:
                v = [r[k] for r in res]
                p = sum(1 for x in v if x >= real[ml][k]) / len(v)
                line.append(f'{k}: real {real[ml][k]} null mean {sum(v)/len(v):.2f} 95% {pct(v,0.95)} P(>=) {p:.3f}')
            say(f'{name} (n={len(res)}):', ' | '.join(line))
    # ---- per-rule family-wise: real rules vs N1 max distribution
    for ml in (2, 3):
        S, P = stats_for(UNITS, None, ml)
        R = rule_support(P, False)
        n1 = [r['any_max'] for r in load(f'N1_m{ml}')]
        say(f'top rules minlen{ml} (family-wise P = share of N1 permutations whose best rule >= k):')
        for r, v in sorted(R.items(), key=lambda kv: -kv[1][0])[:12]:
            fw = sum(1 for x in n1 if x >= v[0]) / len(n1)
            say('  ', r, 'k', v[0], 'FWER-P', f'{fw:.3f}', [('-'.join(a), '-'.join(b)) for a, b in v[2][:6]])
    # ---- controls
    rnd = random.Random(13)
    res = load('controls')
    if res is None:
        res = {}
        # (a) planted in LA Khania, three rules on mid-frequency signs, rate 1.0 and 0.5
        sc = collections.Counter(s for u in UNITS if u[1] == 'KH' for w in u[3] for s in w)
        cand = [s for s, c in sc.items() if 4 <= c <= 15 and not s.startswith('*')]
        for rate in (1.0, 0.5):
            for rep in range(3):
                xs = rnd.sample(cand, 3); ys = rnd.sample([s for s in sc if s not in xs and not s.startswith('*')], 3)
                rules = list(zip(xs, ys))
                U2 = plant(UNITS, 'KH', rules, rate, rnd)
                res[f'LA_plant_KH_r{rate}_{rep}'] = detect(U2, rules, 100, rnd)
                say('LA planted', rate, rep, res[f'LA_plant_KH_r{rate}_{rep}'])
        # (b) Linear B: real sites subsampled to LA type counts (KN~HT, PY~KH, TH~ZA, other~OTH)
        LBU = lb_units()
        tgt = {'KN': 439, 'PY': 87, 'TH': 112}
        def subsample(seed):
            r = random.Random(seed); outu = []
            for site, n in tgt.items():
                docs = [u for u in LBU if u[1] == site]; r.shuffle(docs); seen = set()
                for u in docs:
                    if len(seen) >= n: break
                    outu.append(u); seen.update(u[3])
            docs = [u for u in LBU if u[1] not in tgt]; r.shuffle(docs); seen = set()
            for u in docs:
                if len(seen) >= 249: break
                outu.append((u[0], 'OTH', u[2], u[3])); seen.update(u[3])
            return outu
        for rep in range(3):
            LBs = subsample(100 + rep)
            res[f'LB_real_{rep}'] = detect(LBs, [], 100, rnd)
            say('LB real (unplanted) LA-sized', rep, res[f'LB_real_{rep}'])
            sc = collections.Counter(s for u in LBs if u[1] == 'PY' for w in u[3] for s in w)
            cand = [s for s, c in sc.items() if 4 <= c <= 15]
            for rate in (1.0, 0.5):
                xs = rnd.sample(cand, 3); ys = rnd.sample([s for s in sc if s not in xs], 3)
                rules = list(zip(xs, ys))
                U2 = plant(LBs, 'PY', rules, rate, rnd)
                res[f'LB_plant_r{rate}_{rep}'] = detect(U2, rules, 100, rnd)
                say('LB planted', rate, rep, res[f'LB_plant_r{rate}_{rep}'])
        # (c) LB full KN vs PY, unplanted
        LBk = [u for u in LBU if u[1] in ('KN', 'PY')]
        res['LB_full_KN_PY'] = detect(LBk, [], 30, rnd)
        say('LB full KN vs PY', res['LB_full_KN_PY'])
        # (d) LA random split into two pseudo-sites
        for rep in range(3):
            labs = [rnd.choice(('S1', 'S2')) for _ in UNITS]
            U2 = [(u[0], l, u[2], u[3]) for u, l in zip(UNITS, labs)]
            res[f'LA_split_{rep}'] = detect(U2, [], 100, rnd)
            say('LA random split', rep, res[f'LA_split_{rep}'])
        ckpt('controls', res)
    out.close()

if __name__ == '__main__':
    main()
