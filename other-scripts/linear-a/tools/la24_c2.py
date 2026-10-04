#!/usr/bin/env python3
"""LA-24 cycle 2: do apportionment fits constrain the fraction values?

If clerks divided totals by shares and let fractions absorb remainders, then under the right
fraction value set V the fraction-bearing lists should compress better as apportionments.
Score of V = summed positive MDL gain (and count of lists with gain > 0) over the Linear A lists
that carry fraction letters. The baseline codes letters and is V-independent; the scale term of
the baseline uses CONV for every V so that only the model side changes.
Candidates: CONV (lineara.xyz values), LA1BIN (attack-1 larger-first binary example), and
NV random value sets (each letter drawn from a 22-value pool). Null for 'V matters': the rank of
CONV and LA1BIN among the random sets.
Power control: 25 planted EXACT/ROUND apportionments written with CONV letters (J, E, JE)
replace 25 fraction lists; CONV must then rank at the top.
Letter pinning: for each letter, value distribution in the top 5 % of random sets vs all sets
(permutation P from the random sets themselves).
"""
import json, os, random, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la24_common import *

NV = int(os.environ.get('NV', 600))
G = {}


def score_V(args):
    tag, k, V = args
    lists = G[tag]['lists']; B = G[tag]['B']
    nv = lambda a: val(a, V)
    gains = []; rules = Counter()
    for L in lists:
        toks = L['amts']
        x = np.array([nv(t) for t in toks]); bc = B.costs(toks)
        mb, info = model_best(x, bc, (1.0, 0.5, 0.25))
        g = bc.sum() - mb; gains.append(g)
        if g > 0 and info: rules[info['rule']] += 1
    gains = np.array(gains)
    return tag, k, float(gains[gains > 0].sum()), int((gains > 0).sum()), dict(rules), gains.tolist()


def main():
    la = la_lists()
    fl = [l for l in la if any(a[1] for a in l['amts'])]
    B = Baseline([l['amts'] for l in la], la_numval(CONV))
    G['LA'] = {'lists': fl, 'B': B}
    rng = random.Random(2424)
    # planted corpus: 25 fraction lists replaced by apportionments with fractional unit share
    pl = [dict(l) for l in fl]
    idx = rng.sample(range(len(fl)), min(25, len(fl)))
    for i in idx:
        n = min(max(len(fl[i]['amts']), 3), 10)
        while True:
            w = [rng.choice([1, 2, 3, 4]) for _ in range(n)]
            u = rng.choice([1, 2, 3, 5]) + rng.choice([0.25, 0.5, 0.75])
            vals = [u * x for x in w]
            a = [to_amount(v) for v in vals]
            if all(z is not None for z in a) and any(z[1] for z in a): break
        pl[i] = {'id': 'PL', 'amts': a}
    G['PL'] = {'lists': pl, 'B': Baseline([l['amts'] for l in la if l not in fl] + [l['amts'] for l in pl], la_numval(CONV))}
    Vs = [('CONV', CONV), ('LA1BIN', LA1BIN)] + [('R%d' % i, random_V(rng)) for i in range(NV)]
    jobs = [(tag, k, V) for tag in ('LA', 'PL') for k, (nm, V) in enumerate(Vs)]
    res = {'LA': {}, 'PL': {}}
    t0 = time.time()
    with Pool(2) as pool:
        for tag, k, s, c, rules, gains in pool.imap_unordered(score_V, jobs, chunksize=8):
            res[tag][k] = (s, c, rules, gains)
    out = {'n_frac_lists': len(fl), 'NV': NV, 'secs': time.time() - t0}
    for tag in ('LA', 'PL'):
        S = np.array([res[tag][k][0] for k in range(len(Vs))])
        Cn = np.array([res[tag][k][1] for k in range(len(Vs))])
        rnd = S[2:]
        o = {}
        for k, nm in ((0, 'CONV'), (1, 'LA1BIN')):
            o[nm] = {'sum': float(S[k]), 'cnt': int(Cn[k]), 'rules': res[tag][k][2],
                     'P_sum': float(((rnd >= S[k]).sum() + 1) / (len(rnd) + 1)),
                     'P_cnt': float(((Cn[2:] >= Cn[k]).sum() + 1) / (len(rnd) + 1))}
        o['random_sum_mean'] = float(rnd.mean()); o['random_sum_max'] = float(rnd.max())
        o['random_cnt_mean'] = float(Cn[2:].mean())
        # letter pinning on the random sets
        top = np.argsort(-rnd)[:max(1, len(rnd) // 20)] + 2
        pin = {}
        for let in ('J', 'E', 'JE', 'D', 'B', 'K', 'F', 'A', 'H', 'L2'):
            allv = [float(Vs[k][1][let]) for k in range(2, len(Vs))]
            tv = [float(Vs[k][1][let]) for k in top]
            # statistic: share of top sets whose value equals the modal top value
            mc = Counter(tv).most_common(1)[0]
            base_rate = sum(1 for v in allv if v == mc[0]) / len(allv)
            # binomial-ish permutation: random subsets of the same size
            prng = np.random.default_rng(1)
            sims = [Counter([allv[j] for j in prng.choice(len(allv), len(tv), replace=False)]).most_common(1)[0][1]
                    for _ in range(2000)]
            pin[let] = {'mode': mc[0], 'n_top': mc[1], 'of': len(tv), 'base_rate': base_rate,
                        'P_max_mode': float((np.sum(np.array(sims) >= mc[1]) + 1) / 2001),
                        'top_mean': float(np.mean(tv)), 'all_mean': float(np.mean(allv))}
        o['pin'] = pin
        # which lists drive CONV
        gl = res[tag][0][3]
        o['conv_pos_lists'] = [(G[tag]['lists'][i].get('id'), round(gl[i], 2)) for i in range(len(gl)) if gl[i] > 0]
        gl1 = res[tag][1][3]
        o['la1bin_pos_lists'] = [(G[tag]['lists'][i].get('id'), round(gl1[i], 2)) for i in range(len(gl1)) if gl1[i] > 0]
        # per-list: share of random sets giving gain > 0 (how V-sensitive each list is)
        out[tag] = o
    out['top_random_sets'] = [{let: str(v) for let, v in Vs[k][1].items()} | {'sum': float(res['LA'][k][0])}
                              for k in (np.argsort(-np.array([res['LA'][k][0] for k in range(2, len(Vs))]))[:5] + 2)]
    json.dump(out, open(os.path.join(CK, 'c2.json'), 'w'), indent=1)
    print(json.dumps({t: {k: v for k, v in out[t].items() if k in ('CONV', 'LA1BIN', 'random_sum_mean', 'random_sum_max', 'random_cnt_mean')} for t in ('LA', 'PL')}))


if __name__ == '__main__':
    main()
