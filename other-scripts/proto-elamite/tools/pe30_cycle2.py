"""pe30 cycle 2: the real search on PE, with search-corrected nulls.

Tier 1 (49 clean) and tier 1+2 (91, damaged but every numeral readable).
Statistics: full-data best gain over all-entries (search-inflated) and held-out
gain (fit on half, score the other half, 100 splits).
Nulls (20 each, 40 splits each): random totals of the same size; random sign
labels (sign-derived features re-dealt among all entries); entries shuffled
among tablets (values + features, count per tablet kept); entry order shuffled
within tablets (position null). Also: fit on tier 1, re-test on tier 2 (an
independent held-out set). Per-family maxima are reported so that the search
over 100k rules is corrected by the same search on the nulls.
Usage: python3 pe30_cycle2.py [tier]
"""
import sys, time
from multiprocessing import Pool
from pe30_common import *
from pe30_cycle1 import run, pv

KINDS = ('rand', 'lab', 'ent', 'ord')


def fam_best(sc):
    S = sc.M.sum(1)
    return {f: int(S[sc.fam == f].max() - S[0]) for f in ('EX1', 'KEEP', 'C', 'EXC', 'EX2')}


def job(args):
    tier, kind, i = args
    C = load_cases(tier)
    rng = np.random.default_rng(500 + 31 * i + KINDS.index(kind))
    D = {'rand': rand_totals, 'lab': shuffle_labels, 'ent': shuffle_entries, 'ord': shuffle_order}[kind](C, rng)
    sc, r = run(D, nsplit=40, seed=i)
    return kind, r['gain_full'], r['heldout_gain'], fam_best(sc), r['base']


if __name__ == '__main__':
    tier = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    t0 = time.time()
    C = load_cases(tier)
    sc, r = run(C, nsplit=100)
    r['fam_best'] = fam_best(sc)
    print(json.dumps(r, default=str)[:2500], flush=True)
    with Pool(2) as pool:
        res = pool.map(job, [(tier, k, i) for k in KINDS for i in range(20)])
    nl = {}
    for k in KINDS:
        rr = [x for x in res if x[0] == k]
        g = [x[1] for x in rr]; h = [x[2] for x in rr]
        nl[k] = {'gain_full': g, 'heldout': h, 'base': [x[4] for x in rr],
                 'p_full': pv(r['gain_full'], g), 'p_held': pv(r['heldout_gain'], h),
                 'null_full_mean': float(np.mean(g)), 'null_held_mean': float(np.mean(h)),
                 'null_held_q95': float(np.quantile(h, 0.95)),
                 'fam_best_mean': {f: float(np.mean([x[3][f] for x in rr])) for f in r['fam_best']}}
        print(k, {a: b for a, b in nl[k].items() if a not in ('gain_full', 'heldout', 'base')}, flush=True)
    out = {'tier': tier, 'real': r, 'nulls': nl}
    if tier == 2:
        # fit on tier 1 only, re-test on the 42 damaged tablets
        t1 = np.array([c['tier'] == 1 for c in C]); t2 = ~t1
        S1 = sc.M[:, t1].sum(1); S2 = sc.M[:, t2].sum(1)
        best = S1.max(); cand = np.where(S1 == best)[0]
        cand = cand[sc.cx[cand] == sc.cx[cand].min()]
        out['t1_to_t2'] = {'t1_base': int(S1[0]), 't1_best': int(best), 'fits': [sc.names[i] for i in cand[:10]],
                           'n_fits': int(len(cand)), 't2_base': int(S2[0]),
                           't2_gain_mean': float(S2[cand].mean() - S2[0]),
                           't2_gain_each': [int(S2[i] - S2[0]) for i in cand[:10]]}
        print('t1->t2', out['t1_to_t2'], flush=True)
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(DATA, 'pe30_cycle2_t%d.json' % tier), 'w'), indent=1, default=str)
