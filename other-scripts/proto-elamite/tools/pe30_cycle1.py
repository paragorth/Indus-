"""pe30 cycle 1: controls for the exclusion-rule search.

(a) Ur III balanced accounts (known: the debit summary leaves out the credit
    section). Blind search over all rule families; held-out gain over 100 splits;
    nulls: features re-dealt among entries (20), totals replaced by random numbers
    of the same size (20).
(b) Planted exclusion rules on the PE tablets (tier 1+2) that close at baseline:
    totals recomputed leaving out entries with a mid-frequency sign, the last
    entry, the largest entry, bare-number entries, or adding +1 / +10 N01.
    Recovery = planted rule (or an equivalent with identical closure) is the
    full-data best, and held-out gain > 0.
Usage: python3 pe30_cycle1.py
"""
import time
from pe30_common import *
from pe30_ur3 import build_balanced


def ur_shuffle(C, rng):
    allf = [fl for c in C for fl in c['hyps'][0][0]['F']]
    perm = [allf[i] for i in rng.permutation(len(allf))]
    it = iter(perm); out = []
    for c in C:
        p = c['hyps'][0][0]
        out.append(dict(c, hyps=[[dict(p, F=[next(it) for _ in p['F']])]]))
    return out


def run(C, nsplit=100, seed=0, pairs=True):
    fl, _ = feature_list(C)
    sc = Scorer(C, fl, pairs=pairs)
    return sc, summary(sc, C, np.random.default_rng(seed), nsplit)


def null_stats(C, kind, n, seed=100):
    rng = np.random.default_rng(seed)
    g, h = [], []
    for i in range(n):
        D = {'shuf': ur_shuffle, 'rand': rand_totals, 'lab': shuffle_labels,
             'ent': shuffle_entries, 'ord': shuffle_order}[kind](C, rng)
        _, r = run(D, nsplit=40, seed=seed + i)
        g.append(r['gain_full']); h.append(r['heldout_gain'])
    return {'gain_full': g, 'heldout': h, 'gain_mean': float(np.mean(g)), 'held_mean': float(np.mean(h)),
            'held_q95': float(np.quantile(h, 0.95))}


def pv(x, arr):
    arr = np.array(arr)
    return float((1 + (arr >= x).sum()) / (1 + len(arr)))


if __name__ == '__main__':
    t0 = time.time()
    out = {}
    rows = []
    # (a) Ur III
    U = build_balanced()
    sc, r = run(U)
    nl = {k: null_stats(U, k, 20) for k in ('shuf', 'rand')}
    r['p'] = {k: {'p_full': pv(r['gain_full'], v['gain_full']), 'p_held': pv(r['heldout_gain'], v['heldout']),
                  'null_held_mean': v['held_mean'], 'null_held_q95': v['held_q95'], 'null_full_mean': v['gain_mean']}
              for k, v in nl.items()}
    out['ur3'] = r
    print('ur3', json.dumps(r, default=str)[:1500], flush=True)
    # (b) plants on PE tier 1+2
    C = load_cases(2)
    fl, cnt = feature_list(C)
    base_sc = Scorer(C, fl, pairs=False)
    closing = base_sc.M[0]
    # mid-frequency signs among closing tablets
    ccount = collections.Counter()
    for k, c in enumerate(C):
        if closing[k]:
            ccount.update({f for h in c['hyps'] for p in h for x in p['F'] for f in x if f.startswith('B_')})
    signs = [f for f, n in ccount.most_common() if 5 <= n <= 9][:3]
    plants = [('EX(%s)' % s, (lambda s: lambda x: s in x)(s), 0) for s in signs] + [
        ('EX(P_last)', lambda x: 'P_last' in x, 0),
        ('EX(Z_max)', lambda x: 'Z_max' in x, 0),
        ('EX(N0)', lambda x: 'N0' in x, 0),
        ('C(+1)', lambda x: False, 1),
        ('C(+10)', lambda x: False, 10)]
    out['plant'] = {}
    for name, pred, const in plants:
        Cp, n = plant(C, pred, const)
        sc, r = run(Cp, nsplit=60)
        ix = sc.names.index(name)
        S = sc.M.sum(1)
        same = [sc.names[i] for i in np.where((sc.M == sc.M[ix]).all(1))[0]][:6]
        r.update({'planted': name, 'n_planted_tablets': n, 'planted_score': int(S[ix]),
                  'planted_rank': int((S > S[ix]).sum()) + 1, 'equivalent_rules': same,
                  'recovered_full': bool(S[ix] == S.max())})
        out['plant'][name] = r
        print('plant', name, n, r['base'], r['best'], r['planted_score'], r['planted_rank'],
              r['best_rules'][:3], r['heldout_gain'], r['top_split_picks'][:3], flush=True)
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(DATA, 'pe30_cycle1.json'), 'w'), indent=1, default=str)
