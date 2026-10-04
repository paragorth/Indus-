"""pe32 cycle 2: FROZEN PREDICTION OF NEW CLASS MEMBERS.  Affinity of every final sign to C14 from the cycle-1
features (families that were coherent against the selection-matched null N2: pos, size, hdr, next, herd, tabs;
office excluded so the list owes nothing to pe15).  Candidates = signs never seen before an M288 entry in the
pairs pe27/pe28 could use.  The top 20 are frozen and hashed BEFORE the held-out M288 entries are scored.
Held-out = M288 entries pe27/pe28 never used (a line of the pair unclean or the predecessor not a clean count).
Outcome 'std' = y = 60 x (x known) or, x unknown, the amount ends in 2(N39B) 1(N24) (y mod 120 = 60: the
half-N01 tail an odd count leaves under the standard allotment); 'tail' is that ending alone.
Control: 300 seeds of 14 random signs drawn frequency-matched from the other M288 predecessors (N2 pool) go
through the same pipeline; their frozen top-20 lists are scored on the same held-out entries."""
import os, sys, json, time
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe32_common import *  # noqa

FAM2 = ['pos', 'size', 'hdr', 'next', 'herd', 'tabs']
K = 20
t0 = time.time()


def tail(e):
    return e['y'] % 120 == 60


def outcome(e):
    return is_std(e) if e['x'] else tail(e)


def affinity(M, ix, seed, univ):
    sidx = [ix[s] for s in seed]
    Z = []
    for f in FAM2:
        X = M[f]
        a = np.nanmean(X[:, sidx], 1)
        a = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-12)
        Z.append(np.nan_to_num(a))
    return np.mean(Z, 0)


def topk(aff, univ, cand):
    order = np.argsort(-aff)
    return [univ[i] for i in order if univ[i] in cand][:K]


def outcome2(e):
    return is_std(e, mult=(1, 2)) if e['x'] else (tail(e) or e['y'] in (60, 120))


def score(lst, H):
    s = set(lst)
    ev = [e for e in H if e['pfin'] in s]
    return (len(ev), sum(outcome(e) for e in ev), sum(tail(e) for e in ev), sum(outcome2(e) for e in ev),
            sum(1 for e in ev if e['bare']), sum(outcome2(e) for e in ev if e['bare']))


if __name__ == '__main__':
    T = table()
    E = m288_events(T)
    excl = {(e['tid'], e['line'] - 1) for e in E}
    F = sign_features(T, exclude=excl)
    univ = sorted(s for s, n in F['fin_n'].items() if n >= 3 and s not in ('x', '-'))
    ix = {s: i for i, s in enumerate(univ)}
    M = sim_matrix(F, univ)
    seenpred = {e['pfin'] for e in E if e['seen']}
    cand = set(univ) - seenpred - set(C14)
    aff = affinity(M, ix, C14, univ)
    P = topk(aff, univ, cand)
    frozen = {'list': P, 'rule': 'M288 after a line ending in one of these signs holds 60 N39C per counted unit '
              '(2(N39B) 1(N24) per unit; amount ends in 2(N39B) 1(N24) when the count is odd)'}
    old = 'f742591cac572a50'
    h = sha(frozen)
    json.dump({'frozen': frozen, 'sha': h, 'time': time.strftime('%H:%M:%S')},
              open(os.path.join(CK, 'cycle2_frozen.json'), 'w'), indent=1)
    print('FROZEN', h, P, 'same as first freeze' if h == old else 'CHANGED', flush=True)
    # ---- only now: held-out scoring
    H = [e for e in E if not e['seen']]
    base = [e for e in H if e['pfin'] not in C14 and e['pfin'] not in set(P)]
    res = {'frozen': frozen, 'sha': h, 'n_heldout': len(H)}
    res['P'] = score(P, H)
    res['C14_heldout'] = score(C14, H)
    res['other'] = score(sorted({e['pfin'] for e in base}), base)
    res['score_fields'] = 'events, std(60x or tail), tail, std-or-double, bare events, bare std-or-double'
    res['seen_C14'] = score(C14, [e for e in E if e['seen']])
    res['seen_other'] = (lambda b: (len(b), sum(outcome(e) for e in b), sum(tail(e) for e in b)))(
        [e for e in E if e['seen'] and e['pfin'] not in C14])
    res['seen_other'] = score(sorted({e['pfin'] for e in E if e['seen'] and e['pfin'] not in C14}),
                              [e for e in E if e['seen'] and e['pfin'] not in C14])
    res['P_events'] = [(e['tid'], e['praw'], e['raw'], outcome(e)) for e in H if e['pfin'] in set(P)]
    res['C14_events'] = [(e['tid'], e['praw'], e['raw'], outcome(e)) for e in H if e['pfin'] in C14]
    # ---- control seeds
    rng = np.random.default_rng(232)
    pool2 = sorted({e['pfin'] for e in E} - set(C14))
    pool2 = [s for s in pool2 if s in ix]
    nul = []
    for r in range(int(os.environ.get('NS', 300))):
        seed = freq_matched(rng, C14, pool2, F['fin_n'])
        cand_r = set(univ) - seenpred - set(seed) - set(C14)
        Pr = topk(affinity(M, ix, seed, univ), univ, cand_r)
        nul.append(score(Pr, H))
    nul = np.array(nul)
    n, k = res['P'][0], res['P'][3]
    res['null'] = {'events_mean': float(nul[:, 0].mean()), 'hits2_mean': float(nul[:, 3].mean()),
                   'p_hits2': float((1 + (nul[:, 3] >= k).sum()) / (1 + len(nul))),
                   'rate2_mean': float(np.nanmean(np.where(nul[:, 0] > 0, nul[:, 3] / np.maximum(nul[:, 0], 1), np.nan)))}
    # predicted signs' standing in the whole corpus (any context): how often they end count lines etc.
    res['P_finals'] = {s: F['fin_n'][s] for s in P}
    res['P_overlap_C14_offices'] = {s: OFF_OF.get(s, '-') for s in P}
    print(json.dumps({k: v for k, v in res.items() if k not in ('P_events', 'C14_events')}, default=str))
    for x in res['P_events'] + [('--C14--',)] + res['C14_events']:
        print(x)
    json.dump(res, open(os.path.join(CK, 'cycle2.json'), 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))
