"""LA-35 cycle 4: kill test for the cycle-3 lead (in LA the extended form W+X carries MORE '1'
entries than the bare form W; in LB the opposite, as a plural would). Confound: longer words
(personal names on HT lists of '1') go with 1 anyway. Control: fake extension pairs (W, V) where
V is a random word with len(W)+k signs that does NOT start with W, matched to the real pairs'
k and to V's site profile; 5,000 fake pair sets. Also leave-one-pair-out, per-site, and a
length-only logistic baseline. Same for LB (full and LA-sized)."""
import json, sys, random
import numpy as np
from la35_common import *
from la35_c1 import lb_sub
from la35_c3 import pairs_of


def one_rate(E):
    by = defaultdict(list)
    for e in E: by[e['word']].append(e['v'] == 1)
    return {w: (np.mean(v), len(v)) for w, v in by.items()}


def d1(P, R):
    return float(np.mean([R[b][0] - R[a][0] for a, b, x in P])) if P else 0.0


def fake_sets(E, P, R, n, rng, mode):
    """mode 'len': V any word of the same length as the real long form, not an extension of W.
    mode 'lenone': additionally V matched on the real long form's site (first site seen)."""
    words = list(R)
    bylen = defaultdict(list)
    site = {}
    for e in E: site.setdefault(e['word'], e['site'])
    for w in words: bylen[len(w)].append(w)
    bylensite = defaultdict(list)
    for w in words: bylensite[(len(w), site[w])].append(w)
    pools = []
    for a, b, x in P:
        pool = bylensite[(len(b), site[b])] if mode == 'lensite' else bylen[len(b)]
        pool = [v for v in pool if v[:len(a)] != a] or [v for v in bylen[len(b)] if v[:len(a)] != a]
        pools.append((np.array([R[v][0] for v in pool]), R[a][0]))
    out = np.zeros(n)
    for pv, ra in pools:
        out += pv[rng.integers(len(pv), size=n)] - ra
    return out / len(pools)


def analyse(name, E, nfake, seed, detail=False):
    rng = np.random.default_rng(seed)
    P = pairs_of(E, 'ext'); R = one_rate(E)
    real = d1(P, R)
    res = dict(name=name, npairs=len(P), real=round(real, 4))
    for mode in ('len', 'lensite'):
        F = fake_sets(E, P, R, nfake, rng, mode)
        res[mode] = dict(mean=round(float(F.mean()), 4), p_hi=round(float(((F >= real).sum() + 1) / (nfake + 1)), 4),
                         p_lo=round(float(((F <= real).sum() + 1) / (nfake + 1)), 4))
    # length baseline: corpus-wide share of '1' by word length
    L = defaultdict(list)
    for e in E: L[len(e['word'])].append(e['v'] == 1)
    res['one_by_len'] = {k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(L.items()) if len(v) >= 10}
    # leave-one-pair-out range and sign count
    loo = [d1(P[:i] + P[i + 1:], R) for i in range(len(P))]
    res['loo'] = (round(min(loo), 4), round(max(loo), 4)) if loo else None
    diffs = [R[b][0] - R[a][0] for a, b, x in P]
    res['sign'] = dict(pos=int(sum(d > 0 for d in diffs)), neg=int(sum(d < 0 for d in diffs)), zero=int(sum(d == 0 for d in diffs)))
    # by site of the long form
    site = {}
    for e in E: site.setdefault(e['word'], e['site'])
    bs = defaultdict(list)
    for (a, b, x), d in zip(P, diffs): bs['HT' if site[b] == 'HT' else 'nonHT'].append(d)
    res['by_site'] = {k: (round(float(np.mean(v)), 3), len(v)) for k, v in bs.items()}
    if detail:
        res['pairs'] = sorted([('-'.join(a), '%.2f/%d' % R[a], '-'.join(b), '%.2f/%d' % R[b]) for a, b, x in P],
                              key=lambda t: t[2])
    print(json.dumps(res), flush=True)
    return res


if __name__ == '__main__':
    NF = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    LA = la_entries(); LB = lb_entries()
    out = [analyse('LA', LA, NF, 1, detail=True)]
    out.append(analyse('LA_noHT', [e for e in LA if e['site'] != 'HT'], NF, 2))
    out.append(analyse('LA_HT', [e for e in LA if e['site'] == 'HT'], NF, 3))
    out.append(analyse('LBfull', LB, NF // 5, 4))
    for s in range(6):
        out.append(analyse('LBsub%d' % s, lb_sub(LB, len(LA), s), NF // 5, 10 + s))
    json.dump(out, open(os.path.join(CK, 'c4.json'), 'w'), indent=1)
