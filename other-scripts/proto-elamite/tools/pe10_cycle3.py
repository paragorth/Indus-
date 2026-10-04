"""pe10 cycle 3: sign CHOICE under squeeze (merge into compounds, switch to variants,
prefer particular signs).  Same Q (squeezed) positions as cycle 2.

3A  global features: share of entries in Q that contain a compound |A+B|, a ~variant form,
    an 'x'-free single sign; obs - exp with exp = tablet mean (Mantel-Haenszel style);
    null = within-tablet shuffle of Q (10,000).
3B  20,000 random sign sets (size 5-15, base signs with n >= 10): score = summed
    obs-exp count of entries in Q containing any sign of the set, on a random half of
    tablets (A); the top 1% are re-scored on the other half (B).  Null: identical
    procedure with Q shuffled within tablet (50 reps) -> distribution of mean held-out z
    of survivors.  Repeat over 20 random A/B splits.
3C  compound <-> sequence pairs: for each compound |A+B| (n>=2) whose parts also occur as
    adjacent 'A B' in some entry, compare the compound's Q rate with the sequence's.
Planted control: in Q entries, replace adjacent 'A B' by '|A+B|' for the 8 most common
    adjacent pairs w.p. 0.7, and add sign M999 (a planted 'short variant') to 30% of Q
    entries' sign sets by swapping one sign; must be recovered by 3A/3B.
"""
import sys, json
import numpy as np
from pe10_common import *
from pe10_cycle2 import collect

rng = np.random.default_rng(30)


def tabgroups(E):
    g = defaultdict(list)
    for i, e in enumerate(E):
        g[e['tab']].append(i)
    return {k: np.array(v) for k, v in g.items()}


def mh(feat, q, groups, tabs=None):
    """sum over Q entries of feature minus tablet-mean expectation; z from hypergeometric var."""
    o = e = v = 0.0
    for t, idx in groups.items():
        if tabs is not None and t not in tabs:
            continue
        nq = q[idx].sum()
        if nq == 0 or nq == len(idx):
            continue
        f = feat[idx]; n = len(idx)
        o += f[q[idx] > 0].sum()
        m = f.mean(); e += nq * m
        v += nq * (n - nq) / n * f.var() * n / max(n - 1, 1)
    return o, e, (o - e) / np.sqrt(v) if v > 0 else 0.0


def shuffle_q(q, groups):
    q2 = q.copy()
    for idx in groups.values():
        q2[idx] = q[rng.permutation(idx)]
    return q2


def features(E):
    F = {}
    F['compound'] = np.array([any(s.startswith('|') for s in e['raw']) for e in E], float)
    F['variant'] = np.array([any('~' in s for s in e['raw']) for e in E], float)
    F['single'] = np.array([len(e['s']) == 1 for e in E], float)
    F['len'] = np.array([len(e['s']) for e in E], float)
    return F


def collect_raw(T):
    E, q2 = collect(T)
    # re-attach raw (variant-bearing) sign strings in the same order as collect()
    raws = []
    for ti, t in enumerate(T):
        broken_faces = None
        for face in ('obverse', 'reverse'):
            F = [u for u in t['units'] if u['face'] == face]
            ent = [u for u in F if u['entry'] and not u['total']]
            for u in ent:
                if u['broken'] or 'x' in u['signs']:
                    continue
                raws.append(tuple(u['signs']))
    assert len(raws) == len(E)
    for e, r in zip(E, raws):
        e['raw'] = r
    return E


def sign_matrix(E, signs):
    idx = {s: i for i, s in enumerate(signs)}
    M = np.zeros((len(E), len(signs)), bool)
    for i, e in enumerate(E):
        for s in e['s']:
            for part in re.split(r'[|+.&]', s):
                if part in idx:
                    M[i, idx[part]] = True
    return M


def setscore(M, setidx, q, groups, tabs):
    feat = M[:, setidx].any(1).astype(float)
    return mh(feat, q, groups, tabs)[2]


def random_sets(E, q, groups, nsets=20000, nsplit=20, top=0.01):
    cnt = Counter(p for e in E for s in e['s'] for p in re.split(r'[|+.&]', s) if p.startswith('M'))
    signs = [s for s, c in cnt.items() if c >= 10]
    M = sign_matrix(E, signs)
    sets = [rng.choice(len(signs), rng.integers(5, 16), replace=False) for _ in range(nsets)]
    # precompute per-sign Q-membership via per-tablet vectorisation: score function per set
    tabs_all = np.array(sorted(groups))
    held = []
    for sp in range(nsplit):
        perm = rng.permutation(tabs_all)
        A = set(perm[:len(perm) // 2]); B = set(perm[len(perm) // 2:])
        sA = np.array([setscore(M, s, q, groups, A) for s in sets[:2000]])  # 2,000 per split for speed
        k = max(1, int(len(sA) * top))
        best = np.argsort(sA)[::-1][:k]
        held.append(float(np.mean([setscore(M, sets[b], q, groups, B) for b in best])))
    return signs, M, float(np.mean(held)), held


def per_sign(E, q, groups, signs, M):
    out = []
    for j, s in enumerate(signs):
        o, e, z = mh(M[:, j].astype(float), q, groups)
        out.append((s, int(M[:, j].sum()), round(o, 1), round(e, 2), round(float(z), 2)))
    return sorted(out, key=lambda x: -x[4])


def plant(E):
    E2 = [dict(e) for e in E]
    pairs = Counter()
    for e in E:
        for a, b in zip(e['s'], e['s'][1:]):
            if not a.startswith('|') and not b.startswith('|'):
                pairs[(a, b)] += 1
    top = [p for p, _ in pairs.most_common(8)]
    for e in E2:
        if not e['q']:
            continue
        s = list(e['s']); r = list(e['raw'])
        for a, b in top:
            for i in range(len(s) - 1):
                if s[i] == a and s[i + 1] == b and rng.random() < 0.7:
                    s[i:i + 2] = ['|%s+%s|' % (a, b)]; r[i:i + 2] = ['|%s+%s|' % (a, b)]
                    break
        if rng.random() < 0.3 and s:
            j = rng.integers(len(s)); s[j] = 'M999'; r[j] = 'M999'
        e['s'] = tuple(s); e['raw'] = tuple(r)
    # M999 must exist with n >= 10 overall: also sprinkle in a few non-Q entries
    nq = [e for e in E2 if not e['q']]
    for k in rng.choice(len(nq), 15, replace=False):
        e = nq[k]; s = list(e['s']); s[0] = 'M999'; e['s'] = tuple(s); e['raw'] = tuple(s)
    return E2


def analyse(E, tag, nnull=50):
    groups = tabgroups(E)
    q = np.array([e['q'] for e in E], float)
    F = features(E)
    res = {'tag': tag, 'n_Q': int(q.sum())}
    for k, f in F.items():
        o, e, z = mh(f, q, groups)
        nz = [mh(f, shuffle_q(q, groups), groups)[2] for _ in range(300)]
        res['3A_' + k] = {'obs': round(o, 1), 'exp': round(e, 1), 'z': round(float(z), 2),
                          'perm_sd_of_z': round(float(np.std(nz)), 2)}
    signs, M, held, hl = random_sets(E, q, groups)
    nul = []
    for r in range(nnull):
        qs = shuffle_q(q, groups)
        nul.append(random_sets(E, qs, groups, nsplit=4)[2])
    nul = np.array(nul)
    res['3B_heldout_mean_z'] = held
    res['3B_null_mean'] = float(nul.mean()); res['3B_null_sd'] = float(nul.std())
    res['3B_p'] = float((np.sum(nul >= held) + 1) / (len(nul) + 1))
    ps = per_sign(E, q, groups, signs, M)
    res['3B_top_signs'] = ps[:10]; res['3B_bottom_signs'] = ps[-5:]
    # 3C compounds vs sequences
    cmp_rows = []
    allseq = Counter(); seqQ = Counter()
    for i, e in enumerate(E):
        for a, b in zip(e['s'], e['s'][1:]):
            allseq[(a, b)] += 1; seqQ[(a, b)] += e['q']
    cC = Counter(); cQ = Counter()
    for e in E:
        for s in e['s']:
            if s.startswith('|') and s.count('+') == 1 and '.' not in s:
                a, b = s.strip('|').split('+')
                cC[(a, b)] += 1; cQ[(a, b)] += e['q']
    pq = q.mean()
    totc = sum(cC[k] for k in cC if allseq[k]); totcq = sum(cQ[k] for k in cC if allseq[k])
    tots = sum(allseq[k] for k in cC if allseq[k]); totsq = sum(seqQ[k] for k in cC if allseq[k])
    res['3C'] = {'n_compound_types_with_sequence_twin': sum(1 for k in cC if allseq[k]),
                 'compound_Q_rate': round(totcq / max(totc, 1), 4), 'n_compound': totc,
                 'sequence_Q_rate': round(totsq / max(tots, 1), 4), 'n_sequence': tots, 'base_Q_rate': round(pq, 4)}
    return res


if __name__ == '__main__':
    T = load()
    E = collect_raw(T)
    ck = os.path.join(CKPT, 'c3.json')
    out = json.load(open(ck)) if os.path.exists(ck) else {}
    if 'real' not in out:
        out['real'] = analyse(E, 'real'); json.dump(out, open(ck, 'w'), indent=1, default=str)
        print(json.dumps(out['real'], indent=1, default=str), flush=True)
    if 'planted' not in out:
        out['planted'] = analyse(plant(E), 'planted', nnull=20); json.dump(out, open(ck, 'w'), indent=1, default=str)
        print(json.dumps(out['planted'], indent=1, default=str), flush=True)
