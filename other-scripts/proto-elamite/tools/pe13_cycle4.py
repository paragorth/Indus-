"""pe13 cycle 4: the memory is a SEQUENCE?  A learned canonical order of entries.

Cycle 2 found tablets (e.g. P008295) that cycle through a fixed run of entries
(M362+X, M269, M106, M009, M206, M102, M309, ...).  If scribes held a memorised order of
categories (as Ur III herd scribes did for animal classes, or as lexical lists fix a
sequence), the precedence of two signs on a tablet should be the same across tablets.

(A) Precedence consistency: for each tablet, first-occurrence line of each sign; for every
    sign pair on >= 5 tablets (different first lines), p = share of tablets with a before b.
    Consistency = n-weighted mean |2p - 1|.  Null: 50 within-tablet line shuffles.
(B) Held-out prediction: a global rank (mean relative first position, fitted on half A)
    predicts the order of every co-occurring pair on half B tablets: accuracy vs 50 shuffles
    of half B.
(C) Massive random guessing: 4,000 random sign sets (size 3-6, signs on >= 10 half-A tablets);
    statistic on half A = share of tablets (with >= 3 of the set on distinct first lines) whose
    order matches the set's modal order; FWER from the same search on 10 shuffled half A;
    top 30 re-tested on half B (modal order frozen from A) against 200 shuffles.
Controls: Ur III Drehem (known standard order of animal classes), proto-cuneiform lexical
(copied canonical order), Linear B, proto-cuneiform admin; PLANTED on PE skeleton: TOPIC (no
order) and ORDER (topic tablets whose lines are sorted by a hidden global rank of their first
sign, with Gaussian rank noise; on 30% of tablets).
usage: python3 pe13_cycle4.py [workers]
"""
import itertools, json, math, os, random, sys
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe13_common as C  # noqa
from pe13_cycle2 import variant  # noqa

OUT = os.path.join(C.CK, 'c4')
os.makedirs(OUT, exist_ok=True)


def firsts(t):
    f = {}
    for i, l in enumerate(t['lines']):
        for s in l['toks']:
            if s not in f:
                f[s] = i
    return f


def pair_table(tabs):
    P = defaultdict(lambda: [0, 0])
    for t in tabs:
        f = firsts(t)
        S = sorted(f)
        for a, b in itertools.combinations(S, 2):
            if f[a] == f[b]:
                continue
            P[(a, b)][0 if f[a] < f[b] else 1] += 1
    return P


def consistency(tabs, nmin=5):
    P = pair_table(tabs)
    num = den = 0.0
    npairs = 0
    for (a, b), (x, y) in P.items():
        n = x + y
        if n >= nmin:
            num += abs(x - y)
            den += n
            npairs += 1
    return (num / den if den else 0.0), npairs


def global_rank(tabs):
    pos = defaultdict(list)
    for t in tabs:
        f = firsts(t)
        L = max(len(t['lines']) - 1, 1)
        for s, i in f.items():
            pos[s].append(i / L)
    return {s: np.mean(v) for s, v in pos.items() if len(v) >= 3}


def predict_acc(rank, tabs):
    ok = n = 0
    for t in tabs:
        f = firsts(t)
        S = [s for s in f if s in rank]
        for a, b in itertools.combinations(S, 2):
            if f[a] == f[b] or rank[a] == rank[b]:
                continue
            n += 1
            ok += (f[a] < f[b]) == (rank[a] < rank[b])
    return ok / n if n else 0.5, n


def job_ab(args):
    name, tabs, nshuf = args
    fn = os.path.join(OUT, 'ab_' + name + '.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    rng = random.Random(3)
    c, npairs = consistency(tabs)
    cs = [consistency(C.shuffle_lines(tabs, rng))[0] for _ in range(nshuf)]
    ids = sorted(range(len(tabs)))
    A = set(random.Random(11).sample(ids, len(ids) // 2))
    TA = [tabs[i] for i in ids if i in A]
    TB = [tabs[i] for i in ids if i not in A]
    rk = global_rank(TA)
    acc, n = predict_acc(rk, TB)
    accs = [predict_acc(rk, C.shuffle_lines(TB, rng))[0] for _ in range(nshuf)]
    res = {'name': name, 'cons': c, 'npairs': npairs, 'cons_null_mean': float(np.mean(cs)), 'cons_null_max': float(np.max(cs)),
           'cons_z': float((c - np.mean(cs)) / (np.std(cs) + 1e-9)), 'acc': acc, 'n_pred': n,
           'acc_null_mean': float(np.mean(accs)), 'acc_null_max': float(np.max(accs)),
           'acc_z': float((acc - np.mean(accs)) / (np.std(accs) + 1e-9))}
    json.dump(res, open(fn, 'w'))
    print('AB %-12s cons %.3f (null %.3f max %.3f z %.1f, %d pairs) | held-out order acc %.3f (null %.3f max %.3f z %.1f, n %d)' % (
        name, c, res['cons_null_mean'], res['cons_null_max'], res['cons_z'], npairs, acc, res['acc_null_mean'],
        res['acc_null_max'], res['acc_z'], n), flush=True)
    return res


def plant_order(skel, seed, uni, share=0.3, noise=0.15):
    rng = random.Random(seed)
    T = C.gen_planted(skel, 'TOPIC', seed, uni)
    keys = sorted(uni)
    hid = {k: rng.random() for k in keys}
    out = []
    for t in T:
        L = [dict(l) for l in t['lines']]
        if rng.random() < share:
            toks = sorted([l['toks'] for l in L], key=lambda tk: hid[tk[0]] + rng.gauss(0, noise))
            for l, tk in zip(L, toks):
                l['toks'] = tk
        out.append({'id': t['id'], 'lines': L})
    return out


# ------------- random sign-set search -------------
def set_stat(tabs, S, modal=None):
    """share of tablets (>= 3 of S on distinct first lines) whose order of S matches modal."""
    orders = []
    FF = tabs if (tabs and isinstance(tabs[0], dict) and 'lines' not in tabs[0]) else [firsts(t) for t in tabs]
    for f in FF:
        pres = [s for s in S if s in f]
        if len(pres) < 3 or len(set(f[s] for s in pres)) < len(pres):
            continue
        orders.append(tuple(sorted(pres, key=lambda s: f[s])))
    if not orders:
        return 0.0, 0, modal
    if modal is None:
        # modal = global order by mean pairwise wins
        wins = Counter()
        for o in orders:
            for i, s in enumerate(o):
                wins[s] += len(o) - 1 - i
        modal = sorted(S, key=lambda s: -wins[s])
    pos = {s: i for i, s in enumerate(modal)}
    ok = sum(1 for o in orders if all(pos[o[i]] < pos[o[i + 1]] for i in range(len(o) - 1)))
    return ok / len(orders), len(orders), modal


def expected_chance(tabs, S):
    """chance share for random order = mean over qualifying tablets of 1/k!."""
    v = []
    FF = tabs if (tabs and isinstance(tabs[0], dict) and 'lines' not in tabs[0]) else [firsts(t) for t in tabs]
    for f in FF:
        pres = [s for s in S if s in f]
        if len(pres) >= 3 and len(set(f[s] for s in pres)) == len(pres):
            v.append(1 / math.factorial(len(pres)))
    return float(np.mean(v)) if v else 0.0


def search_stats(tabs, sets):
    out = np.zeros(len(sets))
    tabs = [firsts(t) for t in tabs]
    for i, S in enumerate(sets):
        st, n, _ = set_stat(tabs, S)
        out[i] = st - expected_chance(tabs, S) if n >= 5 else 0.0
    return out


def null_search(args):
    seed, TA, sets = args
    return search_stats(C.shuffle_lines(TA, random.Random(seed)), sets).max()


def search(name, tabs, pool, nsets=4000):
    ids = list(range(len(tabs)))
    A = set(random.Random(23).sample(ids, len(ids) // 2))
    TA = [tabs[i] for i in ids if i in A]
    TB = [tabs[i] for i in ids if i not in A]
    tc = Counter(s for t in TA for s in firsts(t))
    pool_s = sorted(s for s, n in tc.items() if n >= 10)
    rng = random.Random(5)
    sets = []
    seen = set()
    while len(sets) < nsets:
        k = rng.randint(3, 6)
        S = tuple(sorted(rng.sample(pool_s, k)))
        if S not in seen:
            seen.add(S)
            sets.append(S)
    obs = search_stats(TA, sets)
    mx = np.array(pool.map(null_search, [(100 + s, TA, sets) for s in range(10)]))
    thr = float(np.quantile(mx, 0.95)) if len(mx) else 0
    order = [int(i) for i in np.argsort(-obs)[:30]]
    res = {'name': name, 'nsets': nsets, 'thr': thr, 'null_max': mx.tolist(), 'top': []}
    rng2 = random.Random(9)
    for i in order:
        S = sets[i]
        sa, na, modal = set_stat(TA, S)
        sb, nb, _ = set_stat(TB, S, modal)
        eb = expected_chance(TB, S)
        null = []
        for _ in range(200):
            null.append(set_stat(C.shuffle_lines(TB, rng2), S, modal)[0])
        pb = (1 + sum(1 for x in null if x >= sb)) / 201 if nb >= 3 else 1.0
        res['top'].append({'set': modal, 'A_excess': float(obs[i]), 'A_share': sa, 'nA': na, 'B_share': sb, 'nB': nb,
                           'B_chance': eb, 'pB': pb, 'surv': bool(obs[i] > thr)})
    json.dump(res, open(os.path.join(OUT, 'search_' + name + '.json'), 'w'))
    ns = sum(1 for x in res['top'] if x['surv'])
    npass = sum(1 for x in res['top'] if x['pB'] <= 0.05 / 30)
    print('SEARCH %s thr %.3f survivors %d, B-pass (Bonferroni 30) %d' % (name, thr, ns, npass), flush=True)
    for x in res['top'][:10]:
        print('   ', x, flush=True)
    return res


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    CO = C.corpora()
    ent = variant(CO['PE'], 'ENT')
    mid = variant(CO['PE'], 'MID')
    uni = Counter(x for t in ent for l in t['lines'] for x in l['toks'])
    J = [('PE_ENT', ent, 50), ('PE_MID', mid, 50),
         ('PL_TOPIC', C.gen_planted(ent, 'TOPIC', 71, uni), 50), ('PL_ORDER', plant_order(ent, 72, uni), 50)]
    for k in ('UR3', 'LINB', 'ARCH_ADM', 'ARCH_LEX'):
        J.append((k, random.Random(5).sample(CO[k], min(1000, len(CO[k]))), 50))
    with Pool(nw) as p:
        p.map(job_ab, J, chunksize=1)
        search('PL_ORDER', plant_order(ent, 73, uni), p)
        search('PL_TOPIC', C.gen_planted(ent, 'TOPIC', 74, uni), p)
        search('PE_ENT', ent, p)
