"""pe47 cycle 1: random search over hidden-spreadsheet shapes (sign -> axis role).

A hypothesis H gives each candidate token one role: U (row / unit axis), C (column / commodity axis),
P (period / slice axis) or X (not an index). Under H an entry sits in cell
   (U-tokens of entry + header, C-tokens of entry, P-tokens of header);
a missing axis means the entry is a sum over that axis (an aggregate cell).
The hidden table is estimated by sparse cell counts with hierarchical back-off
   cell(U,C,P) -> (U,C) -> (C) -> baseline (last token -> system marginal).
Score = held-out (tablet-disjoint) mean log-likelihood gain of the entry values over the baseline
'commodity-only' model (values depend only on the last token).
Search: N random role vectors + hill-climbing of the best, fitted on half A of the tablets
(inner 2-fold CV), tested once on half B.
"""
import collections, json, math, os, random, sys, time
import numpy as np
from pe47_common import CK, build_pe, build_ur3, plant, sample_like, ent_lengths

ROLES = 'XUCP'
ALPHA = 2.0


def vkey(v):
    return '%.4f' % v


def prep(corpus, K=120, seed=0):
    tf = collections.Counter()
    for t in corpus:
        s = set(t['ctx'])
        for e in t['ents']:
            s |= set(e[0])
        tf.update(s)
    cand = [w for w, n in tf.most_common(K) if n >= 5]
    idx = {w: i for i, w in enumerate(cand)}
    tabs = []
    for t in corpus:
        ctx = sorted({idx[w] for w in t['ctx'] if w in idx})
        E = []
        for toks, v, sysn, rev in t['ents']:
            et = sorted({idx[w] for w in toks if w in idx})
            E.append((tuple(et), tuple(ctx), toks[-1], sysn, vkey(v)))
        tabs.append(E)
    return cand, tabs


def score(rho, train, test):
    """rho: np array role per token. Nested refinement of the baseline cell (sys, last token):
    k1 = + C-axis tokens, k2 = + U-axis tokens (entry and header), k3 = + P-axis header tokens.
    Witten-Bell back-off; a level whose key adds nothing is skipped. Returns mean gain (nats/entry)."""
    def keys(e):
        et, ctx, last, sysn, v = e
        al = set(et) | set(ctx)
        U = tuple(sorted(i for i in al if rho[i] == 1))
        C = tuple(sorted(i for i in al if rho[i] == 2))
        P = tuple(sorted(i for i in al if rho[i] == 3))
        b = (sysn, last)
        return (b, C) if C else None, (b, C, U) if U else None, (b, C, U, P) if P else None
    nsys = collections.Counter(); nsv = collections.Counter()
    nl = collections.Counter(); nlv = collections.Counter()
    N = [collections.Counter() for _ in range(3)]
    NV = [collections.Counter() for _ in range(3)]
    D = [collections.Counter() for _ in range(3)]
    vocab = collections.defaultdict(set)
    dl = collections.Counter()
    for e in train:
        _, _, last, sysn, v = e
        nsys[sysn] += 1
        if nsv[(sysn, v)] == 0:
            vocab[sysn].add(v)
        nsv[(sysn, v)] += 1
        if nlv[(sysn, last, v)] == 0:
            dl[(sysn, last)] += 1
        nl[(sysn, last)] += 1; nlv[(sysn, last, v)] += 1
        for j, k in enumerate(keys(e)):
            if k is None:
                continue
            if NV[j][(k, v)] == 0:
                D[j][k] += 1
            N[j][k] += 1; NV[j][(k, v)] += 1
    tot = 0.0
    for e in test:
        _, _, last, sysn, v = e
        pm = (nsv[(sysn, v)] + 1.0) / (nsys[sysn] + len(vocab[sysn]) + 1.0)
        b = (sysn, last)
        d = dl[b]
        p0 = (nlv[(sysn, last, v)] + d * pm) / (nl[b] + d) if nl[b] else pm
        p = p0
        for j, k in enumerate(keys(e)):
            if k is None or N[j][k] == 0:
                continue
            d = D[j][k]
            p = (NV[j][(k, v)] + d * p) / (N[j][k] + d)
        tot += math.log(p) - math.log(p0)
    return tot / max(1, len(test))


def flat(tabs, ids):
    return [e for i in ids for e in tabs[i]]


def inner(rho, tabs, A, folds):
    s = 0.0
    for f in range(2):
        tr = flat(tabs, [i for i, g in zip(A, folds) if g != f])
        te = flat(tabs, [i for i, g in zip(A, folds) if g == f])
        s += score(rho, tr, te)
    return s / 2


def random_rho(K, r):
    p = r.choice([(0.4, 0.2, 0.2, 0.2), (0.6, 0.15, 0.15, 0.1), (0.25, 0.25, 0.25, 0.25), (0.7, 0.1, 0.1, 0.1)])
    return np.array(r.choices(range(4), weights=p, k=K))


def search(tabs, K, seed, n_rand=1000, n_top=4, n_climb=120):
    r = random.Random(seed)
    ids = list(range(len(tabs)))
    r.shuffle(ids)
    A, B = ids[:len(ids) // 2], ids[len(ids) // 2:]
    folds = [r.randrange(2) for _ in A]
    pool = []
    for k in range(n_rand):
        rho = random_rho(K, r)
        pool.append((inner(rho, tabs, A, folds), k, rho))
    z = np.zeros(K, dtype=int)
    pool.append((inner(z, tabs, A, folds), -1, z))
    pool.sort(key=lambda x: -x[0])
    rand_best = pool[0][0]
    best = []
    for s0, _, rho in pool[:n_top]:
        rho = rho.copy(); s = s0
        for it in range(n_climb):
            j = r.randrange(K); old = rho[j]
            rho[j] = r.choice([x for x in range(4) if x != old])
            s2 = inner(rho, tabs, A, folds)
            if s2 >= s:
                s = s2
            else:
                rho[j] = old
        best.append((s, rho.copy()))
    best.sort(key=lambda x: -x[0])
    trA, teB = flat(tabs, A), flat(tabs, B)
    held = [score(rho, trA, teB) for s, rho in best]
    # ablations on the best solution: remove P axis, remove U axis
    rho = best[0][1]
    abl = {}
    for name, role in (('noP', 3), ('noU', 1), ('noC', 2)):
        r2 = rho.copy(); r2[r2 == role] = 0
        abl[name] = score(r2, trA, teB)
    return dict(rand_best_inner=rand_best, climb_inner=[b[0] for b in best], heldout=held,
                heldout_best=held[0], ablation=abl, rho=[int(x) for x in rho],
                rhos=[[int(x) for x in b[1]] for b in best])


def null_shuffle_values(corpus, seed):
    """N1: values shuffled across tablets within (last token, system)."""
    r = random.Random(seed)
    groups = collections.defaultdict(list)
    for ti, t in enumerate(corpus):
        for ei, e in enumerate(t['ents']):
            groups[(e[0][-1], e[2])].append((ti, ei))
    new = [dict(t, ents=[list(e) for e in t['ents']]) for t in corpus]
    for g, L in groups.items():
        vals = [corpus[ti]['ents'][ei][1] for ti, ei in L]
        r.shuffle(vals)
        for (ti, ei), v in zip(L, vals):
            new[ti]['ents'][ei][1] = v
    return new


def null_permute_entries(corpus, seed):
    """N2: whole entries (tokens + value) re-dealt between tablets, tablet sizes and headers kept."""
    r = random.Random(seed)
    allE = [e for t in corpus for e in t['ents']]
    r.shuffle(allE)
    new, k = [], 0
    for t in corpus:
        n = len(t['ents'])
        new.append(dict(t, ents=allE[k:k + n])); k += n
    return new


def null_random_tensor(corpus, seed):
    """N3: plant-shaped world whose values are drawn iid from the corpus value distribution."""
    r = random.Random(seed)
    vals = [e[1] for t in corpus for e in t['ents']]
    P = plant(len(corpus), ent_lengths(corpus), seed, U=30, C=20, P=6)
    for t in P:
        for e in t['ents']:
            e[1] = r.choice(vals)
    return P


def get_corpus(name, seed=0):
    if name == 'PE':
        return build_pe()
    pe = build_pe()
    if name == 'PLANT':
        return plant(len(pe), ent_lengths(pe), 1000 + seed, U=30, C=20, P=6)
    if name.startswith('UR3'):
        site = 'drehem' if 'D' in name else 'umma'
        return sample_like(build_ur3(site), len(pe), 77 + seed)
    raise ValueError(name)


def run(job):
    name, null, seed, n_rand = job
    corpus = get_corpus(name, seed)
    if null == 'N1':
        corpus = null_shuffle_values(corpus, 500 + seed)
    elif null == 'N2':
        corpus = null_permute_entries(corpus, 500 + seed)
    elif null == 'N3':
        corpus = null_random_tensor(corpus, 500 + seed)
    cand, tabs = prep(corpus)
    t0 = time.time()
    res = search(tabs, len(cand), seed, n_rand=n_rand)
    res.update(name=name, null=null, seed=seed, cand=cand, secs=time.time() - t0)
    return res


if __name__ == '__main__':
    from multiprocessing import Pool
    out = sys.argv[1]
    n_rand = int(sys.argv[2])
    jobs = []
    for seed in (0, 1):
        for name in ('PLANT', 'UR3D', 'UR3U', 'PE'):
            for null in (('real', 'N1', 'N2') if seed == 0 else ('real', 'N1')):
                jobs.append((name, null, seed, n_rand))
        jobs.append(('PE', 'N3', seed, n_rand))
    done = {}
    if os.path.exists(out):
        done = {(d['name'], d['null'], d['seed']): d for d in json.load(open(out))}
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    with Pool(2) as p:
        for res in p.imap_unordered(run, jobs):
            done[(res['name'], res['null'], res['seed'])] = res
            json.dump(list(done.values()), open(out, 'w'))
            print(res['name'], res['null'], res['seed'], 'held %.4f' % res['heldout_best'],
                  'abl', {k: round(v, 4) for k, v in res['ablation'].items()}, '%.0fs' % res['secs'], flush=True)
