"""pe70 cycle 3: invert the problem -- predict who signed from the text alone.

Thousands of random 'party marker' hypotheses: a hypothesis is a random set of 1-3 tokens (drawn
idf-weighted from tokens on seal-id tablets) read as 'tablets carrying ANY of these belong to one party'.
Score on TRAIN seals: among tablet pairs with a seal id where both carry the marker, the excess share of
pairs that share a seal over all seal-id pairs (lift).  Survivors (top 1% by train lift with >= 1 train
same-seal pair) are re-scored on HELD-OUT seals (seals split in half; held-out pairs involve only tablets of
held-out seals plus singletons).  Statistic: mean held-out lift of survivors and share with held-out lift > 0.
Nulls: seal ids shuffled among seal-id tablets (free, and within volume); same search, same splits.
Controls: Ur III Umma at PE size (validated: survivor tokens in the seal legend), planted PE marker.
usage: pe70_c3.py -> data/pe70_ckpt/c3.json
"""
import json, os, collections, math
import numpy as np
from multiprocessing import Pool
from pe70_common import get, CK
from pe70_c2 import groups_of, shuffle_groups

NH = 20000
NSPLIT = 10


def prep(R, G):
    """index tablets with a seal id; label vector of group id (-1 singleton)"""
    idx = [i for i, r in enumerate(R) if r['seals']]
    gid = {}
    for k, (s, ix) in enumerate(sorted(G.items())):
        for i in ix:
            gid.setdefault(i, k)
    lab = np.array([gid.get(i, -1 - n) for n, i in enumerate(idx)])
    vocab = sorted({t for i in idx for t in R[i]['toks']})
    vi = {t: j for j, t in enumerate(vocab)}
    X = np.zeros((len(idx), len(vocab)), dtype=bool)
    for n, i in enumerate(idx):
        for t in R[i]['toks']:
            X[n, vi[t]] = True
    return idx, lab, vocab, X


def lift(mask, lab, allowed):
    """among allowed tablets carrying the marker: same-seal pair share minus base same-seal pair share"""
    m = mask & allowed
    L = lab[m]
    n = len(L)
    if n < 2:
        return np.nan, 0
    c = collections.Counter(L[L >= 0])
    same = sum(v * (v - 1) / 2 for v in c.values())
    La = lab[allowed]
    ca = collections.Counter(La[La >= 0])
    base = sum(v * (v - 1) / 2 for v in ca.values()) / (len(La) * (len(La) - 1) / 2)
    return same / (n * (n - 1) / 2) - base, same


def _job(args):
    X, lab, hyps, splits = args
    out = []
    for tr_allowed, te_allowed in splits:
        res = []
        for H in hyps:
            mask = X[:, H].any(1)
            a, s = lift(mask, lab, tr_allowed)
            res.append((a if s >= 1 else -9, s))
        res = np.array(res)
        top = np.argsort(-res[:, 0])[:max(1, len(hyps) // 100)]
        te = [lift(X[:, hyps[i]].any(1), lab, te_allowed)[0] for i in top]
        te = np.array([x for x in te if not np.isnan(x)])
        out.append((float(te.mean()) if len(te) else 0.0, float((te > 0).mean()) if len(te) else 0.0,
                    [int(i) for i in top[:20]]))
    return out


def make_splits(lab, rng, n=NSPLIT):
    groups = sorted(set(lab[lab >= 0]))
    S = []
    for _ in range(n):
        g = set(rng.permutation(groups)[:len(groups) // 2])
        single = lab < 0
        coin = rng.random(len(lab)) < .5
        tr = np.array([l in g for l in lab]) | (single & coin)
        te = np.array([(l >= 0 and l not in g) for l in lab]) | (single & ~coin)
        S.append((tr, te))
    return S


def search(R, G, rng, pool, nh=NH, nnull=4, label=''):
    idx, lab, vocab, X = prep(R, G)
    df = X.sum(0)
    w = np.where(df >= 2, 1.0 / np.log1p(df), 0)   # markers must occur at least twice; rarer preferred
    w = w / w.sum()
    hyps = [rng.choice(len(vocab), rng.integers(1, 4), replace=False, p=w) for _ in range(nh)]
    splits = make_splits(lab, rng)
    vol = [R[i]['vol'] for i in range(len(R))]
    pool_idx = sorted(idx)
    labsets = {'real': lab}
    for k in range(nnull):
        G2 = shuffle_groups(G, pool_idx, vol, rng, within_vol=(k % 2 == 1))
        R_lab = {}
        for kk, (s, ix) in enumerate(sorted(G2.items())):
            for i in ix:
                R_lab.setdefault(i, kk)
        labsets['null%d_%s' % (k, 'vol' if k % 2 else 'free')] = np.array([R_lab.get(i, -1 - n) for n, i in enumerate(idx)])
    out = dict(label=label, n_seal_tablets=len(idx), groups=len(G))
    for nm, L in labsets.items():
        half = len(splits) // 2
        parts = pool.map(_job, [(X, L, hyps, splits[:half]), (X, L, hyps, splits[half:])])
        P = parts[0] + parts[1]
        mt = float(np.mean([p[0] for p in P])); sh = float(np.mean([p[1] for p in P]))
        rec = collections.Counter(vocab[j] for p in P for i in p[2] for j in hyps[i])
        out[nm] = dict(test_lift=mt, share_pos=sh, recurrent=rec.most_common(10))
    real = out['real']['test_lift']
    nulls = [v['test_lift'] for k, v in out.items() if k.startswith('null')]
    out['rank'] = int(sum(n >= real for n in nulls))
    print(label, {k: (round(v['test_lift'], 4), round(v['share_pos'], 2)) for k, v in out.items() if isinstance(v, dict)},
          'nulls>=real', out['rank'], 'recurrent', out['real']['recurrent'][:6], flush=True)
    return out


def main():
    rng = np.random.default_rng(703)
    pe, ur = get('pe'), get('ur3')
    G = groups_of(pe)
    res = {}
    with Pool(2) as pool:
        res['pe'] = search(pe, G, rng, pool, nnull=8, label='PE')
        res['plant'] = []
        df = collections.Counter(t for r in pe for t in r['toks'])
        rare = sorted(t for t, c in df.items() if 3 <= c <= 10)
        for k in range(4):
            P = [dict(r) for r in pe]
            for s in rng.choice(sorted(G), 3, replace=False):   # three parties get a private marker each
                t = rng.choice(rare)
                for i in G[s]:
                    P[i]['toks'] = sorted(set(P[i]['toks']) | {t})
            o = search(P, G, rng, pool, nh=5000, nnull=2, label='PE plant %d' % k)
            res['plant'].append({kk: vv for kk, vv in o.items()})
        U = [r for r in ur if r['vol'] == 'Umma']
        GU = groups_of(U)
        sizes = sorted(len(v) for v in G.values())
        res['ur3_pe'] = []
        for k in range(4):
            cand = [s for s, v in GU.items() if len(v) >= max(sizes)]
            pick = rng.choice(cand, len(sizes), replace=False)
            sub, newG, j = [], {}, 0
            for s, m in zip(pick, sizes):
                for i in rng.choice(GU[s], m, replace=False):
                    sub.append(dict(U[i], seals=[s]))
                newG[s] = list(range(j, j + m)); j += m
            chosen = set(id(x) for x in sub)
            single = [i for i, r in enumerate(U) if r['seals'] and not set(r['seals']) & set(pick)]
            for q, i in enumerate(rng.choice(single, 97 - len(sub), replace=False)):
                sub.append(dict(U[i], seals=['single%d' % q]))
            o = search(sub, newG, rng, pool, nh=5000, nnull=2, label='UR3 PE-size %d' % k)
            leg = collections.Counter()
            for s, ix in newG.items():
                for i in ix:
                    leg.update(set(sub[i]['legend']))
            o['recurrent_in_legend'] = [(t, c, t in leg) for t, c in o['real']['recurrent']]
            res['ur3_pe'].append(o)
            print('   legend check', o['recurrent_in_legend'][:8], flush=True)
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
