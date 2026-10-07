"""pe82 cycle 2: WHAT THE READER COULD NOT HAVE KNOWN (novelty tolerance).

A reader can absorb a sign never seen before only where the context fixes its role.  Discovery batches D,
held-out batches H.  A token in H is NOVEL if its base sign never occurs in D.  A context definition maps a sign
token to a key (features: entry position, entry length, left / right neighbour (through a random sign partition),
line system, line index, surface, header).  Tolerance(key) = smoothed rate, in D, of tokens whose sign occurs in
only one tablet of D (proxy novelty).  Score = AUC of tolerance for H-novel vs H-known tokens.

Controls: (kill) H novelty labels shuffled within (entry length x position) strata - the AUC that position+length
alone give; (planted) in H, tokens right after a chosen mid-frequency sign are replaced by brand-new signs.
Calibration: proto-cuneiform, same pipeline; openness of PC_WORLD (goods) signs vs others.

usage: python3 pe82_c2.py CORPUS MODE seed     MODE real | plant
"""
import sys, os, json, math, random, itertools, collections
import numpy as np
import pe82_common as pc

FEATS = ['epos', 'elen', 'left', 'right', 'sys', 'lidx', 'surf', 'hdr']
SPLITS = {'PE': [({'MDP26', 'MDP26S', 'TCL31'}, {'MDP17', 'MDP06', 'OTHER'}),
                 ({'MDP17', 'MDP06', 'OTHER'}, {'MDP26', 'MDP26S', 'TCL31'})],
          'PC': [({'U3'}, {'U4', 'OTHER'}), ({'U4', 'OTHER'}, {'U3'})]}


def tokens(T):
    out = []
    for ti, t in enumerate(T):
        for li, l in enumerate(t['lines']):
            sg = l['sg']
            n = len(sg)
            for k, s in enumerate(sg):
                ep = 'only' if n == 1 else ('ini' if k == 0 else ('fin' if k == n - 1 else 'mid'))
                out.append(dict(tab=ti, batch=t['batch'], s=s, epos=ep, elen=str(min(n, 5)),
                                left=sg[k - 1] if k else '^', right=sg[k + 1] if k + 1 < n else '$',
                                sys=l['sys'], lidx=str(min(li, 4)), surf=l['surf'], hdr=t['hdr'] or '-'))
    return out


def auc(score, lab):
    score = np.asarray(score); lab = np.asarray(lab, bool)
    if lab.sum() == 0 or (~lab).sum() == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    # ties: average ranks
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r); r = (sums / cnt)[inv]
    n1 = lab.sum(); n0 = (~lab).sum()
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def make_hyps(rng, signs, n):
    """hypothesis = (features tuple, partition for neighbour features: dict sign->cluster or None=identity)."""
    hyps = []
    for k in (1, 2, 3):
        for f in itertools.combinations(FEATS, k):
            hyps.append((f, None, 0))
    while len(hyps) < n:
        k = rng.choice([1, 2, 3])
        f = tuple(sorted(rng.sample(FEATS, k)))
        if not ({'left', 'right'} & set(f)):
            f = tuple(sorted(set(f) | {rng.choice(['left', 'right'])}))
        K = rng.choice([2, 3, 4, 6, 8, 12, 20])
        hyps.append((f, K, rng.randrange(1 << 30)))
    return hyps


def keyfun(f, K, pseed, signs):
    part = None
    if K:
        r = random.Random(pseed)
        part = {s: r.randrange(K) for s in signs}
    def key(t):
        out = []
        for x in f:
            v = t[x]
            if part is not None and x in ('left', 'right') and v not in ('^', '$'):
                v = part.get(v, 'U')
            out.append(v)
        return tuple(out)
    return key


def run(corpus, mode, seed, n_h=3000):
    rng = random.Random(seed)
    T = pc.load_pe() if corpus == 'PE' else pc.load_pc()
    tok = tokens(T)
    signs = sorted({t['s'] for t in tok})
    res = []
    for si, (D, H) in enumerate(SPLITS[corpus]):
        Dt = [t for t in tok if t['batch'] in D]
        Ht = [dict(t) for t in tok if t['batch'] in H]
        seenD = {t['s'] for t in Dt}
        if mode == 'plant':
            cnt = collections.Counter(t['s'] for t in Dt)
            mids = [s for s, c in cnt.items() if 40 <= c <= 150]
            Q = random.Random(seed + si).choice(sorted(mids))
            k = 0
            for t in Ht:
                if t['left'] == Q and t['s'] in seenD and random.Random(seed * 7 + k).random() < 0.5:
                    t['s'] = 'NEW%d' % k; k += 1
            plantQ = (Q, k)
        tabs = collections.defaultdict(set)
        for t in Dt:
            tabs[t['s']].add(t['tab'])
        proxy = np.array([len(tabs[t['s']]) == 1 for t in Dt])
        novel = np.array([t['s'] not in seenD for t in Ht])
        base = proxy.mean()
        # null strata
        strat = collections.defaultdict(list)
        for i, t in enumerate(Ht):
            strat[(t['elen'], t['epos'])].append(i)
        nulls = []
        for rep in range(2):
            nn = novel.copy(); r2 = random.Random(seed * 101 + rep + 17 * si)
            for idx in strat.values():
                v = [novel[i] for i in idx]; r2.shuffle(v)
                for i, x in zip(idx, v):
                    nn[i] = x
            nulls.append(nn)
        hyps = make_hyps(random.Random(seed * 13 + si), signs, n_h)
        # cut the H tokens into two random-tablet-free halves by batch for survivor re-test: H1, H2
        Hb = sorted(H)
        h1 = np.array([t['batch'] == Hb[0] for t in Ht])
        for f, K, ps in hyps:
            kf = keyfun(f, K, ps, signs)
            c = collections.Counter(); n = collections.Counter()
            for t, p in zip(Dt, proxy):
                kk = kf(t); n[kk] += 1; c[kk] += p
            a = 5.0
            sc = np.array([(c.get(kf(t), 0) + a * base) / (n.get(kf(t), 0) + a) for t in Ht])
            res.append((si, list(f), K, ps, round(auc(sc[h1], novel[h1]), 4), round(auc(sc[~h1], novel[~h1]), 4),
                        round(auc(sc, novel), 4),
                        [[round(auc(sc[h1], nn[h1]), 4), round(auc(sc[~h1], nn[~h1]), 4)] for nn in nulls]))
        # null: novelty shuffled within (elen, epos) -> AUC of a pos+len-only reader for reference
        info = dict(si=si, nD=len(Dt), nH=len(Ht), novel=int(novel.sum()), novel_h1=int(novel[h1].sum()),
                    novel_h2=int(novel[~h1].sum()), H1=Hb[0])
        if mode == 'plant':
            info['plant'] = plantQ
        res.append(('info', info))
    return res


if __name__ == '__main__':
    corpus, mode, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    res = run(corpus, mode, seed)
    fn = os.path.join(pc.CK, 'c2_%s_%s_%d.json' % (corpus, mode, seed))
    json.dump(res, open(fn, 'w'))
    for r in res:
        if r[0] == 'info':
            print(corpus, mode, seed, r[1])
