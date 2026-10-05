"""v60 cycle 2: the complexion as a hidden class label.
In real herbals hot vs cold is (a) a ~75/25 split, (b) stated once per entry, and (c) predictable from the rest of the
entry (the uses). Search: binary slot = which of two items comes first in a window (whole entry, first 12, last 12
tokens); gates on part A: coverage >= 0.5, majority share 0.70-0.90. Naive Bayes on the rest of the entry (word types
containing either item removed) fitted on part A, candidates ranked by AUC on part B, confirmed on part C (model refit
on A+B). Pages split in thirds."""
import sys, random, json, collections, time
import numpy as np
import v60_lib as L
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

WINS = ('all', 'head12', 'tail12')


def thirds(ents, seed):
    pages = sorted(set(e['page'] for e in ents))
    rng = random.Random(seed); rng.shuffle(pages)
    k = len(pages) // 3
    part = {p: (0 if i < k else 1 if i < 2 * k else 2) for i, p in enumerate(pages)}
    return [[i for i, e in enumerate(ents) if part[e['page']] == j] for j in range(3)]


def window(toks, w):
    return toks if w == 'all' else toks[:12] if w == 'head12' else toks[-12:]


def firstpos(ents, keep, w):
    idx = {it: j for j, it in enumerate(keep)}
    F = np.full((len(ents), len(keep)), 10 ** 6, np.int32)
    for i, e in enumerate(ents):
        for t, tok in enumerate(window(e['toks'], w)):
            for it in L.items_of(tok):
                j = idx.get(it)
                if j is not None and F[i, j] > t: F[i, j] = t
    return F


def labels(F, a, b):
    """y (n, K): 1 = a first, 0 = b first, -1 uncovered."""
    Fa, Fb = F[:, a], F[:, b]
    pa, pb = Fa < 10 ** 6, Fb < 10 ** 6
    y = np.where(pa | pb, (Fa < Fb).astype(np.int8), -1).astype(np.int8)
    # same token carries both -> ambiguous
    y[(Fa == Fb) & pa] = -1
    return y


def auc_cols(S, Y, z=False):
    """AUC per column of scores S for labels Y in {0,1,-1}; NaN if fewer than 4 of a class. z=True: (AUC-0.5)/SE
    with the rank-sum null SE sqrt((n1+n0+1)/(12 n1 n0))."""
    out = np.full(S.shape[1], np.nan); zz = np.full(S.shape[1], np.nan)
    for k in range(S.shape[1]):
        m = Y[:, k] >= 0
        y = Y[m, k]; s = S[m, k]
        n1 = int(y.sum()); n0 = len(y) - n1
        if n1 < 4 or n0 < 4: continue
        r = np.argsort(np.argsort(s)) + 1.0
        out[k] = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        zz[k] = (out[k] - 0.5) / np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0))
    return (out, zz) if z else out


def nb_weights(X, Y, mask, alpha=0.5):
    """X (n, V) counts; Y (n, K) in {0,1,-1}; mask (V, K) True = feature removed. Returns W (V, K) log-ratio."""
    Y1 = (Y == 1).astype(np.float32); Y0 = (Y == 0).astype(np.float32)
    C1 = X.T @ Y1; C0 = X.T @ Y0
    C1 = np.where(mask, 0, C1); C0 = np.where(mask, 0, C0)
    V = (~mask).sum(0)
    W = np.log((C1 + alpha) / (C1.sum(0) + alpha * V)) - np.log((C0 + alpha) / (C0.sum(0) + alpha * V))
    return np.where(mask, 0, W)


def run(ents, seed=1, max_items=600, max_cand=150_000, share=(0.70, 0.90), min_cov=0.5, top=50, truth=None):
    A, B, C = thirds(ents, seed)
    EA = [ents[i] for i in A]
    keep, _, _ = L.token_items(EA, max_items=max_items)
    I = len(keep)
    nest = np.array([[L.nested(x, y) for y in keep] for x in keep])
    # vocabulary of word types (fitted on A+B)
    vocab = sorted(set(t for i in A + B for t in ents[i]['toks']))
    vi = {t: j for j, t in enumerate(vocab)}
    def counts(ix):
        X = np.zeros((len(ix), len(vocab)), np.float32)
        for r, i in enumerate(ix):
            for t in ents[i]['toks']:
                j = vi.get(t)
                if j is not None: X[r, j] += 1
        return X
    XA, XB, XC = counts(A), counts(B), counts(C)
    TI = np.zeros((len(vocab), I), bool)
    ki = {it: j for j, it in enumerate(keep)}
    for t, j in vi.items():
        for it in L.items_of(t):
            if it in ki: TI[j, ki[it]] = True
    rng = np.random.default_rng(seed)
    cands = []
    n_total = 0
    for w in WINS:
        FA = firstpos([ents[i] for i in A], keep, w)
        P = FA < 10 ** 6
        for a in range(I):
            b = np.arange(a + 1, I)
            b = b[~nest[a, b]]
            if len(b) == 0: continue
            Fa = FA[:, [a]]; Fb = FA[:, b]
            cov = ((P[:, [a]] | P[:, b]) & ~((Fa == Fb) & P[:, [a]])).mean(0)
            afirst = (Fa < Fb).sum(0) / np.maximum(1, ((P[:, [a]] | P[:, b]) & ~((Fa == Fb) & P[:, [a]])).sum(0))
            maj = np.maximum(afirst, 1 - afirst)
            ok = (cov >= min_cov) & (maj >= share[0]) & (maj <= share[1])
            n_total += len(b)
            for bb in b[ok]:
                cands.append((w, a, int(bb)))
    n_pass = len(cands)
    if len(cands) > max_cand:
        sel = rng.choice(len(cands), max_cand, replace=False); cands = [cands[i] for i in sel]
    if truth is not None:
        tset = set((w, a, b) for (w, a, b) in cands if truth(keep[a], keep[b]))
    res = []
    F = {(w, part): firstpos([ents[i] for i in ix], keep, w) for w in WINS for part, ix in (('A', A), ('B', B), ('C', C))}
    for s in range(0, len(cands), 4000):
        cb = cands[s:s + 4000]
        K = len(cb)
        mask = np.zeros((len(vocab), K), bool)
        YA = np.zeros((len(A), K), np.int8); YB = np.zeros((len(B), K), np.int8); YC = np.zeros((len(C), K), np.int8)
        for k, (w, a, b) in enumerate(cb):
            mask[:, k] = TI[:, a] | TI[:, b]
            YA[:, k] = labels(F[(w, 'A')], [a], [b])[:, 0]
            YB[:, k] = labels(F[(w, 'B')], [a], [b])[:, 0]
            YC[:, k] = labels(F[(w, 'C')], [a], [b])[:, 0]
        W = nb_weights(XA, YA, mask)
        aucB, zB = auc_cols(XB @ W, YB, z=True)
        res += [dict(w=w, a=a, b=b, aucB=float(x), zB=float(y)) for (w, a, b), x, y in zip(cb, aucB, zB)]
        if s == 0:
            pass
    good = [r for r in res if not np.isnan(r['aucB'])]
    good.sort(key=lambda r: -r['zB'])
    topr = good[:top]
    # confirm on C with NB refit on A+B
    if topr:
        K = len(topr)
        mask = np.zeros((len(vocab), K), bool)
        YAB = np.zeros((len(A) + len(B), K), np.int8); YC = np.zeros((len(C), K), np.int8)
        for k, r in enumerate(topr):
            mask[:, k] = TI[:, r['a']] | TI[:, r['b']]
            YAB[:, k] = np.concatenate([labels(F[(r['w'], 'A')], [r['a']], [r['b']])[:, 0], labels(F[(r['w'], 'B')], [r['a']], [r['b']])[:, 0]])
            YC[:, k] = labels(F[(r['w'], 'C')], [r['a']], [r['b']])[:, 0]
        W = nb_weights(np.vstack([XA, XB]), YAB, mask)
        aucC, zC = auc_cols(XC @ W, YC, z=True)
        for r, x, y in zip(topr, aucC, zC):
            r['aucC'] = float(x); r['zC'] = float(y)
            r['items'] = [keep[r['a']], keep[r['b']]]
            yc = YC[:, topr.index(r)]
            r['shareC'] = float((yc == 1).sum() / max(1, (yc >= 0).sum())); r['covC'] = float((yc >= 0).mean())
    out = dict(n_hyp=n_total, n_pass=n_pass, n_scored=len(good), top=topr,
               aucB_q=[float(x) for x in np.percentile([r['aucB'] for r in good], [50, 95, 99, 99.9])] if good else None,
               zB_q=[float(x) for x in np.percentile([r['zB'] for r in good], [50, 95, 99, 99.9])] if good else None)
    if truth is not None:
        ranks = [k for k, r in enumerate(good) if (r['w'], r['a'], r['b']) in tset]
        out['truth_n'] = len(tset); out['truth_best_rank'] = ranks[0] if ranks else None
        out['truth_pct'] = (1 - ranks[0] / len(good)) if ranks else None
        tr = [r for r in good if (r['w'], r['a'], r['b']) in tset][:5]
        out['truth_top'] = [dict(r, items=[keep[r['a']], keep[r['b']]]) for r in tr]
    return out, keep


def summ(o):
    t = o['top']
    d = dict(n_hyp=o['n_hyp'], n_pass=o['n_pass'], n_scored=o['n_scored'], aucB_q=[round(x, 3) for x in (o['aucB_q'] or [])])
    if t:
        cs = [r['aucC'] for r in t if not np.isnan(r.get('aucC', np.nan))]
        zs = [r['zC'] for r in t if not np.isnan(r.get('zC', np.nan))]
        d.update(top_zB=round(t[0]['zB'], 2), zB_q=[round(x, 2) for x in o['zB_q']], med_zC=round(float(np.median(zs)), 2) if zs else None,
                 frac_zC_gt2=round(float(np.mean([z > 2 for z in zs])), 2) if zs else None,
                 top_aucB=round(t[0]['aucB'], 3), med_aucC=round(float(np.median(cs)), 3) if cs else None,
                 frac_aucC_gt_0_65=round(float(np.mean([c > 0.65 for c in cs])), 2) if cs else None,
                 best=[t[0]['w']] + t[0]['items'] + [round(t[0].get('aucC', np.nan), 3)])
    for k in ('truth_n', 'truth_best_rank', 'truth_pct'):
        if k in o: d[k] = o[k]
    return d


def hotcold_truth(C):
    dec = {}
    def f(x, y):
        for it in (x, y):
            if it not in dec:
                d = L.decode_item(C, it, 1); dec[it] = d[0] if d else ''
        q = lambda s: bool(L.HOT.match(s) or L.COLD.match(s))
        return q(dec[x]) and q(dec[y]) and (bool(L.HOT.match(dec[x])) != bool(L.HOT.match(dec[y])))
    return f


if __name__ == '__main__':
    out = {}
    vh = [e for e in L.voynich_entries('ZL3b', ('H', 'P')) if e['strat'][1] == 'A']
    vi = [e for e in L.voynich_entries('IT2a', ('H', 'P')) if e['strat'][1] == 'A']
    ci = L.encode_entries(L.herbal_entries('CI'), seed=61, pad=0.35, max_len=128)
    lyt = L.encode_entries([dict(e, toks=e['toks']) for e in L.herbal_entries('LYT') if e['cx']][:300], seed=65, pad=0.35, max_len=200)
    runs = [('P_CI', ci, hotcold_truth(ci)), ('P_LYT', lyt, hotcold_truth(lyt)),
            ('V_herbA', vh, None), ('N_markov', L.markov_null(vh, 1), None), ('N_wordshuf', L.wordshuf_null(vh, 1), None),
            ('V_starsB', L.voynich_entries('ZL3b', ('S',)), None), ('V_herbA_IT2a', vi, None),
            ('NEG_CURY', L.encode_entries(L.plain_entries('CURY'), seed=63, pad=0.35, max_len=128), None),
            ('NEG_AST', L.encode_entries(L.plain_entries('AST')[:250], seed=64, pad=0.35, max_len=128), None)]
    only = sys.argv[1:] or None
    for name, ents, truth in runs:
        if only and name not in only: continue
        for seed in (1, 2):
            t = time.time()
            o, keep = run(ents, seed=seed, truth=truth)
            out['%s|%d' % (name, seed)] = dict(summary=summ(o), top=o['top'][:20], truth_top=o.get('truth_top'))
            print(name, seed, len(ents), round(time.time() - t), json.dumps(summ(o)), flush=True)
            L.jsave('cycle2.json', out)
