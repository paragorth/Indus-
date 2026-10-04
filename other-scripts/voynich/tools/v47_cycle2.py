"""v47 cycle 2: which words carry the picture-text link, and do they predict unseen pages?
(a) WORD-VISUAL MANTEL: for each word type w (on >= 4 pages), partial Mantel r between the
    four-net visual similarity and the page co-occurrence matrix of w (w on both pages), with the
    usual confounds; z against within-stratum page permutations (shared across words).
(b) LEAVE-ONE-WORD-OUT: drop of the composite r when w is deleted from every page.
(c) FROZEN LIST, HELD-OUT QUIRES: 2-fold over quires (fold 1 = A, C, E, G, O; fold 2 = B, D, F, Q).
    On the discovery fold, rank words by (a); freeze the top K = 15. On the test fold, for each
    frozen word, score each test page by its mean visual similarity (four nets, z-scored) to the
    discovery pages that contain w minus that to discovery pages without w, residualised on test-
    page log length and drawing area; AUC for presence of w on the test page. Statistic: mean AUC
    over frozen words. Null: the whole pipeline (selection + test) rerun with discovery-fold
    visual rows permuted (200x).
(d) VISUAL TRAITS (CLIP zero-shot prompts: root forms, leaf forms, flowers, berries, colour):
    trait score = CLIP image-prompt similarity minus mean over prompts; word x trait point-biserial
    correlation (partialled on log length and quire) on discovery fold; freeze the top 15 (word,
    trait, sign) pairs; test the sign on the test fold (AUC of presence vs trait score). Null as (c).
Run on Voynich A1 herbal, on Gerard and Dodoens (identical pipeline; pseudo-quires).
Writes data/v47_ckpt/c2.json and loops/v47_cycle2.txt.
"""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v47_setup import *
from v47_cycle1 import comp, sub

NPERM = int(os.environ.get('NPERM', 200))
K = 15


def vis_comp(S, nets=NETS4):
    """Average of z-scored (off-diagonal) cosine similarities of the four networks."""
    out = 0
    for f in nets:
        M = emb_sim_m([S['emb'][k][f] for k in S['keys']])
        iu = np.triu_indices(len(M), 1)
        M = (M - M[iu].mean()) / M[iu].std()
        out = out + M
    return out / len(nets)


def presence(words, mindf=4):
    df = Counter(w for ws in words for w in set(ws))
    vocab = sorted(w for w, c in df.items() if c >= mindf and c <= 0.7 * len(words))
    B = np.array([[w in set(ws) for w in vocab] for ws in words], float)
    return vocab, B


def word_mantel(V, B, part, rng, strata, nperm):
    """Partial r of residualised V with each word's co-occurrence (vectorised over words)."""
    n = V.shape[0]
    iu = np.triu_indices(n, 1)
    C = (B[iu[0]] * B[iu[1]])                         # pairs x words
    C = C[part.m]
    C = C - part.X @ (part.P @ C)
    sd = C.std(0); sd[sd == 0] = 1
    C = (C - C.mean(0)) / sd
    def r_of(p):
        a = zs(part.res(upper(V[np.ix_(p, p)])))
        return a @ C / len(a)
    obs = r_of(np.arange(n))
    null = np.array([r_of(perm(n, rng, strata)) for _ in range(nperm)])
    z = (obs - null.mean(0)) / np.where(null.std(0) > 0, null.std(0), 1)
    return obs, z


def fold_split(quire, src):
    q = np.array(quire)
    if src == 'voynich':
        f1 = np.isin(q, ['A', 'C', 'E', 'G', 'O'])
    else:
        u = sorted(set(q), key=lambda s: int(s[1:]))
        f1 = np.isin(q, u[0::2])
    return f1


def clip_traits(S):
    names, T = clip_text_matrix()
    X = np.array([S['emb'][k]['clip'] for k in S['keys']], float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sc = X @ T.T
    sc = sc - sc.mean(1, keepdims=True)
    return names, (sc - sc.mean(0)) / sc.std(0)


def resid_on(y, covs):
    X = np.column_stack([np.ones(len(y))] + covs)
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def auc(score, lab):
    lab = lab.astype(bool)
    if lab.all() or (~lab).all():
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = lab.sum(); n0 = (~lab).sum()
    return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def heldout_words(S, V, vocab, B, d, t, rng, perm_rows=None, nperm_sel=0):
    """Select top-K words on discovery pages d by word-visual Mantel; test on t."""
    Sd = sub(S, d)
    Vd = V[np.ix_(d, d)]
    if perm_rows is not None:
        Vd = Vd[np.ix_(perm_rows, perm_rows)]
    part = MPartial(list(Sd['conf'].values()), len(d))
    Bd = B[d]
    ok = (Bd.sum(0) >= 3) & (B[t].sum(0) >= 2)
    # selection score: observed word-visual partial r on the discovery fold
    n = len(d); iu = np.triu_indices(n, 1)
    C = Bd[:, ok][iu[0]] * Bd[:, ok][iu[1]]
    C = C - part.X @ (part.P @ C)
    sd = C.std(0); sd[sd == 0] = 1
    a = zs(part.res(upper(Vd)))
    r = a @ ((C - C.mean(0)) / sd) / len(a)
    widx = np.where(ok)[0][np.argsort(-r)[:K]]
    ln = np.log([len(S['words'][i]) for i in t]); ar = np.log([S['emb'][S['keys'][i]]['area'] + 1e-4 for i in t])
    Vtd = V[np.ix_(t, d)]
    if perm_rows is not None:
        Vtd = Vtd[:, perm_rows]
    aucs = []
    for w in widx:
        has = Bd[:, w] > 0
        sc = Vtd[:, has].mean(1) - Vtd[:, ~has].mean(1)
        sc = resid_on(sc, [ln, ar])
        aucs.append(auc(sc, B[t, w]))
    return [vocab[w] for w in widx], np.array(aucs, float)


def trait_words(S, traits, vocab, B, d, t, quire, rotate=None):
    """Select top-K (word, trait, sign) on d; test on t. rotate: permutation of discovery rows of traits."""
    ln = np.log([len(ws) for ws in S['words']])
    qd = np.array(quire)
    Td = traits[d] if rotate is None else traits[d][rotate]
    Bd = B[d]
    ok = (Bd.sum(0) >= 3) & (B[t].sum(0) >= 2)
    qcats = sorted(set(qd[d]))
    cov_d = [ln[d]] + [(qd[d] == c).astype(float) for c in qcats[1:]]
    Xd = np.column_stack([np.ones(len(d))] + cov_d)
    Pd = Xd @ np.linalg.pinv(Xd)
    Rt = Td - Pd @ Td
    Rb = Bd - Pd @ Bd
    sdt = Rt.std(0); sdt[sdt == 0] = 1
    Rt = (Rt - Rt.mean(0)) / sdt
    sdb = Rb.std(0); sdb[sdb == 0] = 1
    Rb = (Rb - Rb.mean(0)) / sdb
    Cr = Rb.T @ Rt / len(d)                       # words x traits
    Cr[~ok] = 0
    flat = np.argsort(-np.abs(Cr).ravel())[:K]
    pairs = [(w, j, np.sign(Cr[w, j])) for w, j in (np.unravel_index(f, Cr.shape) for f in flat)]
    qt = sorted(set(qd[t]))
    cov_t = [ln[t]] + [(qd[t] == c).astype(float) for c in qt[1:]]
    aucs = []
    for w, j, s in pairs:
        sc = resid_on(traits[t][:, j], cov_t) * s
        aucs.append(auc(sc, B[t, w]))
    return pairs, np.array(aucs, float)


def run(S, src, rng):
    V = vis_comp(S)
    vocab, B = presence(S['words'])
    n = len(S['keys'])
    part = MPartial(list(S['conf'].values()), n)
    obs, z = word_mantel(V, B, part, rng, S['strata'], 500)
    top = np.argsort(-z)[:20]
    # leave-one-word-out on the composite r (four nets x 5 metrics), top 300 words by df
    T0 = tsims(S['words'])
    base_V = sims(S, NETS4)
    Tres0 = [zs(part.res(upper(T))) for T in T0.values()]
    A = [zs(part.res(upper(M))) for M in base_V.values()]
    r0 = np.mean([np.mean(a * t) for a in A for t in Tres0])
    df = B.sum(0)
    cand = [vocab[i] for i in np.argsort(-df)[:300]]
    drops = {}
    for w in cand:
        ws = [[x for x in p if x != w] for p in S['words']]
        Tw = tsims(ws)
        Tr = [zs(part.res(upper(T))) for T in Tw.values()]
        drops[w] = float(r0 - np.mean([np.mean(a * t) for a in A for t in Tr]))
    dtop = sorted(drops.items(), key=lambda kv: -kv[1])[:12]
    dvals = np.array(list(drops.values()))
    # held-out words, both directions
    f1 = fold_split(S['quire'], src)
    res = {}
    for name, d, t in [('1->2', np.where(f1)[0], np.where(~f1)[0]), ('2->1', np.where(~f1)[0], np.where(f1)[0])]:
        words_sel, a = heldout_words(S, V, vocab, B, d, t, rng)
        null = []
        for _ in range(NPERM):
            pr = rng.permutation(len(d))
            null.append(np.nanmean(heldout_words(S, V, vocab, B, d, t, rng, perm_rows=pr)[1]))
        null = np.array(null)
        names, traits = clip_traits(S)
        pairs, ta = trait_words(S, traits, vocab, B, d, t, S['quire'])
        tnull = np.array([np.nanmean(trait_words(S, traits, vocab, B, d, t, S['quire'], rotate=rng.permutation(len(d)))[1]) for _ in range(NPERM)])
        res[name] = dict(words=words_sel, aucs=a.tolist(), mean=float(np.nanmean(a)), null_mean=float(null.mean()), null_sd=float(null.std()),
                         z=float((np.nanmean(a) - null.mean()) / null.std()), p=float((1 + (null >= np.nanmean(a)).sum()) / (1 + NPERM)),
                         trait_pairs=[(vocab[w], names[j], int(s)) for w, j, s in pairs], trait_aucs=ta.tolist(),
                         trait_mean=float(np.nanmean(ta)), trait_z=float((np.nanmean(ta) - tnull.mean()) / tnull.std()),
                         trait_p=float((1 + (tnull >= np.nanmean(ta)).sum()) / (1 + NPERM)))
    return dict(n=n, nvocab=len(vocab), top=[(vocab[i], float(z[i]), int(df[i])) for i in top],
                nz3=int((z >= 3).sum()), nz_expected=float(len(z) * 0.00135), r0=float(r0),
                drops=dtop, drop_sd=float(dvals.std()), heldout=res)


if __name__ == '__main__':
    rng = np.random.default_rng(4702)
    out, rows = {}, []
    # usage: v47_cycle2.py SRC (runs one source, saves c2_SRC.json) | v47_cycle2.py report
    if sys.argv[1] != 'report':
        src = sys.argv[1]
        S = vsetup('A1') if src == 'voynich' else hsetup(src, 8 if src == 'gerard' else 6)
        r = run(S, src, rng)
        json.dump(r, open(os.path.join(CK47, 'c2_%s.json' % src), 'w'), default=str)
        sys.exit()
    for src in ['voynich', 'gerard', 'dodoens']:
        out[src] = json.load(open(os.path.join(CK47, 'c2_%s.json' % src)))
    tag = {'voynich': 'Voynich A1 herbal', 'gerard': 'Gerard 1636 control', 'dodoens': 'Dodoens 1583 control'}
    k = 1
    for src, r in out.items():
        rows.append(('V-47.2.%d' % k, 'WORD-VISUAL MANTEL, %s (n %d, %d word types on >= 4 pages): per-word partial Mantel of four-net visual similarity vs co-occurrence of the word, 500 shared permutations' % (tag[src], r['n'], r['nvocab']),
                     'words with z >= 3: %d (chance ~%.1f); top: %s' % (r['nz3'], r['nz_expected'], ', '.join('%s %+.1f (df %d)' % tuple(t) for t in r['top'][:10])), '')); k += 1
        rows.append(('V-47.2.%d' % k, 'LEAVE-ONE-WORD-OUT, %s: drop of the composite r (base %+.4f) when each of the 300 commonest words is deleted' % (tag[src], r['r0']),
                     'largest drops: %s; sd of drops %.5f' % (', '.join('%s %+.5f' % tuple(t) for t in r['drops'][:8]), r['drop_sd']), '')); k += 1
        for d, h in r['heldout'].items():
            rows.append(('V-47.2.%d' % k, 'HELD-OUT QUIRES %s, %s: freeze top %d words by word-visual Mantel on the discovery fold; on unseen pages, AUC of word presence from visual similarity to the discovery pages carrying it (length, area partialled); null: discovery visual rows permuted, whole pipeline rerun (%d)' % (d, tag[src], K, NPERM),
                         'frozen: %s; mean AUC %.3f (null %.3f +- %.3f), z %+.2f, p %.3f' % (' '.join(h['words']), h['mean'], h['null_mean'], h['null_sd'], h['z'], h['p']),
                         'held-out support' if h['p'] < 0.05 else 'no held-out support')); k += 1
            rows.append(('V-47.2.%d' % k, 'TRAIT WORDS %s, %s: CLIP zero-shot traits (bulbous/tuber/fibrous/tap root, round/narrow/lobed/serrate leaf, flower, berries, tree, climber, blue, red); freeze the top %d (word, trait, sign) by partial correlation (length, quire) on the discovery fold; held-out AUC; null: trait rows permuted in discovery (%d)' % (d, tag[src], K, NPERM),
                         'frozen: %s; mean AUC %.3f, z %+.2f, p %.3f' % ('; '.join('%s~%s%s' % (w, t, '+' if s > 0 else '-') for w, t, s in h['trait_pairs'][:8]), h['trait_mean'], h['trait_z'], h['trait_p']),
                         'held-out support' if h['trait_p'] < 0.05 else 'no held-out support')); k += 1
    json.dump(out, open(os.path.join(CK47, 'c2.json'), 'w'), default=str)
    with open(os.path.join(LOOPS, 'v47_cycle2.txt'), 'w') as fh:
        fh.write('# v47 cycle 2 - which words carry the link; frozen word lists tested on held-out quires\n')
        fh.write('| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            fh.write('| %s | %s | %s | %s |\n' % r)
    print(open(os.path.join(LOOPS, 'v47_cycle2.txt')).read())
