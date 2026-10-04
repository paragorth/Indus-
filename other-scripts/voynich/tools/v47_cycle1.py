"""v47 cycle 1: the pre-registered support and kill tests of the v38 picture-text link.
Statistic everywhere: composite partial Mantel = mean partial r over (network x 5 text metrics),
z against a page-permutation null (within language x hand strata for the full herbal; free inside
a single stratum). Confounds as in v38 (language, hand, quire, bifolio, leaf, adjacency, log page
distance, text length, drawing area) for the Voynich herbal; folio, paragraph adjacency, leaf,
log leaf distance, length, area, vertical position for pharma units; adjacency, log distance,
length, area for the printed herbals.
Tests (pre-registered in loops/v38_final.txt and the v47 brief):
 K1 CLIP image tower alone, herbal A1 (kill: z < 1).
 S1 MAE (fifth network) alone (support: z >= 3).
 S2 pharmaceutical register units (support: composite z > 0, ideally >= 2).
 S3 held-out quires: each quire's pairs with the other quires only; per-quire z, Stouffer.
 K2 hand 1 split three ways (quires A-C vs D-Q; paint/production 2-means; glyph-style 2-means).
 C  Gerard and Dodoens through the identical pipeline; shuffled page assignment; planted 10%.
Writes data/v47_ckpt/c1.json and loops/v47_cycle1.txt.
"""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v47_setup import *

NPERM = int(os.environ.get('NPERM', 1000))
TM = ['jaccard', 'tfidf', 'rare', 'midcos', 'tri']


def comp(S, nets, rng, mask=None, words=None, nperm=NPERM, extra_conf=()):
    V = sims(S, nets)
    T = tsims(words if words is not None else S['words'])
    part = MPartial(list(S['conf'].values()) + list(extra_conf), len(S['keys']), mask)
    return part.test(V, T, nperm, rng, S['strata'])


def sub(S, idx):
    out = dict(S)
    out['keys'] = [S['keys'][i] for i in idx]; out['words'] = [S['words'][i] for i in idx]
    out['conf'] = {k: C[np.ix_(idx, idx)] for k, C in S['conf'].items()}
    out['strata'] = None if S['strata'] is None else [S['strata'][i] for i in idx]
    out['quire'] = [S['quire'][i] for i in idx]
    return out


def heldout_quires(S, nets, rng, nperm=400):
    q = np.array(S['quire'])
    res = {}
    for x in sorted(set(q)):
        a = q == x
        if a.sum() < 3:
            continue
        mask = np.logical_xor(a[:, None], a[None, :])          # pairs with exactly one page in quire x
        r = comp(S, nets, rng, mask=mask, nperm=nperm)
        res[x] = (int(a.sum()), r['z'])
    zs_ = [z for _, z in res.values()]
    w = [np.sqrt(n) for n, _ in res.values()]
    stouffer = float(np.dot(w, zs_) / np.sqrt(np.dot(w, w)))
    return res, stouffer, sum(z > 0 for z in zs_)


def kmeans2(X, rng, it=50):
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    best = None
    for _ in range(20):
        c = X[rng.choice(len(X), 2, replace=False)]
        for _ in range(it):
            lab = ((X[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            if len(set(lab)) < 2:
                break
            c = np.array([X[lab == k].mean(0) for k in range(2)])
        if len(set(lab)) < 2:
            continue
        sse = sum(((X[lab == k] - c[k]) ** 2).sum() for k in range(2))
        bal = min((lab == 0).sum(), (lab == 1).sum())
        if bal >= 0.3 * len(X) and (best is None or sse < best[0]):
            best = (sse, lab)
    if best is None:   # force a balanced split along the first PC
        u = np.linalg.svd(X - X.mean(0), full_matrices=False)[0][:, 0]
        return (u > np.median(u)).astype(int)
    return best[1]


def glyph_profile(words):
    from v38_cycle5 import glyphs_of
    G = [Counter(g for w in ws for g in glyphs_of(w)) for ws in words]
    al = sorted(set().union(*G))
    M = np.array([[c[a] for a in al] for c in G], float)
    return M / M.sum(1, keepdims=True)


def fz(r):
    return 'z %+.2f (p %.3f, n pairs %d)' % (r['z'], r['p'], r['npairs'])


if __name__ == '__main__':
    rng = np.random.default_rng(4701)
    rows, out = [], {}
    A1 = vsetup('A1')
    ALL = vsetup(None)
    n = len(A1['keys'])
    # ---------------- replication of the v38 composite on the new embeddings
    r4 = comp(A1, NETS4, rng); r4all = comp(ALL, NETS4, rng)
    out['rep'] = dict(A1=r4, all=r4all)
    per = {f: comp(A1, [f], rng) for f in NETS6}
    out['per_net'] = per
    rows.append(('V-47.1', 'REPLICATION on re-fetched images: v38 four-network composite (ResNet-18, DINO, EfficientNet-B0, DINOv2) x 5 text metrics, Currier A hand 1 herbal pages (n %d), v38 confounds, %d within-stratum permutations; all 119 herbal pages likewise' % (n, NPERM),
                 'A1 %s; all pages %s. Single networks (A1): %s' % (fz(r4), fz(r4all), ', '.join('%s %+.2f' % (f, per[f]['z']) for f in NETS6)),
                 'replicates' if r4['z'] >= 2 else 'does not replicate at z >= 2'))
    rows.append(('V-47.2', 'KILL TEST K1 (pre-registered in v38: z < 1 kills). CLIP ViT-B/32 image tower (openai weights) alone, A1, same confounds and null',
                 fz(per['clip']), 'KILLED' if per['clip']['z'] < 1 else ('survives (z >= 1)' + (', supports (z >= 2)' if per['clip']['z'] >= 2 else ''))))
    rows.append(('V-47.3', 'SUPPORT TEST S1 (pre-registered: a fifth network at z >= 3). MAE ViT-B/16 (self-supervised masked autoencoder, mean patch token) alone, A1',
                 fz(per['mae']), 'supports' if per['mae']['z'] >= 3 else ('weak (1-3)' if per['mae']['z'] >= 1 else 'no support')))
    r6 = comp(A1, NETS6, rng)
    out['nets6'] = r6
    # ---------------- shuffled and planted controls on A1
    sh = []
    for s in range(20):
        p = rng.permutation(n)
        sh.append(comp(A1, NETS4, rng, words=[A1['words'][i] for i in p], nperm=200)['z'])
    allw = Counter(w for ws in A1['words'] for w in ws)
    pool = [w for w, c in allw.items() if 5 <= c <= 40]
    Z = np.array([A1['emb'][k]['clip'] for k in A1['keys']], float)
    U, Sv, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pl = comp(A1, NETS4, rng, words=plant_text(A1['words'], U[:, :10] * Sv[:10], 0.10, rng, pool), nperm=400)
    out['shuffled'] = sh; out['planted'] = pl
    rows.append(('V-47.4', 'CONTROLS on A1: six-network composite; text pages shuffled (20 random reassignments, four-net composite); planted link (10% of tokens drawn from a vocabulary driven by the CLIP embedding, four-net composite: a link the four networks did not make)',
                 'six-net composite %s; shuffled z mean %+.2f, max %+.2f (20 draws); planted %s' % (fz(r6), np.mean(sh), np.max(sh), fz(pl)),
                 'shuffle null behaves; planted link recovered' if pl['z'] > 3 and np.mean(sh) < 1 else 'check controls'))
    # ---------------- pharmaceutical units
    PH = psetup()
    rp4 = comp(PH, NETS4, rng); rp6 = comp(PH, NETS6, rng)
    perp = {f: comp(PH, [f], rng, nperm=500)['z'] for f in NETS6}
    # pooled herbal A1 + pharma? (different kind of image: whole plant vs tiled parts) -> pharma alone is primary
    out['pharma'] = dict(n=len(PH['keys']), c4=rp4, c6=rp6, per=perp)
    rows.append(('V-47.5', 'SUPPORT TEST S2: PHARMACEUTICAL plant parts. %d units = one paragraph + the small root/leaf drawings of its register (jars removed; parts tiled into one 224 px image), 14 panels, f88r-f102v2 (f101v left out); confounds same folio, adjacent paragraph, same leaf, log leaf distance, length, drawing area, vertical position; free permutation' % len(PH['keys']),
                 'four-net %s; six-net %s; single nets %s' % (fz(rp4), fz(rp6), ', '.join('%s %+.1f' % kv for kv in perp.items())),
                 'supports' if rp4['z'] >= 2 else ('weakly consistent' if rp4['z'] > 0.5 else 'no support')))
    # ---------------- held-out quires
    hq, st, npos = heldout_quires(A1, NETS4, rng)
    out['heldout'] = dict(per=hq, stouffer=st, npos=npos)
    rows.append(('V-47.6', 'SUPPORT TEST S3: HELD-OUT QUIRES. For each quire, the composite on pairs with exactly one page in that quire (quire vs rest; within-quire pairs excluded), A1, 400 perms each; Stouffer over quires (weights sqrt n)',
                 'per quire (n, z): ' + '; '.join('%s %d %+.2f' % (q, a, z) for q, (a, z) in hq.items()) + '; positive %d/%d; Stouffer %+.2f' % (npos, len(hq), st),
                 'spread across quires' if npos >= len(hq) - 2 and st >= 2 else ('carried by a few quires' if st >= 2 else 'not robust to quire hold-out')))
    # ---------------- hand-1 splits
    q = np.array(A1['quire'])
    splits = {'quires A-C | D-Q': np.isin(q, ['A', 'B', 'C']).astype(int)}
    P = np.array([A1['emb'][k]['prod'] for k in A1['keys']], float)
    splits['paint/production 2-means'] = kmeans2(P, rng)
    splits['glyph-style 2-means'] = kmeans2(glyph_profile(A1['words']), rng)
    sp = {}
    for name, lab in splits.items():
        zz = []
        for g in (0, 1):
            idx = np.where(lab == g)[0]
            zz.append((len(idx), comp(sub(A1, idx), NETS4, rng, nperm=500)['z']))
        sp[name] = zz
    out['splits'] = sp
    killed = any(max(z for _, z in v) < 1 for v in sp.values())
    rows.append(('V-47.7', 'KILL TEST K2 (pre-registered: disappearance when hand 1 is split). Three splits of the A1 herbal pages; four-net composite inside each half (500 perms). Expected under a uniform link: z about 3/sqrt(2) ~ 2 per half',
                 '; '.join('%s: %s' % (k, ' / '.join('n %d z %+.2f' % t for t in v)) for k, v in sp.items()),
                 'KILLED (both halves z < 1 in a split)' if killed else 'survives'))
    # ---------------- real herbals through the identical pipeline
    for src, adjw, tag in [('gerard', 8, 'V-47.8'), ('dodoens', 6, 'V-47.9')]:
        H = hsetup(src, adjw)
        rh4 = comp(H, NETS4, rng); rh6 = comp(H, NETS6, rng)
        perh = {f: comp(H, [f], rng, nperm=300)['z'] for f in NETS6}
        hq2, st2, np2 = heldout_quires(H, NETS4, rng, nperm=200)
        idx = np.random.default_rng(1).choice(len(H['keys']), min(n, len(H['keys'])), replace=False)
        rsz = comp(sub(H, np.sort(idx)), NETS4, rng, nperm=500)
        out[src] = dict(n=len(H['keys']), c4=rh4, c6=rh6, per=perh, heldout=dict(per=hq2, stouffer=st2), at_n=rsz)
        rows.append((tag, 'CONTROL: %s (%d pages with a woodcut and >= 30 OCR words), identical pipeline and composite' % ('Gerard, Herball 1636 (English)' if src == 'gerard' else 'Dodoens, Stirpium historiae pemptades sex 1583 (Latin)', len(H['keys'])),
                     'four-net %s; six-net %s; single nets %s; at the A1 page count (n %d) %s; held-out pseudo-quires Stouffer %+.2f (%d/%d positive)' % (
                         fz(rh4), fz(rh6), ', '.join('%s %+.1f' % kv for kv in perh.items()), len(idx), fz(rsz), st2, np2, len(hq2)),
                     'positive control works' if rh4['z'] >= 3 else 'positive control weak'))
    json.dump(out, open(os.path.join(CK47, 'c1.json'), 'w'), default=str)
    with open(os.path.join(LOOPS, 'v47_cycle1.txt'), 'w') as fh:
        fh.write('# v47 cycle 1 - pre-registered support and kill tests of the v38 picture-text link\n')
        fh.write('| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            fh.write('| %s | %s | %s | %s |\n' % r)
    print(open(os.path.join(LOOPS, 'v47_cycle1.txt')).read())
