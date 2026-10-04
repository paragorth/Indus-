"""v29 cycle 2: are the spreading Voynich features real stroke features?
(a) frequency-matched relabel twins (100 per feature): does the stroke-defined class spread more than
    classes of glyphs with the same frequencies?  (b) held-out folio halves and ZL->IT replication.
(c) blocking: which intervening glyph classes block tier agreement (vs Markov-2 texts).
usage: python3 v29_cycle2.py NAME   (NAME = ZL, IT, tr, hu, plant7 ...) -> data/v29_ckpt/c2_NAME.pkl"""
import sys, os, pickle, random
import numpy as np
import v29_lib as L, v25_shapes as S
from v29_cycle1 import corpus

SEL = ('tier1', 'tier2', 'tier3', 'raw3', 'junc')


def select(r, z, thr, k=8):
    kinds = np.array([f['kind'] for f in r['feats']])
    st = np.isin(kinds, ['stroke', 'rstroke'])
    out = {}
    for s in SEL:
        i = L.STATS.index(s); zz = np.where(st, np.nan_to_num(z[:, i], nan=-99), -99)
        o = [int(j) for j in np.argsort(-zz)[:k] if zz[j] > thr[i]]
        out[s] = o
    return out


def excess(Es, f):
    o = L.stats(Es[0], f['m'], f['v'])
    n = np.array([L.stats(E, f['m'], f['v']) for E in Es[1:]])
    return o - np.nanmean(n, 0), (o - np.nanmean(n, 0)) / np.maximum(np.nanstd(n, 0, ddof=1), 0.02)


def blocking(E, f, shapes, alph):
    """tier-1 pairs with >=1 transparent glyph in between: phi with vs without an intervening glyph that has primitive X."""
    ids, wid = E.ids, E.wid
    T = np.flatnonzero(f['m'][ids]); ww = wid[T]
    s = (ww[1:] == ww[:-1]) & (T[1:] - T[:-1] > 1)
    a, b = T[:-1][s], T[1:][s]
    x = f['v'][ids[a]].astype(float); y = f['v'][ids[b]].astype(float)
    prims = sorted({p for g in alph for p in shapes[g]})
    out = {}
    for p in prims:
        ind = np.array([shapes[g].get(p, 0) > 0 for g in alph])[ids] & ~f['m'][ids]
        cs = np.r_[0, np.cumsum(ind)]
        has = (cs[b] - cs[a + 1]) > 0
        if has.sum() >= 40 and (~has).sum() >= 40:
            out[p] = (L._phi(x[has], y[has]), L._phi(x[~has], y[~has]), int(has.sum()))
    return out


if __name__ == '__main__':
    name = sys.argv[1]
    r = pickle.load(open(os.path.join(L.CK, f'c1_{name}.pkl'), 'rb'))
    z, ex = L.zscores(r); thr = np.nanmax(L.maxnull(r), 0)
    sel = select(r, z, thr)
    feats = sorted({j for v in sel.values() for j in v})
    lines, sh = corpus(name)
    alph = r['alph']; cnt = r['cnt']
    Es = [L.Enc(lines, alph)] + [L.Enc(L.markov2(lines, 1000 + q), alph) for q in range(10)]
    Esj = [Es[0]] + [L.Enc(L.wshuffle(lines, 1000 + q), alph) for q in range(10)]
    bins = L.freq_bins(alph, cnt, 4)
    rng = random.Random(5)
    out = dict(sel=sel, feats={}, alph=alph)
    for j in feats:
        f = r['feats'][j]
        e0, z0 = excess(Es, f); ej0, zj0 = excess(Esj, f)
        twins = []
        for t in range(100):
            p = L.relabel(bins, rng, len(alph))
            g = dict(m=f['m'][p], v=f['v'][p])
            if not ((g['m'] & g['v']).any() and (g['m'] & ~g['v']).any()):
                continue
            e, _ = excess(Es, g); ej, _ = excess(Esj, g)
            e[6] = ej[6]; twins.append(e)
        e0[6] = ej0[6]; z0[6] = zj0[6]
        twins = np.array(twins)
        pct = np.nanmean(twins < e0[None, :], 0)
        blk = blocking(Es[0], f, sh, alph)
        blkn = [blocking(E, f, sh, alph) for E in Es[1:4]]
        out['feats'][j] = dict(name=L.fname(f, alph), e=e0, z=z0, twin_pct=pct, twin_mean=np.nanmean(twins, 0), blk=blk, blkn=blkn)
        print(j, L.fname(f, alph), 'pct', np.round(pct, 2), flush=True)
    pickle.dump(out, open(os.path.join(L.CK, f'c2_{name}.pkl'), 'wb'))
    if name in ('ZL', 'IT'):   # held-out halves
        for fold in (0, 1):
            lf = L.voynich_lines('ZL3b' if name == 'ZL' else 'IT2a', fold=fold)
            Ef = [L.Enc(lf, alph)] + [L.Enc(L.markov2(lf, 2000 + q), alph) for q in range(10)]
            Efj = [Ef[0]] + [L.Enc(L.wshuffle(lf, 2000 + q), alph) for q in range(10)]
            for j in feats:
                f = r['feats'][j]
                _, zf = excess(Ef, f); _, zfj = excess(Efj, f); zf[6] = zfj[6]
                out['feats'][j][f'z_fold{fold}'] = zf
    pickle.dump(out, open(os.path.join(L.CK, f'c2_{name}.pkl'), 'wb'))
    print('done', name)
