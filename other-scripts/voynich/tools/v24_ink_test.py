"""v24 cycle 3b: does the local ink-anomaly detector find corrections, and if so what do flagged glyphs enforce?
(1) Residualise segment darkness on glyph type x page; anomaly = residual - word median residual, / page MAD.
(2) Planted twins (double-ink simulation on one glyph): detection (top-2% flag) and localisation.
(3) Validation on real marks: ZL correction-note words on these pages, and transcriber-disagreement glyphs
    (ZL [a:b] alternatives; ZL vs IT single-glyph differences, word-aligned lines) vs other glyphs of the same
    type (permutation of the disagreement label within glyph type x page).
(4) Rule position test: for flagged glyphs, constraint C_R = R(word) - mean R(random edit at that glyph);
    compared with all other glyph positions (permute flags within word length x page).
"""
import os, sys, json, random, math, re, pickle
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v24_lib as L
import v24_data as D
import v18_lib as V18


def anomalies(recs):
    real = [r for r in recs if r['plant'] is None]
    # glyph type x page means from real words
    acc = defaultdict(list)
    for r in real:
        for g, v in zip(r['glyphs'], r['vals']):
            if v is not None:
                acc[(r['folio'], g)].append(v)
    mu = {k: float(np.mean(v)) for k, v in acc.items()}
    gm = defaultdict(list)
    for r in real:
        for g, v in zip(r['glyphs'], r['vals']):
            if v is not None:
                gm[g].append(v)
    gmu = {g: float(np.mean(v)) for g, v in gm.items()}
    for r in recs:
        res = [(v - mu.get((r['folio'], g), gmu.get(g, v))) if v is not None else None for g, v in zip(r['glyphs'], r['vals'])]
        ok = [x for x in res if x is not None]
        med = float(np.median(ok)) if ok else 0
        r['res'] = [None if x is None else x - med for x in res]
    pmad = defaultdict(list)
    for r in real:
        pmad[r['folio']] += [x for x in r['res'] if x is not None]
    mad = {f: float(np.median(np.abs(v))) * 1.4826 + 1e-6 for f, v in pmad.items()}
    for r in recs:
        r['z'] = [None if x is None else x / mad[r['folio']] for x in r['res']]
        zz = [x for x in r['z'] if x is not None]
        r['zmax'] = max(zz) if zz else None
        r['kmax'] = int(np.nanargmax([-9 if x is None else x for x in r['z']])) if zz else None
    return recs


def it_glyph_diffs():
    """ZL vs IT: lines with equal word counts, single-unit edits per word -> {(folio,n,k): glyph index in ZL}."""
    zl = {(r['folio'], r['n']): [D.reading(t, 0) for t in D.zl_variants(r['raw'])] for r in D.zl_lines()}
    it = {(r['folio'], r['n']): [D.reading(t, 0) for t in D.zl_variants(r['raw'])] for r in D.zl_lines(os.path.join(L.DATA, 'IT2a-n.txt'))}
    out = {}
    for key, zw in zl.items():
        iw = it.get(key)
        if not iw or len(iw) != len(zw):
            continue
        for k, (a, b) in enumerate(zip(zw, iw)):
            if a == b or '?' in a + b or '@' in a + b or not a or not b:
                continue
            ga, gb = V18.glyphs(a), V18.glyphs(b)
            if len(ga) == len(gb):
                d = [i for i in range(len(ga)) if ga[i] != gb[i]]
                if len(d) == 1:
                    out[key + (k,)] = (d[0], ''.join(gb))
    return out


def zl_alt_glyphs():
    out = {}
    for r in D.zl_lines():
        for k, t in enumerate(D.zl_variants(r['raw'])):
            if '[' in t:
                a0, a1 = D.reading(t, 0), D.reading(t, 1)
                ga, gb = V18.glyphs(a0), V18.glyphs(a1)
                if len(ga) == len(gb):
                    d = [i for i in range(len(ga)) if ga[i] != gb[i]]
                    if len(d) == 1:
                        out[(r['folio'], r['n'], k)] = (d[0], a1)
    return out


def perm_test(items, label, strata, nperm=2000, seed=0):
    """mean z of labelled glyphs minus mean of unlabelled, null = label permuted within strata."""
    rng = np.random.default_rng(seed)
    z = np.array([x for x in items]); lab = np.array(label, bool); st = [str(s) for s in strata]
    obs = z[lab].mean() - z[~lab].mean()
    groups = defaultdict(list)
    for i, s in enumerate(st):
        groups[s].append(i)
    groups = [np.array(v) for v in groups.values() if lab[v].any() and (~lab[v]).any()]
    null = []
    for _ in range(nperm):
        l2 = lab.copy()
        for g in groups:
            l2[g] = rng.permutation(lab[g])
        null.append(z[l2].mean() - z[~l2].mean())
    null = np.array(null)
    return float(obs), float(null.mean()), float(null.std()), float(((null >= obs).sum() + 1) / (nperm + 1)), int(lab.sum())


if __name__ == '__main__':
    recs = anomalies(json.load(open(os.path.join(L.CK, 'ink_glyphs.json'))))
    real = [r for r in recs if r['plant'] is None and r['zmax'] is not None]
    pl = [r for r in recs if r['plant'] is not None and r['zmax'] is not None]
    thr = float(np.percentile([r['zmax'] for r in real], 98))
    out = dict(n_words=len(real), n_glyphs=sum(len([x for x in r['z'] if x is not None]) for r in real), thr98=thr)
    out['planted_flag'] = float(np.mean([r['zmax'] >= thr for r in pl]))
    out['planted_loc'] = float(np.mean([r['kmax'] == r['plant'] for r in pl]))
    out['planted_loc_pm1'] = float(np.mean([abs(r['kmax'] - r['plant']) <= 1 for r in pl]))
    out['chance_loc'] = float(np.mean([1 / len(r['glyphs']) for r in pl]))
    out['planted_z_at_k'] = float(np.mean([r['z'][r['plant']] for r in pl if r['z'][r['plant']] is not None]))
    # (3) ZL correction notes on these pages
    notes = {('f107v', 44): 'kaich', ('f108r', 43): 'cheam', ('f112r', 1): 'ykair', ('f58v', 3): None}
    rank = sorted(r['zmax'] for r in real)
    nr = []
    for r in real:
        key = (r['folio'], r['n'])
        if key in notes and (notes[key] is None or r['word'] == notes[key] or r['word'] == 'chcphy'):
            if r['word'] in ('kaich', 'cheam', 'ykair', 'chcphy') or r['word'].startswith('sho'):
                q = np.searchsorted(rank, r['zmax']) / len(rank)
                nr.append((r['folio'], r['word'], round(r['zmax'], 2), round(float(q), 3)))
    out['corr_note_words'] = nr
    # transcriber disagreement glyphs
    itd = it_glyph_diffs(); alt = zl_alt_glyphs()
    items, labA, labI, strata = [], [], [], []
    for r in real:
        key = (r['folio'], r['n'], r['k'])
        for i, (g, z) in enumerate(zip(r['glyphs'], r['z'])):
            if z is None:
                continue
            items.append(z); strata.append((r['folio'], g))
            labA.append(key in alt and alt[key][0] == i)
            labI.append(key in itd and itd[key][0] == i)
    out['alt_test'] = perm_test(items, labA, strata, seed=1)
    out['it_test'] = perm_test(items, labI, strata, seed=2)
    # top flagged words: fraction with any disagreement in the word, vs rest
    fl = [r for r in real if r['zmax'] >= thr]
    def anyd(r):
        key = (r['folio'], r['n'], r['k']); return key in alt or key in itd
    out['flag_dis_rate'] = float(np.mean([anyd(r) for r in fl])); out['rest_dis_rate'] = float(np.mean([anyd(r) for r in real if r['zmax'] < thr]))
    out['n_flagged'] = len(fl)
    json.dump(out, open(os.path.join(L.CK, 'c3_ink.json'), 'w'), indent=1)
    pickle.dump(dict(real=real, thr=thr, alt=alt, itd=itd), open(os.path.join(L.CK, 'c3_ink.pkl'), 'wb'))
    print(json.dumps(out, indent=1))
