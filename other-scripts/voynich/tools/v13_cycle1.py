"""v13 cycle 1: does the text follow the drawing on the same page?

Image features (blind) x text features, partial correlation given section + Currier
language + hand; null = image vectors permuted among pages of the same section x language
(and, second null, same quire x section). Family-wise via max |r|.
Controls: (a) segmentation check (blind text-ink area vs real glyph count);
(b) planted token-level texts whose length / q-rate are generated from the image;
(c) derangement: each page's text against another page's image of the same section.
"""
import json, os, sys, math, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v13_lib import *
from collections import Counter

NPERM = 3000
rng = random.Random(13)
nrng = np.random.default_rng(13)
OUTD = os.path.join(DER, 'v13')
os.makedirs(OUTD, exist_ok=True)


def build():
    img = json.load(open(os.path.join(DER, 'v13_imgfeat.json')))
    pages = load_pages()
    gc = Counter(w for p in pages.values() for w in p['P'] + p['L'])
    rows = []
    for lab, f in img.items():
        fol = 'f' + lab
        if 'error' in f or fol not in pages:
            continue
        p = pages[fol]
        if len(p['P']) + len(p['L']) < 5:
            continue
        t = text_features(p['P'], p['L'], p['nlines'], gc, rng)
        rows.append({'folio': fol, 'meta': p['meta'], 'img': f, 'txt': t, 'P': p['P'], 'L': p['L']})
    return rows, gc


def run_matrix(rows, ifeats, tfeats, strata_key, nperm=NPERM, extra=None, seed=1):
    r_ = np.random.default_rng(seed)
    keep = [i for i, r in enumerate(rows) if all(np.isfinite(r['txt'][t]) for t in tfeats)]
    rows = [rows[i] for i in keep]
    metas = [r['meta'] for r in rows]
    Z = design(metas, None if extra is None else [[e(r) for r in rows] for e in extra])
    X = np.array([[r['img'][f] for f in ifeats] for r in rows])
    X = np.log1p(np.maximum(X, 0) * 100)  # tame skew; monotone
    Y = np.array([[r['txt'][t] for t in tfeats] for r in rows])
    R = partial_r_matrix(Z, X, Y)
    strata = [strata_key(r) for r in rows]
    exceed = np.zeros_like(R); maxnull = []
    pyrng = random.Random(seed)
    for _ in range(nperm):
        pi = strata_perm(strata, pyrng)
        # permuting X rows breaks the X-Z link only within strata (Z constant inside strata
        # except hand), so residualise permuted X again
        Rn = partial_r_matrix(Z, X[pi], Y)
        exceed += (np.abs(Rn) >= np.abs(R))
        maxnull.append(np.abs(Rn).max())
    P = (exceed + 1) / (nperm + 1)
    maxnull = np.array(maxnull)
    Pfw = np.array([[(np.sum(maxnull >= abs(R[i, j])) + 1) / (nperm + 1) for j in range(R.shape[1])]
                    for i in range(R.shape[0])])
    return dict(n=len(rows), R=R, P=P, Pfw=Pfw, max95=float(np.quantile(maxnull, 0.95)))


def top(res, ifeats, tfeats, k=8):
    R, P, Pfw = res['R'], res['P'], res['Pfw']
    idx = sorted(((abs(R[i, j]), i, j) for i in range(len(ifeats)) for j in range(len(tfeats))), reverse=True)[:k]
    return ['%s~%s r=%+.2f p=%.4f pFW=%.3f' % (ifeats[i], tfeats[j], R[i, j], P[i, j], Pfw[i, j]) for _, i, j in idx]


def sec_lang(r):
    return (r['meta']['illus'], r['meta']['lang'])


def quire_sec(r):
    return (r['meta']['quire'], r['meta']['illus'])


def planted(rows, gc, beta_len, beta_q, feat_len='draw_edges', feat_q='frac_green', seed=0):
    """Token-level planted texts: length and q-word share driven by the image."""
    r_ = random.Random(seed)
    vocab = [w for w in gc for _ in range(gc[w])]
    qv = [w for w in vocab if w.startswith('qo')]; nq = [w for w in vocab if not w.startswith('qo')]
    xl = np.array([math.log1p(100 * r['img'][feat_len]) for r in rows]); xl = (xl - xl.mean()) / xl.std()
    xq = np.array([math.log1p(100 * r['img'][feat_q]) for r in rows]); xq = (xq - xq.mean()) / xq.std()
    out = []
    base_q = len(qv) / len(vocab)
    for k, r in enumerate(rows):
        n0 = max(5, len(r['P']))
        n = max(5, int(round(math.exp(math.log(n0) + beta_len * xl[k] + r_.gauss(0, 0.0)))))
        pq = min(0.9, max(0.0, base_q * math.exp(beta_q * xq[k])))
        ws = [r_.choice(qv) if r_.random() < pq else r_.choice(nq) for _ in range(n)]
        rr = dict(r); rr['txt'] = text_features(ws, r['L'], max(1, n // 9), gc, r_)
        out.append(rr)
    return out


if __name__ == '__main__':
    rows, gc = build()
    ifeats, tfeats = IMG_FEATS, TXT_FEATS
    log = []
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s); log.append(s)
    P('pages joined', len(rows), Counter(r['meta']['illus'] for r in rows))
    # (a) segmentation check
    res = run_matrix(rows, ['text_ink', 'draw_area'], ['log_words', 'n_lines'], sec_lang, nperm=1000)
    P('SEG CHECK text_ink~log_words r=%+.2f p=%.4f ; draw_area~log_words r=%+.2f p=%.4f' % (
        res['R'][0, 0], res['P'][0, 0], res['R'][1, 0], res['P'][1, 0]))
    # main, all sections
    res = run_matrix(rows, ifeats, tfeats, sec_lang)
    P('ALL n=%d max|r|95=%.3f' % (res['n'], res['max95']))
    for s in top(res, ifeats, tfeats, 10): P('  ', s)
    nsig = int((res['P'] < 0.01).sum()); P('  cells p<0.01: %d of %d (expected %.1f)' % (nsig, res['P'].size, 0.01 * res['P'].size))
    json.dump({'R': res['R'].tolist(), 'P': res['P'].tolist(), 'Pfw': res['Pfw'].tolist(), 'ifeats': ifeats, 'tfeats': tfeats},
              open(os.path.join(OUTD, 'c1_all.json'), 'w'))
    # quire x section null
    res2 = run_matrix(rows, ifeats, tfeats, quire_sec, seed=2)
    P('ALL quire-null max|r|95=%.3f' % res2['max95'])
    for s in top(res2, ifeats, tfeats, 5): P('  ', s)
    # complexity given drawing area (layout removed)
    cf = [f for f in ifeats if f != 'draw_area']
    res3 = run_matrix(rows, cf, tfeats, sec_lang, extra=[lambda r: math.log1p(100 * r['img']['draw_area'])], seed=3)
    P('ALL | draw_area covariate, max|r|95=%.3f' % res3['max95'])
    for s in top(res3, cf, tfeats, 6): P('  ', s)
    # herbal only
    herb = [r for r in rows if r['meta']['illus'] == 'H']
    res4 = run_matrix(herb, ifeats, tfeats, lambda r: (r['meta']['lang'], r['meta']['hand']), seed=4)
    P('HERBAL n=%d max|r|95=%.3f' % (res4['n'], res4['max95']))
    for s in top(res4, ifeats, tfeats, 8): P('  ', s)
    # (c) derangement within section x language: shift image by a cyclic offset of 3 pages
    der = []
    by = {}
    for r in rows:
        by.setdefault(sec_lang(r), []).append(r)
    for g in by.values():
        k = len(g)
        for i, r in enumerate(g):
            rr = dict(r); rr['img'] = g[(i + 3) % k]['img'] if k > 4 else r['img']
            der.append(rr)
    der = [r for r in der if len(by[sec_lang(r)]) > 4]
    res5 = run_matrix(der, ifeats, tfeats, sec_lang, seed=5)
    P('DERANGED (+3 pages within section x lang) n=%d cells p<0.01: %d of %d; min pFW=%.3f' % (
        res5['n'], int((res5['P'] < 0.01).sum()), res5['P'].size, res5['Pfw'].min()))
    for s in top(res5, ifeats, tfeats, 3): P('  ', s)
    # (b) planted power curve
    for bl, bq in ((0.10, 0.15), (0.20, 0.30), (0.30, 0.45)):
        hits_l = hits_q = 0; rl = []; rq = []
        for t in range(6):
            pr = planted(rows, gc, bl, bq, seed=t)
            rp = run_matrix(pr, ifeats, tfeats, sec_lang, nperm=300, seed=100 + t)
            i1, j1 = ifeats.index('draw_edges'), tfeats.index('log_words')
            i2, j2 = ifeats.index('frac_green'), tfeats.index('w_qo')
            rl.append(rp['R'][i1, j1]); rq.append(rp['R'][i2, j2])
            hits_l += rp['Pfw'][i1, j1] < 0.05; hits_q += rp['Pfw'][i2, j2] < 0.05
        P('PLANTED beta_len=%.2f beta_q=%.2f: len r=%.2f detect(FW) %d/6 ; q r=%.2f detect(FW) %d/6' % (
            bl, bq, np.mean(rl), hits_l, np.mean(rq), hits_q))
    open(os.path.join(OUTD, 'c1_log.txt'), 'w').write('\n'.join(log) + '\n')
