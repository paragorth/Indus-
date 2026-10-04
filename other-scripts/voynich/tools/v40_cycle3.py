"""v40 cycle 3: THE PEN CHOOSES THE TWIN? Ink load as a predictor of the twin choice.
For the 25 v18 image pages (11,599 words with ink measures), each twin slot gets
  load = mean content-residualised darkness ('top', darkest 30% ink) of the 2 previous words in the line,
  next = same for the 2 following words, own = residual of the word itself, and 'since' = words since the
  last detected dip (top 12% residual).
Base model (SCRIBE + CTX + FRAME) is trained on the OTHER ZL pages and predicts these pages.
Gain of each ink feature: offset logistic, page-grouped 5-fold, bits/token; null = 2,000 permutations of the
feature among twin slots of the same line (keeps line-level ink, breaks within-line pen state).
PLANTED: y resampled from the base model + 'heavy pen (load above median) -> B member with prob s'."""
import sys, os, json
import numpy as np
from scipy.sparse import hstack
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v40_lib import *
from v40_lib import _fit_offset, _ll
from v40_cycle1 import vpairs

W = json.load(open(os.path.join(DATA, 'derived', 'v18_words.json')))
pairs = vpairs()


def residual_dark():
    rows, ys = [], []
    allw = [w for P in W for w in P['words']]
    inv = sorted(set(c for w in allw for c in vglyphs(w['word'])))
    gi = {c: i for i, c in enumerate(inv)}
    X = np.zeros((len(allw), len(inv) + 2))
    y = np.full(len(allw), np.nan)
    for i, w in enumerate(allw):
        g = vglyphs(w['word'])
        for c in g:
            X[i, gi[c]] += 1.0 / max(1, len(g))
        X[i, -2] = len(g); X[i, -1] = 1
        if w['m']:
            y[i] = w['m']['top']
    ok = ~np.isnan(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    r = np.full(len(allw), np.nan); r[ok] = y[ok] - X[ok] @ beta
    # remove line means? no: keep, the null is within-line
    return allw, r


def main():
    log = open(os.path.join(CKPT, 'cycle3.log'), 'a')
    rng = np.random.default_rng(403)
    allw, r = residual_dark()
    folios = set(P['folio'] for P in W)
    # base model on the other pages
    pages = load_voynich('ZL3b')
    toks_all = extract(pages, pairs, V_TALL)
    out = {}
    # map (folio, line n, word k) for v18 words
    lines = defaultdict(list)
    for i, w in enumerate(allw):
        lines[(w['folio'], w['li'])].append(i)
    thr = np.nanquantile(r, 0.88)
    feats = {}
    tokw = []  # (pair, y, frame-token-dict, features)
    for (f, li), idx in lines.items():
        idx = sorted(idx, key=lambda i: allw[i]['k'])
        since = 99
        for pos, i in enumerate(idx):
            w = allw[i]
            prev = [r[idx[j]] for j in range(max(0, pos - 2), pos) if not np.isnan(r[idx[j]])]
            nxt = [r[idx[j]] for j in range(pos + 1, min(len(idx), pos + 3)) if not np.isnan(r[idx[j]])]
            g = vglyphs(w['word'])
            for j, c in enumerate(g):
                for pname, (A, B, maskc) in pairs.items():
                    if c in A or c in B:
                        fr = ''.join(g[:j]) + maskc(c) + ''.join(g[j + 1:])
                        tokw.append(dict(pair=pname, y=int(c in B), frame=pname + ':' + fr,
                                         prev=(g[j - 1] if j else '#'), nxt=(g[j + 1] if j + 1 < len(g) else '#'),
                                         jw=j, wl=len(g), hand='?', sect='?', lang='?', line=(f, li), page=f,
                                         load=np.mean(prev) if prev else np.nan, nextl=np.mean(nxt) if nxt else np.nan,
                                         own=r[i], since=since))
            since = 0 if (not np.isnan(r[i]) and r[i] > thr) else since + 1
    meta = {P['page']: P for P in pages}
    for t in tokw:
        P = meta.get(t['page'])
        if P:
            t['hand'], t['sect'], t['lang'] = P['hand'], P['sect'], P['lang']
    for pname in pairs:
        tr = [t for t in toks_all if t['pair'] == pname and pages[t['page']]['page'] not in folios]
        te = [t for t in tokw if t['pair'] == pname]
        if len(te) < 150:
            continue
        cols_tr = [family_cols(tr, f) for f in ('SCRIBE', 'CTX', 'FRAME')]
        cols_te = [family_cols(te, f) for f in ('SCRIBE', 'CTX', 'FRAME')]
        mats = []
        for a, b in zip(cols_tr, cols_te):
            M = onehot([ca + cb for ca, cb in zip(a, b)], min_count=3)
            mats.append(M)
        X = hstack(mats).tocsr()
        ntr = len(tr)
        from sklearn.linear_model import LogisticRegression
        ytr = np.array([t['y'] for t in tr]); y = np.array([t['y'] for t in te])
        m = LogisticRegression(C=1.0, max_iter=400, solver='liblinear').fit(X[:ntr], ytr)
        p0 = np.clip(m.predict_proba(X[ntr:])[:, 1], 1e-4, 1 - 1e-4)
        H0 = bits(y, np.full(len(y), y.mean())); Hb = bits(y, p0)
        pg = np.array([hash(t['page']) % 1000003 for t in te]); ug = np.unique(pg)
        fold = {g: i % 5 for i, g in enumerate(rng.permutation(ug))}
        f = np.array([fold[g] for g in pg])
        lines_te = defaultdict(list)
        for i, t in enumerate(te):
            lines_te[t['line']].append(i)
        res = dict(n=len(y), rateB=float(y.mean()), H0=H0, Hbase=Hb)
        for fn in ('load', 'nextl', 'own', 'since'):
            x = np.array([t[fn] for t in te], float)
            if fn == 'since':
                x = np.minimum(x, 30)
            ok = ~np.isnan(x)
            xz = np.where(ok, (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9), 0.0)
            obs = offset_gain(y, p0, xz, f)
            null = []
            for _ in range(500):
                x2 = xz.copy()
                for idx in lines_te.values():
                    idx = np.array(idx); x2[idx] = xz[rng.permutation(idx)]
                null.append(offset_gain(y, p0, x2, f))
            null = np.array(null)
            # sign of the effect
            a, b = _fit_offset(y, np.log(p0 / (1 - p0)), xz)
            res[fn] = dict(gain=obs, b=b, p=float((null >= obs).mean()), z=float((obs - null.mean()) / (null.std() + 1e-12)))
        # PLANTED on 'load'
        x = np.array([t['load'] for t in te], float); med = np.nanmedian(x)
        xz = np.where(np.isnan(x), 0.0, (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9))
        pl = {}
        for s in (0.1, 0.2, 0.4):
            hits = []
            for rep in range(3):
                ys = (rng.random(len(y)) < p0).astype(int)
                ys[(x > med) & (rng.random(len(y)) < s)] = 1
                obs = offset_gain(ys, p0, xz, f)
                null = []
                for _ in range(100):
                    x2 = xz.copy()
                    for idx in lines_te.values():
                        idx = np.array(idx); x2[idx] = xz[rng.permutation(idx)]
                    null.append(offset_gain(ys, p0, x2, f))
                hits.append(float((np.array(null) >= obs).mean()))
            pl[s] = hits
        res['plant_p'] = pl
        out[pname] = res
        msg = '%s n=%d B=%.2f H0 %.3f base %.3f | ' % (pname, len(y), y.mean(), H0, Hb) + ' '.join(
            '%s gain %.4f b %+.2f p %.3f' % (k, res[k]['gain'], res[k]['b'], res[k]['p']) for k in ('load', 'nextl', 'own', 'since')) + \
            ' | planted p (s .1/.2/.4): ' + str(pl)
        print(msg, flush=True); log.write(msg + '\n'); log.flush()
    json.dump(out, open(os.path.join(CKPT, 'cycle3.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
