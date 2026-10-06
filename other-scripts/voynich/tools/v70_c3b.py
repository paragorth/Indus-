"""v70 cycle 3b: what the ink says about EVA's segmentation, and whether an ink-informed alphabet tracks
sections better than EVA.

(a) Ink join map. At theta 0 (units = blank-column-separated ink), each EVA glyph is placed in the word by
    least-squares glyph widths (as cycle 2) and assigned to the ink unit holding its centre. For every EVA
    bigram inside words: joined rate = share of tokens whose two glyphs sit in one ink unit. Expected rate from
    a logistic model with a right-edge term for the first glyph and a left-edge term for the second
    (pen-lift propensities); excess = observed - expected (z by binomial SD). Big positive excess = an ink
    ligature EVA splits; big negative = a boundary EVA hides.
    The same estimator on SV (synthetic hand: random touching only) gives the false-positive level.
(b) Ink-informed EVA: merge the top-N excess bigrams (N = 3, 6, 10) into single units over the whole ZL3b
    corpus (paragraph text, all sections). Held-out line -> section (ZL illustration code) and -> page with
    naive Bayes on unit unigrams+bigrams, even/odd lines. Null: merge N random bigrams of matched frequency
    (40 draws). Also h2/h1 and mean word length.
(c) Width-only null for cycle-2 page tracking: units labelled only by width quantile (no shape).
(d) Anatomy at theta 4, K 40: NMI of ink clusters vs estimated EVA spans in V, and in SV both vs estimated
    spans and vs exact truth (calibrates the estimator).
"""
import sys, os, json, collections, random, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v70_lib import *
from v70_c1 import embed, words_from, nearest
from v70_c2 import lines_of, nb_acc, Q20
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score as NMI
ARR = os.path.join(SCR, 'v70', 'arr')


def glyph_lists(name, D):
    if name == 'V':
        return [vglyphs(re.sub(r'[^a-z]', '', r['word'])) for r in D['recs']]
    return [list(r['truth']) for r in D['recs']]


import re


def glyph_units(name, th):
    """per word: list of unit index (position) for each EVA glyph, from least-squares widths."""
    D = json.load(open(os.path.join(ARR, name + '.json')))
    U = D['units'][str(th)]
    gl = glyph_lists(name, D)
    spans = collections.defaultdict(list)
    for i, u in enumerate(U):
        spans[u[0]].append((u[2], u[1], i))
    inv = sorted(set(g for w in gl for g in w))
    rows, yv = [], []
    for wi, sp in spans.items():
        if gl[wi]:
            c = collections.Counter(gl[wi]); rows.append([c[g] for g in inv]); yv.append(sum(x[1] for x in sp))
    gw = dict(zip(inv, np.clip(np.linalg.lstsq(np.array(rows, float), np.array(yv), rcond=None)[0], 0.1, None)))
    G2U = {}
    for wi, sp in spans.items():
        g = gl[wi]
        if not g:
            continue
        sp = sorted(sp); tot = sum(x[1] for x in sp)
        exp = np.array([gw[x] for x in g]); cen = (np.cumsum(exp) - exp / 2) / exp.sum() * tot
        edges = np.cumsum([0] + [x[1] for x in sp])
        G2U[wi] = [int(min(len(sp) - 1, np.searchsorted(edges, c, side='right') - 1)) for c in cen]
    return D, U, gl, G2U, spans


def join_map(name):
    D, U, gl, G2U, _ = glyph_units(name, 0)
    obs = collections.Counter(); tot = collections.Counter()
    X, y, keys = [], [], []
    for wi, m in G2U.items():
        g = gl[wi]
        for j in range(len(g) - 1):
            b = (g[j], g[j + 1]); j1 = m[j] == m[j + 1]
            tot[b] += 1; obs[b] += j1
            X.append((g[j], g[j + 1])); y.append(j1)
    inv = sorted(set(a for a, _ in X) | set(b for _, b in X)); ix = {g: i for i, g in enumerate(inv)}
    M = np.zeros((len(X), 2 * len(inv)))
    for r, (a, b) in enumerate(X):
        M[r, ix[a]] = 1; M[r, len(inv) + ix[b]] = 1
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(M, y)
    p = clf.predict_proba(M)[:, 1]
    expd = collections.defaultdict(float)
    for (a, b), pp in zip(X, p):
        expd[(a, b)] += pp
    res = []
    for b, n in tot.items():
        if n < 40:
            continue
        e = expd[b] / n; o = obs[b] / n
        z = (o - e) / math.sqrt(max(1e-6, e * (1 - e) / n))
        res.append({'bigram': '+'.join(b), 'n': n, 'obs': round(o, 3), 'exp': round(e, 3), 'z': round(z, 2)})
    res.sort(key=lambda r: -r['z'])
    return res, float(np.mean(y))


def zl_lines():
    L = json.load(open(os.path.join(DER, 'ZL3b_lines.json')))
    out = []
    for r in L:
        if r['ltype'] != 'P':
            continue
        ws = [vglyphs(re.sub(r'[^a-z]', '', w)) for w in r['words']]
        ws = [w for w in ws if w]
        if ws:
            out.append({'folio': r['folio'], 'sec': r['illus'], 'n': r['n'], 'words': ws})
    return out


def merge_words(ws, pairs):
    ps = set(pairs); out = []
    for w in ws:
        o, j = [], 0
        while j < len(w):
            if j + 1 < len(w) and (w[j], w[j + 1]) in ps:
                o.append(w[j] + '|' + w[j + 1]); j += 2
            else:
                o.append(w[j]); j += 1
        out.append(tuple(o))
    return out


def nb_fast(L, alpha=0.5):
    from scipy.sparse import csr_matrix
    keys = sorted(L, key=str)
    fidx = {}; rows, cols, vals = [], [], []
    for r, k in enumerate(keys):
        c = collections.Counter()
        for w in L[k]:
            c.update(('u', u) for u in w); c.update(('b', a, b) for a, b in zip(w, w[1:]))
        for f, n in c.items():
            rows.append(r); cols.append(fidx.setdefault(f, len(fidx))); vals.append(n)
    X = csr_matrix((vals, (rows, cols)), shape=(len(keys), len(fidx)), dtype=np.float64)
    cls = [k[0] for k in keys]; par = np.array([k[1] % 2 for k in keys])
    labs = sorted(set(cls), key=str); li = {c: i for i, c in enumerate(labs)}; y = np.array([li[c] for c in cls])
    accs = []
    for p in (0, 1):
        tr = par == p; te = ~tr
        C = np.zeros((len(labs), X.shape[1]))
        for c in range(len(labs)):
            m = tr & (y == c)
            if m.any():
                C[c] = np.asarray(X[m].sum(0)).ravel()
        V = (C.sum(0) > 0).sum()
        logp = np.log(C + alpha) - np.log(C.sum(1, keepdims=True) + alpha * V)
        present = np.array([(tr & (y == c)).any() for c in range(len(labs))])
        sc = X[te] @ logp.T
        sc[:, ~present] = -np.inf
        accs.append(float(np.mean(np.argmax(sc, 1) == y[te])))
    return float(np.mean(accs))


def track(lines, key):
    L = collections.defaultdict(list)
    for i, r in enumerate(lines):
        for w in r['words']:
            L[(r[key], i)].append(w)
    return nb_fast(L)


def ivals(lines, pairs):
    ml = [dict(r, words=merge_words(r['words'], pairs)) for r in lines]
    ws = [w for r in ml for w in r['words']]
    b = battery(ws, ntok=10 ** 9)
    fc = collections.Counter(r['folio'] for r in ml)
    return {'sec': track(ml, 'sec'), 'page': track([r for r in ml if fc[r['folio']] >= 20], 'folio'), 'h2r': b['h2r'], 'Lm': b['Lm'], 'keff': b['keff']}


def main():
    out = {}
    jm, base = join_map('V'); jms, bases = join_map('SV')
    out['join_V'] = jm[:15] + jm[-10:]; out['join_SV'] = jms[:8] + jms[-5:]
    out['base_join'] = {'V': base, 'SV': bases}
    zsv = sorted(abs(r['z']) for r in jms)
    out['SV_absz_95'] = zsv[int(0.95 * len(zsv))] if zsv else None
    print('join V top', jm[:10], '\nbottom', jm[-6:], '\nSV top', jms[:5], 'SV |z| 95%', out['SV_absz_95'], flush=True)
    # (b) ink-informed EVA over the whole corpus; page task restricted to pages with >= 20 lines
    lines = zl_lines()
    fc = collections.Counter(r['folio'] for r in lines)
    big = [r for r in lines if fc[r['folio']] >= 20]
    bc = collections.Counter(b for r in lines for w in r['words'] for b in zip(w, w[1:]))
    out['eva'] = ivals(lines, [])
    print('EVA', out['eva'], flush=True)
    rng = random.Random(0)
    pos = [tuple(r['bigram'].split('+')) for r in jm if r['z'] > 0]
    allb = [b for b, n in bc.items() if n >= 40]
    for N in (3, 6, 10):
        top = pos[:N]
        out[f'ink_top{N}'] = dict(ivals(lines, top), pairs=['+'.join(p) for p in top])
        nulls = []
        for d in range(40):
            # frequency-matched random pairs: for each top pair pick a random bigram within +-30% frequency
            pick = []
            for p in top:
                cand = [b for b in allb if 0.7 * bc[p] <= bc[b] <= 1.3 * bc[p] and b not in pick and b not in top] or allb
                pick.append(rng.choice(cand))
            nulls.append(ivals(lines, pick))
        for k in ('sec', 'page', 'h2r', 'Lm'):
            arr = np.array([x[k] for x in nulls])
            out[f'ink_top{N}'][k + '_null'] = [float(arr.mean()), float(arr.std())]
            out[f'ink_top{N}'][k + '_z'] = float((out[f'ink_top{N}'][k] - arr.mean()) / (arr.std() + 1e-9))
        print(N, out[f'ink_top{N}'], flush=True)
    # (c) width-only page tracking null (V, Q20 pages)
    for th in (0, 2.5, 4):
        D = json.load(open(os.path.join(ARR, 'V.json'))); U = D['units'][str(th)]
        keep = {r['pg'] for r in D['recs'] if r['folio'] in Q20}
        wd = np.array([u[1] for u in U])
        for K in (10, 20, 40):
            lab = np.searchsorted(np.quantile(wd, np.linspace(0, 1, K + 1)[1:-1]), wd)
            W = words_from(lab, U, len(D['recs']))
            out[f'width_only_t{th}_K{K}'] = nb_acc(lines_of(D, W, keep))
        print('width-only', th, [out[f'width_only_t{th}_K{K}'] for K in (10, 20, 40)], flush=True)
    # (d) anatomy at theta 4, K 40
    for name in ('V', 'SV'):
        D, U, gl, G2U, spans = glyph_units(name, 4)
        E, _, _ = embed(name, 4)
        lab = MiniBatchKMeans(40, random_state=0, n_init=1, batch_size=4096).fit_predict(E)
        est = [''] * len(U)
        for wi, m in G2U.items():
            sp = sorted(spans[wi])
            for pos_, (_, _, ui) in enumerate(sp):
                est[ui] = '+'.join(g for g, mm in zip(gl[wi], m) if mm == pos_) or '0'
        ok = [i for i, t in enumerate(est) if t]
        r = {'nmi_est': float(NMI([est[i] for i in ok], lab[ok]))}
        if name == 'SV':
            tr = [u[3] for u in U]; ok2 = [i for i, t in enumerate(tr) if t]
            r['nmi_truth'] = float(NMI([tr[i] for i in ok2], lab[ok2]))
        # per-cluster majority EVA string and purity, sorted by size
        cl = []
        for c in range(40):
            idx = [i for i in ok if lab[i] == c]
            cn = collections.Counter(est[i] for i in idx)
            if idx:
                t, n = cn.most_common(1)[0]; cl.append((len(idx), t, round(n / len(idx), 2), cn.most_common(3)))
        r['clusters'] = sorted(cl, reverse=True)
        out[f'anat_{name}'] = r
        print(name, 'anatomy', r['nmi_est'], r.get('nmi_truth'), flush=True)
    json.dump(out, open(os.path.join(CK, 'c3b.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
