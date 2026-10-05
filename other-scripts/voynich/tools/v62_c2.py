"""v62 cycle 2: METRE. Under thousands of random syllable/weight assignments to glyphs, is the per-line
weight total more constant than the same words re-broken into lines of the same glyph lengths (residual
variance after regression on line glyph count)? And do word boundaries fall at fixed metrical positions
(caesura / cadence): concentration of boundary positions counted in weight units from line start and
from line end, against the same re-flow?

Weight modes: 'bin' (0/1 summed), 'int' (0..3 summed), 'runs' (number of runs of weight-1 glyphs per
word = nuclei). Search on a training half, best re-scored on held-out half and IT2a. Family-wise nulls:
the same search on Markov resynthesis with line effects, and on the Voynich re-flowed (lines destroyed).
Controls: verse (Regimen, Macer, Dante), litany, prose herbal laid out by word count, AND prose herbal
JUSTIFIED to a fixed ink width with random glyph widths (positive control for the justification
artefact: a filled line looks metrical under a width-like weight).
"""
import sys, os, json, random, pickle, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v62_lib as L

NW = int(os.environ.get('NW', 700))     # weight vectors per mode
K = int(os.environ.get('K', 8))         # re-flow draws
A = pickle.load(open(os.path.join(L.CK, 'corpora.pkl'), 'rb'))
MAXP = 64


def justified(name, seed=11):
    rng = random.Random(seed)
    words = [w for pg in A[name]['pages'] for l in pg for w in l]
    alpha = L.alphabet(A[name]['pages'])
    wid = {a: rng.uniform(0.5, 1.6) for a in alpha}
    ww = [sum(wid[s] for s in w) + 0.8 for w in words]
    target = np.mean(ww) * 8.5
    pages, cur, line, acc = [], [], [], 0.0
    for w, x in zip(words, ww):
        if line and acc + x > target:
            cur.append(line); line, acc = [], 0.0
            if len(cur) >= 24: pages.append(cur); cur = []
        line.append(w); acc += x
    return pages


def width_reflow(pages, rng):
    """re-break each page's word stream so line i gets glyph length closest to the original line i."""
    out = []
    for pg in pages:
        ws = [w for l in pg for w in l]
        tg = [sum(len(w) for w in l) for l in pg]
        # small random offset so draws differ
        start = rng.randrange(0, max(1, min(3, len(pg[0]))))
        ws = ws[start:] + ws[:start]
        res, i = [], 0
        for j, t in enumerate(tg):
            if j == len(tg) - 1:
                res.append(ws[i:]); break
            acc, line = 0, []
            while i < len(ws):
                x = len(ws[i])
                if line and abs(acc + x - t) > abs(acc - t):
                    break
                line.append(ws[i]); acc += x; i += 1
            res.append(line)
        out.append([l for l in res if l])
    return out


class Corpus:
    def __init__(self, pages, alpha):
        self.alpha = alpha; ai = {a: i for i, a in enumerate(alpha)}
        types = {}
        for pg in pages:
            for l in pg:
                for w in l:
                    if w not in types: types[w] = len(types)
        T, Aa = len(types), len(alpha)
        U = np.zeros((T, Aa)); B = np.zeros((T, Aa * Aa))
        for w, t in types.items():
            idx = [ai[s] for s in w if s in ai]
            for i in idx: U[t, i] += 1
            for a, b in zip(idx[:-1], idx[1:]): B[t, a * Aa + b] += 1
        self.types, self.U, self.B = types, U, B
        self._lay = {}

    def word_weights(self, Wb, mode):
        if mode == 'runs':
            pair = np.einsum('aw,bw->abw', Wb, Wb).reshape(-1, Wb.shape[1])
            return self.U @ Wb - self.B @ pair
        return self.U @ Wb

    def layout(self, pages):
        """precomputed flat arrays for a line partition."""
        key = id(pages)
        if key in self._lay: return self._lay[key]
        ids, ls, le, glen, keep, bpos, bline = [], [], [], [], [], [], []
        for pg in pages:
            med = np.median([len(l) for l in pg])
            for li, l in enumerate(pg):
                s0 = len(ids)
                ids += [self.types[w] for w in l]
                ok = (li < len(pg) - 1) and len(l) >= 0.6 * med and len(l) >= 3
                ls.append(s0); le.append(len(ids)); glen.append(sum(len(w) for w in l)); keep.append(ok)
                if ok:
                    for b in range(s0 + 1, len(ids)):
                        bpos.append(b); bline.append(len(ls) - 1)
        lay = dict(ids=np.array(ids), ls=np.array(ls), le=np.array(le), glen=np.array(glen, float),
                   keep=np.array(keep), bpos=np.array(bpos), bline=np.array(bline))
        self._lay[key] = (pages, lay)
        return self._lay[key]

    def lines_matrix(self, pages, wt):
        lay = self.layout(pages)[1]
        X = wt[lay['ids']]
        C = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
        tot = C[lay['le']] - C[lay['ls']]
        bs = C[lay['bpos']] - C[lay['ls'][lay['bline']]]
        be = C[lay['le'][lay['bline']]] - C[lay['bpos']]
        k = lay['keep']
        return tot[k], lay['glen'][k], bs, be


def resvar(tot, glen):
    X = np.vstack([np.ones_like(glen), glen]).T
    beta, *_ = np.linalg.lstsq(X, tot, rcond=None)
    r = tot - X @ beta
    return r.var(axis=0) + 1e-9


def simpson(pos):
    p = np.clip(np.rint(pos).astype(int), 0, MAXP)
    out = np.zeros(p.shape[1])
    for j in range(p.shape[1]):
        h = np.bincount(p[:, j], minlength=MAXP + 1).astype(float)
        out[j] = (h * h).sum() / h.sum() ** 2
    return out


_RF = {}


def reflows(pages, seed):
    key = (id(pages), seed)
    if key not in _RF:
        rng = random.Random(seed)
        _RF[key] = [width_reflow(pages, rng) for _ in range(K)]
    return _RF[key]


def score(pages, Wb, mode, rng, corp):
    wt = corp.word_weights(Wb, mode)
    t, g, bs, be = corp.lines_matrix(pages, wt)
    rv = resvar(t, g); ss = simpson(bs); se = simpson(be)
    nrv, nss, nse = 0, 0, 0
    for q in reflows(pages, 1234):
        t2, g2, bs2, be2 = corp.lines_matrix(q, wt)
        nrv += resvar(t2, g2) / K; nss += simpson(bs2) / K; nse += simpson(be2) / K
    return dict(metre=np.log(nrv / rv), caes=np.log(ss / nss), cad=np.log(se / nse))


def weights(alpha, mode, n, rng):
    Aa = len(alpha)
    if mode == 'int':
        return rng.integers(0, 4, size=(Aa, n)).astype(float)
    p = rng.uniform(0.2, 0.6, size=n)
    return (rng.random((Aa, n)) < p[None, :]).astype(float)


def split(name):
    pg = A[name]['pages']
    if A[name]['kind'] == 'voynich':
        meta = A[name]['meta']
        return ([p for p, m in zip(pg, meta) if L.folio_num(m['folio']) % 2 == 1],
                [p for p, m in zip(pg, meta) if L.folio_num(m['folio']) % 2 == 0])
    h = len(pg) // 2
    return pg[:h], pg[h:]


def climb(tr, w0, mode, stat, corp, steps=8):
    """greedy hill-climb from w0: evaluate every single-glyph change in one batch, keep the best."""
    w = w0.copy(); cur = score(tr, w[:, None], mode, None, corp)[stat][0]
    levels = [0.0, 1.0] if mode != 'int' else [0.0, 1.0, 2.0, 3.0]
    for _ in range(steps):
        cands = []
        for a in range(len(w)):
            for v in levels:
                if v != w[a]:
                    c = w.copy(); c[a] = v; cands.append(c)
        C = np.array(cands).T
        sc = score(tr, C, mode, None, corp)[stat]
        i = int(np.argmax(sc))
        if sc[i] <= cur + 1e-4: break
        w, cur = C[:, i].copy(), float(sc[i])
    return w, cur


def run_search(tag, tr, te_list, seed):
    alpha = L.alphabet(tr + sum([t for _, t in te_list], []))
    rng = np.random.default_rng(seed)
    corp = Corpus(tr + sum([t for _, t in te_list], []), alpha)
    out = {}
    for mode in ('bin', 'runs'):
        Wb = weights(alpha, mode, NW, rng)
        s = score(tr, Wb, mode, None, corp)
        r = {}
        for stat in ('metre', 'caes', 'cad'):
            v = s[stat]; i = int(np.argmax(v))
            row = dict(max=round(float(v.max()), 4), mean=round(float(v.mean()), 4),
                       p99=round(float(np.percentile(v, 99)), 4))
            for tn, te in te_list:
                st = score(te, Wb[:, [i]], mode, None, corp)[stat]
                row['test_' + tn] = round(float(st[0]), 4)
                rnd = score(te, Wb[:, :100], mode, None, corp)[stat]
                row['test_rand_mean_' + tn] = round(float(rnd.mean()), 4)
            # hill-climb the top 3 random starts on training, test the climbed weights
            cl = []
            for j in np.argsort(-v)[:2]:
                w, c = climb(tr, Wb[:, j], mode, stat, corp)
                cl.append((c, w))
            c, w = max(cl, key=lambda x: x[0])
            row['climb_train'] = round(c, 4)
            for tn, te in te_list:
                row['climb_test_' + tn] = round(float(score(te, w[:, None], mode, None, corp)[stat][0]), 4)
            r[stat] = row
        out[mode] = r
    print(tag, json.dumps(out), flush=True)
    return out


def _task(a):
    return a[0], run_search(*a)


def main():
    t0 = time.time()
    from multiprocessing import Pool
    zl_tr, zl_te = split('V-ZL3b'); it_tr, it_te = split('V-IT2a')
    A['Hildegard-justified'] = dict(pages=justified('Hildegard(prose herbal)'), kind='prose')
    A['Caesar-justified'] = dict(pages=justified('Caesar(prose)'), kind='prose')
    tasks = [('ZL', zl_tr, [('zl', zl_te), ('it', it_te)], 1),
             ('null-markov', L.markov_line_generator(zl_tr, random.Random(9)), [('zl', L.markov_line_generator(zl_te, random.Random(10)))], 1),
             ('null-reflowV', L.reflow_page(zl_tr, random.Random(9)), [('zl', L.reflow_page(zl_te, random.Random(10)))], 1)]
    for name in ['Dante(verse,terza)', 'Macer(verse,hexam)', 'Regimen(verse,rhymed)', 'Litany(refrain)',
                 'Hildegard(prose herbal)', 'Caesar(prose)', 'Hildegard-justified', 'Caesar-justified', 'Dante-reflowed']:
        tr, te = split(name)
        tasks.append((name, tr, [('half2', te)], 2))
    res = {}
    with Pool(2) as pool:
        for tag, r in pool.imap_unordered(_task, tasks):
            res[tag] = r
            json.dump(res, open(os.path.join(L.CK, 'cycle2.json'), 'w'), indent=1)
    res['secs'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(L.CK, 'cycle2.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
