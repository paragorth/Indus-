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

    def word_weights(self, Wb, mode):
        if mode == 'runs':
            pair = np.einsum('aw,bw->abw', Wb, Wb).reshape(-1, Wb.shape[1])
            return self.U @ Wb - self.B @ pair
        return self.U @ Wb

    def lines_matrix(self, pages, wt):
        """per line: total weight (rows), glyph count, and boundary positions from start / end."""
        tot, glen, bs, be, keep = [], [], [], [], []
        for pg in pages:
            med = np.median([len(l) for l in pg])
            for li, l in enumerate(pg):
                ids = [self.types[w] for w in l]
                X = wt[ids]                    # words x W
                c = np.cumsum(X, axis=0)
                tot.append(c[-1]); glen.append(sum(len(w) for w in l))
                ok = (li < len(pg) - 1) and len(l) >= 0.6 * med and len(l) >= 3
                keep.append(ok)
                if ok:
                    bs.append(c[:-1]); be.append(c[-1][None, :] - c[:-1])
        tot = np.array(tot); glen = np.array(glen, float); keep = np.array(keep)
        return tot[keep], glen[keep], np.vstack(bs), np.vstack(be)


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


def score(pages, Wb, mode, rng, corp=None):
    corp = corp or Corpus(pages, Wb_alpha[0])
    wt = corp.word_weights(Wb, mode)
    t, g, bs, be = corp.lines_matrix(pages, wt)
    rv = resvar(t, g); ss = simpson(bs); se = simpson(be)
    nrv, nss, nse = 0, 0, 0
    for k in range(K):
        q = width_reflow(pages, rng)
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


def run_search(tag, tr, te_list, seed):
    alpha = L.alphabet(tr + sum([t for _, t in te_list], []))
    rng = np.random.default_rng(seed)
    corp = Corpus(tr + sum([t for _, t in te_list], []), alpha)
    out = {}
    for mode in ('bin', 'int', 'runs'):
        Wb = weights(alpha, mode, NW, rng)
        s = score(tr, Wb, mode, random.Random(seed), corp)
        r = {}
        for stat in ('metre', 'caes', 'cad'):
            v = s[stat]; i = int(np.argmax(v))
            row = dict(max=round(float(v.max()), 4), mean=round(float(v.mean()), 4),
                       p99=round(float(np.percentile(v, 99)), 4))
            for tn, te in te_list:
                st = score(te, Wb[:, [i]], mode, random.Random(seed + 1), corp)[stat]
                row['test_' + tn] = round(float(st[0]), 4)
                # held-out reference: mean of random weights on test
                if stat == 'metre' and mode == 'bin':
                    pass
            r[stat] = row
        out[mode] = r
    print(tag, json.dumps(out), flush=True)
    return out


def main():
    t0 = time.time()
    res = {}
    zl_tr, zl_te = split('V-ZL3b'); it_tr, it_te = split('V-IT2a')
    res['ZL'] = run_search('ZL', zl_tr, [('zl', zl_te), ('it', it_te)], 1)
    res['null-markov'] = run_search('null-markov', L.markov_line_generator(zl_tr, random.Random(9)),
                                    [('zl', L.markov_line_generator(zl_te, random.Random(10)))], 1)
    res['null-reflowV'] = run_search('null-reflowV', L.reflow_page(zl_tr, random.Random(9)),
                                     [('zl', L.reflow_page(zl_te, random.Random(10)))], 1)
    A['Hildegard-justified'] = dict(pages=justified('Hildegard(prose herbal)'), kind='prose')
    A['Caesar-justified'] = dict(pages=justified('Caesar(prose)'), kind='prose')
    for name in ['Regimen(verse,rhymed)', 'Macer(verse,hexam)', 'Dante(verse,terza)', 'Litany(refrain)',
                 'Hildegard(prose herbal)', 'Caesar(prose)', 'Hildegard-justified', 'Caesar-justified', 'Dante-reflowed']:
        tr, te = split(name)
        res[name] = run_search(name, tr, [('half2', te)], 2)
    res['secs'] = round(time.time() - t0)
    json.dump(res, open(os.path.join(L.CK, 'cycle2.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
