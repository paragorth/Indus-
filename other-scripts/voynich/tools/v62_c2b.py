"""v62 cycle 2b: is the 'caesura / cadence' signal of cycle 2 metre, or line-edge and position effects?

Cycle 2 found that word boundaries in Voynich lines fall at concentrated weight positions (from line
start and from line end) far more than in the same words re-flowed, more than in Dante or Macer. The
re-flow null destroys the known special line-initial and line-final words. Two sharper nulls here:
  N1 interior shuffle: keep each line's first and last word, shuffle the words between them;
  N2 slot swap: for lines with the same word count n, the word at slot j is exchanged with the slot-j
     word of another such line (pooled over the corpus). Keeps every slot's word distribution
     (edges, any gradient along the line) and destroys only co-ordination INSIDE a line, which is what
     a metre (a fixed sum, a caesura at a fixed beat) needs.
Statistic: log(Simpson concentration of interior boundary positions in weight units, real / null),
boundaries 2..n-2 only (the first and last are invariant under N1). Also the line weight total
residual variance (metre) under N2. Random weights (bin, runs) + hill-climb on training half, tested on
held-out half and IT2a; Markov generator with planted edges as a further null; verse and prose
controls.
"""
import sys, os, json, random, pickle, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v62_lib as L
import v62_c2 as C2

NW = int(os.environ.get('NW', 400)); K = int(os.environ.get('K', 6))
A = C2.A


def interior_shuffle(pages, rng):
    out = []
    for pg in pages:
        q = []
        for l in pg:
            if len(l) > 3:
                mid = l[1:-1]; rng.shuffle(mid); q.append([l[0]] + mid + [l[-1]])
            else:
                q.append(list(l))
        out.append(q)
    return out


def slot_swap(pages, rng):
    slots = {}
    for pi, pg in enumerate(pages):
        for li, l in enumerate(pg):
            for j in range(len(l)):
                slots.setdefault((len(l), j), []).append((pi, li))
    out = [[list(l) for l in pg] for pg in pages]
    for (n, j), locs in slots.items():
        perm = locs[:]; rng.shuffle(perm)
        for (pi, li), (pj, lj) in zip(locs, perm):
            out[pi][li][j] = pages[pj][lj][j]
    return out


class Lay:
    """boundary positions 2..n-2 and line totals for a partition, via flat cumsums."""
    def __init__(self, corp, pages):
        ids, ls, le, bpos, bline, glen, keep = [], [], [], [], [], [], []
        for pg in pages:
            med = np.median([len(l) for l in pg])
            for li, l in enumerate(pg):
                s0 = len(ids); ids += [corp.types[w] for w in l]
                ok = (li < len(pg) - 1) and len(l) >= 0.6 * med and len(l) >= 4
                ls.append(s0); le.append(len(ids)); glen.append(sum(len(w) for w in l)); keep.append(ok)
                if ok:
                    for b in range(s0 + 2, len(ids) - 1):
                        bpos.append(b); bline.append(len(ls) - 1)
        self.ids = np.array(ids); self.ls = np.array(ls); self.le = np.array(le)
        self.bpos = np.array(bpos); self.bline = np.array(bline); self.glen = np.array(glen, float)
        self.keep = np.array(keep)

    def stats(self, wt):
        X = wt[self.ids]
        Cm = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
        bs = Cm[self.bpos] - Cm[self.ls[self.bline]]
        be = Cm[self.le[self.bline]] - Cm[self.bpos]
        tot = (Cm[self.le] - Cm[self.ls])[self.keep]
        return C2.simpson(bs), C2.simpson(be), C2.resvar(tot, self.glen[self.keep])


def make_scorer(corp, pages, seed):
    real = Lay(corp, pages)
    n1 = [Lay(corp, interior_shuffle(pages, random.Random(seed + k))) for k in range(K)]
    n2 = [Lay(corp, slot_swap(pages, random.Random(seed + 100 + k))) for k in range(K)]
    def f(W, mode):
        wt = corp.word_weights(W, mode)
        ss, se, rv = real.stats(wt)
        a1 = [x.stats(wt) for x in n1]; a2 = [x.stats(wt) for x in n2]
        m = lambda arr, i: np.mean([x[i] for x in arr], axis=0)
        return dict(caes_N1=np.log(ss / m(a1, 0)), cad_N1=np.log(se / m(a1, 1)),
                    caes_N2=np.log(ss / m(a2, 0)), cad_N2=np.log(se / m(a2, 1)), metre_N2=np.log(m(a2, 2) / rv))
    return f


def climb(f, w0, mode, stat, steps=8):
    w = w0.copy(); cur = f(w[:, None], mode)[stat][0]
    for _ in range(steps):
        cands = []
        for a in range(len(w)):
            c = w.copy(); c[a] = 1.0 - c[a]; cands.append(c)
        Cc = np.array(cands).T; sc = f(Cc, mode)[stat]; i = int(np.argmax(sc))
        if sc[i] <= cur + 1e-4: break
        w, cur = Cc[:, i].copy(), float(sc[i])
    return w, cur


def run(args):
    tag, tr, tests, seed = args
    allp = tr + sum([t for _, t in tests], [])
    corp = C2.Corpus(allp, L.alphabet(allp))
    ftr = make_scorer(corp, tr, seed)
    fte = {tn: make_scorer(corp, te, seed + 7) for tn, te in tests}
    rng = np.random.default_rng(seed)
    out = {}
    for mode in ('bin', 'runs'):
        W = C2.weights(corp.alpha, mode, NW, rng)
        s = ftr(W, mode)
        st_te = {tn: g(W[:, :150], mode) for tn, g in fte.items()}
        for stat in ('caes_N1', 'cad_N1', 'caes_N2', 'cad_N2', 'metre_N2'):
            v = s[stat]
            row = dict(train_max=round(float(v.max()), 4), train_mean=round(float(v.mean()), 4))
            for tn in fte:
                row['rand_mean_' + tn] = round(float(st_te[tn][stat].mean()), 4)
            best = max((climb(ftr, W[:, j], mode, stat) for j in np.argsort(-v)[:2]), key=lambda x: x[1])
            row['climb_train'] = round(best[1], 4)
            for tn, g in fte.items():
                row['climb_test_' + tn] = round(float(g(best[0][:, None], mode)[stat][0]), 4)
            out[f'{mode}:{stat}'] = row
    print(tag, json.dumps(out), flush=True)
    return tag, out


def main():
    zl_tr, zl_te = C2.split('V-ZL3b'); it_tr, it_te = C2.split('V-IT2a')
    tasks = [('ZL', zl_tr, [('zl', zl_te), ('it', it_te)], 1),
             ('null-markov', L.markov_line_generator(zl_tr, random.Random(9)), [('zl', L.markov_line_generator(zl_te, random.Random(10)))], 1)]
    for name in ['Dante(verse,terza)', 'Macer(verse,hexam)', 'Regimen(verse,rhymed)', 'Hildegard(prose herbal)', 'Caesar(prose)']:
        tr, te = C2.split(name); tasks.append((name, tr, [('half2', te)], 2))
    # Voynich by Currier language (train odd / test even folios within the language)
    for lang in ('A', 'B'):
        pg, meta = A['V-ZL3b']['pages'], A['V-ZL3b']['meta']
        tr = [p for p, m in zip(pg, meta) if m['lang'] == lang and L.folio_num(m['folio']) % 2 == 1]
        te = [p for p, m in zip(pg, meta) if m['lang'] == lang and L.folio_num(m['folio']) % 2 == 0]
        tasks.append((f'ZL-lang{lang}', tr, [('zl', te)], 3))
    res = {}
    with Pool(2) as pool:
        for tag, r in pool.imap_unordered(run, tasks):
            res[tag] = r
            json.dump(res, open(os.path.join(L.CK, 'cycle2b.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
