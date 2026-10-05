"""v69 cycle 5: massive random guessing of 'slow-down' word classes.
5,000 random word classes per corpus: random sets of word types drawn by (a) frequency band x
distributional cue (page-concentrated, line-initial-leaning, paragraph-first-line-leaning, next to a
given glyph), (b) random type subsets, (c) context classes (word after / before a given type, word
repeated on the page, word whose type first appears on this page). Each class is scored by the mean
residual care (c1 index) of its tokens minus the rest, on odd-numbered pages; the top 1% re-tested
on even pages. Null: the same pipeline on care permuted within (glyph count, first, last glyph)
strata (10 null runs) -> how many held-out survivors chance gives. Latin control run identically.
Out: data/v69_ckpt/c5.json"""
import sys, os, json, collections, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v69_lib as X
import v69_c1 as C1

RNG = np.random.default_rng(694)
NC = int(os.environ.get('NCLS', 5000))


def classes(D, freq, rng):
    N = D.N
    types = sorted(set(D.word))
    tid = {w: i for i, w in enumerate(types)}
    t = np.array([tid[w] for w in D.word])
    lf = np.array([math.log(freq.get(w, 0) + 1) for w in types])
    # distributional cues per type (computed on all pages: they use no care values)
    pages_of = collections.defaultdict(set); cnt = collections.Counter(D.word)
    li0 = collections.Counter(); pf = collections.Counter()
    for i, w in enumerate(D.word):
        pages_of[w].add(D.page[i])
        li0[w] += D.k[i] == 0
        pf[w] += bool(D.R[i]['para_start'])
    conc = np.array([cnt[w] / len(pages_of[w]) for w in types])
    lini = np.array([li0[w] / cnt[w] for w in types]); pfs = np.array([pf[w] / cnt[w] for w in types])
    prev = [''] + D.word[:-1]; nxt = D.word[1:] + ['']
    gls = [D.gl(w) for w in D.word]
    glyphset = [g for g, _ in collections.Counter(x for g in gls for x in g).most_common(20)]
    # first appearance on page
    seen = set(); firstp = np.zeros(N, bool)
    for i in np.lexsort((D.k, D.li, D.page)):
        kk = (D.page[i], D.word[i])
        if kk not in seen:
            firstp[i] = True; seen.add(kk)
    out, names = [], []
    for c in range(NC):
        kind = rng.integers(0, 4)
        if kind == 0:
            lo, hi = np.sort(rng.uniform(lf.min(), lf.max() + 1e-9, 2))
            cue = [conc, lini, pfs][rng.integers(0, 3)]
            thr = np.quantile(cue, rng.uniform(0.3, 0.9))
            sel = (lf >= lo) & (lf <= hi) & (cue >= thr)
            m = sel[t]; nm = 'band_cue'
        elif kind == 1:
            sel = rng.random(len(types)) < rng.uniform(0.02, 0.3)
            m = sel[t]; nm = 'random_types'
        elif kind == 2:
            g = glyphset[rng.integers(0, len(glyphset))]
            side = rng.integers(0, 2)
            ctx = prev if side == 0 else nxt
            m = np.array([g in (ctx[i][-len(g):] if side == 0 else ctx[i][:len(g)]) for i in range(N)])
            if rng.random() < 0.5:
                m &= firstp
            nm = 'context_glyph'
        else:
            m = firstp.copy() if rng.random() < 0.5 else ~firstp
            lo, hi = np.sort(rng.uniform(lf.min(), lf.max() + 1e-9, 2))
            m &= ((lf >= lo) & (lf <= hi))[t]
            nm = 'first_on_page_band'
        if 30 <= m.sum() <= N - 30:
            out.append(m); names.append(nm)
    return np.array(out), names


def score(M, care, pages_mask):
    c = care[pages_mask]; Mm = M[:, pages_mask]
    n1 = Mm.sum(1); n0 = Mm.shape[1] - n1
    s1 = Mm @ c; s0 = c.sum() - s1
    d = s1 / np.maximum(n1, 1) - s0 / np.maximum(n0, 1)
    se = np.sqrt(1 / np.maximum(n1, 1) + 1 / np.maximum(n0, 1))
    return d / se  # approx z (care has unit variance)


def pipeline(D, M, care):
    odd = np.array([hash(f) % 2 == 0 for f in D.folio])  # fixed page halves (hash stable within a run)
    z1 = score(M.astype(float), care, odd)
    top = np.argsort(-np.abs(z1))[:max(10, len(z1) // 100)]
    z2 = score(M[top].astype(float), care, ~odd)
    surv = int(((np.sign(z2) == np.sign(z1[top])) & (np.abs(z2) > 2.5)).sum())
    return surv, top, z1, z2


def run(which, freq, tag, nnull=10):
    D = C1.Data(which)
    M, names = classes(D, freq, RNG)
    surv, top, z1, z2 = pipeline(D, M, D.care)
    nulls = []
    Sc = D.strata('content')
    for r in range(nnull):
        nulls.append(pipeline(D, M, D.perm(D.care, Sc))[0])
    best = sorted([(float(z2[j]), float(z1[top[j]]), names[top[j]], int(M[top[j]].sum())) for j in range(len(top))],
                  key=lambda x: -abs(x[0]))[:10]
    out = {'nclasses': len(M), 'survivors': surv, 'null_survivors': nulls, 'best': best,
           'kinds': collections.Counter(names[j] for j in top)}
    print(tag, {k: v for k, v in out.items()}, flush=True)
    return out


if __name__ == '__main__':
    vf, _ = X.voynich_freq(); lf = X.latin_freq()
    res = {'L': run('L', lf, 'L'), 'V': run('V', vf, 'V')}
    json.dump(res, open(os.path.join(X.CK, 'c5.json'), 'w'), indent=1, default=str)
