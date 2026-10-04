"""v11 cycle 3: does the line-initial chain (FINDINGS section 10) correspond to fixed starting regions of the map?
Embedding = dist2 fitted on train pages (cycle-1 checkpoints; fitted here for the planted start-region control).
On held-out pages, paragraphs; pairs (line n, line n+1) for lines 2..n as in v10 (paragraph-first line kept fixed).
 S1 region chain: k-means regions (k=4, 6) of the map; MI(region of start word n, region of start word n+1);
    null = lines 2..n permuted within paragraph (500x).
 S2 distance between consecutive start words; S3 distance end-of-line n -> start-of-line n+1 (walk continues?).
 S4 glyph chain MI (first glyph) on the same pairs and null, for comparison.
 S5 who carries whom: region MI after re-dealing start words among lines with the same first glyph (keeps the glyph
    chain, randomises the word/region) and glyph MI after re-dealing among lines with the same region.
 S6 start concentration: JS divergence of region distribution of line-start tokens vs all tokens.
Controls: Planted-gridstart-30 (starts cycle through quadrants: must be found), Planted-grid-30 (no start rule),
Latin-Isidore poured into the layout."""
import sys, os, time, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v11_lib as L, vlib
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool

CORPORA = ['Planted-gridstart-30', 'Planted-grid-30', 'Voynich-ZL', 'Voynich-IT', 'Latin-Isidore', 'Voynich-A', 'Voynich-B']
NPERM = 500


def kmeans(X, w, k, seed=0, it=100):
    rng = np.random.default_rng(seed); best = None
    for r in range(10):
        Cn = X[rng.choice(len(X), k, replace=False, p=w / w.sum())]
        for _ in range(it):
            lab = ((X[:, None] - Cn[None]) ** 2).sum(-1).argmin(1)
            Cn = np.array([np.average(X[lab == j], 0, w[lab == j]) if (lab == j).any() else Cn[j] for j in range(k)])
        inert = (w * ((X - Cn[lab]) ** 2).sum(1)).sum()
        if best is None or inert < best[0]: best = (inert, lab)
    return best[1]


def paragraphs(lines):
    paras, cur, fol = [], None, None
    for l in lines:
        if l['para_start'] or l['folio'] != fol or cur is None:
            if cur and len(cur) >= 2: paras.append(cur)
            cur = []
        fol = l['folio']; cur.append(l)
    if cur and len(cur) >= 2: paras.append(cur)
    return paras


def mi(pairs):
    return L_mi(pairs)


def L_mi(pairs):
    n = len(pairs)
    if not n: return 0.0
    a = Counter(x for x, _ in pairs); b = Counter(y for _, y in pairs); ab = Counter(pairs)
    return sum(v / n * math.log2(v * n / (a[x] * b[y])) for (x, y), v in ab.items())


def stats(paras, idx, X, reg):
    """returns dict of statistics for one arrangement of paragraphs."""
    rp, gp, dS, dE = [], [], [], []
    for p in paras:
        for l1, l2 in zip(p, p[1:]):
            if l1 is p[0]: pass
            s1, s2 = l1['words'][0], l2['words'][0]
            g1, g2 = vlib.glyphs(s1)[0], vlib.glyphs(s2)[0]
            gp.append((g1, g2))
            if s1 in idx and s2 in idx:
                rp.append((reg[idx[s1]], reg[idx[s2]])); dS.append(((X[idx[s1]] - X[idx[s2]]) ** 2).sum())
            e1 = [w for w in l1['words'] if w in idx]
            if e1 and s2 in idx: dE.append(((X[idx[e1[-1]]] - X[idx[s2]]) ** 2).sum())
    return {'region_mi': L_mi(rp), 'glyph_mi': L_mi(gp), 'start_dist': float(np.mean(dS)), 'end_start_dist': float(np.mean(dE)), 'n_region_pairs': len(rp)}


def permute(paras, rng):
    out = []
    for p in paras:
        body = p[1:]; out.append([p[0]] + rng.sample(body, len(body)))
    return out


def redeal(paras, key, rng):
    """re-deal the first word of lines 2..n among all lines (test set) with the same key(first word)."""
    pool = defaultdict(list)
    for p in paras:
        for l in p[1:]: pool[key(l['words'][0])].append(l['words'][0])
    for v in pool.values(): rng.shuffle(v)
    out = []
    for p in paras:
        q = [p[0]]
        for l in p[1:]:
            q.append(dict(l, words=[pool[key(l['words'][0])].pop()] + l['words'][1:]))
        out.append(q)
    return out


def z(o, xs):
    xs = np.array(xs); return float((o - xs.mean()) / (xs.std() + 1e-12)), float(xs.mean())


def job(args):
    name, k = args
    ck = f'c3_{name}_k{k}.json'
    r = L.jload(ck)
    if r: return r
    lines, truth = L.corpus(name)
    out = {'corpus': name, 'k': k, 'folds': []}
    for fold in (0, 1):
        c1 = L.jload(f'c1_{name}_{fold}.json')
        tr, te = L.split(lines, fold)
        if c1:
            voc = c1['vocab']; X = np.array(c1['emb']['2'])
        else:
            voc = L.vocab(lines, 400); idx = {w: i for i, w in enumerate(voc)}
            C = L.bigram_counts(tr, idx); X = L.best_fit(C, 'dist', 2, 2, iters=800)['P'][0]
        idx = {w: i for i, w in enumerate(voc)}
        freq = Counter(w for l in tr for w in l['words']); wts = np.array([freq[w] + 1.0 for w in voc])
        reg = kmeans(X, wts, k, seed=fold)
        paras = paragraphs(te); rng = random.Random(100 + fold)
        obs = stats(paras, idx, X, reg)
        null = [stats(permute(paras, rng), idx, X, reg) for _ in range(NPERM)]
        res = {'fold': fold, 'obs': obs}
        for key in ('region_mi', 'glyph_mi', 'start_dist', 'end_start_dist'):
            res[key + '_z'], res[key + '_null'] = z(obs[key], [n[key] for n in null])
        # S5: re-deal (keeps glyph chain, randomises region) and (keeps region, randomises glyph)
        gkey = lambda w: vlib.glyphs(w)[0]
        rkey = lambda w: ('R', reg[idx[w]]) if w in idx else ('W', w)
        rd_g = [stats(redeal(paras, gkey, rng), idx, X, reg)['region_mi'] for _ in range(100)]
        rd_r = [stats(redeal(paras, rkey, rng), idx, X, reg)['glyph_mi'] for _ in range(100)]
        res['region_mi_keep_glyph'] = float(np.mean(rd_g)); res['glyph_mi_keep_region'] = float(np.mean(rd_r))
        # S6 concentration
        allc = Counter(reg[idx[w]] for l in te for w in l['words'] if w in idx)
        stc = Counter(reg[idx[l['words'][0]]] for p in paras for l in p[1:] if l['words'][0] in idx)
        pa = np.array([allc[j] for j in range(k)], float); pa /= pa.sum()
        ps = np.array([stc[j] for j in range(k)], float); ps /= ps.sum(); m = (pa + ps) / 2
        js = 0.5 * sum(x * math.log2(x / y) for x, y in zip(pa, m) if x) + 0.5 * sum(x * math.log2(x / y) for x, y in zip(ps, m) if x)
        res['start_region_JS'] = js; res['start_region_dist'] = ps.round(3).tolist(); res['all_region_dist'] = pa.round(3).tolist()
        out['folds'].append(res)
    L.jdump(out, ck)
    return out


if __name__ == '__main__':
    jobs = [(c, k) for c in CORPORA for k in (4, 6)]
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            for f in r['folds']:
                o = f['obs']
                print(r['corpus'], 'k', r['k'], 'fold', f['fold'], 'regionMI %.3f (null %.3f z %.1f; keep-glyph %.3f)' % (o['region_mi'], f['region_mi_null'], f['region_mi_z'], f['region_mi_keep_glyph']),
                      'glyphMI %.3f (null %.3f z %.1f; keep-region %.3f)' % (o['glyph_mi'], f['glyph_mi_null'], f['glyph_mi_z'], f['glyph_mi_keep_region']),
                      'startdist z %.1f endstart z %.1f JS %.3f n %d' % (f['start_dist_z'], f['end_start_dist_z'], f['start_region_JS'], o['n_region_pairs']), flush=True)
