"""v96 cycle 1 (N1): DOES THE PAGE'S SPELLING REACH ITS COMMONEST WORDS?
W1 (topic): the page's content changes WHICH rare words it uses; its commonest words (function words, formula words)
are spelled the same on every page, so they carry little of the page's spelling profile.
W2 (spelling key): the key respells every word, so the commonest words carry the page's profile as much as the rare ones.
Per page, two equal-size token samples, one from the commonest types (C) and one from rare types (count lo-hi, R), are
each put back on their page from the glyph profile of the page's MIDDLE-frequency tokens (M, disjoint from both).
a = acc_C - 0.5 (key axis), t = acc_R - acc_C (topic axis).
Random views (glyph feature x merge x strata cut-offs x sample cap x smoothing) are scored on train folios (leaf parity)
by how well they separate the planted worlds on both axes; the top 20 are frozen (with thresholds) and applied once to
held-out folios of every corpus, including the Voynich."""
import os, sys, json, time
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v96_lib as L

NV = int(os.environ.get('V96_NV', '300'))


def strata_view(rng, glyphs):
    v = L.random_view(rng, glyphs)
    v.update(cfrac=float(rng.choice([0.2, 0.3, 0.4])), lo=int(rng.choice([2, 3])), hi=int(rng.choice([10, 20, 40])),
             cap=int(rng.choice([12, 20, 30])), seed=int(rng.integers(1 << 30)))
    return v


def stats(pages, v, h):
    """(a, t, n) for view v on half h of a corpus."""
    cnt = Counter(w for p in pages for l in p['lines'] for w in l['w'])
    ranked = cnt.most_common(); tot = sum(cnt.values()); acc = 0; C = set()
    for w, n in ranked:
        if acc >= v['cfrac'] * tot: break
        C.add(w); acc += n
    ff = L.featf(v); cache = {}
    def fv(w):
        x = cache.get(w)
        if x is None: x = cache[w] = Counter(ff(w))
        return x
    rng = np.random.default_rng(v['seed'])
    rows = []
    for pi, p in enumerate(pages):
        if L.half(p['id']) != h: continue
        ws = [w for l in p['lines'] for w in l['w']]
        c = [w for w in ws if w in C]; r = [w for w in ws if w not in C and v['lo'] <= cnt[w] <= v['hi']]
        m = [w for w in ws if w not in C and not (v['lo'] <= cnt[w] <= v['hi'])]
        B = 4 * v['cap']                                  # glyph budget: both samples carry the same number of glyphs
        def take(lst):
            out = []; g = 0
            for i in rng.permutation(len(lst)):
                if g >= B: break
                out.append(lst[i]); g += len(lst[i])
            return out if g >= B else None
        cs = take(c); rs = take(r)
        if cs is None or rs is None or len(m) < 10: continue
        def prof(lst):
            z = Counter()
            for w in lst: z.update(fv(w))
            return z
        rows.append((L.group(p), prof(cs), prof(rs), prof(m), pi))
    if len(rows) < 12: return None
    feats = sorted({f for _, a, b, c, _ in rows for f in list(a) + list(b) + list(c)}); fi = {f: i for i, f in enumerate(feats)}
    def vec(z):
        x = np.zeros(len(feats))
        for f, n in z.items(): x[fi[f]] = n
        return x
    QC = np.array([vec(r[1]) for r in rows]); QR = np.array([vec(r[2]) for r in rows]); M = np.array([vec(r[3]) for r in rows])
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[r[0]].append(i)
    aC, n = L.rank_acc(QC, M, groups, v['lam']); aR, _ = L.rank_acc(QR, M, groups, v['lam'])
    out = dict(a=aC - 0.5, t=aR - aC, aC=aC, aR=aR, n=n)
    if h == 1:
        pos = np.array([r[4] for r in rows])
        out['aC_far'], out['n_far'] = L.rank_acc(QC, M, groups, v['lam'], pos, 4)
        out['aR_far'], _ = L.rank_acc(QR, M, groups, v['lam'], pos, 4)
    return out


def run(name):
    out = os.path.join(L.CK, 'c1_%s_h%s.json' % (name, os.environ.get('V96_H', '0')))
    if os.path.exists(out): return name, 0.0
    t0 = time.time(); h = int(os.environ.get('V96_H', '0'))
    pages = L.corpus(name)
    glyphs = sorted({c for p in pages for l in p['lines'] for w in l['w'] for c in w})
    if h == 0:
        rng = np.random.default_rng(96001)
        views = [strata_view(rng, sorted(set('abcdefghijklmnopqrstuvwxyzCSTKPF'))) for _ in range(NV)]
    else:
        views = json.load(open(os.path.join(L.DATA, 'v96_frozen_c1.json')))['views']
    res = [stats(pages, v, h) for v in views]
    if h == 1:   # kill battery: layout-free core (no para-first lines, no line-first or line-last word) and no page-first paragraph
        core = [dict(p, lines=[dict(l, w=l['w'][1:-1]) for l in p['lines'] if not l['ps'] and len(l['w']) > 2]) for p in pages]
        def nofirst(p):
            ls = p['lines']; k = next((i for i in range(1, len(ls)) if ls[i]['ps']), len(ls))
            return dict(p, lines=ls[k:])
        nf = [nofirst(p) for p in pages]
        for r, v in zip(res, views):
            if r is None: continue
            a = stats(core, v, 1); b = stats(nf, v, 1)
            r['aC_core'] = a['aC'] if a else None; r['aR_core'] = a['aR'] if a else None
            r['aC_nf'] = b['aC'] if b else None
    json.dump(dict(name=name, h=h, res=res, secs=time.time() - t0), open(out, 'w'))
    return name, time.time() - t0


if __name__ == '__main__':
    from multiprocessing import Pool
    names = sys.argv[1:] or L.all_names()
    for n in names: L.corpus(n)
    with Pool(2) as P:
        for n, s in P.imap_unordered(run, names):
            print(n, '%.0fs' % s, flush=True)
