"""v73 cycle 3: THE LINE POINTS AT ITS OWN ITEM. Pointer marking rules: a key read off the line (the first glyph of
its first word, the last glyph of its last word, the line length mod 4, or the first glyph of the previous line)
selects the position of the item through a random lookup table (positions 0,1,2,3, last, second-to-last).
4 keys x 750 random tables = 3,000 pointer rules. Control plants: the Antidotarium list placed by a hidden random
first-glyph pointer table (and by a hidden last-glyph table) inside SELFCIT and MK2 filler with the VOY codebook
(items spelled as ordinary Voynich words, so only the pointer finds them). Nulls as cycle 1."""
import os, sys, time, random
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v73_lib as L, v72_lib as V

NULLSET = ['MK2', 'SELFCIT', 'SC10', 'JUNC', 'LSHUF']
KEYS = ['first', 'last', 'lenmod', 'prevfirst']
GLY = list('oqdsyCSkt') + ['*']          # key alphabet for glyph keys (other = '*')
NPOS = 6                                  # 0,1,2,3,last,second-to-last
NT = 750


def keyval(key, lines, li):
    l = lines[li]
    if key == 'first': g = l[0][0]
    elif key == 'last': g = l[-1][-1] if l[-1][-1] in GLY else ('y' if l[-1][-1] in 'ym' else '*')
    elif key == 'lenmod': return len(l) % 4
    else: g = lines[li - 1][0][0] if li > 0 else '*'
    return GLY.index(g) if g in GLY else len(GLY) - 1


def posof(code, n):
    p = [0, 1, 2, 3, n - 1, n - 2][code]
    return min(max(p, 0), n - 1)


def tables(seed=373):
    rng = np.random.default_rng(seed)
    out = []
    for key in KEYS:
        nk = 4 if key == 'lenmod' else len(GLY)
        for _ in range(NT): out.append((key, rng.integers(0, NPOS, nk)))
    return out


def select_ptr(C, key, tab):
    idx = np.zeros(C.nl, int); li = 0
    for p in C.pages:
        lines = [l['w'] for l in p['lines'] if l['w']]
        for k in range(len(lines)):
            idx[li] = C.line_start[li] + posof(int(tab[keyval(key, lines, k)]), len(lines[k])); li += 1
    return idx


def plant_ptr(skeleton, filler, key, seed=373):
    rng = np.random.default_rng(seed + 1)
    nk = len(GLY); allowed = [1, 2, 3, 5] if key == 'first' else [0, 1, 2, 3]   # never overwrite the key word
    tab = rng.choice(allowed, nk)
    fill = L.NULLS[filler](skeleton, seed)
    coded = L.code_items(L.item_stream('antid'), skeleton, 'VOY', seed)
    k = 0; out = []; truth = []
    for pi, p in enumerate(fill):
        lines = [list(l['w']) for l in p['lines']]
        for li, ws in enumerate(lines):
            if ws and k < len(coded):
                j = posof(int(tab[keyval(key, lines, li)]), len(ws))
                ws[j] = coded[k]; k += 1; truth.append((pi, li, j))
        out.append(dict(p, lines=[dict(l, w=w) for l, w in zip(p['lines'], lines)]))
    last = truth[-1][0]
    return out[:last + 1], truth, tab.tolist()


def targets():
    P = V.voynich('ZL3b')
    T = {'V_ZL3b': P, 'V_IT2a': V.voynich('IT2a')}
    for fil in ('SELFCIT', 'MK2'):
        for key in ('first', 'last'):
            pp, tr, tab = plant_ptr(P, fil, key)
            nm = 'PP_%s_%s' % (key, fil); T[nm] = pp; L.jsave('truth_%s.json' % nm, tr); L.jsave('tab_%s.json' % nm, tab)
    T['FP3_SELFCIT'] = L.NULLS['SELFCIT'](P, 701)
    T['FP3_MK2'] = L.NULLS['MK2'](P, 702)
    return T


def job(args):
    tname, kind, pages = args
    out = os.path.join(L.CK, 'S3_%s__%s.npy' % (tname, kind))
    if os.path.exists(out): return out
    if kind != 'REAL': pages = L.NULLS[kind](pages, 31)
    C = L.Corpus(pages, kind)
    masks = [C.half == 0, C.half == 1]; bases = [L.baseline(C, m) for m in masks]
    TB = tables(); S = np.zeros((len(TB), 2, len(L.STATS)))
    for i, (key, tab) in enumerate(TB):
        idx = select_ptr(C, key, tab)
        for j, m in enumerate(masks):
            st = L.stream_stats(C, idx, m); S[i, j] = [st[k] - bases[j][k] for k in L.STATS]
    # coordinate ascent per key on the discovery half (same procedure on every null)
    ext = []
    for key in KEYS:
        nk = 4 if key == 'lenmod' else len(GLY)
        best = None
        for start in range(2):
            tab = np.random.default_rng(start).integers(0, NPOS, nk)
            def obj(tb):
                st = L.stream_stats(C, select_ptr(C, key, tb), masks[0])
                return (st['XP'] - bases[0]['XP']) / 0.02 + (st['VS'] - bases[0]['VS']) / 0.05
            cur = obj(tab)
            for sweep in range(3):
                for v in range(nk):
                    for c in range(NPOS):
                        if c == tab[v]: continue
                        tb = tab.copy(); tb[v] = c; o = obj(tb)
                        if o > cur: cur, tab = o, tb
            if best is None or cur > best[0]: best = (cur, tab)
        idx = select_ptr(C, key, best[1])
        ext.append([[L.stream_stats(C, idx, m)[k] - b[k] for k in L.STATS] for m, b in zip(masks, bases)])
        L.jsave('tab3_%s__%s__%s.json' % (tname, kind, key), best[1].tolist())
    S = np.concatenate([S, np.array(ext)])
    np.save(out, S.astype(np.float32))
    return out


if __name__ == '__main__':
    T = targets()
    jobs = [(t, k, P) for t, P in T.items() for k in ['REAL'] + NULLSET]
    t0 = time.time()
    with Pool(2) as pool:
        for i, o in enumerate(pool.imap_unordered(job, jobs)):
            print('%3d/%d %6.0fs %s' % (i + 1, len(jobs), time.time() - t0, os.path.basename(o)), flush=True)
