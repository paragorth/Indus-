"""v79 cycle 3: is the line-start chain an INDEX or CROSS-REFERENCE key?

(A) Cross-reference keys. Random key definitions = 1-3 'slots' read off the left margin and line edges (line-initial
    glyph of the line itself and of lines -2..+2, the line's second glyph, the first glyph of word 2, the last glyph of
    the line). If a key names the item a line is about, lines with the same key on OTHER pages share body vocabulary
    (TF-IDF cosine of words 2..n) more than other lines of the same section|language x line mode. Score per key =
    mean same-key cross-page cosine / mean cross-page cosine in the stratum - 1. Discovery on pairs inside half 0,
    held-out pairs inside half 1 (and swapped). Null: the key values permuted among lines of the same page (keeps each
    page's key mix, breaks the line link); generators.
(B) Index letters. Is the tag the first (or second, or last) glyph of a selected word of its own line (the rarest word,
    a word that recurs on another page, the longest word, word k)? Rate of tag == selected glyph vs the same with tags
    permuted within paragraph.
Controls: Brumati with a genus key (hash of the genus, 10 opaque glyphs) on 80% of lines (a real cross-reference key:
species of one genus share vocabulary across pages); Isidore with an index letter = initial of the line's rarest word
(planted index column through the same surface); Dante verse numbers (no content link).
"""
import sys, os, time, random, re, zlib
import numpy as np
import scipy.sparse as sp
from collections import Counter, defaultdict
import v79_lib as L

NK = int(os.environ.get('V79_NK', '3000'))
SLOTS = ['t0', 't-1', 't+1', 't-2', 't+2', 'g2', 'w2', 'end']


def records(pages):
    R = []; para = -1
    for pi, p in enumerate(pages):
        for li, l in enumerate(p['lines']):
            if l['ps'] or para < 0: para += 1
            if not l['w']: continue
            body = l['w'][1:]
            from v79_c2 import MODE
            ms = Counter(MODE.get(w[0], 3) for w in body if w)
            R.append(dict(page=pi, para=para, ps=l['ps'], w=l['w'], body=body, half=L.leaf_half(p['id']),
                          sec='%s|%s' % (p['sec'], p.get('lang', '-')), mode=ms.most_common(1)[0][0] if ms else 3))
    return R


def slot_values(R):
    n = len(R)
    V = {}
    t0 = [r['w'][0][0] for r in R]
    def nb(i, d):
        j = i + d
        if 0 <= j < n and R[j]['para'] == R[i]['para']: return t0[j]
        return '#'
    V['t0'] = t0
    for d, nm in ((-1, 't-1'), (1, 't+1'), (-2, 't-2'), (2, 't+2')):
        V[nm] = [nb(i, d) for i in range(n)]
    V['g2'] = [r['w'][0][1] if len(r['w'][0]) > 1 else '#' for r in R]
    V['w2'] = [r['w'][1][0] if len(r['w']) > 1 else '#' for r in R]
    V['end'] = [r['w'][-1][-1] for r in R]
    return V


def tfidf(R):
    vocab = {}
    rows, cols = [], []
    for i, r in enumerate(R):
        for w in set(r['body']):
            rows.append(i); cols.append(vocab.setdefault(w, len(vocab)))
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(R), len(vocab)))
    df = np.asarray((X > 0).sum(0)).ravel()
    idf = np.log(len(R) / df); idf[df < 2] = 0   # hapaxes cannot link pages
    X = X.multiply(idf).tocsr()
    nr = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); nr[nr == 0] = 1
    return sp.diags(1 / nr) @ X


def cross_cos(X, grp, page, mask):
    """mean cosine over pairs of lines (both in mask) in the same group and on different pages."""
    idx = np.where(mask)[0]
    g = grp[idx]; pg = page[idx]
    ug, gi = np.unique(g, return_inverse=True)
    Ig = sp.csr_matrix((np.ones(len(idx)), (gi, np.arange(len(idx)))), shape=(len(ug), len(idx)))
    Sg = Ig @ X[idx]
    tot = np.asarray(Sg.multiply(Sg).sum(1)).ravel()
    gp = np.unique(np.stack([gi, pg]), axis=1, return_inverse=True)[1].ravel()
    ngp = gp.max() + 1
    Igp = sp.csr_matrix((np.ones(len(idx)), (gp, np.arange(len(idx)))), shape=(ngp, len(idx)))
    Sgp = Igp @ X[idx]
    same = np.asarray(Sgp.multiply(Sgp).sum(1)).ravel()
    cnt_g = np.bincount(gi); cnt_gp = np.bincount(gp)
    num = tot.sum() - same.sum()
    den = (cnt_g ** 2).sum() - (cnt_gp ** 2).sum()
    return num / max(den, 1)


def key_score(X, R, keyv, strat, page, half):
    out = []
    for h in (0, 1):
        m = half == h
        base = cross_cos(X, strat, page, m)
        kk = cross_cos(X, strat * 100000 + keyv, page, m)
        out.append(kk / base - 1 if base > 0 else 0)
    return out


def encode_keys(V, slots):
    cols = [V[s] for s in slots]
    d = {}
    return np.array([d.setdefault(tuple(c[i] for c in cols), len(d)) for i in range(len(cols[0]))])


def part_A(R, nk, seed, nnull=6):
    X = tfidf(R)
    body = np.array([not r['ps'] for r in R])
    sk = {}
    strat = np.array([sk.setdefault((r['sec'], r['mode']), len(sk)) for r in R])
    page = np.array([r['page'] for r in R]); half = np.array([r['half'] for r in R])
    half = np.where(body, half, 9)    # only body lines enter pairs
    V = slot_values(R)
    rng = np.random.default_rng(seed)
    defs = set()
    while len(defs) < min(nk, 92):
        k = int(rng.integers(1, 4)); defs.add(tuple(sorted(rng.choice(len(SLOTS), k, replace=False).tolist())))
    defs = [tuple(SLOTS[i] for i in d) for d in defs]
    # massive part: random keys built as random many-to-one merges of slot tuples (index classes)
    res = []
    for d in defs:
        kv = encode_keys(V, d)
        res.append((d, 'exact', key_score(X, R, kv, strat, page, half)))
    nmerge = max(nk - len(defs), 0)
    for j in range(nmerge):
        d = defs[int(rng.integers(len(defs)))]
        kv = encode_keys(V, d)
        nc = int(rng.integers(2, 12))
        lab = rng.integers(0, nc, kv.max() + 1)
        res.append((d, 'merge%d' % nc, key_score(X, R, lab[kv], strat, page, half)))
    def sel(res):
        b0 = max(res, key=lambda x: x[2][0]); b1 = max(res, key=lambda x: x[2][1])
        return dict(held=0.5 * (b0[2][1] + b1[2][0]), best0=(b0[0], b0[1], b0[2]), best1=(b1[0], b1[1], b1[2]),
                    t0=[x[2] for x in res if x[0] == ('t0',) and x[1] == 'exact'][0])
    real = sel(res)
    # null: permute key slot values within page (every slot column independently keeps its page mix)
    nul = []
    for j in range(nnull):
        Vn = {}
        perm = np.arange(len(R))
        for pg in np.unique(page):
            ix = np.where(page == pg)[0]; perm[ix] = ix[rng.permutation(len(ix))]
        for s in SLOTS: Vn[s] = [V[s][perm[i]] for i in range(len(R))]
        rn = []
        for d in defs:
            rn.append((d, 'exact', key_score(X, R, encode_keys(Vn, d), strat, page, half)))
        for jj in range(nmerge // 4):
            d = defs[int(rng.integers(len(defs)))]
            kv = encode_keys(Vn, d); nc = int(rng.integers(2, 12)); lab = rng.integers(0, nc, kv.max() + 1)
            rn.append((d, 'merge', key_score(X, R, lab[kv], strat, page, half)))
        nul.append(sel(rn))
    hv = np.array([x['held'] for x in nul]); t0v = np.array([np.mean(x['t0']) for x in nul])
    return dict(real=real, null_held=hv.tolist(), z_held=float((real['held'] - hv.mean()) / (hv.std() + 1e-9)),
                t0=float(np.mean(real['t0'])), z_t0=float((np.mean(real['t0']) - t0v.mean()) / (t0v.std() + 1e-9)))


def part_B(R, nnull=30, seed=0):
    wc = Counter(w for r in R for w in r['body'])
    pages_of = defaultdict(set)
    for r in R:
        for w in r['body']: pages_of[w].add(r['page'])
    sels = {
        'rarest': lambda r: [min(r['body'], key=lambda w: wc[w])] if r['body'] else [],
        'recurs_elsewhere': lambda r: [w for w in r['body'] if len(pages_of[w] - {r['page']}) > 0 and wc[w] <= 20],
        'longest': lambda r: [max(r['body'], key=len)] if r['body'] else [],
        'word2': lambda r: r['body'][:1], 'word3': lambda r: r['body'][1:2], 'last': lambda r: r['body'][-1:],
        'any': lambda r: r['body'],
    }
    posf = {'first': lambda w: w[0], 'first_noq': lambda w: w[1] if w[0] == 'q' and len(w) > 1 else w[0], 'second': lambda w: w[1] if len(w) > 1 else '#', 'last': lambda w: w[-1]}
    body = [i for i, r in enumerate(R) if not r['ps']]
    tags = [R[i]['w'][0][0] for i in body]
    rng = random.Random(seed)
    grp = defaultdict(list)
    for j, i in enumerate(body): grp[R[i]['para']].append(j)
    out = {}
    for sn, sf in sels.items():
        S = [sf(R[i]) for i in body]
        for pn, pf in posf.items():
            G = [set(pf(w) for w in s) for s in S]
            obs = np.mean([t in g for t, g in zip(tags, G)])
            nv = []
            for k in range(nnull):
                t2 = list(tags)
                for ix in grp.values():
                    v = [t2[j] for j in ix]; rng.shuffle(v)
                    for j, x in zip(ix, v): t2[j] = x
                nv.append(np.mean([t in g for t, g in zip(t2, G)]))
            nv = np.array(nv)
            out['%s/%s' % (sn, pn)] = (float(obs), float(nv.mean()), float((obs - nv.mean()) / (nv.std() + 1e-9)))
    return out


def genus_pages():
    """Brumati entries with their genus (last genus header 'CIX. Menta.' seen)."""
    import v49_lib
    ents = v49_lib.brumati_entries()
    lines_all = []; genus = 0
    for e in ents:
        if re.match(r'^[IVXLC]+°?\.\s', e['paras'][0]): genus += 1
        ws = []
        for pi, p in enumerate(e['paras']):
            if pi == 0: p = re.sub(r'^(\d+a?\.?|[IVXLC]+°?\.)\s', '', p)
            ws += v49_lib.brumati_words(p)
        for i in range(0, min(len(ws), 64), 8):
            lines_all.append(dict(w=ws[i:i + 8], ps=(i == 0), gen=genus))
    return [dict(id='bg%03d' % (i // 20), sec='B', lang='-', hand='-', quire='-', lines=lines_all[i:i + 20])
            for i in range(0, len(lines_all), 20)]


def index_pages():
    I = L.V72.isidore_plain()[:200]
    wc = Counter(w for p in I for l in p['lines'] for w in l['w'])
    for p in I:
        for l in p['lines']:
            body = l['w'][1:] or l['w']
            l['idx'] = min(body, key=lambda w: wc[w])[0]
    return I


def corpora():
    C = {}
    for nm in ('ZL3b', 'IT2a'):
        C[nm] = ('VOY', L.V77.voy(nm))
    G = genus_pages()
    C['C_GENUS'] = ('CTL', L.surface_channel(G, lambda l: zlib.crc32(str(l['gen']).encode()) % 10, p_write=0.8)[0])
    I = L.V72.isidore_plain()[:200]
    C['C_INDEX'] = ('CTL', _index_surface(I))
    D = L.dante_plain()[:170]
    C['N_DANTE'] = ('NEG', L.surface_channel(D, lambda l: l['ch'] % 10)[0])
    for g in ('SELFCIT', 'MK2', 'JUNC', 'STACK'):
        C['G_' + g] = ('GEN', L.V77.generate('ZL3b', g, 793))
    return C


def _index_surface(I):
    """the line's index letter (initial of its rarest word) written in the margin in the same script: the surfaced
    first glyph of that word (after any q- prefix) is prefixed to the line on 80% of lines."""
    wc = Counter(w for p in I for l in p['lines'] for w in l['w'])
    S, _ = L.surface_channel(I, lambda l: 0, p_write=0.0)
    rng = random.Random(3)
    for p0, p1 in zip(I, S):
        for l0, l1 in zip(p0['lines'], p1['lines']):
            if len(l0['w']) < 2 or rng.random() >= 0.8: continue
            j = 1 + min(range(len(l0['w']) - 1), key=lambda k: wc[l0['w'][k + 1]])
            sw = l1['w'][j]; sw = sw[1:] if sw[0] == 'q' and len(sw) > 1 else sw
            w0 = l1['w'][0]
            if w0[0] in 'ysdt' and len(w0) > 1: w0 = w0[1:]
            l1['w'][0] = sw[0] + w0
    return S


def main():
    t0 = time.time()
    C = corpora()
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    names = list(C) if part == 'all' else list(C)[int(part)::2]
    for nm in names:
        kind, pages = C[nm]
        R = records(pages)
        rA = part_A(R, NK, 1)
        rB = part_B(R)
        L.psave('c3_%s.pkl' % nm, dict(kind=kind, A=rA, B=rB))
        topB = sorted(rB.items(), key=lambda x: -x[1][2])[:3]
        print(nm, '%.0fs' % (time.time() - t0), 'A held %.4f z %.1f t0 %.4f z %.1f' % (rA['real']['held'], rA['z_held'], rA['t0'], rA['z_t0']),
              rA['real']['best0'][:2], 'B', [(k, round(v[0], 3), round(v[1], 3), round(v[2], 1)) for k, v in topB], flush=True)


if __name__ == '__main__':
    main()
