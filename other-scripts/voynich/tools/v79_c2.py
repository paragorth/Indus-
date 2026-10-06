"""v79 cycle 2: is the line-start glyph a CLASS TAG (or a catchword) for the line's payload?

Thousands of random binary payload features of a line's BODY (words 2..n only: the first word, which carries the tag,
is never used): presence of any of a random set of word types, of a random set of glyph bigrams, a random glyph set
at word 2/3 start, at the line-final glyph. Each feature is predicted from the line's tag (first glyph of the line)
on top of a stratum = section|language x line mode (v67: majority first-glyph class of the body words) x line length
band. Held-out (leaf halves) bits gained by knowing the tag; top-50 features chosen on the discovery half.
Variants: SAME (tag -> own line body), NEXT (tag -> body of the line below), PREV (catchword: body of the line above ->
tag of this line, with the tag above in the stratum, so the known chain does not count).
Nulls: tag shuffled within paragraph (line-level null) and within page (paragraph-level null), generators.
Controls (opaque v72 surface, tag on 70-90% of lines): Brumati with its real class letter on every line (paragraph-level
class tag), Brumati with the real part number inside the entry (which paragraph of the entry the line is in:
description / synonyms / habitat / uses), Dante verse numbers (a tag with no content link: must give ~0).
"""
import sys, os, time, random, re
import numpy as np
from collections import Counter, defaultdict
import v79_lib as L

NF = int(os.environ.get('V79_NF', '6000'))
MODE = {}
for g in 'CSkKTPtpfF': MODE[g] = 0
for g in 'oel': MODE[g] = 1
for g in 'qdsy': MODE[g] = 2


def line_records(pages, alpha):
    ix = {g: i for i, g in enumerate(alpha)}
    R = []
    para = -1
    for pi, p in enumerate(pages):
        for li, l in enumerate(p['lines']):
            if l['ps'] or para < 0: para += 1
            if not l['w']: continue
            body = l['w'][1:]
            ms = Counter(MODE.get(w[0], 3) for w in body if w)
            mode = ms.most_common(1)[0][0] if ms else 3
            R.append(dict(page=pi, para=para, ps=l['ps'], tag=ix.get(l['w'][0][0], -1), body=body, mode=mode,
                          sec='%s|%s' % (p['sec'], p.get('lang', '-')), nb=min(len(body) // 3, 3),
                          half=L.leaf_half(p['id'])))
    return R


def feature_bank(R, nf, seed):
    rng = np.random.default_rng(seed)
    wc = Counter(w for r in R for w in r['body'])
    types = [w for w, c in wc.most_common(400) if c >= 5]
    tix = {w: i for i, w in enumerate(types)}
    bc = Counter(w[j:j + 2] for r in R for w in r['body'] for j in range(len(w) - 1))
    bigr = [b for b, c in bc.most_common(150)]
    bix = {b: i for i, b in enumerate(bigr)}
    gl = [g for g, _ in Counter(c for r in R for w in r['body'] for c in w).most_common(20)]
    gix = {g: i for i, g in enumerate(gl)}
    n = len(R)
    W = np.zeros((n, len(types)), np.float32); Bm = np.zeros((n, len(bigr)), np.float32)
    G2 = np.zeros((n, len(gl)), np.float32); G3 = np.zeros((n, len(gl)), np.float32); GF = np.zeros((n, len(gl)), np.float32)
    for i, r in enumerate(R):
        for w in r['body']:
            if w in tix: W[i, tix[w]] = 1
            for j in range(len(w) - 1):
                b = w[j:j + 2]
                if b in bix: Bm[i, bix[b]] = 1
        if len(r['body']) > 0 and r['body'][0][0] in gix: G2[i, gix[r['body'][0][0]]] = 1
        if len(r['body']) > 1 and r['body'][1][0] in gix: G3[i, gix[r['body'][1][0]]] = 1
        if len(r['body']) > 0 and r['body'][-1][-1] in gix: GF[i, gix[r['body'][-1][-1]]] = 1
    mats = [W, Bm, G2, G3, GF]
    F = np.zeros((n, nf), bool); desc = []
    for f in range(nf):
        kind = int(rng.integers(0, 5)); M = mats[kind]
        if M.shape[1] == 0: kind = 0; M = W
        k = int(rng.integers(1, {0: 9, 1: 6, 2: 5, 3: 5, 4: 4}[kind]))
        S = rng.choice(M.shape[1], size=min(k, M.shape[1]), replace=False)
        F[:, f] = M[:, S].sum(1) > 0
        desc.append((kind, S.tolist()))
    keep = (F.mean(0) > 0.03) & (F.mean(0) < 0.97)
    return F[:, keep], [d for d, k in zip(desc, keep) if k]


def gains(F, tag, strat, half, a=4.0):
    """held-out bits per line gained by P(F | strat, tag) over P(F | strat), per feature, per held-out half."""
    out = np.zeros((2, F.shape[1]))
    ok = tag >= 0
    ns = strat.max() + 1; nt = tag.max() + 1
    Ff = F.astype(np.float64)
    for h in (0, 1):
        tr = ok & (half != h); te = ok & (half == h)
        S1 = np.zeros((ns, F.shape[1])); S0 = np.zeros(ns)
        np.add.at(S1, strat[tr], Ff[tr]); np.add.at(S0, strat[tr], 1)
        glob = (Ff[tr].sum(0) + 1) / (tr.sum() + 2)
        ps = (S1 + a * glob) / (S0[:, None] + a)
        C1 = np.zeros((ns * nt, F.shape[1])); C0 = np.zeros(ns * nt)
        cell = strat * nt + tag
        np.add.at(C1, cell[tr], Ff[tr]); np.add.at(C0, cell[tr], 1)
        pc = (C1 + a * ps[np.repeat(np.arange(ns), nt)]) / (C0[:, None] + a)
        x = Ff[te]
        p0 = ps[strat[te]]; p1 = pc[cell[te]]
        l0 = x * np.log2(p0) + (1 - x) * np.log2(1 - p0)
        l1 = x * np.log2(p1) + (1 - x) * np.log2(1 - p1)
        out[h] = (l1 - l0).sum(0) / max(te.sum(), 1)
    return out


def evaluate(F, tag, strat, half, top=50):
    G = gains(F, tag, strat, half)
    # discovery on half 0 -> held-out half 1 = G[1] for features ranked by in-sample?  we need discovery gains:
    # rank features by their gain when half 1 is held out of TRAINING... use G[0] (held-out on half 0) to pick, G[1] to score
    o1 = np.argsort(-G[0])[:top]; o0 = np.argsort(-G[1])[:top]
    sel = 0.5 * (G[1][o1].mean() + G[0][o0].mean())
    return dict(all=float(G.mean()), sel=float(sel) * 1000, allk=float(G.mean()) * 1000)


def variant_arrays(R, var):
    """returns tag, strat, half, row index into R for the payload line."""
    sk = {}
    def S(key):
        return sk.setdefault(key, len(sk))
    tags, strats, halves, rows = [], [], [], []
    for i, r in enumerate(R):
        if r['ps']: continue
        if var == 'SAME':
            tags.append(r['tag']); strats.append(S((r['sec'], r['mode'], r['nb']))); halves.append(r['half']); rows.append(i)
        elif var == 'NEXT':
            if i + 1 < len(R) and R[i + 1]['para'] == r['para'] and not R[i + 1]['ps']:
                q = R[i + 1]
                tags.append(r['tag']); strats.append(S((q['sec'], q['mode'], q['nb'], q['tag']))); halves.append(r['half']); rows.append(i + 1)
        elif var == 'PREV':
            if i > 0 and R[i - 1]['para'] == r['para'] and not R[i - 1]['ps']:
                q = R[i - 1]
                tags.append(r['tag']); strats.append(S((q['sec'], q['mode'], q['nb'], q['tag']))); halves.append(r['half']); rows.append(i - 1)
    return np.array(tags), np.array(strats), np.array(halves), np.array(rows)


def shuffle_tags(R, rng, level):
    R2 = [dict(r) for r in R]
    grp = defaultdict(list)
    for i, r in enumerate(R2):
        if not r['ps']: grp[r['para'] if level == 'para' else r['page']].append(i)
    for g, ix in grp.items():
        t = [R2[i]['tag'] for i in ix]; rng.shuffle(t)
        for i, v in zip(ix, t): R2[i]['tag'] = v
    return R2


def run_corpus(pages, nnull=8, seed=0):
    T, alpha = L.chain_table(pages)
    R = line_records(pages, alpha)
    F, desc = feature_bank(R, NF, seed)
    out = {}
    rng = random.Random(seed + 1)
    for var in ('SAME', 'NEXT', 'PREV'):
        tag, strat, half, rows = variant_arrays(R, var)
        real = evaluate(F[rows], tag, strat, half)
        nul = {}
        for lev in ('para', 'page'):
            vals = []
            for j in range(nnull):
                R2 = shuffle_tags(R, rng, lev)
                t2, s2, h2, r2 = variant_arrays(R2, var)
                vals.append(evaluate(F[r2], t2, s2, h2))
            nul[lev] = vals
        def z(key, lev):
            v = np.array([x[key] for x in nul[lev]]); return float((real[key] - v.mean()) / (v.std() + 1e-12))
        out[var] = dict(real=real, null={lev: dict(sel=float(np.mean([x['sel'] for x in nul[lev]])),
                                                   allk=float(np.mean([x['allk'] for x in nul[lev]]))) for lev in nul},
                        z_sel_para=z('sel', 'para'), z_all_para=z('allk', 'para'),
                        z_sel_page=z('sel', 'page'), z_all_page=z('allk', 'page'), n=int(len(tag)))
    return out


def corpora():
    C = {}
    for nm in ('ZL3b', 'IT2a'):
        C[nm] = ('VOY', L.V77.voy(nm))
    B = L.brumati_entries_plain()
    C['C_CLASS'] = ('CTL', L.surface_channel(B, lambda l: l['cls'] % 10, p_write=0.8)[0])
    BP = L.brumati_parts_plain()
    C['C_PART'] = ('CTL', L.surface_channel(BP, lambda l: min(l['part'], 4), p_write=0.8)[0])
    D = L.dante_plain()[:170]
    C['N_DANTE'] = ('NEG', L.surface_channel(D, lambda l: l['ch'] % 10)[0])
    for g in ('SELFCIT', 'MK2', 'JUNC', 'STACK'):
        C['G_' + g] = ('GEN', L.V77.generate('ZL3b', g, 792))
    return C


def main():
    t0 = time.time()
    C = corpora()
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    names = list(C) if part == 'all' else list(C)[int(part)::2]
    for nm in names:
        kind, pages = C[nm]
        r = run_corpus(pages)
        r['kind'] = kind
        L.psave('c2_%s.pkl' % nm, r)
        print(nm, '%.0fs' % (time.time() - t0), {v: (round(r[v]['real']['sel'], 2), round(r[v]['z_sel_para'], 1),
                                                    round(r[v]['z_all_para'], 1), round(r[v]['z_sel_page'], 1)) for v in ('SAME', 'NEXT', 'PREV')}, flush=True)


if __name__ == '__main__':
    main()
