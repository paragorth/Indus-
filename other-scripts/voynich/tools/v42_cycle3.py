"""v42 cycle 3: does long meaningless writing by one hand HARDEN into a system?

Every text is cut into consecutive 100-token windows in writing order. Per window: the v31 feature battery and the
GEN posterior of the n18-world 5-class logistic (trained on noise-matched training corpora, never on CS or Voynich).
'Hardening' = rising GEN posterior, rising unit-order rigidity, rising word-family density, falling h2 ratio,
falling type-token ratio along the writing order.
Objects: Codex Seraphinianus (page order; volumes 1 and 2 separately), Voynich ZL3b (folio order; Currier A, B,
hand 1), Gaskell-Bowern gibberish writers (first vs last 100 tokens, paired).
Controls: long one-author meaningful texts (Isidore, Latin, Italian, German, English, Spanish literary corpora;
n18 channel) -- must show no hardening; a generator (stationary); a PLANTED hardening (Isidore whose words are
progressively replaced by slot-generator words, 0 -> 40% along the text) -- must be caught; trend null = block
permutation of the window series (blocks of 8 windows, 2000x).
"""
import os, sys, json, random, zlib
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
from scipy.stats import spearmanr, wilcoxon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v42_lib as L, v31_lib as L31, v42_cycle1 as C1

FN = 'v42_cycle3.txt'
W = 100
TRAITS = ['GEN', 'order_rig', 'family', 'h2_ratio', 'ttr', 'pos_mi']
SIGN = {'GEN': 1, 'order_rig': 1, 'family': 1, 'h2_ratio': -1, 'ttr': -1, 'pos_mi': 1}


def windows(lines_with_tag, maxw=None):
    """lines_with_tag: list of (tag, line). -> list of (tag of first token, lines)"""
    out = []; cur = []; n = 0; tag0 = None
    for tag, l in lines_with_tag:
        for w in l:
            if n == 0: tag0 = tag; cur = [[]]; last = id(l)
            if id(l) != last: cur.append([]); last = id(l)
            cur[-1].append(w); n += 1
            if n == W: out.append((tag0, cur)); n = 0
    if maxw and len(out) > maxw:
        idx = np.linspace(0, len(out) - 1, maxw).round().astype(int); out = [out[i] for i in idx]
    return out


def cs_series():
    P = L.cs_pages()
    lt = [((p['page'], p['sec']), l) for p in P for pa in p['paras'] for l in pa]
    return windows(lt)


def voy_series(name='ZL3b'):
    import v21_lib as V
    P = V.voynich_pages(name, minw=10)
    lt = [((i, p['sec'], p.get('hand')), l) for i, p in enumerate(P) for pa in p['paras'] for l in pa]
    return windows(lt)


def text_series(docs, maxw=300):
    lt = [(i, l) for d in docs for i, l in enumerate(d)]
    return windows(lt, maxw)


def planted(docs, gen_docs, fmax=0.4, seed=5, maxw=300):
    rng = random.Random(seed)
    gw = [w for d in gen_docs for l in d for w in l]
    lines = [l for d in docs for l in d]
    tot = sum(len(l) for l in lines); k = 0; out = []
    for l in lines:
        nl = []
        for w in l:
            f = fmax * k / tot; k += 1
            nl.append(rng.choice(gw) if rng.random() < f else w)
        out.append(nl)
    return text_series([out], maxw)


def work(a):
    name, i, tag, lines = a
    rng = random.Random(zlib.crc32(f'{name}|{i}'.encode()))
    return name, i, tag, L31.features(lines, rng, nshuf=4)


def series_jobs():
    C = L.all_corpora()
    S = {}
    S['CS'] = cs_series()
    S['VZL'] = voy_series('ZL3b')
    S['VZL_n18'] = None   # built below from noised pages
    import v21_lib as V
    P = V.voynich_pages('ZL3b', minw=10)
    rngs = 0
    docs = [[l for pa in p['paras'] for l in pa] for p in P]
    nd = L.noise_docs(docs, 0.18, seed=11)
    lt = [((i, P[i]['sec'], P[i].get('hand')), l) for i, d in enumerate(nd) for l in d]
    S['VZL_n18'] = windows(lt)
    for k in ('L_Isidore', 'L_Latin_Lite', 'L_Italian_Lite', 'L_German_Lite', 'L_English_Lite', 'L_Spanish_Lite'):
        d = C[k][1]
        S[k + '_n18'] = text_series(L.noise_docs(d, 0.18, seed=zlib.crc32(k.encode()) & 0xffff))
    g = C['G_slot1'][1]
    S['G_selfcit1_n18'] = text_series(L.noise_docs(C['G_selfcit1'][1], 0.18, seed=3))
    S['P_plant_n18'] = [(t, l) for t, l in planted(L.noise_docs(C['L_Isidore'][1], 0.18, seed=4), L.noise_docs(g, 0.18, seed=5))]
    # gibberish writers: first and last 100 tokens; languages: same on a 400-token chunk
    for k, (cls, d) in C.items():
        if cls == 'GIBB':
            toks = [(0, l) for dd in d for l in dd]
            ws = windows(toks)
            if len(ws) >= 2: S['B|' + k] = [ws[0], ws[-1]]
    for k, (cls, d) in C.items():
        if cls == 'LANG':
            ws = windows([(0, l) for dd in d for l in dd][:200])
            if len(ws) >= 2: S['Lh|' + k] = [ws[0], ws[min(3, len(ws) - 1)]]
    J = [(name, i, tag, lines) for name, ws in S.items() for i, (tag, lines) in enumerate(ws)]
    return J


def block_perm_rho(y, nb=8, nperm=2000, seed=0):
    y = np.asarray(y, float); n = len(y); x = np.arange(n)
    obs = spearmanr(x, y).correlation
    blocks = [y[i:i + nb] for i in range(0, n, nb)]
    rng = np.random.RandomState(seed); null = []
    for _ in range(nperm):
        o = rng.permutation(len(blocks)); null.append(spearmanr(x, np.concatenate([blocks[j] for j in o])).correlation)
    null = np.array(null)
    return float(obs), float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + nperm))


def main():
    out = L.load('cycle3_feats.json')
    if out is None:
        J = series_jobs(); print('windows', len(J), flush=True)
        with Pool(2) as p: R = p.map(work, J, chunksize=8)
        out = [{'name': n, 'i': i, 'tag': t, 'F': {k: float(v) for k, v in F.items()}} for n, i, t, F in R]
        L.save('cycle3_feats.json', out)
    keys, X, corp, cls, var = C1.load()
    tr = (var == 'n18') & np.isin(cls, C1.CLASSES)
    P, cl = C1.fit(X[tr], cls[tr]); gi = cl.index('GEN')
    by = defaultdict(list)
    for r in out: by[r['name']].append(r)
    T = {}
    for name, rs in by.items():
        rs.sort(key=lambda r: r['i'])
        Xs = np.array([[r['F'][k] for k in keys] for r in rs], float); Xs[~np.isfinite(Xs)] = 0
        g = P(Xs)[:, gi]
        T[name] = {'tag': [r['tag'] for r in rs], 'GEN': g.tolist(), **{k: [r['F'][k] for r in rs] for k in TRAITS if k != 'GEN'}}
    L.save('cycle3_series.json', T)
    rows = []
    def trend(name, sel=None, label=None):
        s = T[name]; idx = [i for i in range(len(s['GEN'])) if sel is None or sel(s['tag'][i])]
        if len(idx) < 16: return None
        res = {}
        for k in TRAITS:
            y = [s[k][i] for i in idx]
            res[k] = block_perm_rho(y)
        hz = np.mean([SIGN[k] * res[k][0] for k in TRAITS])
        return label or name, len(idx), res, hz
    jobs = [('CS', None, 'CS whole book'), ('CS', lambda t: t[1] == 'V1', 'CS vol 1'), ('CS', lambda t: t[1] == 'V2', 'CS vol 2'),
            ('VZL_n18', None, 'Voynich ZL n18 folio order'), ('VZL_n18', lambda t: t[1].endswith('A'), 'Voynich A n18'),
            ('VZL_n18', lambda t: t[1].endswith('B'), 'Voynich B n18'), ('VZL_n18', lambda t: t[2] in (1, '1'), 'Voynich hand 1 n18'),
            ('VZL', None, 'Voynich ZL clean folio order')]
    jobs += [(k, None, k) for k in T if k.endswith('_n18') and not k.startswith('VZL')]
    for nm, sel, lab in jobs:
        r = trend(nm, sel, lab)
        if r: rows.append(r); print(r[0], r[1], {k: (round(a, 2), round(b, 3)) for k, (a, b) in r[2].items()}, 'hardening index', round(r[3], 3), flush=True)
    # volume / language step
    def step(name, f1, f2):
        s = T[name]; a = [i for i, t in enumerate(s['tag']) if f1(t)]; b = [i for i, t in enumerate(s['tag']) if f2(t)]
        return {k: (float(np.mean([s[k][i] for i in a])), float(np.mean([s[k][i] for i in b]))) for k in TRAITS}
    stCS = step('CS', lambda t: t[1] == 'V1', lambda t: t[1] == 'V2')
    stV = step('VZL_n18', lambda t: t[1].endswith('A'), lambda t: t[1].endswith('B'))
    stV0 = step('VZL', lambda t: t[1].endswith('A'), lambda t: t[1].endswith('B'))
    print('CS V1->V2', stCS); print('Voy A->B n18', stV); print('Voy A->B clean', stV0)
    # paired halves: gibberish writers vs language chunks
    def paired(prefix):
        names = [k for k in T if k.startswith(prefix)]
        res = {}
        for k in TRAITS:
            d = np.array([SIGN[k] * (T[n][k][-1] - T[n][k][0]) for n in names])
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            res[k] = (float(np.mean(d > 0)), float(np.median(d)), float(p))
        return len(names), res
    gb = paired('B|'); lh = paired('Lh|')
    print('gibberish halves', gb); print('language halves', lh)
    L.save('cycle3.json', {'rows': rows, 'stCS': stCS, 'stV': stV, 'stV0': stV0, 'gb': gb, 'lh': lh})
    # ---------- table rows
    fm = lambda res: '; '.join(f'{k} {a:+.2f} (p {b:.3f})' for k, (a, b) in res.items())
    n = 1
    for lab, nwin, res, hz in rows:
        L.row(FN, f'V-42.3.{n}', f'{lab}: Spearman of 6 traits with writing order over {nwin} 100-token windows; null = block permutation (8-window blocks, 2000x); hardening index = mean signed rho (GEN up, rigidity up, family up, h2 down, TTR down, position-MI up)',
              f'{fm(res)}; hardening index {hz:+.2f}',
              ('hardens along the writing order' if hz > 0.1 else ('softens' if hz < -0.1 else 'no consistent drift')) + (' (control)' if lab.startswith(('L_', 'G_', 'P_')) else ''))
        n += 1
    st = lambda d: '; '.join(f'{k} {a:.3f}->{b:.3f}' for k, (a, b) in d.items())
    L.row(FN, f'V-42.3.{n}', 'Step between halves of a book: CS volume 1 -> 2 (raw OCR); Voynich Currier A -> B (n18 and clean)',
          f'CS: {st(stCS)} || Voynich n18: {st(stV)} || Voynich clean: {st(stV0)}', 'see verdict'); n += 1
    fp = lambda r: f'n {r[0]}: ' + '; '.join(f'{k} up-share {a:.2f}, median {m:+.3f}, p {p:.3f}' for k, (a, m, p) in r[1].items())
    L.row(FN, f'V-42.3.{n}', 'Gibberish writers (Gaskell-Bowern): first vs last 100 tokens of each writer, signed so + = toward hardening; Wilcoxon. Control: first vs 4th 100-token window of each language corpus',
          f'gibberish {fp(gb)} || languages {fp(lh)}', 'see verdict')


if __name__ == '__main__':
    main()
