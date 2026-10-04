"""v18 cycle 4: where does the full-family Voynich 'reset' score come from, and is it an artefact?

Cycle 1 full family (6,048 settings, incl. detectors that use the darkness of the two words at
the boundary): reset score 0.46 vs page-swap surrogates 0.06 +- 0.15 (p = 0.04, the floor with
24 surrogates); junction z -1.10 vs -0.2 +- 0.3, line-initial-likeness +1.07 vs 0.27 +- 0.39.
Latin and content-only darkness showed nothing. Tests here:
(a) decomposition of the Voynich score by detector, measure, residualisation, line-start mode;
(b) EDGE-AWARE residualisation (content2: one-hot first/last glyph of the word and of its
    neighbours) for all detectors; same for Latin;
(c) edge-aware CONTENT-ONLY darkness (prediction from content2 + page-shuffled residuals);
(d) split half: settings ranked on even pages, top 5% tested on odd pages and vice versa,
    each against its own page-swap surrogates.
Checkpoint: data/results/v18/c4.json
"""
import sys, os, json, time, itertools, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v18_dips import *
from v18_lib import glyphs
from v18_cycle1 import summarise, SIGN
from v18_cycle2 import latin, lglyphs
D = os.path.join(HERE, '..', 'data', 'derived')
RES = os.path.join(HERE, '..', 'data', 'results', 'v18')


def dips_for(C, conts=(2,)):
    cache = {}; out = []
    for meas, cont, sm, det, q, sp, ls in itertools.product(MEAS, conts, (False, True), DET, QS, SPACE, ('all', 'noLS')):
        if (meas, cont) not in cache:
            cache[(meas, cont)] = C.residual(meas, cont)
        out.append(((meas, cont, sm, det, q, sp, ls), np.where(C.detect(cache[(meas, cont)], det, q, sp, sm, ls))[0]))
    return out


def score_row(z):
    return sum(SIGN[f] * z[f] for f in SIGN) / 4


def decompose(ev):
    """mean reset score (real minus surrogate mean) by setting factor."""
    out = {}
    names = ['meas', 'cont', 'smooth', 'det', 'q', 'space', 'ls']
    for fi, name in enumerate(names):
        g = collections.defaultdict(list)
        for st, n, row in ev:
            real = score_row(row[0]); sur = np.mean([score_row(r) for r in row[1:]])
            g[str(st[fi])].append(real - sur)
        out[name] = {k: round(float(np.mean(v)), 3) for k, v in g.items()}
    return out


def voy_pages():
    return json.load(open(os.path.join(D, 'v18_words.json')))


def model():
    ZL = json.load(open(os.path.join(D, 'ZL3b_lines.json')))
    return [r['words'] for r in ZL if r['ltype'] == 'P']


def main():
    out = {}
    fn = os.path.join(RES, 'c4.json')
    t0 = time.time()
    M = model()
    C = Corpus(voy_pages(), glyphs, M)
    maps = [page_swap_map(C, k) for k in range(1, C.npages)]
    # (a) decomposition of cycle-1 full family
    ds = all_dips(C)
    ev = evaluate(C, ds, maps)
    out['decomp_full'] = decompose(ev)
    print('decomp', out['decomp_full'], round(time.time() - t0), flush=True)
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    # (b) edge-aware residualisation, Voynich
    ds2 = dips_for(C, (2,))
    ev2 = evaluate(C, ds2, maps)
    out['voy_content2'] = summarise(ev2, C.npages); out['voy_content2_decomp'] = decompose(ev2)
    print('voy content2', {k: out['voy_content2'][k] for k in ('max_abs_z', 'p_max', 'reset_score', 'p_score', 'sur_score_mean', 'sur_score_sd')}, flush=True)
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    # (b') Latin
    L, _ = latin()
    lmaps = [page_swap_map(L, k) for k in range(1, L.npages)]
    evl = evaluate(L, dips_for(L, (2,)), lmaps)
    out['lat_content2'] = summarise(evl, L.npages)
    print('lat content2', {k: out['lat_content2'][k] for k in ('max_abs_z', 'p_max', 'reset_score', 'p_score', 'sur_score_mean', 'sur_score_sd')}, flush=True)
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    # (c) edge-aware content-only darkness
    V = Corpus(voy_pages(), glyphs, M)
    Cm = V.content_matrix2()
    rng = np.random.default_rng(9)
    for key in ('med', 'top', 'p90', 'gray', 'areag', 'rb'):
        y = np.array([w['m'][key] if w['m'] else np.nan for w in V.W]); ok = ~np.isnan(y)
        X = np.hstack([np.ones((ok.sum(), 1)), Cm[ok]])
        b = np.linalg.solve(X.T @ X + np.eye(X.shape[1]), X.T @ y[ok])
        pred = np.full(V.N, np.nan); pred[ok] = X @ b
        res_ = y - pred; perm = np.where(ok)[0].copy(); rng.shuffle(perm)
        fake = pred.copy(); fake[np.where(ok)[0]] += res_[perm]
        for i, w in enumerate(V.W):
            if w['m']:
                w['m'] = dict(w['m']); w['m'][key] = float(fake[i])
    V._C = None; V._C2 = None
    evc = evaluate(V, all_dips(V), [page_swap_map(V, k) for k in range(1, V.npages)])
    out['content2_only_full'] = summarise(evc, V.npages)
    print('content2-only full', {k: out['content2_only_full'][k] for k in ('max_abs_z', 'p_max', 'reset_score', 'p_score', 'sur_score_mean', 'sur_score_sd')}, flush=True)
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    # (d) split half on the cycle-1 full family
    P = voy_pages()
    halves = {'even': [p for i, p in enumerate(P) if i % 2 == 0], 'odd': [p for i, p in enumerate(P) if i % 2 == 1]}
    hres = {}
    for h, pages in halves.items():
        H = Corpus(pages, glyphs, M)
        hev = evaluate(H, all_dips(H), [page_swap_map(H, k) for k in range(1, H.npages)])
        hres[h] = [(st, score_row(row[0]), [score_row(r) for r in row[1:]]) for st, n, row in hev]
        print('half', h, 'done', round(time.time() - t0), flush=True)
    sp = {}
    for a, b in (('even', 'odd'), ('odd', 'even')):
        ra = sorted(hres[a], key=lambda x: -(x[1] - np.mean(x[2])))
        top = set(st for st, _, _ in ra[:len(ra) // 20])
        rb = [x for x in hres[b] if x[0] in top]
        real = float(np.mean([x[1] for x in rb])); nsur = len(rb[0][2])
        sur = [float(np.mean([x[2][j] for x in rb])) for j in range(nsur)]
        sp[a + '->' + b] = {'n_settings': len(rb), 'score': real, 'sur_mean': float(np.mean(sur)), 'sur_sd': float(np.std(sur)),
                            'p': (1 + sum(s >= real for s in sur)) / (1 + nsur)}
        # all-settings correlation between halves
    ea = {str(st): s - np.mean(su) for st, s, su in hres['even']}; oa = {str(st): s - np.mean(su) for st, s, su in hres['odd']}
    ks = list(ea)
    sp['corr_settings_even_odd'] = float(np.corrcoef([ea[k] for k in ks], [oa[k] for k in ks])[0, 1])
    sp['even_full_score'] = float(np.mean([ea[k] for k in ks])); sp['odd_full_score'] = float(np.mean([oa[k] for k in ks]))
    out['split_half'] = sp
    print('split', sp, flush=True)
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))


if __name__ == '__main__':
    main()
