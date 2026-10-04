"""v18 cycle 5: is the Voynich pen-load coherence (cycle 3: same-load word pairs more alike,
edit-similarity z 3.2, p 0.04 = surrogate floor) a content artefact?

Words that look alike may also ink alike, so a darkness 'dip' may simply fall less often
between look-alike words. Controls:
(a) coherence with dips from EDGE-AWARE content2-residualised darkness (all clean settings);
(b) coherence on CONTENT-ONLY darkness (linear content, and content2), page-shuffled noise;
(c) split half (even / odd pages), each with its own page-swap surrogates;
(d) Latin with content2 residualisation.
Checkpoint: data/results/v18/c5.json
"""
import sys, os, json, time, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from v18_dips import *
from v18_lib import glyphs
from v18_cycle2 import latin
from v18_cycle3 import pairs, simmats, coherence
from v18_cycle4 import voy_pages, model
D = os.path.join(HERE, '..', 'data', 'derived')
RES = os.path.join(HERE, '..', 'data', 'results', 'v18')


def clean_dips(C, cont):
    cache = {}; out = []
    for meas, sm, det, q, sp, ls in itertools.product(MEAS, (False, True), CLEAN, QS, SPACE, ('all', 'noLS')):
        if meas not in cache:
            cache[meas] = C.residual(meas, cont)
        out.append(((meas, cont, sm, det, q, sp, ls), np.where(C.detect(cache[meas], det, q, sp, sm, ls))[0]))
    return out


def coh(C, cont):
    I, J, Dd = pairs(C); S = simmats(C, I, J)
    maps = [page_swap_map(C, k) for k in range(1, C.npages)]
    return coherence(C, clean_dips(C, cont), maps, I, J, Dd, S)


def fake_content(C, which, seed):
    Cm = C.content_matrix() if which == 1 else C.content_matrix2()
    rng = np.random.default_rng(seed)
    for key in ('med', 'top', 'p90', 'gray', 'areag', 'rb'):
        y = np.array([w['m'][key] if w['m'] else np.nan for w in C.W]); ok = ~np.isnan(y)
        X = np.hstack([np.ones((ok.sum(), 1)), Cm[ok]])
        b = np.linalg.solve(X.T @ X + np.eye(X.shape[1]), X.T @ y[ok])
        pred = np.full(C.N, np.nan); pred[ok] = X @ b
        res_ = y - pred; perm = np.where(ok)[0].copy(); rng.shuffle(perm)
        fake = pred.copy(); fake[np.where(ok)[0]] += res_[perm]
        for i, w in enumerate(C.W):
            if w['m']:
                w['m'] = dict(w['m']); w['m'][key] = float(fake[i])
    C._C = None; C._C2 = None


def short(r):
    return {k: (round(v['z'], 2), round(v['p_one_sided'], 3)) for k, v in r.items()}


def main():
    out = {}; fn = os.path.join(RES, 'c5.json'); t0 = time.time()
    M = model()
    C = Corpus(voy_pages(), glyphs, M)
    out['voy_content1'] = coh(C, True); print('voy content1', short(out['voy_content1']), round(time.time() - t0), flush=True)
    out['voy_content2'] = coh(C, 2); print('voy content2', short(out['voy_content2']), flush=True)
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    for which in (1, 2):
        for seed in (1, 2):
            V = Corpus(voy_pages(), glyphs, M); fake_content(V, which, seed)
            k = 'content%d_only_s%d' % (which, seed)
            out[k] = coh(V, True); print(k, short(out[k]), flush=True)
            json.dump(out, open(fn, 'w'), indent=1, default=str)
    P = voy_pages()
    for h, sel in (('even', 0), ('odd', 1)):
        H = Corpus([p for i, p in enumerate(P) if i % 2 == sel], glyphs, M)
        out['half_' + h] = coh(H, True); print('half', h, short(out['half_' + h]), flush=True)
        json.dump(out, open(fn, 'w'), indent=1, default=str)
    L, _ = latin()
    out['lat_content2'] = coh(L, 2); print('lat content2', short(out['lat_content2']), flush=True)
    json.dump(out, open(fn, 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))


if __name__ == '__main__':
    main()
