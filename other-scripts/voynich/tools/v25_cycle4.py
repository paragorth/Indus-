"""v25 cycle 4: tune the copy-and-edit generator (relabelled seeds, shape-driven edits) over a grid of
edit strength, edit rate and window, to see whether any setting reproduces ALL Voynich signatures at
once: image r, hand r, bench parallelogram, compositional LOO prediction."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v25_lib as L, v25_shapes as S, v25_image as I
from v25_cycle3 import gen_copy_edit
from multiprocessing import Pool

OUT = os.path.join(L.LOOPS, 'v25_cycle4.txt')


def run(arg):
    kappa, pe, win, seed = arg
    rng = np.random.default_rng(seed)
    vw = L.corpus('voynich'); A = L.alphabet(vw, S.VOYNICH)
    Sim = I.image_sims('voynich', A)['combo']
    pm = dict(zip(A, rng.permutation(A)))
    seeds = [tuple(pm[g] for g in x) for x in vw]
    P = Sim * kappa / 5.0  # gen uses exp(5*(Sim-mean)); scale similarity to emulate kappa
    g = gen_copy_edit(seeds, len(vw), P, A, rng, shape=True, p_edit=pe, window=win)
    B = L.behaviour(g, A, models=('ppmi', 'svd'))
    F, _ = L.shape_matrix(S.VOYNICH, A); Sh = L.jaccard_sim(F)
    iu = np.triu_indices(len(A), 1)
    ri = L.spearman_vec(Sim[iu], B['ppmi'][iu]); rh = L.spearman_vec(Sh[iu], B['ppmi'][iu])
    bench = L.analogy_score(B['_emb'], A, S.V_ANALOGIES['bench (+c_h)'])
    wide = L.analogy_score(B['_emb'], A, S.V_ANALOGIES['wide (+WIDE)'])
    ttr = len(set(g)) / len(g)
    return (kappa, pe, win, ri, rh, bench, wide, ttr)


if __name__ == '__main__':
    grid = [(k, pe, w, 100 + i) for i, (k, pe, w) in enumerate(itertools.product((5, 10, 20), (0.3, 0.7, 0.95), (10, 30, 100)))]
    with Pool(2) as pool:
        res = pool.map(run, grid)
    vw = L.corpus('voynich'); A = L.alphabet(vw, S.VOYNICH)
    B = L.behaviour(vw, A, models=('ppmi', 'svd'))
    Sim = I.image_sims('voynich', A)['combo']; F, _ = L.shape_matrix(S.VOYNICH, A); Sh = L.jaccard_sim(F)
    iu = np.triu_indices(len(A), 1)
    V = (L.spearman_vec(Sim[iu], B['ppmi'][iu]), L.spearman_vec(Sh[iu], B['ppmi'][iu]),
         L.analogy_score(B['_emb'], A, S.V_ANALOGIES['bench (+c_h)']), L.analogy_score(B['_emb'], A, S.V_ANALOGIES['wide (+WIDE)']), len(set(vw)) / len(vw))
    rows = []
    for k, pe, w, ri, rh, b, wd, t in res:
        allok = ri >= V[0] and rh >= V[1] and b >= V[2]
        rows.append((f'V-25.4.1 k{k} e{pe} w{w}', 'copy-edit generator, relabelled Voynich seeds, shape-driven edits (strength k, edit rate e, window w)',
                     f'image r {ri:+.3f}, hand r {rh:+.3f}, bench {b:+.2f}, wide {wd:+.2f}, TTR {t:.2f}', 'matches all Voynich signatures' if allok else 'falls short'))
    n_ok = sum(r[3] == 'matches all Voynich signatures' for r in rows)
    best = max(res, key=lambda x: min(x[3] / V[0], x[4] / V[1], x[5] / V[2]))
    rows.append(('V-25.4.2 Voynich reference', 'same statistics on ZL', f'image r {V[0]:+.3f}, hand r {V[1]:+.3f}, bench {V[2]:+.2f}, wide {V[3]:+.2f}, TTR {V[4]:.2f}', ''))
    rows.append(('V-25.4.V', f'Verdict: can stroke-edit copying from a shape-neutral start reproduce the Voynich? 27 settings', f'{n_ok}/27 settings match image r, hand r and bench at once; closest k{best[0]} e{best[1]} w{best[2]}: image {best[3]:+.3f}, hand {best[4]:+.3f}, bench {best[5]:+.2f}, TTR {best[7]:.2f}', ''))
    hdr = ('# v25 cycle 4 - kill attempt: tuned copy-and-edit generator vs all Voynich shape signatures.\n| row | method and control | result | verdict |\n|---|---|---|---|')
    L.write_rows(OUT, rows, hdr)
    print(open(OUT).read())
