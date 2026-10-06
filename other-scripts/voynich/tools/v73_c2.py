"""v73 cycle 2: stranger markers and a learned marker.
(a) 2,514 random rules over model-based features (the word the filler cannot predict: left/right context
    surprisal, odd spelling, first mention on the page / in the book, odd-one-out of the line);
(b) a LEARNED marker: on discovery pages, iterated conditional modes choose one word per line that makes the
    stream most page- and neighbour-concentrated; a conditional-logit rule is fitted to those choices and applied
    unchanged to held-out pages (the nulls get the identical procedure).
Targets: Voynich ZL3b and IT2a; four planted list texts (Antidotarium Nicolai and Apicius ingredient lists);
two generator false-positive targets."""
import os, sys, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v73_lib as L, v72_lib as V

NR = 2500
NULLSET = ['MK2', 'SELFCIT', 'SC10', 'JUNC', 'LSHUF']
PLANTS = [('antid', 'FREE', 'SELFCIT', 'VOY'), ('antid', 'FREE', 'MK2', 'NOVEL'), ('antid', 'AFTER', 'SELFCIT', 'VOY'),
          ('apic', 'POS2', 'SELFCIT', 'VOY'), ('apic', 'FREE', 'SELFCIT', 'NOVEL')]


def targets():
    P = V.voynich('ZL3b')
    T = {'V_ZL3b': P, 'V_IT2a': V.voynich('IT2a')}
    for which, sch, fil, code in PLANTS:
        pp, truth = L.plant(P, sch, fil, code, which=which, seed=273)
        nm = 'Q_%s_%s_%s_%s' % (which, sch, fil, code)
        T[nm] = pp; L.jsave('truth_%s.json' % nm, truth)
    T['FP2_SELFCIT'] = L.NULLS['SELFCIT'](P, 601)
    T['FP2_JUNC'] = L.NULLS['JUNC'](P, 602)
    return T


def job(args):
    tname, kind, pages = args
    out = os.path.join(L.CK, 'S2_%s__%s.npy' % (tname, kind))
    if os.path.exists(out): return out
    if kind != 'REAL': pages = L.NULLS[kind](pages, 21)
    C = L.extend(L.Corpus(pages, kind))
    bank, _ = L.make_bank2(NR)
    S = L.score_bank(C, bank)
    # learned marker (discovery half only)
    disc = C.half == 0
    idx = L.icm_select(C, disc)
    w = L.fit_rule(C, idx, disc)
    S_l = L.score_bank(C, w[None, :])
    np.save(out, np.concatenate([S, S_l]).astype(np.float32))
    np.save(os.path.join(L.CK, 'W2_%s__%s.npy' % (tname, kind)), w)
    return out


if __name__ == '__main__':
    T = targets()
    jobs = [(t, k, P) for t, P in T.items() for k in ['REAL'] + NULLSET]
    t0 = time.time()
    with Pool(2) as pool:
        for i, o in enumerate(pool.imap_unordered(job, jobs)):
            print('%3d/%d %6.0fs %s' % (i + 1, len(jobs), time.time() - t0, os.path.basename(o)), flush=True)
