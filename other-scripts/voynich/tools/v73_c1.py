"""v73 cycle 1: massive random marking rules (one word per line) on Voynich ZL3b, six planted list texts
(Antidotarium Nicolai ingredient lists, one item per line, inside generator filler fitted to the Voynich; three
hidden marking schemes x two filler types) and two generator-only false-positive targets.
Each target is scored with its own nulls (generators fitted to the target, and within-page line shuffle)."""
import os, sys, time, json
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v73_lib as L, v72_lib as V

NR = int(os.environ.get('V73_NR', 2500))
NULLSET = ['MK2', 'SELFCIT', 'SC10', 'JUNC', 'LSHUF']


def targets(src='ZL3b'):
    P = V.voynich(src)
    T = {('V_' + src): P}
    if src == 'ZL3b':
        for filler in ('MK2', 'SELFCIT'):
            for code in ('VOY', 'NOVEL'):
                for sch in ('POS2', 'AFTER', 'FREE'):
                    pp, truth = L.plant(P, sch, filler, code)
                    nm = 'PL_%s_%s_%s' % (sch, filler, code)
                    T[nm] = pp
                    L.jsave('truth_%s.json' % nm, truth)
        T['FP_SELFCIT'] = L.NULLS['SELFCIT'](P, 501)
        T['FP_MK2'] = L.NULLS['MK2'](P, 502)
    return T


def job(args):
    tname, kind, pages = args
    out = os.path.join(L.CK, 'S_%s__%s.npy' % (tname, kind))
    if os.path.exists(out): return out
    if kind != 'REAL': pages = L.NULLS[kind](pages, 11)
    C = L.Corpus(pages, kind)
    bank, _ = L.make_bank(NR)
    S = L.score_bank(C, bank)
    np.save(out, S.astype(np.float32))
    if kind == 'REAL':
        np.save(os.path.join(L.CK, 'half_%s.npy' % tname), C.half)
    return out


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'ZL3b'
    T = targets(src)
    jobs = [(t, k, P) for t, P in T.items() for k in ['REAL'] + NULLSET]
    t0 = time.time()
    with Pool(2) as pool:
        for i, o in enumerate(pool.imap_unordered(job, jobs)):
            print('%3d/%d %6.0fs %s' % (i + 1, len(jobs), time.time() - t0, os.path.basename(o)), flush=True)
