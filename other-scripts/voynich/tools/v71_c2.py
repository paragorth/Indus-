"""v71 cycle 2: the scrambling writer.  Real reference chunks (encoded, laid out as in cycle 1)
are scrambled by a simulated writer who permutes words within a line (S1), lines within a
paragraph (S2) and paragraphs within a page (S3), each with probability f in {0, 0.5, 1}; their
fingerprints use the cycle-1 model set, so they live in the same space as the Voynich.
Corners (f in {0,1}^3) + midpoints are computed for real; a multilinear surrogate over the cube
then scores thousands of random (text, f1, f2, f3) writer hypotheses against each Voynich section
(cycle-2 report).  Results: data/v71_ckpt/c2/<job>.json."""
import os, sys, json, random, time, itertools
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v71_lib as L
import v71_c1 as C1

OUTD = os.path.join(L.CK, 'c2'); os.makedirs(OUTD, exist_ok=True)
TEXTS = ['forme_of_cury', 'antidotarium_nl', 'culpeper', 'circa_fr', 'manzoni', 'apicius_index']
PLANT = ['forme_of_cury', 'culpeper']   # chunk 1, corners only: hidden fractions to recover
GRID = list(itertools.product([0, 1], repeat=3)) + [(0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0.5)]

def jobs():
    R = json.load(open(os.path.join(L.CK, 'refs.json')))
    J = []
    for k in TEXTS:
        rng = random.Random('ref' + k)
        chs = L.ref_chunks(R, k, rng, nmax=4)
        for ci in (0, 1):
            if ci >= len(chs) or (ci == 1 and k not in PLANT): continue
            for f in (GRID if ci == 0 else GRID[:8]):
                if f == (0, 0, 0): continue          # = cycle-1 job ref__k__ci
                P = L.partial_shuffle(chs[ci], {'S1': f[0], 'S2': f[1], 'S3': f[2]}, random.Random('%s%d%s' % (k, ci, f)))
                J.append(('scr__%s__%d__%g_%g_%g' % (k, ci, f[0], f[1], f[2]),
                          {'kind': 'scr', 'text': k, 'chunk': ci, 'f': list(f), 'coarse': R[k]['coarse']}, P))
    return J

if __name__ == '__main__':
    C1.OUTD = OUTD
    J = jobs(); print('jobs', len(J), flush=True)
    with Pool(2) as p:
        for name, st in p.imap_unordered(C1.run, J):
            print(name, st, flush=True)
    print('DONE', flush=True)
