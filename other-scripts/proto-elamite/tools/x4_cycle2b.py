"""X-4 cycle 2b: more seeds (2-5) for the gap ratio of each corpus and of its own fitted sign-trigram generator
(TRI), so that the generator-relative gap ratio  G = g_tri(real) / g_tri(TRI generator)  gets a spread.
G < 1: the text leaves lexical gaps its own generator does not; G ~ 1: generator-like (v33 Voynich reading).
Same pipeline and files as x4_cycle2.py (data/x4_ckpt/c2/)."""
import os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x4_lib as X
import x4_cycle2 as C2

if __name__ == '__main__':
    C = X.corpora(); J = []
    for budget in (3000, 8000):
        for name in X.LIST + ['VOY'] + X.PROSE:
            if X.ntok(C[name]) < budget * 0.95: continue
            for seed in (2, 3, 4, 5):
                for cond in ('real', 'TRI'):
                    J.append((name, budget, seed, cond))
    print(len(J), 'jobs', flush=True)
    with Pool(int(os.environ.get('W', '2'))) as p:
        for _ in p.imap_unordered(C2.run, J): pass
