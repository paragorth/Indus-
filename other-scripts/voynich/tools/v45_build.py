"""v45: build every corpus and its cross-fitted residual (checkpointed)."""
import sys, time
from multiprocessing import Pool
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from v45_lib import *

def job(n):
    t = time.time(); R = residual(n); return n, len(R), time.time() - t

if __name__ == '__main__':
    names = sys.argv[1:] or ['V', 'GEN0', 'PL', 'LAw', 'LAl', 'GEw', 'GEl', 'GEN1', 'VI']
    for n in ('V',): get_corpus(n)
    with Pool(2) as P:
        for r in P.imap_unordered(job, names): print(*r, flush=True)
