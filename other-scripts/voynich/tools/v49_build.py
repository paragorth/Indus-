"""v49: build control corpora and their cross-fitted residuals (checkpointed, 2 workers)."""
import sys, time, os
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v49_lib import *


def job(n):
    t = time.time(); R = residual(n); return n, len(R), time.time() - t


if __name__ == '__main__':
    names = sys.argv[1:] or ['BRe', 'BRf', 'GENN', 'GENT']
    for n in names: get_corpus(n)
    with Pool(2) as P:
        for r in P.imap_unordered(job, names): print(*r, flush=True)
