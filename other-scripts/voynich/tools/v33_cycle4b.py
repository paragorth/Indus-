"""v33 cycle 4b: gap ratios (cycle-3 job) for the extra contracted / truncated / verbose sources."""
import time
from multiprocessing import Pool
from v33_lib import *
import v33_cycle3 as c3
import v33_cycle4 as c4

if __name__ == '__main__':
    la, it = lang_lines('la'), lang_lines('it')
    C = c4.corpora()
    # rebuild full source lines (not blocks) for the generators
    rng = random.Random(4)
    letters = sorted(set(c for l in la for w in l for c in w))
    codes = [a + b for a in 'oeyadk' for b in 'oeyadk']; rng.shuffle(codes)
    VC = dict(zip(letters, codes))
    S = {'C-itSusp': [[w if len(w) <= 4 else w[:2] + w[-2:] for w in l] for l in it],
         'C-laTrunc': [[w[:4] for w in l] for l in la],
         'C-laVerb': [[''.join(VC.get(c, 'kk') for c in w) for w in l] for l in la]}
    t0 = time.time()
    with Pool(2) as P:
        for n, s in P.imap_unordered(c3.job, sorted(S.items())):
            print(n, s, round(time.time() - t0), flush=True)
