"""v33 cycle 4c: full 6-feature profile of the contracted Latin / Italian texts' OWN generators (glyph trigram,
in-word slot), so that 'distance to own generators' can be compared with the Voynich (9.4-10.2) and Latin (21)."""
import time
from multiprocessing import Pool
from v33_lib import *
import v33_cycle1 as c1

if __name__ == '__main__':
    C = {}
    for nm, code in (('laSusp', 'la'), ('itSusp', 'it')):
        L = [[w if len(w) <= 4 else w[:2] + w[-2:] for w in l] for l in lang_lines(code)]
        for i in range(2):
            rng = random.Random(400 + i)
            C[f'H-tri-{nm}_{i}'] = Trigram(L).gen(NTOK, rng)
            C[f'H-slot-{nm}_{i}'] = SlotGen(L).gen(NTOK, rng)
    t0 = time.time()
    with Pool(2) as P:
        for n, s in P.imap_unordered(c1.job, sorted(C.items())):
            print(n, s, round(time.time() - t0), flush=True)
