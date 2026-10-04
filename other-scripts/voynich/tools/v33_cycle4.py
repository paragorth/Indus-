"""v33 cycle 4: KILL TEST for the cycle-2 verdict ('Voynich ecology = glyph-generator ecology').
Do short or recoded LANGUAGE texts also fall in the generator cluster with the full 6-feature profile?
Queries (full cycle-1 profile with 30 curveball nulls, classified against the cycle-1/2 references):
  C-laSusp   Latin, words > 4 letters contracted to first 2 + last 2 (keeps endings)
  C-itSusp   Italian, the same
  C-laTrunc  Latin, words cut to their first 4 letters (suspension, endings lost)
  C-la8cls   Latin letters merged into 8 classes (small alphabet)
  C-laVerb   Latin, every letter written as a 2-glyph code from a 6-glyph alphabet (verbose cipher, longer words)
  P-decl100 / P-decl50   Voynich trigram text with planted declension classes (cycle 3)
Usage: python3 v33_cycle4.py [nworkers]"""
import sys, time, zlib
from multiprocessing import Pool
from v33_lib import *
import v33_cycle1 as c1
from v33_cycle3 import CLS, declension_text


def corpora():
    la, it = lang_lines('la'), lang_lines('it')
    rng = random.Random(4)
    letters = sorted(set(c for l in la for w in l for c in w))
    codes = [a + b for a in 'oeyadk' for b in 'oeyadk']; rng.shuffle(codes)
    VC = dict(zip(letters, codes))
    S = {'C-laSusp': [[w if len(w) <= 4 else w[:2] + w[-2:] for w in l] for l in la],
         'C-itSusp': [[w if len(w) <= 4 else w[:2] + w[-2:] for w in l] for l in it],
         'C-laTrunc': [[w[:4] for w in l] for l in la],
         'C-la8cls': [[''.join(CLS.get(c, 'h') for c in w) for w in l] for l in la],
         'C-laVerb': [[''.join(VC[c] for c in w) for w in l] for l in la],
         'P-decl100': declension_text(voy_lines('ZL3b'), 1.0, 1),
         'P-decl50': declension_text(voy_lines('ZL3b'), 0.5, 2)}
    C = {}
    for n, L in S.items():
        for i, b in enumerate(blocks(L, seed=33)): C[f'{n}_{i}'] = b
    return C


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    t0 = time.time()
    C = corpora()
    with Pool(nw) as P:
        for n, s in P.imap_unordered(c1.job, sorted(C.items())):
            print(n, s, round(time.time() - t0), flush=True)
