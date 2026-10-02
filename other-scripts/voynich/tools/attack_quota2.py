"""N1b: line quotas/clumping corrected for line length. Statistic: chi-square-like dispersion of a class's count per
line around its expectation given the line's glyph total and the paragraph's class rate; observed / mean of null
(words re-dealt among the paragraph's lines, line word-counts kept, 200x). >1 = class clumps in some lines,
<1 = spread evenly (quota)."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from attack_quota import paras, VP, VC

def disp(P, cls):
    s = 0.0
    for p in P:
        tot = [sum(len(w) for w in l) for l in p]; c = [sum(1 for w in l for ch in w if ch in cls) for l in p]
        T = sum(tot); C = sum(c)
        if T == 0 or C == 0: continue
        r = C / T
        for t, x in zip(tot, c):
            e = t * r
            if e > 0: s += (x - e) ** 2 / e
    return s

def test(P, cls, reps=200, seed=1):
    rng = random.Random(seed); o = disp(P, cls); sims = []
    for _ in range(reps):
        NP = []
        for p in P:
            ws = [w for l in p for w in l]; rng.shuffle(ws); k = 0; q = []
            for l in p: q.append(ws[k:k+len(l)]); k += len(l)
            NP.append(q)
        sims.append(disp(NP, cls))
    m = sum(sims)/len(sims)
    return round(o/m, 3), sum(s >= o for s in sims)/reps, sum(s <= o for s in sims)/reps

if __name__ == '__main__':
    print('Voynich ZL (ratio, p_clump, p_even)')
    for k, c in VC.items():
        if c != 'ALL': print(f'  {k:8s}', test(VP, c))
    for key in ('Latin-Caesar', 'Italian-Manzoni', 'German-Kafka', 'Italian-Dante'):
        P = paras(vlib.load_ref(key))[:400]
        print(key, {k: test(P, c, reps=100) for k, c in {'vowels': set('aeiou'), 's': {'s'}, 't': {'t'}, 'n': {'n'}, 'r': {'r'}, 'c': {'c'}, 'm': {'m'}}.items()})
