"""v62 cycle 2c: planted controls in Voynich vocabulary for the cycle 2b result.

Cycle 2b: Voynich boundary positions are concentrated beyond slot-swapped and interior-shuffled lines
(held-out, both transcriptions) but line weight totals vary MORE than slot-swapped lines (anti-metre).
Two readings: (M) a metre/caesura; (H) lines are internally homogeneous (similar words in a line, the
known neighbour coupling), which puts boundaries on a per-line lattice and inflates total variance.
Planted texts built from ZL words, same page/line layout:
  P-metre: words dealt into lines so that each line's total under a HIDDEN random 0/1 weight hits a
           fixed target (a planted metre; no similarity);
  P-homog: each line picks an anchor word; every other word is drawn from words sharing the anchor's
           last two glyphs (p 0.5) or at random (no metre);
  P-caes:  planted caesura: the first half of each line is filled to hit a hidden-weight target, the
           rest free.
Each goes through the same search as cycle 2b (random weights + hill-climb, train/test halves).
Reading (M) predicts the Voynich resembles P-metre / P-caes (metre_N2 > 0); (H) predicts P-homog
(caes/cad N2 > 0 with metre_N2 < 0).
"""
import sys, os, json, random
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v62_lib as L
import v62_c2 as C2
import v62_c2b as B

A = C2.A


def words_and_layout(pages):
    ws = [w for pg in pages for l in pg for w in l]
    lay = [[len(l) for l in pg] for pg in pages]
    return ws, lay


def p_metre(pages, seed, half=False):
    rng = random.Random(seed)
    ws, lay = words_and_layout(pages)
    alpha = L.alphabet(pages)
    hw = {a: rng.random() < 0.4 for a in alpha}
    wt = lambda w: sum(hw.get(s, 0) for s in w)
    pool = ws[:]; rng.shuffle(pool)
    bywt = {}
    for w in pool: bywt.setdefault(wt(w), []).append(w)
    mean_w = np.mean([wt(w) for w in ws])
    out = []
    for pg in lay:
        q = []
        for n in pg:
            target = int(round(mean_w * (n if not half else n // 2)))
            line = [rng.choice(pool) for _ in range(n)]
            k = n if not half else min(n, max(2, n // 2))
            # adjust the first k words so their hidden total hits the target
            for _ in range(200):
                tot = sum(wt(w) for w in line[:k])
                if tot == target: break
                j = rng.randrange(k)
                need = wt(line[j]) + (target - tot)
                need = max(min(need, max(bywt)), min(bywt))
                cand = bywt.get(need) or bywt.get(need - 1) or bywt.get(need + 1)
                if cand: line[j] = rng.choice(cand)
            q.append(line)
        out.append(q)
    return out


def p_homog(pages, seed, p=0.5):
    rng = random.Random(seed)
    ws, lay = words_and_layout(pages)
    byend = {}
    for w in ws: byend.setdefault(w[-2:], []).append(w)
    out = []
    for pg in lay:
        q = []
        for n in pg:
            a = rng.choice(ws); fam = byend[a[-2:]]
            q.append([a] + [rng.choice(fam) if rng.random() < p else rng.choice(ws) for _ in range(n - 1)])
        out.append(q)
    return out


def main():
    zl_tr, zl_te = C2.split('V-ZL3b')
    tasks = [('P-metre', p_metre(zl_tr, 1), [('half2', p_metre(zl_te, 1))], 5),
             ('P-caes', p_metre(zl_tr, 2, half=True), [('half2', p_metre(zl_te, 2, half=True))], 5),
             ('P-homog', p_homog(zl_tr, 3), [('half2', p_homog(zl_te, 3))], 5),
             ('P-homog-weak', p_homog(zl_tr, 4, 0.25), [('half2', p_homog(zl_te, 4, 0.25))], 5)]
    res = {}
    with Pool(2) as pool:
        for tag, r in pool.imap_unordered(B.run, tasks):
            res[tag] = r
            json.dump(res, open(os.path.join(L.CK, 'cycle2c.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
