"""Cross-line junction coupling: MI(last glyph of line-final word; first glyph of next line's first word)
within paragraphs, against a null that pairs line ends with random line starts of the same page."""
import math, random, sys
from collections import Counter
import numpy as np
import v61_lib as L

def mi(pairs):
    c = Counter(pairs); n = sum(c.values()); a = Counter(); b = Counter()
    for (x, y), v in c.items(): a[x] += v; b[y] += v
    return sum(v / n * math.log2(v * n / (a[x] * b[y])) for (x, y), v in c.items())

def run(lines, reps=200):
    within = [(a[-1], b[0]) for Ln in lines for a, b in zip(Ln['words'], Ln['words'][1:])]
    # subsample within to same n as cross
    cross, ends, starts = [], [], []
    for i in range(len(lines) - 1):
        A, B = lines[i], lines[i + 1]
        if A['page'] == B['page'] and not A['para_end'] and not B['para_start']:
            cross.append((A['words'][-1][-1], B['words'][0][0])); ends.append((A['page'], A['words'][-1][-1])); starts.append((B['page'], B['words'][0][0]))
    rng = random.Random(0)
    m_cross = mi(cross)
    nullc = []
    for r in range(reps):
        # permute starts within page
        bypage = {}
        for k, (p, g) in enumerate(starts): bypage.setdefault(p, []).append(k)
        perm = list(range(len(starts)))
        for p, ks in bypage.items():
            sh = ks[:]; rng.shuffle(sh)
            for a, b in zip(ks, sh): perm[a] = b
        nullc.append(mi([(ends[k][1], starts[perm[k]][1]) for k in range(len(ends))]))
    w_sub = []
    for r in range(50):
        s = rng.sample(within, len(cross)); w_sub.append(mi(s))
    # null for within at same n: shuffle pairing
    wn = []
    for r in range(50):
        s = rng.sample(within, len(cross)); xs = [a for a, b in s]; ys = [b for a, b in s]; rng.shuffle(ys); wn.append(mi(list(zip(xs, ys))))
    return {'n_cross': len(cross), 'cross_mi': m_cross, 'cross_null': float(np.mean(nullc)), 'cross_null_sd': float(np.std(nullc)),
            'within_mi_same_n': float(np.mean(w_sub)), 'within_null_same_n': float(np.mean(wn))}

if __name__ == '__main__':
    for nm, f in [('VMS-ZL', lambda: L.load_vms('ZL3b')), ('VMS-IT', lambda: L.load_vms('IT2a')),
                  ('Sanskrit', L.load_sanskrit), ('Italian', L.load_italian)]:
        X = f(); r = run(X)
        ex_c = r['cross_mi'] - r['cross_null']; ex_w = r['within_mi_same_n'] - r['within_null_same_n']
        print(nm, {k: round(v, 4) for k, v in r.items()}, 'excess cross %.4f (z %.1f) within %.4f ratio %.2f' % (ex_c, ex_c / r['cross_null_sd'], ex_w, ex_c / ex_w))
