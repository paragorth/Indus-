"""pe14 cycle 3d: anatomy of the fixed two-line records. In tablets where one entry
string recurs at every second position >= 4 times (20 tablets; >= 4 repeats occur
on 7.5 tablets under the DIP null), the recurring line is the COMPANION and the line
before it the MAIN line. Does the companion's number depend on the main line's?
Statistic: Spearman rho between main and companion numeral glyph counts (and share
of companions = 1); null: companions permuted among the records of the same tablet
(10,000 permutations, tablet structure kept).
"""
import json, os, random, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa
from pe14_common import CK  # noqa
from pe14_cycle1 import corpus, run_spaced  # noqa


def glyphs(nm):
    return sum(n for n, _ in nm if isinstance(n, int))


def main():
    raw = {t['id']: t for t in load()}
    T = corpus('PE_ENT')
    recs = []
    per = {}
    for t in T:
        if run_spaced(t['u'], 2) < 4:
            continue
        S = [tuple(u['toks']) for u in t['u']]
        comp = Counter(S[i] for i in range(len(S) - 2) if S[i] == S[i + 2] and S[i + 1] != S[i]).most_common(1)[0][0]
        # numerals from the raw lines in the same order as units (entries = numbered lines with signs)
        L = [l for l in raw[t['id']]['lines'] if l['numerals'] and any(is_sign(s) for s in l['signs'])]
        nums = []
        for l in L:
            sg = tuple(base(s) for s in l['signs'] if is_sign(s))
            nums.append((sg, glyphs(l['numerals']), l['numerals'][0][1]))
        pairs = [(nums[i][1], nums[i + 1][1], nums[i + 1][2]) for i in range(len(nums) - 1)
                 if nums[i + 1][0] == comp and nums[i][0] != comp]
        if len(pairs) >= 3:
            per[t['id']] = {'companion': ' '.join(comp), 'n': len(pairs),
                            'companion_counts': Counter(p[1] for p in pairs).most_common(),
                            'companion_unit': Counter(p[2] for p in pairs).most_common(1)[0][0]}
            recs += [(t['id'], a, b) for a, b, _ in pairs]
    ids = [r[0] for r in recs]
    a = np.array([r[1] for r in recs], float)
    b = np.array([r[2] for r in recs], float)

    def spear(x, y):
        rx = np.argsort(np.argsort(x + 1e-6 * np.random.RandomState(0).rand(len(x))))
        ry = np.argsort(np.argsort(y + 1e-6 * np.random.RandomState(1).rand(len(y))))
        return np.corrcoef(rx, ry)[0, 1]
    obs = spear(a, b)
    rng = random.Random(3)
    groups = {}
    for i, t in enumerate(ids):
        groups.setdefault(t, []).append(i)
    null = []
    for _ in range(10000):
        bb = b.copy()
        for g in groups.values():
            v = [b[i] for i in g]
            rng.shuffle(v)
            for i, x in zip(g, v):
                bb[i] = x
        null.append(spear(a, bb))
    null = np.array(null)
    out = {'tablets': per, 'nrec': len(recs), 'share_comp_1': float(np.mean(b == 1)),
           'mean_main': float(a.mean()), 'mean_comp': float(b.mean()),
           'spearman': float(obs), 'null_m': float(null.mean()), 'null_sd': float(null.std()),
           'p_two': float((1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / 10001)}
    json.dump(out, open(os.path.join(CK, 'c3d_records.json'), 'w'), indent=1)
    for k, v in per.items():
        print(k, v)
    print({k: v for k, v in out.items() if k != 'tablets'})


if __name__ == '__main__':
    main()
