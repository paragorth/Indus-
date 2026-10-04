"""pe27 cycle 7: held-out test of the M288 rate found in cycle 4 (single-line windows).
Discovery set: M288 lines written with capacity signs (N39B/N24/...) right after a count line:
9 of them = 60 N39C x the count (60 = 2(N39B) 1(N24) a-priori units).
Held-out set: M288 lines written ONLY with N01/N14 (cycle 1-6 parsed these as counts), right after
a count line (within 2 lines, count >= 2).  If the rate is real and N01/N14 there are capacity
signs, y_cap = 60 x (N01 = 120, N14 = 720) must hold more often than chance.
Null: M288 values re-dealt among all M288 lines of the held-out set (100,000 times) and, second,
counts re-dealt among the preceding count lines.  Also every value set's verdict and the full
ratio list (no cherry-picking)."""
import json, os, sys
import numpy as np
from fractions import Fraction as Fr
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe27_common import CK, CAPV, load, base, is_sign, _clean_line, SEX, DEC  # noqa

T = load()
C_CODES = {'N39B', 'N30C', 'N24', 'N30D', 'N39C'}


def collect():
    disc, held = [], []
    for t in T:
        L = t['lines']
        num = [(i, l) for i, l in enumerate(L) if l['numerals'] and _clean_line(l)]
        for k, (i, l) in enumerate(num):
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            if not sg or sg[-1] != 'M288':
                continue
            codes = {c.split('@')[0] for _, c in l['numerals']}
            if any('@' in c for _, c in l['numerals']):
                continue
            # nearest preceding count line within 2 numeric lines (N01/N14 only, not M288)
            prev = None
            for kk in range(k - 1, max(-1, k - 3), -1):
                i2, l2 = num[kk]
                sg2 = [base(s) for s in l2['signs'] if is_sign(s)]
                c2 = {c for _, c in l2['numerals']}
                if sg2 and sg2[-1] != 'M288' and c2 <= {'N01', 'N14'}:
                    prev = sum(n * SEX[c] for n, c in l2['numerals'])
                    break
            if not prev or prev < 2:
                continue
            rec = {'tab': t['id'], 'des': t['designation'], 'x': prev, 'nums': l['numerals'], 'raw': l['raw']}
            if codes & C_CODES:
                disc.append(rec)
            elif codes <= {'N01', 'N14'}:
                held.append(rec)
    return disc, held


def yval(nums, vs):
    if not all(c in vs for _, c in nums):
        return None
    return sum(n * vs[c] for n, c in nums)


if __name__ == '__main__':
    disc, held = collect()
    res = {'n_disc': len(disc), 'n_held': len(held)}
    rng = np.random.default_rng(0)
    for name, S in (('discovery', disc), ('heldout', held)):
        for vsn, vs in CAPV.items():
            ys = [yval(r['nums'], vs) for r in S]
            xs = [r['x'] for r in S]
            ok = [i for i in range(len(S)) if ys[i] is not None]
            ratios = Counter(str(Fr(ys[i], xs[i])) for i in ok)
            r60 = Fr(vs['N01'], 2)   # the cycle-4 rate, expressed as half an N01
            hit = sum(1 for i in ok if Fr(ys[i], xs[i]) == r60)
            nul = []
            yv = [ys[i] for i in ok]
            xv = [xs[i] for i in ok]
            for _ in range(100000 if len(ok) < 200 else 20000):
                p = rng.permutation(len(yv))
                nul.append(sum(1 for j in range(len(yv)) if Fr(yv[p[j]], xv[j]) == r60))
            nul = np.array(nul)
            res[f'{name}_{vsn}'] = {'n': len(ok), 'rate_tested': str(r60), 'hits': hit,
                                    'null_mean': round(float(nul.mean()), 2),
                                    'p': float(((nul >= hit).sum() + 1) / (len(nul) + 1)),
                                    'ratio_counts': ratios.most_common(8)}
            print(name, vsn, res[f'{name}_{vsn}'], flush=True)
    res['heldout_cases'] = [(r['tab'], r['des'], r['x'], r['raw']) for r in held]
    res['disc_cases'] = [(r['tab'], r['des'], r['x'], r['raw']) for r in disc]
    for r in held:
        print('  H', r['tab'], r['des'], r['x'], '->', r['raw'])
    json.dump(res, open(os.path.join(CK, 'cycle7.json'), 'w'), indent=1, default=str)
