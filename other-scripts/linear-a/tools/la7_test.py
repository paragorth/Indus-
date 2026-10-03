#!/usr/bin/env python3
"""LA-7 cycle 1: does the depicted object of a word's first sign predict the commodity beside it?
Statistics: H = pairs whose first-sign class hits its pre-set commodity family (PRED);
MI = mutual information (bits) between first-sign class and commodity family.
Null: permute picture labels among coded signs (class sizes kept), N runs.
Same test on Linear B (DAMOS) using the same codes via AB numbers."""
import json, math, os, random, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la7_common import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
rng = random.Random(77)


def stats(pairs, lab):
    rows = [(lab[s], f) for s, f in pairs if s in lab]
    H = sum(1 for c, f in rows if f in PRED.get(c, ()))
    n = len(rows)
    if not n: return 0, 0.0, 0
    cc, ff, cf = Counter(c for c, _ in rows), Counter(f for _, f in rows), Counter(rows)
    mi = sum(v / n * math.log2(v * n / (cc[c] * ff[f])) for (c, f), v in cf.items())
    return H, mi, n


def perm_test(pairs, lab):
    H0, M0, n = stats(pairs, lab)
    keys = sorted(lab); vals = [lab[k] for k in keys]
    hs, ms = [], []
    for _ in range(N):
        rng.shuffle(vals); L = dict(zip(keys, vals))
        h, m, _ = stats(pairs, L); hs.append(h); ms.append(m)
    ph = (1 + sum(h >= H0 for h in hs)) / (N + 1)
    pm = (1 + sum(m >= M0 for m in ms)) / (N + 1)
    mh = sum(hs) / N; mm = sum(ms) / N
    return dict(n=n, H=H0, H_null=round(mh, 1), pH=round(ph, 4), MI=round(M0, 3), MI_null=round(mm, 3), pMI=round(pm, 4))


def subsets(P):
    """P: list of (sign, n_signs, family, doc, word)"""
    tok = [(s, f) for s, k, f, d, w in P]
    typ = sorted({(w, f): (s, f) for s, k, f, d, w in P}.values())
    one = [(s, f) for s, k, f, d, w in P if k == 1]
    multi = [(s, f) for s, k, f, d, w in P if k > 1]
    return {'tokens': tok, 'word-commodity types': typ, 'single-sign words': one, 'multi-sign words (first sign)': multi}


def main():
    res = {}
    out = []
    LA = la_pairs()
    LAP = [(s, k, COM_FAMILY[c], d, w) for s, k, c, d, w in LA]
    v2ab = lb_value_to_ab()
    for strict in (False, True):
        lab, glyph = picture_codes(strict)
        tag = 'strict' if strict else 'full'
        for name, pr in subsets(LAP).items():
            r = perm_test(pr, lab); res[('LA', tag, name)] = r
            out.append('LA %-6s %-30s %s' % (tag, name, r))
        # LB with the same codes, keyed by AB number
        labab = {}
        for s, c in lab.items():
            a = ab_number(glyph[s])
            if a is not None: labab[a] = c
        LBP = lb_pairs()
        for name, pr in subsets(LBP).items():
            r = perm_test(pr, labab); res[('LB', tag, name)] = r
            out.append('LB %-6s %-30s %s' % (tag, name, r))
        # LA restricted to AB signs only (like-for-like with LB)
        labA = {s: c for s, c in lab.items() if ab_number(glyph[s]) is not None}
        for name, pr in list(subsets(LAP).items())[:2]:
            r = perm_test(pr, labA); res[('LA-ABonly', tag, name)] = r
            out.append('LA-ABonly %-6s %-30s %s' % (tag, name, r))
    print('\n'.join(out))
    json.dump({' | '.join(k): v for k, v in res.items()}, open(os.path.join(D, 'la7_c1.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
