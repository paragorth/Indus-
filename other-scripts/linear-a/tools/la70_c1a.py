#!/usr/bin/env python3
"""LA-70 cycle 1a: does la66's 'KI marks a reduced amount' (C) survive before it enters the v2 reading?
Case = KI (single-sign word) directly followed by a number, with an earlier number on the document.
Statistic: share of cases where the KI amount is below the amount just before it.
Nulls: (i) 2,000 within-document shuffles of all amounts; (ii) every single-sign decoy with >= 4 cases
through the same statistic and shuffle (decoy false-survival). Held-out: without HT 118 (the source).
Survive (fixed in advance): within-document P <= 0.05, share without HT 118 > 0.5, and the KI P in the
lowest 10 % of decoy P values."""
import json, os, random
from collections import defaultdict
from la70_common import load_la, admin_docs, CK, seed


def cases(docs, w, vals=None):
    out = []
    for d in docs:
        toks = d['toks']
        nums = [i for i, t in enumerate(toks) if t[0] == 'N']
        v = vals[d['id']] if vals else {i: toks[i][1] for i in nums}
        for i, t in enumerate(toks):
            if t[0] == 'W' and t[1] == w and i + 1 < len(toks) and toks[i + 1][0] == 'N':
                prev = [k for k in nums if k < i]
                if prev and v[prev[-1]] > 0:
                    out.append((d['id'], v[i + 1] < v[prev[-1]]))
    return out


def share(c):
    return sum(x for _, x in c) / len(c) if c else float('nan')


def perm_p(docs, w, real, rng, n=2000):
    ge = 0
    rel = [d for d in docs if any(t[0] == 'W' and t[1] == w for t in d['toks'])]
    for _ in range(n):
        vals = {}
        for d in rel:
            nums = [i for i, t in enumerate(d['toks']) if t[0] == 'N']
            vs = [d['toks'][i][1] for i in nums]; rng.shuffle(vs)
            vals[d['id']] = dict(zip(nums, vs))
        ge += share(cases(rel, w, vals)) >= real
    return (ge + 1) / (n + 1)


def main():
    A = admin_docs(load_la())
    rng = random.Random(seed('la70-c1a'))
    ki = cases(A, 'KI'); s = share(ki)
    ki_wo = [c for c in ki if c[0] != 'HT118']
    p = perm_p(A, 'KI', s, rng)
    singles = sorted({t[1] for d in A for t in d['toks'] if t[0] == 'W' and '-' not in t[1]} - {'KI'})
    dec = []
    for w in singles:
        c = cases(A, w)
        if len(c) >= 4:
            dec.append((w, len(c), share(c), perm_p(A, w, share(c), rng, 500)))
    dps = sorted(x[3] for x in dec)
    rank = sum(1 for x in dps if x <= p) / max(1, len(dps))
    fs = sum(1 for x in dps if x <= 0.05) / max(1, len(dps))
    surv = p <= 0.05 and share(ki_wo) > 0.5 and rank <= 0.10
    res = dict(n=len(ki), share=s, p_within=p, n_wo_HT118=len(ki_wo), share_wo=share(ki_wo),
               cases=ki, decoys=dec, decoy_rank=rank, decoy_false_survival=fs, survives=surv)
    json.dump(res, open(os.path.join(CK, 'c1a_ki.json'), 'w'), indent=1)
    flag = os.path.join(CK, 'ki_survives.flag')
    if surv:
        open(flag, 'w').write('1')
    elif os.path.exists(flag):
        os.remove(flag)
    print(json.dumps({k: v for k, v in res.items() if k not in ('cases', 'decoys')}))
    for x in dec:
        print('decoy', x)


if __name__ == '__main__':
    main()
