"""pe7 cycle 4b: (e) are co-tablet sharers variants of one name (one string contained in the
other as a subsequence) rather than two people?  (f) herd-office names across DIFFERENT
herd tablets vs random cross-tablet pairs (edit model distance, 5,000 random-pair draws)."""
import sys, os, json, random
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe7_common import *  # noqa
from pe7_cycle4 import pe_lines, linb_lines
from pe7_build import HERD


def subseq(a, b):
    it = iter(b)
    return all(x in it for x in a)


def contain_stats(tabs, rng, nperm=200):
    def stat(tl):
        share = cont = 0
        for v in tl:
            v = list(dict.fromkeys(v))
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    if set(v[i]) & set(v[j]):
                        share += 1
                        a, b = sorted((v[i], v[j]), key=len)
                        cont += subseq(a, b)
        return share, cont
    s, c = stat(tabs)
    flat = [w for v in tabs for w in dict.fromkeys(v)]
    sizes = [len(list(dict.fromkeys(v))) for v in tabs]
    nc = []
    for _ in range(nperm):
        rng.shuffle(flat)
        k = 0; tl = []
        for L in sizes:
            tl.append(flat[k:k + L]); k += L
        nc.append(stat(tl))
    return {'sharing_pairs': s, 'contained': c, 'contained_share': c / s if s else None,
            'null_sharing': float(np.mean([x[0] for x in nc])), 'null_contained': float(np.mean([x[1] for x in nc])),
            'sharing_minus_contained_lift': (s - c) / max(1e-9, np.mean([x[0] - x[1] for x in nc]))}


def main():
    rng = random.Random(11)
    res = {'PE': contain_stats([v for v in pe_lines().values() if len(set(v)) >= 2], rng),
           'LINB': contain_stats([v for v in linb_lines().values() if len(v) >= 2], rng)}
    C = corpora()
    names = [tuple(x['seq']) for x in C['PE']]
    em = EditModel(names)
    em.set_rates(0.40, 0.30, 0.15, 0.15)
    herd = [(tuple(x['seq']), set(x['tablets'])) for x in C['PE'] if set(x['tablets']) & HERD]
    hd = [em.align(a, b) + em.align(b, a) - em.align(a, a) - em.align(b, b)
          for i, (a, ta) in enumerate(herd) for (b, tb) in herd[i + 1:] if not ta & tb]
    allx = [(tuple(x['seq']), set(x['tablets'])) for x in C['PE']]
    rd = []
    while len(rd) < 5000:
        (a, ta), (b, tb) = rng.sample(allx, 2)
        if ta & tb:
            continue
        rd.append(em.align(a, b) + em.align(b, a) - em.align(a, a) - em.align(b, b))
    rd = np.array(rd) / 2
    hd = np.array(hd) / 2
    # null: random sets of 12 names (cross-tablet pairs), 2,000 draws
    null = []
    for _ in range(2000):
        S = rng.sample(allx, len(herd))
        null.append(np.mean([(em.align(a, b) + em.align(b, a) - em.align(a, a) - em.align(b, b)) / 2
                             for i, (a, ta) in enumerate(S) for (b, tb) in S[i + 1:] if not ta & tb]))
    null = np.array(null)
    res['herd_cross_tablet'] = {'herd_names': len(herd), 'pairs': len(hd), 'mean_bits': float(hd.mean()),
                                'random_mean': float(null.mean()), 'p_low': float((1 + (null <= hd.mean()).sum()) / 2001),
                                'herd_names_list': [' '.join(a) for a, _ in herd]}
    json.dump(res, open(os.path.join(DATA, 'pe7_cycle4b.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
