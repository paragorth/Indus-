#!/usr/bin/env python3
"""LA-14 cycle 3, part S: is the 'same final sign' signal (long:same_ls) more than repeated words?
For each tablet, pairs of DISTINCT word types (>= 2 signs) on it; statistic = share of pairs sharing the final sign
(and, separately, the first sign). Null: word tokens permuted across tablets within the same site (positions kept),
2,000 runs. Positive control LB (lists of names / ethnics with shared endings expected); negative control FW_FLAT
(a forger's world, nothing expected); planted control: LA where 25% of tablets get every word's last sign set
to that tablet's first word's last sign."""
import random
from itertools import combinations
import numpy as np
import la14_common as C


def stat(docs, pos):
    hit = n = 0
    for d in docs:
        W = sorted({t for t in d if t.startswith('W:') and '-' in t})
        for a, b in combinations(W, 2):
            sa, sb = a[2:].split('-'), b[2:].split('-'); n += 1
            hit += sa[pos] == sb[pos]
    return hit / max(1, n), n


def perm(docs, sites, rng):
    pool = {}
    for d, s in zip(docs, sites): pool.setdefault(s, []).extend(t for t in d if t.startswith('W:'))
    for s in pool: rng.shuffle(pool[s])
    return [[pool[s].pop() if t.startswith('W:') else t for t in d] for d, s in zip(docs, sites)]


def run(name, docs, sites, R=2000):
    rng = random.Random(2); out = []
    for pos, lab in ((-1, 'final'), (0, 'first')):
        o, n = stat(docs, pos)
        nul = [stat(perm(docs, sites, rng), pos)[0] for _ in range(R)]
        p = (1 + sum(x >= o for x in nul)) / (1 + R)
        out.append('%-12s %-5s share %.4f of %d distinct pairs | null %.4f +- %.4f | ratio %.2f P=%.4f'
                   % (name, lab, o, n, np.mean(nul), np.std(nul), o / np.mean(nul), p))
    return out


if __name__ == '__main__':
    la, ids = C.load_la(); s = [i[:2] for i in ids]
    lb = C.corpus('LB'); slb = [C.SITE.get(' '.join(d), '??') for d in lb]
    fw = C.corpus('FW_FLAT')
    rng = random.Random(9); pl = []
    for d in la:
        W = [t for t in d if t.startswith('W:') and '-' in t]
        if W and rng.random() < 0.25:
            f = W[0][2:].split('-')[-1]
            d = ['W:' + '-'.join(t[2:].split('-')[:-1] + [f]) if (t.startswith('W:') and '-' in t) else t for t in d]
        pl.append(d)
    L = []
    L += run('LA', la, s)
    L += run('LA-HT', [d for d, x in zip(la, s) if x == 'HT'], ['HT'] * sum(x == 'HT' for x in s))
    L += run('LA-nonHT', [d for d, x in zip(la, s) if x != 'HT'], [x for x in s if x != 'HT'])
    L += run('LB', lb, slb)
    L += run('FW_FLAT', fw, ['W'] * len(fw))
    L += run('PLANT', pl, s)
    txt = '\n'.join(L); print(txt)
    open(C.os.path.join(C.OUT, 'c3s_report.txt'), 'w').write(txt + '\n')
