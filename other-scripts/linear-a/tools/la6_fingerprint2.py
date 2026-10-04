#!/usr/bin/env python3
"""LA-6 cycle 4b: checks on cycle 4.
(A') The word-identity F test without totals (KU-RO, KI-RO, PO-TO-KU-RO): numbers shuffled
     within commodity, 2,000 runs.
(C') 'Same as the previous entry': for each word, how many of its entries repeat exactly the
     quantity of the entry before it on the same tablet, against the corpus-wide repeat rate
     (binomial tail), and against a within-tablet shuffle of entry order (2,000 runs).
"""
import sys, os, math, random
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))
from la6_common import la_entries, D
from la6_fingerprint import units

random.seed(65)


def binom_tail(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    out = ['# LA-6 cycle 4b']
    E = la_entries()
    Q = [e for e in E if e['role'] in ('entry', 'post') and e['label'] and units(e)]
    tabs = defaultdict(set)
    for e in Q: tabs[e['label']].add(e['doc'])
    words = {w for w in tabs if len(tabs[w]) >= 3}
    vals = [(i, c, math.log10(v)) for i, e in enumerate(Q) for c, v in units(e)]
    pool = defaultdict(list)
    for i, c, v in vals: pool[c].append(v)
    cm = {c: sum(v) / len(v) for c, v in pool.items()}

    def F(vs):
        byw = defaultdict(list)
        for i, c, v in vs:
            if Q[i]['label'] in words: byw[Q[i]['label']].append(v - cm[c])
        allr = [x for v in byw.values() for x in v]; m = sum(allr) / len(allr)
        b = sum(len(v) * (sum(v) / len(v) - m) ** 2 for v in byw.values())
        w = sum((x - sum(v) / len(v)) ** 2 for v in byw.values() for x in v)
        return (b / (len(byw) - 1)) / (w / (len(allr) - len(byw))), byw
    f0, byw = F(vals)
    idx = defaultdict(list)
    for k, (i, c, v) in enumerate(vals): idx[c].append(k)
    R = 2000; ge = 0; fm = 0
    for _ in range(R):
        sh = list(vals)
        for c, ks in idx.items():
            vs = [vals[k][2] for k in ks]; random.shuffle(vs)
            for k, v in zip(ks, vs): sh[k] = (vals[k][0], c, v)
        f, _b = F(sh); fm += f; ge += f >= f0
    out.append(f"(A') without totals: {len(words)} words, F = {f0:.2f} vs {fm/R:.2f}, P = {(ge+1)/(R+1):.4f}")
    # (C') repeats
    bydoc = defaultdict(list)
    for e in E:
        if e['role'] in ('entry', 'post') and units(e): bydoc[e['doc']].append(e)
    def first_q(e): return units(e)[0]
    rep = Counter(); n = Counter(); tot_rep = tot = 0
    for d, es in bydoc.items():
        for a, b in zip(es, es[1:]):
            same = first_q(a) == first_q(b)
            tot += 1; tot_rep += same
            if b['label']:
                n[b['label']] += 1; rep[b['label']] += same
    p0 = tot_rep / tot
    out.append(f"(C') corpus-wide: {tot_rep}/{tot} entries repeat the previous entry's quantity and class exactly (rate {p0:.3f})")
    rows = []
    for w in words:
        if n[w] >= 3:
            rows.append((binom_tail(rep[w], n[w], p0), w, rep[w], n[w]))
    rows.sort()
    m = len(rows)
    for p, w, k, nn in rows[:12]:
        out.append(f'  {w}: {k}/{nn} repeats, binomial P = {p:.4f} (Bonferroni x{m} = {min(1, p*m):.3f})')
    txt = '\n'.join(out)
    print(txt)
    open(os.path.join(D, 'la6_fingerprint2.out'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
