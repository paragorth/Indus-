#!/usr/bin/env python3
"""LA-8 number attachment, measured directly: what kind of token stands right before each NUM?
Control: the same counts after shuffling tokens inside each document (1,000 runs)."""
import json, os, random
from collections import Counter
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'la8')


def stats(docs):
    prev = Counter(); after = Counter(); n = Counter()
    for d in docs:
        k = [t[1] for t in d]
        for i in range(len(k)):
            if k[i] == 'NUM': prev[k[i-1] if i else '<s>'] += 1
            if i + 1 < len(k):
                n[k[i]] += 1; after[k[i]] += k[i+1] == 'NUM'
    return prev, {x: after[x] / n[x] for x in n}


rng = random.Random(1)
for C in ['LA', 'LB', 'LB2', 'PE', 'PE2', 'PFLAT', 'PREC']:
    docs = [d['toks'] for d in json.load(open(os.path.join(OUT, 'corpus_%s.json' % C)))]
    prev, pn = stats(docs)
    null = []
    for _ in range(1000):
        sh = [rng.sample(d, len(d)) for d in docs]; null.append(stats(sh)[1])
    tot = sum(prev.values())
    line = []
    for kd in ('W2', 'W1', 'L', 'S'):
        if kd not in pn: continue
        nv = [x.get(kd, 0) for x in null]; m = sum(nv) / len(nv)
        p = sum(v >= pn[kd] for v in nv) / len(nv)
        line.append('P(NUM|%s)=%.2f (shuffle %.2f, P=%.3f)' % (kd, pn[kd], m, p))
    print(C, 'NUM preceded by:', {k: round(v / tot, 2) for k, v in prev.most_common()}, '|', '; '.join(line))
