#!/usr/bin/env python3
"""LA-8 planted control helper: build the genome of the true generating classes, and compare an evolved genome to truth."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la8_gp as G


def truth_genome(D, truth, nst_map=None):
    labs = sorted(set(truth.values())); lid = {l: i for i, l in enumerate(labs)}
    tc = np.array([lid[truth[w]] for w in D.lex], dtype=np.int32)
    # novel keys go to the open class that owns most novel tokens of that key
    from collections import Counter, defaultdict
    kv = defaultdict(Counter)
    for d in D.docs:
        for t in d['toks']:
            if t[0] not in D.lexid:
                k = t[2] if t[2] in D.keyid else (t[1] + ':other' if t[1] == 'W2' else t[1])
                kv[k][truth.get(t[0], 'NAME')] += 1
    kc = np.array([lid[kv[k].most_common(1)[0][0]] if kv[k] else 0 for k in D.keys], dtype=np.int32)
    nst = np.array([(nst_map or {}).get(l, 1) for l in labs], dtype=np.int32)
    return G.canon({'K': len(labs), 'tc': tc, 'kc': kc, 'nst': nst}), labs


def ari(a, b):
    from math import comb
    from collections import Counter
    n = len(a); cont = Counter(zip(a, b)); A = Counter(a); B = Counter(b)
    s = sum(comb(v, 2) for v in cont.values()); sa = sum(comb(v, 2) for v in A.values()); sb = sum(comb(v, 2) for v in B.values())
    e = sa * sb / comb(n, 2); m = (sa + sb) / 2
    return (s - e) / (m - e) if m != e else 1.0


if __name__ == '__main__':
    corpus = sys.argv[1]
    D = G.Data(json.load(open(os.path.join(G.OUT, 'corpus_%s.json' % corpus))))
    truth = json.load(open(os.path.join(G.OUT, 'planted_truth.json')))[corpus]
    g, labs = truth_genome(D, truth)
    f, bits, out = G.fitness(D, g)
    print(corpus, 'truth genome K=%d fitness %.1f heldout %.1f DL %.1f' % (g['K'], f, bits, f - bits))
