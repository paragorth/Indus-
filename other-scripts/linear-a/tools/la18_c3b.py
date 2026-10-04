#!/usr/bin/env python3
"""LA-18 cycle 3b: robustness of the LA ending partition.
(i) drop the high-frequency words (document frequency >= 8: KU-RO, SA-RA2, KI-RO, A-TA-I-*301-WA-JA ...);
(ii) Hagia Triada only; (iii) leave each of the 12 endings out of the tracked set in turn is skipped (cost);
(iv) which word pairs carry the within-class excess of the best 2-class partition.
"""
import json, collections, random
import numpy as np
from la18_common import *
import la18_c3 as c3


def main():
    docs = la_corpus(); S, site, ids = units(docs)
    out = {}
    df = collections.Counter(w for s in S for w in s)
    hi = {w for w, c in df.items() if c >= 8}
    S1 = [frozenset(w for w in s if w not in hi) for s in S]
    keep = [i for i, s in enumerate(S1) if len(s) >= 2]
    S1 = [S1[i] for i in keep]; s1 = [site[i] for i in keep]
    r = c3.run(S1, s1, c3.ends(S1), 12, 1000, 51, ngrp=(2, 3), nanneal=200)
    out['drop_hi'] = dict(dropped=['-'.join(w) for w in hi], res=r)
    print('drop_hi', len(S1), r['homogeneity'], {g: (v['score'], v['p'], v['classes']) for g, v in r['partition'].items()}, flush=True)
    ht = [i for i in range(len(S)) if site[i] == 'Haghia Triada']
    S2 = [S[i] for i in ht]; s2 = [site[i] for i in ht]
    r = c3.run(S2, s2, c3.ends(S2), 12, 1000, 52, ngrp=(2, 3), nanneal=200)
    out['HT'] = r
    print('HT', len(S2), r['homogeneity'], {g: (v['score'], v['p'], v['classes']) for g, v in r['partition'].items()}, flush=True)
    # contributors for full-LA best 2-class partition
    full = json.load(open(os.path.join(OUT, 'c3_la.json')))['LA']['partition']['2']['classes']
    cls = {e: k for k, c in enumerate(full) for e in c}
    contrib = collections.Counter()
    for s in S:
        ws = [w for w in s if w[-1] in cls]
        for a in ws:
            for b in ws:
                if a < b and a[-1] != b[-1] and cls[a[-1]] == cls[b[-1]]:
                    contrib[('-'.join(a), '-'.join(b))] += 1
    out['top_contrib'] = contrib.most_common(25)
    print(contrib.most_common(15))
    dump(out, 'c3b.json')


if __name__ == '__main__':
    main()
