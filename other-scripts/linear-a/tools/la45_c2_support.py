#!/usr/bin/env python3
"""LA-45 cycle 2 follow-up: rescore each LA_SITE population's frozen meanings on the out-of-site final set split by
object type (tablets vs everything else), and on the Critic's vault, to see where the meanings fail."""
import sys, os, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la45_common as C
sys.argv = [sys.argv[0], '120', '6', 'c2']
import la45_c2 as J

for name in ['LA_SITE', 'LA_SITE_S1']:
    docs, truth, tr, rest = J.corpus(name)
    B = C.build(docs)
    rows = []
    for k in range(6):
        r = json.load(open(os.path.join(C.CK, 'c2_%s_%d.json' % (name, k))))
        ps = C.seed('la45-c2-%s-%d' % (name, k))
        rng = np.random.default_rng(ps)
        rs = rng.permutation(np.array(rest)); h = len(rs) // 2
        g = C.Game(docs, B, ps, rounds=1, train_docs=np.array(tr), vault_docs=rs[:h], final_docs=rs[h:])
        a = np.zeros(B.T, np.int64)
        for w, m in r['tclus'].items():
            a[B.ti[w]] = m
        fin = rs[h:]
        tab = [d for d in fin if docs[d]['support'] == 'Tablet']
        oth = [d for d in fin if docs[d]['support'] != 'Tablet']
        out = []
        for part in (fin, tab, oth):
            s, c, n = g.score(a, None, np.array(part))
            s1, c1, n1 = g.score(np.zeros_like(a), None, np.array(part))
            out.append((round(s + c, 3), round(c, 3), n, round(s1, 3)))
        rows.append(out)
        print(name, k, 'all', out[0], 'tablets', out[1], 'other', out[2])
    m = np.array([[x[0] for x in o] for o in rows])
    print(name, 'mean gain all/tablets/other', m.mean(0).round(3))
