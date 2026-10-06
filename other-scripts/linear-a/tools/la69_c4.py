#!/usr/bin/env python3
"""LA-69 cycle 4: held-out re-test of the cycle-3 cross-site and site results at ONE fixed pool size
(M = 28 documents, the Zakros limit), 6 fresh seeds, for LA, LB and Ur III sites alike."""
import os, sys, random, collections, pickle
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la69_common as C
from la69_c3 import Reader, doc_cloze

M = 28


def rate(res):
    n = sum(r[0] for r in res); return sum(r[1] for r in res) / max(1, n)


def read(docs, ref, rng, npool=6):
    res = []
    for d in docs:
        sub = rng.sample([x for x in ref if x['id'] != d['id']], M)
        r = Reader(sub); r.index()
        res.append(doc_cloze(d, r, in_extra=False))
    return rate(res)


def main():
    la, lb, ur = C.la_docs(), C.lb_docs(), C.ur3_docs()
    out = collections.defaultdict(list)
    for s in range(6):
        rng = random.Random(6904 + s)
        for tag, corp, k in [('LA', la, 3), ('LB', lb, 2), ('UR3', ur, 3)]:
            for site, n in collections.Counter(d['site'] for d in corp).most_common(k):
                own = [d for d in corp if d['site'] == site]; oth = [d for d in corp if d['site'] != site]
                docs = rng.sample(own, min(29, len(own)))
                same = read(docs, own, rng); other = read(docs, oth, rng)
                out[(tag, site)].append((same, other))
        print('seed', s, flush=True)
    pickle.dump(dict(out), open(os.path.join(C.CK, 'c4.pkl'), 'wb'))
    for k, v in out.items():
        a = np.array(v)
        print('%-4s %-16s same %.3f±%.3f other %.3f±%.3f ratio %.2f [%.2f-%.2f]' % (
            k[0], k[1][:16], a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 1].std(),
            np.mean(a[:, 1] / np.maximum(a[:, 0], 1e-9)), np.min(a[:, 1] / np.maximum(a[:, 0], 1e-9)),
            np.max(a[:, 1] / np.maximum(a[:, 0], 1e-9))))


if __name__ == '__main__':
    main()
