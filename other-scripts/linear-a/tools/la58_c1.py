#!/usr/bin/env python3
"""LA-58 cycle 1: ABC-RF on the LA bank (100k worlds); real LA, 20 site-label shuffles, planted worlds;
Linear B and Ur III controls on their own banks."""
import sys, os, json, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la58_common import *
from la58_run import corpus, nk_of, load_bank
from la58_fit import train, read, TYPES

OUT = {}


def planted(K, nk, Tbank, typ, roles, parents, sd):
    rng = np.random.RandomState(sd)
    cand = np.where(Tbank[:, 0] == typ)[0]
    th = Tbank[rng.choice(cand)].copy()
    th[0] = typ; th[1] = sum(1 for r in roles if r == 1)
    th[2:2 + K] = roles; th[2 + K:2 + 2 * K] = parents
    S, T = simulate(K, nk, 1, sd, force_theta=th)
    return S[0]


def main():
    res = {}
    # ---------------- Linear A
    rf, cal = train('LA')
    res['LA_calib'] = cal
    docs, K, ab, _ = corpus('LA'); nk = nk_of(docs, K)
    res['LA_real'] = read(rf, stats(docs, K), K)
    sh = []
    for i in range(20):
        d, _, _, _ = corpus('LASHUF%d' % i)
        sh.append(read(rf, stats(d, K), K))
    res['LA_shuf'] = sh
    _, Tb, _ = load_bank('LA')
    pl = {}
    HT, KN, KH, ZA = 0, 3, 1, 4
    specs = {
        'single_KN': (0, [2 if k != KN else 1 for k in range(K)], [KN if k != KN else -1 for k in range(K)]),
        'single_HT': (0, [2 if k != HT else 1 for k in range(K)], [HT if k != HT else -1 for k in range(K)]),
        'peers_HT_KH_ZA': (2, [1 if k in (HT, KH, ZA) else 2 for k in range(K)],
                           [-1 if k in (HT, KH, ZA) else (HT, KH, ZA)[k % 3] for k in range(K)]),
        'capsat_KN': (1, [2 if k != KN else 1 for k in range(K)], [KN if k != KN else -1 for k in range(K)]),
        'merch': (4, [0] * K, [-1] * K),
        'null': (5, [0] * K, [-1] * K),
    }
    for name, (t, r, p) in specs.items():
        pl[name] = [read(rf, planted(K, nk, Tb, t, r, p, seed('la58-plant-%s-%d' % (name, j))), K) for j in range(6)]
    res['LA_planted'] = pl
    del rf
    # ---------------- controls
    for c in ['LB', 'UR']:
        rfc, calc = train(c)
        d, Kc, abc, _ = corpus(c)
        res[c + '_calib'] = calc
        res[c + '_real'] = read(rfc, stats(d, Kc), Kc)
        res[c + '_shuf'] = []
        for i in range(10):
            rng = random.Random(seed('la58-%s-shuf-%d' % (c, i)))
            s = [x['site'] for x in d]; rng.shuffle(s)
            res[c + '_shuf'].append(read(rfc, stats([dict(x, site=q) for x, q in zip(d, s)], Kc), Kc))
        del rfc
    jdump(res, 'c1_results.json')


if __name__ == '__main__':
    main()
