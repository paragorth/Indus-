#!/usr/bin/env python3
"""LA-24 cycle 1b: ONE CLERK, ONE RULE. Cycle 1 paid the rule/alphabet choice on every list
(about 10 bits), which no short list can repay; the positive controls failed. Here the rule,
grid and share alphabet are chosen once per corpus (the clerk's algorithm) and each list pays
only its unit share u, its class bits and its misses. Corpus score for hypothesis h =
sum over lists of max(0, baseline bits - model bits under h); the best h is reported.
Nulls (each re-optimises h): N3c frequency-weighted band replacement (each distinct amount ->
a corpus amount within +-25 %, drawn by corpus frequency; repeats kept), N1 amounts shuffled
across lists. Positive controls: Ur III ration lists; 40 planted lists made by ONE rule and
ONE alphabet embedded in the Linear A corpus (and the same with 3 clerks); Linear B lists.
"""
import json, os, random, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la24_common import *

G = {}


def setup(name, lists, numval, grids):
    B = Baseline([l['amts'] for l in lists], numval)
    G[name] = dict(lists=lists, B=B, nv=numval, grids=grids, pool=band_pool(B, numval))


def corpus_gain(name, L):
    g = G[name]
    Gm = []
    for toks in L:
        x = np.array([g['nv'](t) for t in toks]); bc = g['B'].costs(toks)
        Gm.append(bc.sum() - list_matrix(x, bc, g['grids']))
    Gm = np.array(Gm)                        # lists x combos x alphabets
    S = np.where(Gm > 0, Gm, 0).sum(0)
    c, s = np.unravel_index(np.argmax(S), S.shape)
    return float(S[c, s]), (int(c), int(s)), Gm[:, c, s]


def null_job(args):
    name, kind, seed = args
    g = G[name]; rng = random.Random(seed)
    L = [l['amts'] for l in g['lists']]
    if kind == 'N1': L2 = null_shuffle(L, rng)
    else: L2 = [null_band_w(t, g['pool'], g['nv'], rng) for t in L]
    s, h, gl = corpus_gain(name, L2)
    return kind, s, int((gl > 0).sum())


def run(name, pool, R):
    g = G[name]; t0 = time.time()
    s, (c, a), gl = corpus_gain(name, [l['amts'] for l in g['lists']])
    cmb = combos_for(g['grids'])
    acc = {'N3c': [], 'N1': []}
    for kind, sv, cnt in pool.imap_unordered(null_job, [(name, k, 1000 + i) for k in acc for i in range(R)]):
        acc[kind].append((sv, cnt))
    out = {'name': name, 'n': len(gl), 'best_rule': RULES[cmb[c][0]], 'grid': cmb[c][1], 'S': ALPH[a],
           'score': s, 'lists_pos': int((gl > 0).sum())}
    for k, v in acc.items():
        v = np.array(v)
        out[k] = {'score_mean': float(v[:, 0].mean()), 'score_sd': float(v[:, 0].std()),
                  'P_upper': float(((v[:, 0] >= s).sum() + 1) / (len(v) + 1)),
                  'P_lower': float(((v[:, 0] <= s).sum() + 1) / (len(v) + 1)),
                  'pos_mean': float(v[:, 1].mean())}
    out['explained'] = [(g['lists'][i]['id'], [round(g['nv'](t), 3) for t in g['lists'][i]['amts']], round(float(gl[i]), 2))
                        for i in np.argsort(-gl)[:12] if gl[i] > 0]
    out['secs'] = time.time() - t0
    print(json.dumps({k: v for k, v in out.items() if k != 'explained'}), flush=True)
    return out


def planted(la, rng, nclerks):
    pl = [dict(l) for l in la]
    idx = rng.sample(range(len(la)), 40)
    clerks = []
    for _ in range(nclerks):
        clerks.append((rng.choice(RULES), list(rng.choice([a for a in ALPH if len(a) in (2, 3)]))))
    for j, i in enumerate(idx):
        rule, S = clerks[j % nclerks]
        n = min(max(len(la[i]['amts']), 3), 10)
        while True:
            w = [rng.choice(S) for _ in range(n)]
            if rule == 'EXACT':
                vals = [(rng.choice([1, 2, 3, 4, 5, 6, 8, 10]) + rng.choice([0, 0, 0.5])) * 1.0] * 0
                u = rng.choice([1, 2, 3, 4, 5, 6, 8, 10]) + rng.choice([0, 0, 0.5])
                vals = [u * x for x in w]
            else:
                vals = apportion(rng.randint(2 * n, 40 * n), w, rule, 1.0)
            a = [to_amount(v) for v in vals]
            if all(z is not None and (z[0] > 0 or z[1]) for z in a): break
        pl[i] = {'id': 'PLANT', 'site': 'PL', 'key': 'PLANT', 'amts': a, 'total': None}
    return pl, clerks


def main():
    R = int(os.environ.get('R', 60))
    la = la_lists()
    setup('LA', la, la_numval(CONV), (1.0, 0.5, 0.25))
    rng = random.Random(124)
    p1, c1 = planted(la, rng, 1); setup('PLANT1', p1, la_numval(CONV), (1.0, 0.5, 0.25))
    p3, c3 = planted(la, rng, 3); setup('PLANT3', p3, la_numval(CONV), (1.0, 0.5, 0.25))
    ur = ur_lists(max_lists=100, seed=1)
    if ur: setup('UR', ur, lambda a: float(a[0]), (1.0, 5.0, 10.0))
    setup('LB', lb_lists(), lambda a: float(a[0]), (1.0, 1 / 60, 1 / 72))
    res = {'planted_clerks': {'PLANT1': c1, 'PLANT3': c3}}
    with Pool(2) as pool:
        for name in ('LA', 'PLANT1', 'PLANT3', 'UR', 'LB'):
            if name in G:
                res[name] = run(name, pool, R)
                json.dump(res, open(os.path.join(CK, 'c1b.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
