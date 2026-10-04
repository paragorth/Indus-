#!/usr/bin/env python3
"""LA-20 cycle 4: is the logogram pecking order more than 'large amounts first'?
 (a) amount-driven null: every list is re-ordered by a Plackett-Luce draw on log(amount) with a
     strength tuned so the share of larger-first pairs matches the real lists; items without an
     amount get a random key. Held-out BT accuracy of the real lists vs this null.
 (b) robustness of the real result: sides not joined; Hagia Triada only; Khania only; only
     commodity logograms (drop single syllabic signs used as logograms/adjuncts)."""
import os, sys, json, math
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
from la20_common import *
from la20_common import _items_from_tokens

NREP = int(sys.argv[1])
docs = load_la()
ql = [d['items']['L'] for d in docs if len(d['items']['L']) >= 2]
ords = [[x for x, _ in o] for o in ql]


def qfirst(lists):
    n = c = 0
    for o in lists:
        for p in range(len(o)):
            for q in range(p + 1, len(o)):
                a, b = o[p][1], o[q][1]
                if a is not None and b is not None and a != b: n += 1; c += a > b
    return c / n


def amount_draw(lists, beta, rng):
    out = []
    for o in lists:
        key = {}
        for x, q in o:
            g = -math.log(-math.log(rng.random()))
            key[x] = (beta * math.log(q) if q and q > 0 else beta * rng.gauss(math.log(10), 1.5)) + g
        out.append(sorted(o, key=lambda t: -key[t[0]]))
    return out


real_q = qfirst(ql)
best = None
for beta in np.arange(0.1, 3.01, 0.1):
    v = np.mean([qfirst(amount_draw(ql, beta, random.Random(i))) for i in range(30)])
    if best is None or abs(v - real_q) < best[0]: best = (abs(v - real_q), beta, v)
beta = best[1]
real = cv_score(ords, fit_bt)
nul = []
for r in range(NREP):
    s = amount_draw(ql, beta, random.Random(10000 + r))
    nul.append(acc(cv_score([[x for x, _ in o] for o in s], fit_bt, seed=r + 1))[0])
nul = np.array(nul)
res = {'real_q': real_q, 'beta': float(beta), 'null_q': best[2], 'real_acc': acc(real)[0], 'null': nul.tolist()}
print('L amount-driven null: real larger-first share %.3f, matched beta %.1f (null share %.3f); real held-out acc %.3f vs amount null %.3f+-%.3f, P %.4f' % (
    real_q, beta, best[2], acc(real)[0], nul.mean(), nul.std(), (np.sum(nul >= acc(real)[0]) + 1) / (NREP + 1)), flush=True)

# (b) robustness with the shuffle null
c = json.load(open(os.path.join(D, 'corpus.json')))
def side_lists():
    out = []
    for r in c:
        if r['support'] not in SUPPORTS: continue
        toks = []
        for t in r['tokens']:
            if t['t'] == 'word': toks.append(('word', '-'.join(t['s'])))
            elif t['t'] == 'logo': toks.append(('logo', t['v']))
            elif t['t'] == 'num': toks.append(('num', t['v']))
            else: toks.append((t['t'], None))
        o = [x for x, _ in _items_from_tokens(toks)['L']]
        if len(o) >= 2: out.append(o)
    return out
variants = {
    'sides_separate': side_lists(),
    'HT_only': [[x for x, _ in d['items']['L']] for d in docs if d['site'] == 'Haghia Triada' and len(d['items']['L']) >= 2],
    'KH_only': [[x for x, _ in d['items']['L']] for d in docs if d['site'] == 'Khania' and len(d['items']['L']) >= 2],
    'commodity_only': [o for o in ([[x for x in oo if len(x) > 2 and x.isascii()] for oo in ords]) if len(o) >= 2],
}
for k, v in variants.items():
    ra = acc(cv_score(v, fit_bt))[0]
    nn = np.array([acc(cv_score(shuffle_within(v, random.Random(r)), fit_bt, seed=r + 1))[0] for r in range(NREP)])
    res[k] = (len(v), ra, float(np.nanmean(nn)), float(np.nanstd(nn)), float((np.sum(nn >= ra) + 1) / (NREP + 1)))
    print('L %s: %d lists, acc %.3f, shuffle null %.3f+-%.3f, P %.4f' % ((k,) + res[k]), flush=True)
json.dump(res, open(os.path.join(CK, 'c4_L.json'), 'w'))
