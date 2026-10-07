"""LA-78 cycle 3: time travel for entry grammars (commodity order) and for frozen readings.

(a) Commodity order. Items = first occurrences of commodity logograms (base sign) and the
    word NI on a document. Pretend it is 1950 / 1976: fit the past-majority order (Copeland
    ranking of past pairs) and score it on the pairs of documents published later, against
    20,000 random orders and against the frozen la72 staple order (which saw all data).
    Controls: shuffled publication years; planted world (true order with 20 % swaps).
(b) Frozen word roles (la72: heading words KU-RE, KA-NA, DA-..; KU-RO total; KI-RO) refitted
    at each cut and scored on later finds like the random pool of cycle 1 (percentile).
(c) la66 quantity directions (KI down, A-DU up, TA down, E up) on later finds.
"""
import sys, os, json, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la78_engine import *  # noqa
from la78_common import _logo  # noqa

docs = load()
C = json.load(open(os.path.join(os.path.dirname(CK), 'corpus_ra.json')))
byid = {d['id']: d for d in C}
rng = np.random.RandomState(7803)
out = {}


def items(d):
    L = []
    for t in byid[d['id']]['tokens']:
        if t.get('st') != 'read': continue
        x = None
        if t['t'] == 'logo': x = t['v'].split('+')[0]
        elif t['t'] == 'word' and t['s'] == ['NI']: x = 'NI'
        if x and x not in L: L.append(x)
    return L


IT = ['GRA', 'OLE', 'CYP', 'VIN', 'OLIV', 'VIR', 'NI', 'QA2', 'AROM', 'HIDE']
dl = []
for d in docs:
    L = [x for x in items(d) if x in IT]
    if len(L) >= 2: dl.append((d['year'], d['site'], L))
out['n_docs_multi'] = len(dl)
out['n_after_1950'] = sum(1 for y, s, L in dl if y > 1950)
out['n_after_1976'] = sum(1 for y, s, L in dl if y > 1976)


def pairs(L):
    return [(L[i], L[j]) for i in range(len(L)) for j in range(i + 1, len(L))]


def fit_order(D):
    w = collections.Counter()
    for y, s, L in D:
        for a, b in pairs(L): w[(a, b)] += 1
    score = {x: sum(w[(x, o)] - w[(o, x)] for o in IT) for x in IT}
    return sorted(IT, key=lambda x: -score[x])


def agree(order, D):
    pos = {x: i for i, x in enumerate(order)}
    a = n = 0
    for y, s, L in D:
        for u, v in pairs(L):
            n += 1; a += pos[u] < pos[v]
    return a, n


LA72 = ['CYP', 'GRA', 'OLE', 'OLIV', 'VIR', 'NI', 'VIN', 'QA2', 'AROM', 'HIDE']  # la50/la72 staple order (pairs listed in la72)
rand_orders = [list(rng.permutation(IT)) for _ in range(20000)]


def order_test(D, cuts, tag):
    r = {}
    for X in cuts:
        past = [z for z in D if z[0] <= X]; fut = [z for z in D if z[0] > X]
        o = fit_order(past)
        a, n = agree(o, fut)
        ra = np.array([agree(q, fut)[0] for q in rand_orders])
        a72, _ = agree(LA72, fut)
        pa, pn = agree(o, past)
        r[str(X)] = dict(order=o, past_fit=f'{pa}/{pn}', fut=f'{a}/{n}', fut_rate=round(a / max(n, 1), 3),
                         pct_vs_random=round(float((ra < a).mean() + 0.5 * (ra == a).mean()), 3),
                         P_random_ge=round(float((ra >= a).mean()), 4), la72=f'{a72}/{n}',
                         sites_future=dict(collections.Counter(z[1] for z in fut)))
    print(tag, json.dumps(r), flush=True)
    return r


out['order_real'] = order_test(dl, (1950, 1976), 'order_real')
# shuffled years
sh = []
for k in range(200):
    yrs = rng.permutation([z[0] for z in dl])
    D2 = [(y, s, L) for y, (_, s, L) in zip(yrs, dl)]
    pos = {}
    for X in (1950, 1976):
        past = [z for z in D2 if z[0] <= X]; fut = [z for z in D2 if z[0] > X]
        a, n = agree(fit_order(past), fut); pos[X] = a / max(n, 1)
    sh.append(pos)
out['order_shuffled_rate'] = {str(X): [round(float(np.mean([p[X] for p in sh])), 3), round(float(np.percentile([p[X] for p in sh], 5)), 3)] for X in (1950, 1976)}
print('order_shuf', out['order_shuffled_rate'], flush=True)
# site-matched: future docs only from sites seen before X
sm = {}
for X in (1950, 1976):
    seen = {z[1] for z in dl if z[0] <= X}
    past = [z for z in dl if z[0] <= X]; fut = [z for z in dl if z[0] > X and z[1] in seen]
    a, n = agree(fit_order(past), fut); sm[str(X)] = f'{a}/{n}'
out['order_site_matched'] = sm
# planted world: true order, each doc's list is the true order with prob 0.2 adjacent swaps
pl = []
for k in range(200):
    true = list(rng.permutation(IT)); pos = {x: i for i, x in enumerate(true)}
    D3 = []
    for y, s, L in dl:
        L2 = sorted(L, key=lambda x: pos[x])
        for i in range(len(L2) - 1):
            if rng.rand() < 0.2: L2[i], L2[i + 1] = L2[i + 1], L2[i]
        D3.append((y, s, L2))
    past = [z for z in D3 if z[0] <= 1950]; fut = [z for z in D3 if z[0] > 1950]
    o = fit_order(past)
    a, n = agree(o, fut); at, _ = agree(true, fut)
    pl.append((a / n, at / n))
out['order_planted_fit_vs_true'] = [round(float(np.mean([p[0] for p in pl])), 3), round(float(np.mean([p[1] for p in pl])), 3)]
print('order_planted', out['order_planted_fit_vs_true'], flush=True)

# ---------------------------------------------------------------- (b) frozen word roles
T = tokens(docs)
y1 = np.array([t['y1'] for t in T]); Y = [(y1, 4)]
MODE['mode'] = os.environ.get('SMODE', 'prior'); MODE['grp'] = np.array([docs[t['doc']]['site'] for t in T])
docid = np.array([t['doc'] for t in T]); year = np.array([docs[t['doc']]['year'] for t in T])
WIN = [(1950, 1976), (1976, 2100), (1988, 2100)]
la72 = {'KU-RE': 0, 'KA-NA': 0, 'KU-RO': 1, 'KI-RO': 2, 'PO-TO-KU-RO': 1}
f72 = word_feature(T, la72, 3)
res = np.load(os.path.join(CK, 'c1_%s_y1_real.npz' % MODE['mode']))
fut_pool = res['fut']
fb = {}
for wi, (a, b) in enumerate(WIN):
    tr = year <= a; te = (year > a) & (year <= b)
    g = score(f72, Y, tr, te)
    fb[f'W{wi+1}'] = dict(gain=round(g, 4), pct_vs_pool=round(float((fut_pool[:, wi] < g).mean()), 3),
                          n_hits=int(np.isin(f72[te], [0, 1, 2]).sum()))
out['la72_roles'] = fb
print('la72', fb, flush=True)

# ---------------------------------------------------------------- (c) la66 quantity directions
fz = json.load(open(os.path.join(os.path.dirname(CK), 'la66_frozen_factors.json')))
dirs = {f['feature']: f['direction'] for f in fz['factors']}
cases = collections.Counter()
for di, d in enumerate(docs):
    qs = [w['q'] for w in d['words'] if w['q'] is not None]
    if len(qs) < 3: continue
    med = float(np.median(qs))
    for w in d['words']:
        if w['q'] is None or not w['clean']: continue
        word = '-'.join(w['s'])
        for feat, dr in dirs.items():
            kind, val = feat.split(':')
            hit = (kind == 'W' and word == val) or (kind == 'S' and w['s'] and w['s'][-1] == val and len(w['s']) >= 2)
            if not hit: continue
            era = 'past1950' if d['year'] <= 1950 else 'after1950'
            ok = (w['q'] < med) if dr == 'down' else (w['q'] > med)
            cases[(feat, era, 'ok' if ok else 'no')] += 1
out['la66_cases'] = {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in sorted(cases.items())}
print('la66', out['la66_cases'], flush=True)
json.dump(out, open(os.path.join(CK, 'c3.json'), 'w'), indent=1)
print('DONE')
