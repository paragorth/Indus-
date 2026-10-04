"""PE-22 cycle 4b: are the recent tablets 'duller' than any batch of old tablets?
Cycle 3 found fewer distinct known signs (173 vs 205 expected) and fewer new sign pairs (96 vs
123) than predicted. Control: the same two statistics, observed minus expected, for 200 random
pre-2000 tablet sets and 200 contiguous publication-order batches of the same sign-token count,
and for the recent tablets with fragmentary tablets (any lacuna line) removed and with Susa only.
Statistics: K = known sign types present - expected (habitat model); P = new pairs observed -
EXCH median; R = distinct types / tokens.
"""
import sys, json, collections, random, time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from pe22_common import *

rng = np.random.default_rng(2205)
prng = random.Random(2205)
t0 = time.time()
PE = load_pe()
tr, ho = pe_split(PE)
raw = {d['id']: d for d in json.load(open(os.path.join(DATA, 'pe_corpus.json')))}
mS = sum(len(d['signs']) for d in ho)


def stats(trn, hold):
    shell = [dict(site=d['site'], m=len(d['signs'])) for d in hold if d['signs']]
    _, hab, _ = known_scores(trn, shell, 'signs')
    seen = set(w for d in trn for w in d['signs'])
    pres = set(w for d in hold for w in d['signs'])
    K = len(pres & seen) - sum(hab.values())
    A = list(collections.Counter(w for d in trn for w in d['bigr']).values())
    m = sum(len(d['bigr']) for d in hold)
    P = truth_new(trn, hold, 'bigr') - float(np.median(exch_new(A, m, 60, rng)))
    toks = [w for d in hold for w in d['signs']]
    return dict(K=float(K), P=P, R=len(set(toks)) / len(toks), Pm=P / max(m, 1))


def take(pool, m):
    out, t = [], 0
    for d in pool:
        if t >= m:
            break
        if d['signs']:
            out.append(d)
            t += len(d['signs'])
    return out


res = {'real': stats(tr, ho)}
frag = lambda d: any(l.get('lacuna') for l in raw[d['id']]['lines'])
res['real_nofrag'] = stats(tr, [d for d in ho if not frag(d)])
res['real_susa'] = stats(tr, [d for d in ho if d['site'] == 'Susa'])
res['real_tcl32'] = stats(tr, [d for d in ho if d['pub'].startswith('TCL 32')])
res['frag_share'] = dict(recent=float(np.mean([frag(d) for d in ho])), train=float(np.mean([frag(d) for d in tr])))
print(res, flush=True)
TRs = [d for d in tr if d['signs']]
order = sorted(TRs, key=lambda d: (d['pub'].split(',')[0], d['id']))
for name in ('random', 'block'):
    out = []
    for r in range(120):
        if name == 'random':
            pool = TRs[:]
            prng.shuffle(pool)
        else:
            i = prng.randrange(len(order))
            pool = order[i:] + order[:i]
        hold = take(pool, mS)
        ids = set(d['id'] for d in hold)
        out.append(stats([d for d in tr if d['id'] not in ids], hold))
    res[name] = {k: iv([x[k] for x in out]) for k in out[0]}
    res[name + '_p_low'] = {k: float(np.mean([x[k] <= res['real'][k] for x in out])) for k in out[0]}
    print(name, res[name], res[name + '_p_low'], round(time.time() - t0), flush=True)
json.dump(res, open(os.path.join(CK, 'cycle4b.json'), 'w'), indent=1, default=float)
print('done')
