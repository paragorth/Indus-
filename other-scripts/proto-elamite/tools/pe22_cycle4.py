"""PE-22 cycle 4.
(A) POST-HOC (not pre-registered) cleaned re-score: drop sign tokens that are transliteration
    artefacts (',', 'MXXX', 'M1', '3(M39B)', unbarred compounds such as 'M362+x'), applied to
    training and recent tablets alike; re-run the cycle-2 models and score.
(B) Discovery curve in publication order: new base-sign types contributed by each publication
    batch (1905 ... 2019) against 300 random tablet orders (same batch sizes). Is late
    discovery slower than exchangeable (the first editors took the 'rich' tablets)?
(C) OUT-OF-CORPUS pre-registration (hashed, to be scored when the ATF appears): the 89 CDLI
    catalogue entries 'unpublished assigned', Susa, National Museum Tehran, with no ATF in the
    2023 CDLI dump (cdli.earth returned HTTP 500 on 4 Oct 2026). Sizes are unknown, so the
    prediction is a rate per 1,000 cleaned sign tokens plus a known-sign ranking.
"""
import sys, json, collections, random, time, re
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from pe22_common import *

rng = np.random.default_rng(2204)
prng = random.Random(2204)
t0 = time.time()
res = {}
PAT = re.compile(r'^(M\d{3}[a-z]?(@[a-z])?|\|[^|]+\|)$')
PE = load_pe()
for d in PE:
    d['signs'] = [w for w in d['signs'] if PAT.match(w)]
tr, ho = pe_split(PE)
# (A)
A_ = {}
for key in ('signs',):
    meta = [(d['site'], len(d[key])) for d in ho if d[key]]
    P = predict_new(tr, meta, key, rng, B=1000, abc=True)
    A_[key] = scored(P, truth_new(tr, ho, key))
seen = set(w for d in tr for w in d['signs'])
A_['new_types'] = sorted(set(w for d in ho for w in d['signs']) - seen)
for grp, sites in (('newsites', ('Sofalin', 'Ozbaki')),):
    H = [d for d in ho if d['site'] in sites]
    meta = [(d['site'], len(d['signs'])) for d in H if d['signs']]
    hs = collections.Counter()
    for s, k in meta:
        hs[s] += k
    Atr = list(collections.Counter(w for d in tr for w in d['signs']).values())
    m = sum(hs.values())
    P = dict(m=m, EXCH=iv(exch_new(Atr, m, 1000, rng)), HAB=iv(hab_new(tr, hs, 'signs', 1000, rng)))
    A_['site_' + grp] = dict(scored(P, len(set(w for d in H for w in d['signs']) - seen)),
                             types=sorted(set(w for d in H for w in d['signs']) - seen),
                             n_types=len(set(w for d in H for w in d['signs'])))
res['A_cleaned_posthoc'] = A_
print('A', json.dumps(A_, default=float), flush=True)

# (B) discovery curve
docs = [d for d in PE if d['signs'] and d['site'] != 'Larsa']


def batch_of(d):
    if d in ho:
        return 9999 if d['year'] is None else d['year']
    return d['year'] if d['year'] else 1905


B_ = collections.defaultdict(list)
for d in docs:
    B_[batch_of(d)].append(d)
years = sorted(B_)
sizes = [len(B_[y]) for y in years]


def curve(order_docs):
    seen_, out, i = set(), [], 0
    for s in sizes:
        new = set()
        for d in order_docs[i:i + s]:
            new |= set(d['signs'])
        out.append(len(new - seen_))
        seen_ |= new
        i += s
    return out


real = curve([d for y in years for d in B_[y]])
null = []
for r in range(300):
    o = docs[:]
    prng.shuffle(o)
    null.append(curve(o))
null = np.array(null)
res['B_discovery'] = [dict(year=y, n_tablets=s, new_real=int(real[i]), null=iv(null[:, i]),
                           p_low=float(np.mean(null[:, i] <= real[i])), p_high=float(np.mean(null[:, i] >= real[i])))
                      for i, (y, s) in enumerate(zip(years, sizes))]
late = [i for i, y in enumerate(years) if y >= 1950]
lr, ln = sum(real[i] for i in late), null[:, late].sum(1)
res['B_late_total'] = dict(real=int(lr), null=iv(ln), p_low=float(np.mean(ln <= lr)))
print('B', json.dumps(res['B_discovery']), res['B_late_total'], flush=True)

# (C) out-of-corpus pre-registration
full = [d for d in PE if d['site'] != 'Larsa']
A = list(collections.Counter(w for d in full for w in d['signs']).values())
susa = [d for d in full if d['site'] == 'Susa' and d['signs']]
mt = float(np.mean([len(d['signs']) for d in susa]))
pre = dict(target='89 CDLI catalogue entries: primary_publication "unpublished assigned", provenience Susa, '
                  'collection National Museum Tehran, no ATF in the 2023 CDLI dump (P-numbers in catalog.json, '
                  'in_pe False). Units: base signs (variants merged), artefacts dropped as in cycle 4A.',
           mean_tokens_per_susa_tablet=mt)
ex1000 = exch_new(A, 1000, 1000, rng)
abc1000, _ = abc_new(A, 1000, rng)
pre['new_base_types_per_1000_tokens'] = dict(EXCH=iv(ex1000), ABC=iv(abc1000))
m89 = int(round(89 * mt))
pre['new_base_types_if_89_average_tablets'] = dict(m=m89, EXCH=iv(exch_new(A, m89, 1000, rng)),
                                                   HAB=iv(hab_new(full, {'Susa': m89}, 'signs', 1000, rng)))
sc_ex, sc_hab, ec = known_scores(full, [dict(site='Susa', m=m89)], 'signs')
pre['top40_known'] = [w for w, _ in sorted(sc_hab.items(), key=lambda x: -x[1])[:40]]
pre['expected_known_types_present'] = float(sum(sc_hab.values()))
pre['scoring'] = ('rate: count base types absent from all 1,581 corpus tablets per 1,000 tokens, inside 95% '
                  'interval = pass; known ranking: AUC > 0.85 and >= 36 of top 40 present = pass '
                  '(cycle 3 real AUC 0.92, 29/30)')
h = sha(pre)
pre['sha256'] = h
res['C_preregistration'] = pre
json.dump(pre, open(os.path.join(CK, 'cycle4_outofcorpus_predictions.json'), 'w'), indent=1, default=float)
print('C', json.dumps(pre, default=float), flush=True)
json.dump(res, open(os.path.join(CK, 'cycle4.json'), 'w'), indent=1, default=float)
print('done', round(time.time() - t0))
