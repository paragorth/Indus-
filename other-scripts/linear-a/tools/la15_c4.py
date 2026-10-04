"""LA-15 cycle 4: the census turned on the writers and the habitats.
4a How many Linear A findspots (sites) exist? Species = sites, individuals = documents.
   Out-of-corpus test: from the GORILA-published documents only, predict how many NEW sites
   the post-GORILA documents (count known, sites hidden) come from. Controls: random held-out
   GORILA documents of the same count (60 reps); a planted set of 120 sites with Zipf
   document counts sampled at the GORILA size.
4b How many scribal hands? Species = hands, individuals = attributed documents.
   Calibration: Linear B Knossos hands ('?' stripped, '-' dropped). From LA-sized random
   subsamples (592 attributed documents), predict the number of hands in all KN documents.
"""
import sys, json, random, collections, time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from la15_common import *
from la15_c3 import exch_new, iv

rng = np.random.default_rng(154)
prng = random.Random(154)
res = {}
LA = load_la()
LB = load_lb()


def site_of(d):
    return d['site'] if d['site'] not in ('', '?', 'Crete') else None


TR = [d for d in LA if d['pub'] != 'post' and site_of(d)]
HO = [d for d in LA if d['pub'] == 'post' and site_of(d)]
A = list(collections.Counter(site_of(d) for d in TR).values())
seen = set(site_of(d) for d in TR)
pred = exch_new(A, len(HO), 4000)
truth = len(set(site_of(d) for d in HO) - seen)
res['sites'] = dict(S_obs=len(A), f1=A.count(1), f2=A.count(2), chao1=chao1(A), ace=ace(A),
                    m=len(HO), pred_new=iv(pred), truth_new=truth,
                    new=sorted(set(site_of(d) for d in HO) - seen))
print(res['sites'], flush=True)
# control: random held-out GORILA docs
cov, err = [], []
for r in range(60):
    pool = TR[:]
    prng.shuffle(pool)
    ho, tr = pool[:len(HO)], pool[len(HO):]
    A2 = list(collections.Counter(site_of(d) for d in tr).values())
    p = iv(exch_new(A2, len(ho), 500))
    t = len(set(site_of(d) for d in ho) - set(site_of(d) for d in tr))
    cov.append(p[1] <= t <= p[2])
    err.append(p[0] - t)
res['sites_random_control'] = dict(cover=float(np.mean(cov)), mean_err=float(np.mean(err)))
print(res['sites_random_control'], flush=True)
# planted
w = (np.arange(1, 121) + 0.5) ** -1.6
w /= w.sum()
pc = []
for r in range(200):
    x = rng.multinomial(len(TR), w)
    pc.append((chao1(x[x > 0]), (x > 0).sum()))
pc = np.array(pc)
res['sites_planted120'] = dict(Sobs=float(pc[:, 1].mean()), chao1=float(np.median(pc[:, 0])),
                               chao1_95=[float(np.percentile(pc[:, 0], 2.5)), float(np.percentile(pc[:, 0], 97.5))])
print(res['sites_planted120'], flush=True)

# 4b scribes
la_h = collections.Counter(d['scribe'] for d in LA if d['scribe'])
Ah = list(la_h.values())
res['la_hands'] = dict(S_obs=len(Ah), n=sum(Ah), f1=Ah.count(1), f2=Ah.count(2), chao1=chao1(Ah), ace=ace(Ah),
                       ht=chao1([c for h, c in la_h.items() if h.startswith('HT')]),
                       ht_obs=sum(1 for h in la_h if h.startswith('HT')))
print(res['la_hands'], flush=True)


def norm(h):
    h = h.replace('?', '').strip()
    return None if h in ('', '-') else h


kn = [norm(d['scribe']) for d in LB if d['site'] == 'KN']
kn = [h for h in kn if h]
truth_kn = len(set(kn))
rows = []
for r in range(200):
    smp = prng.sample(kn, sum(Ah))
    c = list(collections.Counter(smp).values())
    nn = exch_new(c, len(kn) - sum(Ah), 200) if r < 40 else None
    rows.append((len(c), chao1(c), ace(c), None if nn is None else len(c) + np.median(nn),
                 None if nn is None else (len(c) + np.percentile(nn, 2.5), len(c) + np.percentile(nn, 97.5))))
c1 = np.array([x[1] for x in rows])
ex = [x for x in rows if x[3] is not None]
res['kn_hands_calib'] = dict(truth=truth_kn, n_full=len(kn), Sobs=float(np.mean([x[0] for x in rows])),
                             chao1=iv(c1), chao1_ge_truth=float(np.mean(c1 >= truth_kn)),
                             extrap=iv([x[3] for x in ex]),
                             extrap_cover=float(np.mean([x[4][0] <= truth_kn <= x[4][1] for x in ex])))
print(res['kn_hands_calib'], flush=True)
json.dump(res, open(os.path.join(OUT, 'c4.json'), 'w'), indent=1, default=float, ensure_ascii=False)
print('done')
