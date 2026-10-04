"""PE-22 cycle 2, stage 1 (pre-registration). From the tablets published up to 1999 only, plus
the SIZES and SITES of the recent tablets (TCL 32 2019, unpublished Susa and Malyan entries,
Tepe Sofalin, Ozbaki), predict:
  P1 new base-sign types, P2 new graph types (variants), P3 new entry strings, P4 new sign pairs,
     in all recent tablets (EXCH community bootstrap, HAB habitat novelty, ABC Zipf-Mandelbrot);
  P5 new sign types at the new sites (Sofalin + Ozbaki) and at Malyan;
  P6 which known signs appear: top 30 by habitat and exchangeable scores, and the expected number
     of known sign types present; P7 the share of recent sign tokens that are new types.
The JSON (without the hash field) is hashed with sha256 BEFORE any held-out sign is read.
Stage 2 (cycle 3) reads the recent tablets' signs and scores.
"""
import sys, json, collections, time
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from pe22_common import *

rng = np.random.default_rng(2202)
t0 = time.time()
PE = load_pe()
tr, ho = pe_split(PE)
pre = dict(split='train: published <= 1999 (+ undated old); recent: >= 2000, unpublished, Sofalin, Ozbaki',
           n_train=len(tr), n_recent=len(ho), recent_ids=sorted(d['id'] for d in ho))
for key in ('signs', 'graphs', 'words', 'bigr'):
    meta = [(d['site'], len(d[key])) for d in ho if d[key]]
    pre[key] = predict_new(tr, meta, key, rng, B=1000, abc=key in ('signs', 'graphs', 'words'))
    print(key, pre[key], round(time.time() - t0), flush=True)
A = list(collections.Counter(w for d in tr for w in d['signs']).values())
for grp, sites in (('newsites', ('Sofalin', 'Ozbaki')), ('Malyan', ('Malyan',)), ('Susa', ('Susa',))):
    meta = [(d['site'], len(d['signs'])) for d in ho if d['site'] in sites and d['signs']]
    m = sum(k for _, k in meta)
    hs = collections.Counter()
    for s, k in meta:
        hs[s] += k
    pre['site_' + grp] = dict(m=m, EXCH=iv(exch_new(A, m, 1000, rng)), HAB=iv(hab_new(tr, hs, 'signs', 1000, rng)))
    print(grp, pre['site_' + grp], flush=True)
shell = [dict(site=d['site'], m=len(d['signs'])) for d in ho if d['signs']]
sc_ex, sc_hab, ec = known_scores(tr, shell, 'signs')
pe_ = np.array(list(sc_hab.values()))
sims = [(rng.random(len(pe_)) < pe_).sum() for _ in range(2000)]
pre['known_types_expected'] = dict(hab=iv(sims), exch_mean=float(sum(sc_ex.values())))
pre['top30_hab'] = [w for w, _ in sorted(sc_hab.items(), key=lambda x: -x[1])[:30]]
pre['top30_exch'] = [w for w, _ in sorted(sc_ex.items(), key=lambda x: -x[1])[:30]]
# moderately common signs the habitat model says will NOT appear (risky predictions)
mid = [w for w in sc_hab if 0.4 < sc_ex[w] < 0.9]
pre['risky_absent_hab'] = [w for w, _ in sorted(((w, sc_hab[w]) for w in mid), key=lambda x: x[1])[:15]]
m_all = pre['signs']['m']
pre['P7_new_token_share'] = dict(EXCH_types_over_tokens=[x / m_all for x in pre['signs']['EXCH']])
pre_hash = sha(pre)
pre['sha256'] = pre_hash
json.dump(pre, open(os.path.join(CK, 'cycle2_predictions.json'), 'w'), indent=1, default=float)
print('HASH', pre_hash, round(time.time() - t0), flush=True)
