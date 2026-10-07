"""LA-78 freeze: survivors of all three time-travel windows (prior-shift scoring, y1), their
count against a label-shuffled null, and a frozen, hashed ranking for RILA-S1 / future finds."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la78_engine import *  # noqa

docs = load()
T = tokens(docs)
pool, signs, words, F = make_pool(T, 800, seed=78)
y1 = np.array([t['y1'] for t in T]); Y = [(y1, 4)]
year = np.array([docs[t['doc']]['year'] for t in T])
MODE['mode'] = 'prior'; MODE['grp'] = np.array([docs[t['doc']]['site'] for t in T])
WIN = [(1950, 1976), (1976, 2100), (1988, 2100)]
R = np.load(os.path.join(CK, 'c1_prior_y1_real.npz'))
fut = R['fut']; cv = R['cv']
surv = np.where((fut > 0).all(1))[0]
out = dict(n_hyp=len(pool), n_survivors=int(len(surv)))
rng = np.random.RandomState(7804)
null = []
for r in range(int(os.environ.get('NNULL', 30))):
    ys = y1.copy()
    for a, b in WIN[:2]:
        te = np.where((year > a) & (year <= b))[0]
        ys[te] = ys[rng.permutation(te)]
    Ys = [(ys, 4)]
    ok = np.ones(len(pool), bool)
    for a, b in WIN:
        tr = year <= a; te = (year > a) & (year <= b)
        ok &= np.array([score(p['f'], Ys, tr, te) > 0 for p in pool])
    null.append(int(ok.sum())); print('null', r, null[-1], flush=True)
out['null_survivors_mean_p95'] = [float(np.mean(null)), float(np.percentile(null, 95))]
out['P_survivors_ge'] = float((1 + sum(n >= len(surv) for n in null)) / (1 + len(null)))
# frozen ranking: survivors ordered by mean future gain; fitted on ALL documents (for new finds)
labels = ['num', 'logo', 'word', 'end']
rows = []
allm = np.ones(len(T), bool)
for i in sorted(surv, key=lambda i: -fut[i].mean()):
    p = pool[i]; f = p['f']
    cnt = np.zeros((int(f.max()) + 1, 4)); np.add.at(cnt, (f, y1), 1)
    entry = dict(name=p['name'], family=p['fam'], K=p['K'], future_gain_bits=[round(float(x), 4) for x in fut[i]],
                 cv_gain_bits=[round(float(x), 4) for x in cv[i]],
                 next_token_counts_by_class={str(c): dict(zip(labels, map(int, cnt[c]))) for c in range(len(cnt))})
    if p['fam'] == 'WORD':
        entry['classes'] = {str(c): [w for w, k in zip(words, p['part']) if k == c] for c in range(p['K'])}
        entry['unseen_word_class'] = 'class of an unseen word = its own class index (fitted to the marginal)'
    else:
        entry['classes'] = {str(c): [s for s, k in zip(signs, p['part']) if k == c] for c in range(p['K'])}
        entry['feature'] = {'FIN': 'class of the last syllabic sign', 'INI': 'class of the first sign',
                            'PAIR': '2*class(first)+class(last)', 'LENFIN': '3*class(last)+min(len,4)-2'}[p['fam']]
    rows.append(entry)
frozen = dict(name='la78 time-travel survivors (Linear A)', date='2026-10-07',
              observable='what follows a clean sign-group of >= 2 syllabic signs: number / logogram / sign-group / end of text',
              score='bits per token over the test-period marginal (label-shift corrected), classes fitted on the past',
              test_rule='on >= 50 new clean sign-groups (RILA-S1 or new finds), a hypothesis is supported if its gain is > 0 and in the top 25 % of 1,000 fresh random partitions of the same family; killed if gain <= 0.',
              n_pool=len(pool), survivors=rows)
h = sha(frozen)
frozen_path = os.path.join(os.path.dirname(CK), 'la78_frozen_ranking.json')
json.dump(frozen, open(frozen_path, 'w'), indent=1, sort_keys=True)
open(frozen_path.replace('.json', '.sha256'), 'w').write(h + '  la78_frozen_ranking.json (sha256 of json.dumps(obj, sort_keys=True))\n')
out['sha256'] = h
out['families'] = dict(collections.Counter(pool[i]['fam'] for i in surv))
out['top5'] = [r['name'] for r in rows[:5]]
json.dump(out, open(os.path.join(CK, 'freeze.json'), 'w'), indent=1)
print('DONE', json.dumps(out))
