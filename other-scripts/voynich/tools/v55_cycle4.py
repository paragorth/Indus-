"""v55 cycle 4: THE CIRCULAR TEXTS AS DAY RUNS. The words of each sign's ring texts (964 words)
are spread over the 30 days the Sun spends in that sign (word k of n -> day 30k/n, rotation,
direction); recurrent words, affixes and random word classes are crossed with the daily sky
(categorical classes and events) for every year 1290-1611. Held-out halves A/B by sign."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v55_lib as V

cond = sys.argv[1]
t0 = time.time()
words = V.load_ring_words()
T = V.get_sky(cond if cond.startswith('fake') else 'real')
dd = V.degree_days(T)
idx = V.alignments(words, dd)
labs = V.make_condition(cond, words, idx, T)
feats = list(V.day_features(T).items()) + V.event_features(T)
classes = V.marker_classifiers(labs, min_type=4, min_affix=10) + V.random_classifiers(labs, 60, seed=57)
maskA = np.array([x['name'] in V.HALF_A for x in labs])
res = V.run_search(labs, classes, feats, idx, maskA)
summ = V.summarize(res)
ci, fi, yi = summ['argmax']
summ.update(cond=cond, ncls=len(classes), nfeat=len(feats), secs=round(time.time() - t0),
            best=dict(classifier=classes[ci][0][:80], feature=feats[fi][0], year=V.Y0 + yi, align=int(res['full_al'][ci, fi, yi])))
np.savez_compressed(os.path.join(V.CK, f'c4_{cond}.npz'), **res)
json.dump(summ, open(os.path.join(V.CK, f'c4_{cond}.json'), 'w'))
print(json.dumps(summ))
