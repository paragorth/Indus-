"""v55 cycle 2: recurrent label words / affixes as EVENT MARKERS on the real sky
(new / full / quarter moons, eclipses, Moon ingress, Moon-planet conjunctions, planet ingress,
Mercury retrograde and stations, Sundays, Easter and movable feasts), every year 1290-1611,
every rotation and direction. Same G-test engine and A/B held-out halves as cycle 1."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v55_lib as V

cond = sys.argv[1]
t0 = time.time()
labels = V.load_labels()
T = V.get_sky(cond if cond.startswith('fake') else 'real')
dd = V.degree_days(T)
idx = V.alignments(labels, dd)
labs = V.make_condition(cond, labels, idx, T)
feats = V.event_features(T)
classes = V.marker_classifiers(labs)
maskA = np.array([x['name'] in V.HALF_A for x in labs])
res = V.run_search(labs, classes, feats, idx, maskA)
summ = V.summarize(res)
ci, fi, yi = summ['argmax']
summ.update(cond=cond, ncls=len(classes), secs=round(time.time() - t0),
            best=dict(classifier=classes[ci][0], feature=feats[fi][0], year=V.Y0 + yi, align=int(res['full_al'][ci, fi, yi])))
np.savez_compressed(os.path.join(V.CK, f'c2_{cond}.npz'), **res)
json.dump(summ, open(os.path.join(V.CK, f'c2_{cond}.json'), 'w'))
print(json.dumps(summ))
