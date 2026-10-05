"""v55 cycle 1: zodiac labels as day-by-day sequences on the real sky (categorical classes).
Usage: v55_cycle1.py COND [NCLS]   COND in real, shuf<k>, fake<k>, plantP1, plantP2, plantW, gen<k>
Writes v55_ckpt/c1_<COND>.json (summary) and .npz (per classifier x feature x year maxima)."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v55_lib as V

cond = sys.argv[1]
ncls = int(sys.argv[2]) if len(sys.argv) > 2 else 300
t0 = time.time()
labels = V.load_labels()
skyk = cond if cond.startswith('fake') else 'real'
T = V.get_sky(skyk)
dd = V.degree_days(T)
assert (dd[:, list(range(0, 270)) + list(range(330, 360))] >= 0).all()
idx = V.alignments(labels, dd)
feats = list(V.day_features(T).items())
rng = np.random.default_rng(abs(hash(cond)) % (1 << 31))
labs = [dict(x) for x in labels]
if cond.startswith('shuf'):
    r = np.random.default_rng(int(cond[4:]))
    for s in set(x['sign'] for x in labs):
        ii = [k for k, x in enumerate(labs) if x['sign'] == s]
        perm = r.permutation(ii)
        vals = [labels[k]['label'] for k in perm]
        for k, v in zip(ii, vals):
            labs[k]['label'] = v
elif cond.startswith('plant'):
    # synthetic almanac for 1421, rotation 0, forward: code glyph reflects that day's sky
    r = np.random.default_rng(7)
    yi = 1421 - V.Y0
    q = {'plantP1': 0.5, 'plantP2': 0.3, 'plantW': 0.5}[cond]
    F = V.day_features(T)
    for k, x in enumerate(labs):
        day = idx[yi, 0, k]
        if r.random() < q:
            if cond == 'plantW':
                code = 'ydlrsmg'[F['weekday'][0][day]]
                x['label'] = x['label'] + code
            else:
                code = ['q', 'y', 'd', 's'][F['moon_elem'][0][day]]
                x['label'] = code + x['label']
elif cond.startswith('gen'):
    # glyph-bigram Markov generator trained on the labels (a 'generator ring')
    r = np.random.default_rng(100 + int(cond[3:]))
    from collections import defaultdict, Counter
    tr = defaultdict(Counter)
    for x in labels:
        w = '^' + x['label'] + '$'
        for a, b in zip(w, w[1:]):
            tr[a][b] += 1
    for x in labs:
        w, c = '', '^'
        while True:
            ks = list(tr[c]); p = np.array([tr[c][k] for k in ks], float)
            c = ks[r.choice(len(ks), p=p / p.sum())]
            if c == '$' or len(w) > 14:
                break
            w += c
        x['label'] = w or 'o'
classes = V.random_classifiers(labs, ncls, seed=55)
maskA = np.array([x['name'] in V.HALF_A for x in labs])
res = V.run_search(labs, classes, feats, idx, maskA)
summ = V.summarize(res)
summ.update(cond=cond, ncls=len(classes), secs=round(time.time() - t0), feats=[f for f, _ in feats])
ci, fi, yi = summ['argmax']
summ['best'] = dict(classifier=classes[ci][0][:120], feature=feats[fi][0], year=V.Y0 + yi,
                    align=int(res['full_al'][ci, fi, yi]))
np.savez_compressed(os.path.join(V.CK, f'c1_{cond}.npz'), **res)
json.dump(summ, open(os.path.join(V.CK, f'c1_{cond}.json'), 'w'))
print(json.dumps(summ))
