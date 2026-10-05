"""v55 cycle 3: THE ZODIAC AS A HOROSCOPE. Nymph i of sign s = ecliptic degree (rotation o,
direction); for every day 1290-1611 the Sun, Moon and five planets each stand on one degree.
Does any word class (recurrent types, affixes, substrings, random binary classes) sit on the
degrees occupied by the bodies on some date, beyond within-sign shuffles, fake skies and a
generator ring? Which year does the best date fall in? (C kernel v55_c/horo.c)"""
import sys, os, json, time, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v55_lib as V

cond = sys.argv[1]
t0 = time.time()
labels = V.load_labels()
T = V.get_sky(cond if cond.startswith('fake') else 'real')
dd = V.degree_days(T)
idx = V.alignments(labels, dd)
yr = T['year']
keep = (yr >= V.Y0) & (yr <= V.Y1)
B = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn']
deg = np.stack([np.floor(T[b][keep]).astype(np.int16) % 360 for b in B], 1)
yi = (yr[keep] - V.Y0).astype(np.int16)
L = len(labels)
amap = np.full((60, 360), -1, np.int16)
for li, x in enumerate(labels):
    s, i, n = x['sign'], x['i'], x['n']
    for di in range(2):
        ii = i if di == 0 else n - 1 - i
        for o in range(30):
            amap[di * 30 + o, 30 * s + int(np.floor(ii * 30.0 / n + o)) % 30] = li
labs = [dict(x) for x in labels]
PLANT_DAY = int(np.where((T['year'][keep] == 1421) & (T['month'][keep] == 5) & (T['day'][keep] == 1))[0][0])
if cond.startswith('plantH'):
    # plantH: 4 of the occupied degrees + 6 decoys; plantH7: every occupied degree + 3 decoys
    r = np.random.default_rng(3)
    occ = [amap[0, d] for d in deg[PLANT_DAY] if amap[0, d] >= 0]
    nh, nd = (4, 6) if cond == 'plantH' else (len(occ), 3)
    pick = list(r.choice(occ, nh, replace=False))
    others = [k for k in range(L) if k not in occ]
    pick += list(r.choice(others, nd, replace=False))
    for k in pick:
        labs[k]['label'] = 'q' + labs[k]['label']
else:
    labs = V.make_condition(cond, labels, idx, T)
classes = V.marker_classifiers(labs, prefix1=True) + V.random_classifiers(labs, 150, seed=56, kmax=2)
classes = [(n_, c) for n_, c in classes if c.max() == 1]
C = np.array([c for _, c in classes], np.uint8)
inp = os.path.join(V.CK, f'c3_{cond}.bin'); outp = os.path.join(V.CK, f'c3_{cond}.out')
with open(inp, 'wb') as f:
    np.array([len(deg), 7, 60, len(C), L], np.int32).tofile(f)
    deg.tofile(f); yi.tofile(f); amap.tofile(f); C.tofile(f)
nY = V.Y1 - V.Y0 + 1
subprocess.run([os.path.join(V.CK, 'horo'), inp, outp, str(nY)], check=True)
raw = open(outp, 'rb').read()
nc = len(C)
best = np.frombuffer(raw, np.float32, nc * nY).reshape(nc, nY)
bd = np.frombuffer(raw, np.int32, nc * nY, offset=4 * nc * nY).reshape(nc, nY)
ba = np.frombuffer(raw, np.int16, nc * nY, offset=8 * nc * nY).reshape(nc, nY)
os.remove(inp)
years = np.arange(V.Y0, V.Y1 + 1)
win = (years >= V.WINDOW[0]) & (years <= V.WINDOW[1])
ymax = best.max(0)
ci, y = np.unravel_index(best.argmax(), best.shape)
d = bd[ci, y]
summ = dict(cond=cond, ncls=nc, max=float(best.max()), peak_year=int(years[ymax.argmax()]),
            win_z=float((ymax[win].mean() - ymax[~win].mean()) / ymax[~win].std()),
            win_max=float(ymax[win].max()), out_max=float(ymax[~win].max()),
            best=dict(classifier=classes[ci][0][:80], date='%d-%02d-%02d' % (T['year'][keep][d], T['month'][keep][d], T['day'][keep][d]),
                      align=int(ba[ci, y])), secs=round(time.time() - t0))
if cond.startswith('plantH'):
    qi = [k for k, (n_, _) in enumerate(classes) if n_.startswith(('pre1=q', 'pre2=q'))]
    summ['plant_class_best'] = [(classes[k][0], float(best[k].max()), int(years[best[k].argmax()])) for k in qi]
np.save(os.path.join(V.CK, f'c3_{cond}_best.npy'), best)
json.dump(summ, open(os.path.join(V.CK, f'c3_{cond}.json'), 'w'))
print(json.dumps(summ))
