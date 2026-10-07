"""la76 cycle 3: freeze the surviving value vector (full corpus fit), hash it, and test unit-free sign
predictions from Near Eastern price data (Laws of Eshnunna s.1 numbers; Old Babylonian slave prices),
calibrated by the same procedure on planted economies. Also: mixed-goods KU-RO sections."""
import sys, os, json, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '2')
sys.path.insert(0, os.path.dirname(__file__))
import la76_c1 as C
from la76_common import *

mk, K, goods = C.mk, C.K, C.goods
gi = {g: i for i, g in enumerate(goods)}
alld = set(mk.docs); p1, p2 = mk.doc_mask(alld)


def fit(lq, seed, nboot=0):
    rng = np.random.default_rng(seed)
    LV, S = C.climb(lq, p1, p2, rng)
    top = np.argsort(-S)[:50]
    return np.median(LV[top], 0), LV[top]


# Outside predictions (unit-free signs only; Linear A unit sizes unknown):
# LE s.1: per shekel 300 qa barley, 12 qa sesame oil, 3 qa best oil -> oil 25-100x barley per qa.
# A Linear A oil unit would have to exceed 25 grain units for oil to be cheaper per unit: predict OLE > GRA.
# OB slave/person prices 10-30 shekels = 3,000-9,000 qa barley: predict VIR > GRA (per head vs grain unit).
PRED = [('OLE', 'GRA'), ('VIR', 'GRA')]

if __name__ == '__main__':
    best, tops = fit(mk.lq, 11)
    # stability: 20 restarts of the full-corpus search with different seeds
    reps = np.array([fit(mk.lq, 100 + s)[0] for s in range(20)])
    frozen = {'goods': goods, 'log_value_median_top50': np.round(best, 4).tolist(),
              'restart_sd': np.round(reps.std(0), 4).tolist(),
              'sign_share': {f'{a}>{b}': float((reps[:, gi[a]] > reps[:, gi[b]]).mean()) for a, b in PRED},
              'note': 'la76 frozen before outside comparison; log values relative to *304; fit = S1+S2 on all 146 documents'}
    blob = json.dumps(frozen, sort_keys=True).encode()
    frozen['sha256'] = hashlib.sha256(blob).hexdigest()
    json.dump(frozen, open(os.path.join(D, 'la76_values.json'), 'w'), indent=1)
    print('FROZEN', frozen['sha256']); print(dict(zip(goods, np.round(best, 2)))); print('restart sd', dict(zip(goods, np.round(reps.std(0), 2))))
    print('sign shares over 20 restarts', frozen['sign_share'])
    # calibration: on planted full economies (20 draws), how often is the sign of a planted pair recovered?
    hits = {p: [] for p in PRED}; rec = []
    for s in range(20):
        rng = np.random.default_rng(500 + s)
        lv = random_values(rng, 1, K, span=np.log(20))[0]
        lq = planted_quantities(mk, rng, lv, frac=1.0)
        b, _ = fit(lq, 900 + s)
        for a, c in PRED:
            hits[(a, c)].append(np.sign(b[gi[a]] - b[gi[c]]) == np.sign(lv[gi[a]] - lv[gi[c]]))
        from scipy.stats import spearmanr; rec.append(spearmanr(b[1:], lv[1:])[0])
    print('planted full economy: sign recovered', {f'{a}>{c}': float(np.mean(h)) for (a, c), h in hits.items()}, 'rho %.2f' % np.mean(rec))
    # same on no-economy (permuted) data: share of fits that put OLE>GRA and VIR>GRA by chance
    ch = {p: [] for p in PRED}
    for s in range(20):
        b, _ = fit(mk.permute(np.random.default_rng(700 + s)), 1300 + s)
        for a, c in PRED: ch[(a, c)].append(b[gi[a]] > b[gi[c]])
    print('no-economy fits: share with sign as predicted', {f'{a}>{c}': float(np.mean(h)) for (a, c), h in ch.items()})
