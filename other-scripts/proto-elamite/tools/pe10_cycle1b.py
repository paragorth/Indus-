"""pe10 cycle 1b: the between-tablet 'spacious tablets carry longer entries' effect,
re-tested with a stricter null (permute mm/line only among tablets sharing opener
and dominant number system), on all tablets with an intact obverse (not only >=5 entries),
and the Ur III positive control expressed as short-form rates by tablet-height quartile.
"""
import json
import numpy as np
from pe10_common import *
from pe10_cycle2 import system

rng = np.random.default_rng(11)

T = load()
rows = []
for t in T:
    if not t['h']:
        continue
    obv = [u for u in t['units'] if u['face'] == 'obverse']
    if not obv or any(u['prime'] for u in obv):
        continue
    if any(m[0] == 'obverse' and ('broken' in m[3].lower() or 'missing' in m[3].lower()) for m in t['markers']):
        continue
    ent = [u for u in obv if u['entry'] and not u['broken'] and 'x' not in u['signs']]
    if len(ent) < 3:
        continue
    U = t['units']
    hdr = base(U[0]['signs'][0]) if U[0]['header'] and U[0]['signs'] else '-'
    sy = Counter(system(u['nums']) for u in ent).most_common(1)[0][0]
    rows.append({'mml': t['h'] / len(obv), 'len': np.mean([len(u['signs']) for u in ent]),
                 'n': len(ent), 'stratum': hdr + '/' + sy, 'w': t['w'] or np.nan, 'h': t['h']})
mml = np.array([r['mml'] for r in rows]); ml = np.array([r['len'] for r in rows])
st = np.array([r['stratum'] for r in rows])
nb = np.digitize([r['n'] for r in rows], [5, 8, 12, 18])
key = np.array([s + '#' + str(b) for s, b in zip(st, nb)])
def cor(x):
    return float(np.corrcoef(x, ml)[0, 1])
obs = cor(mml)
nul = []
for _ in range(10000):
    x = mml.copy()
    for k in np.unique(key):
        idx = np.where(key == k)[0]
        x[idx] = mml[rng.permutation(idx)]
    nul.append(cor(x))
nul = np.array(nul)
# spacious vs cramped: does WIDTH matter given height spacing? (horizontal room for a long string)
W = np.array([r['w'] for r in rows]); ok = ~np.isnan(W)
rw = float(np.corrcoef(W[ok], ml[ok])[0, 1])
res = {'n_tablets': len(rows), 'r_mm_per_line_vs_meanlen': obs, 'strat_null_mean': float(nul.mean()),
       'strat_null_sd': float(nul.std()), 'z_vs_strat_null': float((obs - nul.mean()) / nul.std()),
       'r_width_vs_meanlen': rw}
# Ur III: short-form rate by height quartile within year
d = json.load(open(os.path.join(CKPT, 'ur3_years.json')))
by = defaultdict(list)
for r in d['rows']:
    by[r['year']].append(r)
sq = defaultdict(list)
for y, g in by.items():
    if len(g) < 20:
        continue
    L = np.array([r['len'] for r in g]); H = np.array([r['h'] for r in g])
    med = np.median(L)
    qs = np.quantile(H, [0.25, 0.5, 0.75])
    for r in g:
        sq[int(np.digitize(r['h'], qs))].append(r['len'] < med)
res['ur3_shortform_rate_by_height_quartile'] = {k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(sq.items())}
json.dump(res, open(os.path.join(CKPT, 'c1b.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
