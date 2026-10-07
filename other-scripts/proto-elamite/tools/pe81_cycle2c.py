"""pe81 cycle 2c: header vs corner fill, controlling size, line count, aspect, thickness, damage; stratified permutation."""
import sys, os, json, csv
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe81_engine as E, common
SCR = sys.argv[1]; rng = np.random.default_rng(8123)
SH = json.load(open(os.path.join(E.CK, 'shapes.json')))
csv.field_size_limit(10 ** 9)
cat = {'P%06d' % int(x['id_text']): x for x in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8', errors='replace')) if x.get('period', '').startswith('Proto-Elamite')}
def num(s):
    try: return float(s)
    except Exception: return None
T = common.load(); rows = []
for t in T:
    sh = SH.get(t['id']); c = cat.get(t['id'])
    if not sh or not c or not t['lines']: continue
    h, w, th = num(c['height']), num(c['width']), num(c['thickness'])
    if not (h and w and th): continue
    nl = len(t['lines']); nx = sum(g == 'x' for l in t['lines'] for g in l['signs'])
    rev = any(l['surface'] != 'obverse' for l in t['lines'])
    seal = any('seal' in (l.get('raw') or '').lower() for l in t['lines'])
    rows.append(dict(id=t['id'], hd=common.header(t) is not None, cf=sh['corner_fill'], la=np.log(h*w), asp=np.log(h/w), tw=th/w,
                     nl=np.log(nl), dx=float(nx > 0), rev=float(rev), comp=c['object_preservation'] == 'complete', site=t['provenience']))
def partial(sub, B=5000):
    y = np.array([r['cf'] for r in sub]); hd = np.array([r['hd'] for r in sub], float)
    Z = np.column_stack([np.ones(len(sub))] + [np.array([r[k] for r in sub]) for k in ('la', 'asp', 'tw', 'nl', 'dx', 'rev')])
    Z = np.column_stack([Z, Z[:, 1] ** 2, Z[:, 4] ** 2])
    yr = rankdata(y); res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    a, b = res(yr), res(hd); r = float(np.corrcoef(a, b)[0, 1])
    # permutation of header within strata of line-count quartile x size quartile
    s = np.digitize(Z[:, 4], np.quantile(Z[:, 4], [.25, .5, .75])) * 4 + np.digitize(Z[:, 1], np.quantile(Z[:, 1], [.25, .5, .75]))
    cnt = 0
    for _ in range(B):
        hp = hd.copy()
        for k in np.unique(s):
            m = np.where(s == k)[0]; hp[m] = rng.permutation(hp[m])
        cnt += abs(np.corrcoef(a, res(hp))[0, 1]) >= abs(r)
    return dict(n=len(sub), n_header=int(hd.sum()), partial_r=round(r, 3), p=(cnt + 1) / (B + 1))
out = {'all': partial(rows), 'complete': partial([r for r in rows if r['comp']]),
       'complete_noX': partial([r for r in rows if r['comp'] and r['dx'] == 0]),
       'susa_noX_fullwidth': partial([r for r in rows if r['dx'] == 0 and 'Susa' in r['site']])}
# planted-null sanity: random fake 'header' with no shape link
fake = [dict(r, hd=bool(rng.random() < 0.5)) for r in rows]
out['fake_label'] = partial(fake, 1000)
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(E.CK, 'cycle2c.json'), 'w'), indent=1)
