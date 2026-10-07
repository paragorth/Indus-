"""pe81 cycle 2b: is header<->corner_fill breakage? restrict to intact tablets."""
import sys, os, json, csv
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe81_engine as E, common
SCR = sys.argv[1]
SH = json.load(open(os.path.join(E.CK, 'shapes.json')))
csv.field_size_limit(10 ** 9)
cat = {'P%06d' % int(x['id_text']): x for x in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8', errors='replace')) if x.get('period', '').startswith('Proto-Elamite')}
T = common.load()
R = []
for t in T:
    sh = SH.get(t['id']); c = cat.get(t['id'])
    if not sh or not c or not t['lines']:
        continue
    obv = [l for l in t['lines'] if l['surface'] == 'obverse']
    first_ok = bool(obv) and not obv[0]['lacuna'] and not obv[0]['damaged'] and obv[0]['label'] in ('1', "1'") and "'" not in obv[0]['label']
    anyprime = any("'" in (l['label'] or '') for l in t['lines'])
    R.append(dict(id=t['id'], header=common.header(t) is not None, cf=sh['corner_fill'], ctb=sh['corner_top_vs_bottom'],
                  rect=sh['rect'], pres=c['object_preservation'], first_ok=first_ok, prime=anyprime,
                  x=sum(g == 'x' for l in t['lines'] for g in l['signs'])))
def rep(name, sub):
    h = np.array([r['header'] for r in sub]); cf = np.array([r['cf'] for r in sub]); ctb = np.array([r['ctb'] for r in sub])
    if h.sum() < 5 or (~h).sum() < 5:
        print(name, len(sub), 'too few'); return None
    p = mannwhitneyu(cf[h], cf[~h]).pvalue; p2 = mannwhitneyu(ctb[h], ctb[~h]).pvalue
    d = dict(n=len(sub), n_header=int(h.sum()), cf_header=float(np.median(cf[h])), cf_none=float(np.median(cf[~h])), p_cf=float(p),
             ctb_header=float(np.median(ctb[h])), ctb_none=float(np.median(ctb[~h])), p_ctb=float(p2))
    print(name, d); return d
out = {}
out['all'] = rep('all', R)
out['complete'] = rep('catalogue complete', [r for r in R if r['pres'] == 'complete'])
out['intact_text'] = rep('no primed labels, first obverse line intact, no x', [r for r in R if not r['prime'] and r['first_ok'] and r['x'] == 0])
out['both'] = rep('complete AND intact text', [r for r in R if r['pres'] == 'complete' and not r['prime'] and r['first_ok'] and r['x'] == 0])
out['fragments'] = rep('catalogue fragment', [r for r in R if r['pres'] == 'fragment'])
json.dump(out, open(os.path.join(E.CK, 'cycle2b.json'), 'w'), indent=1)
