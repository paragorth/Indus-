"""Summarise cycle 1 (random concept-alphabet search) per corpus."""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v64_lib as V
CORP = ['ars', 'med', 'lat', 'voy', 'voyit', 'voy_gshuf', 'voy_mk2', 'voy_sc', 'ars_mk2', 'med_mk2']
rows = {}
for k in CORP:
    p = os.path.join(V.CK, 'c1_%s.json' % k)
    if not os.path.exists(p):
        continue
    r = json.load(open(p)); D = V.all_corpora()['codes']
    sel = r['sel']; n = len(sel)
    g = sorted(x['Gm1'] for x in sel); gf = sorted(x['Gfree'] for x in sel)
    h = r['held']; hb = max(h, key=lambda x: x['Gm1']); hf = max(r['heldf'], key=lambda x: x['Gfree'])
    i0 = r['top'][0]
    row = dict(n=n, sel_best_Gm1=g[-1], sel_p99=g[int(.99 * n)], sel_best_Gfree=gf[-1],
               held_best_Gm1=hb['Gm1'], held_best_Gfree=hf['Gfree'],
               held_top1=h[0], top_alpha=r['al'][i0],
               glyph_held=r['base']['glyph_held'])
    if 'truth' in r['base']:
        row['truth_held'] = r['base']['truth_held']
        cname = [c for c in D if k.startswith(c)][0]
        row['jacc_top1'] = V.jaccard(r['al'][i0], D[cname].values())
        row['jacc_best20'] = max(V.jaccard(r['al'][i], D[cname].values()) for i in r['top'])
    rows[k] = row
    print(k, json.dumps(row))
