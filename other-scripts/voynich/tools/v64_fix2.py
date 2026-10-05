"""Post-fix for cycle-2 files written before the truth-indentation fix:
recompute truth fields for control corpora, drop them for others, add glyph_held."""
import sys, json, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v64_lib as V
D = V.all_corpora()
for f in sorted(os.listdir(V.CK)):
    if not (f.startswith('c2_') and f.endswith('.json')):
        continue
    k = f[3:-5]; p = os.path.join(V.CK, f); r = json.load(open(p))
    if 'glyph_held' in r:
        continue
    ps, ph = os.path.join(V.CK, 'c_%s_sel.txt' % k), os.path.join(V.CK, 'c_%s_held.txt' % k)
    for key in ('truth_sel', 'truth_held', 'jacc_finals'):
        r.pop(key, None)
    if k in D['codes']:
        codes = D['codes'][k]
        r['truth_sel'] = V.score(ps, [sorted(codes.values())])[0]
        r['truth_held'] = V.score(ph, [sorted(codes.values())])[0]
        r['jacc_finals'] = [V.jaccard(a, codes.values()) for a, _ in r['finals']]
    tr, _ = V.type_counts(D['corpora'][k], 'sel')
    gl = Counter(c for w, n in tr.items() for c in w)
    r['glyph_held'] = V.score(ph, [sorted(gl)[:40]])[0]
    json.dump(r, open(p, 'w')); print('fixed', k)
