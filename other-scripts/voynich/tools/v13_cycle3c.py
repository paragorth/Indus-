"""v13 cycle 3c: side-centred same-leaf test (removes any constant recto/verso offset in image or
text; null = permuting image differences across leaves), Voynich and Gerard; Gerard OCR sanity
(blind text-ink area vs cleaned OCR count)."""
import json, os, sys, math, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v13_lib import *
from v13_cycle1 import build
from v13_cycle2 import leaf_pairs, order_key
from v13_cycle3 import leaf_vals, gerard_pairs
from collections import Counter
log = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.append(s)
def perm_r(dx, dy, nperm=20000, seed=0):
    dx = dx - dx.mean(); dy = dy - dy.mean()
    r = np.corrcoef(dx, dy)[0, 1]
    rng = np.random.default_rng(seed)
    rn = np.array([np.corrcoef(rng.permutation(dx), dy)[0, 1] for _ in range(nperm)])
    return r, (np.sum(np.abs(rn) >= abs(r)) + 1) / (nperm + 1), len(dx)
vrows, gc = build()
pairs = leaf_pairs(vrows); herb = [p for p in pairs if p[0]['meta']['illus'] == 'H']
for nm, pp in (('all', pairs), ('herbal', herb)):
    dx, dy = leaf_vals(pp, 'bottom_edges', lambda r: r['txt']['g_benched'])
    P('VOYNICH centred %s: mean d(img)=%+.3f mean d(benched)=%+.4f ; r=%+.3f p=%.4f n=%d' % ((nm, dx.mean(), dy.mean()) + perm_r(dx, dy)))
g = json.load(open(os.path.join(DER, 'v13_gerard.json')))['pages']
norm = lambda w: re.sub(r'[^a-z]', '', w.lower()) if re.fullmatch(r"[A-Za-z]+[.,;:]?", w) else ''
cnt = Counter(norm(w) for v in g.values() for w in v['ocr']); good = {w for w, c in cnt.items() if len(w) >= 3 and c >= 10}
rows = sorted([{'n': int(n), 'img': v['img'], 'txt': {'log_words': math.log1p(sum(norm(w) in good for w in v['ocr']))}} for n, v in g.items()], key=lambda r: r['n'])
ti = np.array([math.log1p(100 * r['img']['text_ink']) for r in rows]); lw = np.array([r['txt']['log_words'] for r in rows])
da = np.array([math.log1p(100 * r['img']['draw_area']) for r in rows])
P('GERARD OCR sanity: text_ink~log_words r=%+.2f ; draw_area~text_ink r=%+.2f ; draw_area~log_words r=%+.2f (pages=%d)' % (
    np.corrcoef(ti, lw)[0, 1], np.corrcoef(da, ti)[0, 1], np.corrcoef(da, lw)[0, 1], len(rows)))
for sh in (0, 1, 3):
    dx, dy = leaf_vals(gerard_pairs(rows, shift=sh), 'draw_area', lambda r: r['txt']['log_words'])
    P('GERARD centred draw_area~log_words shift %d: r=%+.3f p=%.4f n=%d' % ((sh,) + perm_r(dx, dy, 5000)))
open(os.path.join(DER, 'v13', 'c3c_log.txt'), 'w').write('\n'.join(log) + '\n')
