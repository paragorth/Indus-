"""v13 cycle 3b: (1) Gerard control with cleaned OCR (only alphabetic words >=3 letters that occur
>=10 times in the 400-page OCR: woodcut/show-through junk removed), real leaf vs wrong-leaf r for
the layout cell; (2) Voynich root-zone edges ~ benched gallows: offset profile (k=0 spike?) and
independent halves (Currier A herbal vs B herbal; recto-first vs verso-first ordering irrelevant)."""
import json, os, sys, math, re, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v13_lib import *
from v13_cycle1 import build, sec_lang
from v13_cycle2 import leaf_pairs, order_key, leaf_test, IF
from v13_cycle3 import leaf_vals, signflip_r, gerard_pairs
from collections import Counter, defaultdict
log = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.append(s)
g = json.load(open(os.path.join(DER, 'v13_gerard.json')))['pages']
norm = lambda w: re.sub(r'[^a-z]', '', w.lower()) if re.fullmatch(r"[A-Za-z]+[.,;:]?", w) else ''
cnt = Counter(norm(w) for v in g.values() for w in v['ocr'])
good = {w for w, c in cnt.items() if len(w) >= 3 and c >= 10}
rows = []
for n, v in g.items():
    ws = [norm(w) for w in v['ocr']]; ws = [w for w in ws if w in good]
    rows.append({'n': int(n), 'folio': 'g' + n, 'meta': {}, 'img': v['img'],
                 'txt': {'log_words': math.log1p(len(ws)), 'n_words': len(ws)}})
rows.sort(key=lambda r: r['n'])
raw_n = {int(n): len(v['ocr']) for n, v in g.items()}
P('GERARD cleaned OCR: median words/page %d (raw tokens %d)' % (np.median([r['txt']['n_words'] for r in rows]), np.median(list(raw_n.values()))))
for f in ('draw_area', 'draw_edges', 'n_regions', 'bottom_edges'):
    out = []
    for sh in (0, 1, 3, 10):
        dx, dy = leaf_vals(gerard_pairs(rows, shift=sh), f, lambda r: r['txt']['log_words'])
        r, p, n = signflip_r(dx, dy); out.append('shift%d r=%+.2f p=%.4f' % (sh, r, p))
    P('  %s ~ log_words (n=%d leaves): ' % (f, n) + '; '.join(out))
# whole-page correlation in Gerard (no covariates: one hand, one section)
X = np.array([math.log1p(100 * r['img']['draw_area']) for r in rows]); Y = np.array([r['txt']['log_words'] for r in rows])
P('  GERARD page-level draw_area~log_words r=%+.2f (n=%d)' % (np.corrcoef(X, Y)[0, 1], len(rows)))
# ---------------- Voynich
vrows, gc = build()
vrows.sort(key=lambda r: order_key(r['folio']))
groups = defaultdict(list)
for r in vrows:
    groups[sec_lang(r)].append(r)
prof = []
for k in range(-6, 7):
    xs, ys, ms = [], [], []
    for gg in groups.values():
        if len(gg) < 12: continue
        for i, r in enumerate(gg):
            if 0 <= i + k < len(gg):
                xs.append(math.log1p(100 * gg[i + k]['img']['bottom_edges'])); ys.append(r['txt']['g_benched']); ms.append(r['meta'])
    Z = design(ms)
    prof.append('%+d:%+.2f' % (k, partial_r_matrix(Z, np.array(xs)[:, None], np.array(ys)[:, None])[0, 0]))
P('VOYNICH bottom_edges~g_benched offset profile (page-level partial r): ' + ' '.join(prof))
pairs = leaf_pairs(vrows)
for lang in ('A', 'B'):
    pp = [p for p in pairs if p[0]['meta']['illus'] == 'H' and p[0]['meta']['lang'] == lang and p[1]['meta']['lang'] == lang]
    dx, dy = leaf_vals(pp, 'bottom_edges', lambda r: r['txt']['g_benched'])
    if len(dx) > 3:
        P('  herbal Currier %s leaves: r=%+.3f p=%.4f n=%d' % ((lang,) + signflip_r(dx, dy)))
nh = [p for p in pairs if p[0]['meta']['illus'] != 'H']
dx, dy = leaf_vals(nh, 'bottom_edges', lambda r: r['txt']['g_benched'])
P('  non-herbal leaves: r=%+.3f p=%.4f n=%d' % signflip_r(dx, dy))
# what is in the root zone? relation to drawing height and to benched gallows by line position
for f in ('draw_height', 'draw_area', 'bottom_unpainted', 'top_nongreen', 'draw_edges'):
    dx, dy = leaf_vals([p for p in pairs if p[0]['meta']['illus'] == 'H'], f, lambda r: r['txt']['g_benched'])
    P('  herbal %s~g_benched: r=%+.3f p=%.4f n=%d' % ((f,) + signflip_r(dx, dy)))
open(os.path.join(DER, 'v13', 'c3b_log.txt'), 'w').write('\n'.join(log) + '\n')
