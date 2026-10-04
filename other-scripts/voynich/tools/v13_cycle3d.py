"""v13 cycle 3d: show-through check for root-zone edges ~ benched gallows. Each leaf side a has a
reverse side b. If the bold benched gallows of b show through into a's drawing zone, a's root-zone
edges follow b's benched rate (not a's). Partial r (section+lang+hand) over all leaf sides."""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v13_lib import *
from v13_cycle1 import build
from v13_cycle2 import leaf_pairs
log = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.append(s)
rows, gc = build()
pairs = leaf_pairs(rows)
for nm, pp in (('all', pairs), ('herbal', [p for p in pairs if p[0]['meta']['illus'] == 'H'])):
    sides = [(a, b) for a, b in pp] + [(b, a) for a, b in pp]
    y = np.array([math.log1p(100 * a['img']['bottom_edges']) for a, b in sides])
    own = np.array([a['txt']['g_benched'] for a, b in sides]); rev = np.array([b['txt']['g_benched'] for a, b in sides])
    revink = np.array([math.log1p(100 * b['img']['text_ink']) for a, b in sides])
    leaf = [a['folio'][:-1] for a, b in sides]
    Z = design([a['meta'] for a, b in sides])
    Zo = np.column_stack([Z, rev]); Zr = np.column_stack([Z, own])
    r_own = partial_r_matrix(Zo, own[:, None], y[:, None])[0, 0]
    r_rev = partial_r_matrix(Zr, rev[:, None], y[:, None])[0, 0]
    r_ink = partial_r_matrix(Z, revink[:, None], y[:, None])[0, 0]
    # null: permute which leaf the text comes from (keep side pairing), 5000x
    rng = np.random.default_rng(0); L = len(pp)
    no, nr = [], []
    for _ in range(5000):
        pi = rng.permutation(L); pi = np.concatenate([pi, pi + L])
        o2, r2 = own[pi], rev[pi]
        no.append(partial_r_matrix(np.column_stack([Z, r2]), o2[:, None], y[:, None])[0, 0])
        nr.append(partial_r_matrix(np.column_stack([Z, o2]), r2[:, None], y[:, None])[0, 0])
    no, nr = np.array(no), np.array(nr)
    P('%s sides=%d: root-zone edges ~ OWN benched (| reverse) r=%+.3f p=%.4f ; ~ REVERSE benched (| own) r=%+.3f p=%.4f ; ~ reverse text-ink r=%+.3f' % (
        nm, len(sides), r_own, (np.sum(np.abs(no) >= abs(r_own)) + 1) / 5001, r_rev, (np.sum(np.abs(nr) >= abs(r_rev)) + 1) / 5001, r_ink))
open(os.path.join(DER, 'v13', 'c3d_log.txt'), 'w').write('\n'.join(log) + '\n')
