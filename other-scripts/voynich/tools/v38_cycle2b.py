"""v38 cycle 2b: the statistic chosen on the CONTROL. In Gerard only the network embeddings
(ResNet-18, DINO) carry the plant-text link; hand-crafted families dilute the omnibus. Here the
primary statistic becomes the neural composite (2 families x 5 text metrics), fixed from the
control, and applied to: Gerard (full, and subsampled to the Voynich page count), Voynich (all
herbal pages; Currier A hand 1 only), planted Voynich, shuffled Voynich.
Writes data/v38_ckpt/c2b.json and appends to loops/v38_cycle2.txt.
"""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *
from v38_cycle1 import voynich_setup
from v38_cycle2 import gerard_setup

NPERM = int(os.environ.get('NPERM', 2000))
FAMS = ['r18', 'dino']

if __name__ == '__main__':
    rng = np.random.default_rng(3802)
    rows, out = [], {}
    gk, gw, gvis, gconf = gerard_setup()
    GV = vis_sims(gvis, gk, FAMS); GT = text_sims(text_profiles(gw))
    gpart = Partial(list(gconf.values()), len(gk))
    g = mantel_table(GV, GT, gpart, NPERM, rng)
    out['gerard'] = g
    sub = []
    for rep in range(20):
        idx = np.sort(rng.choice(len(gk), 119, replace=False))
        rr = mantel_table({f: M[np.ix_(idx, idx)] for f, M in GV.items()}, {m: M[np.ix_(idx, idx)] for m, M in GT.items()},
                          Partial([C[np.ix_(idx, idx)] for C in gconf.values()], 119), 400, rng)
        sub.append((rr['omni_z'], rr['omni_p']))
    out['gerard_sub'] = sub
    rows.append(('V-38.11', 'NEURAL COMPOSITE fixed on the control (ResNet-18 + DINO x 5 text metrics, partial). Gerard all %d pages; Gerard subsampled to 119 pages (20 draws, 400 perms)' % len(gk),
                 'full: omni %+.4f z %+.1f p %.4f; at n=119: z median %.1f (range %.1f..%.1f), p<0.01 in %d/20, p<0.05 in %d/20' % (
                     g['omni'], g['omni_z'], g['omni_p'], np.median([z for z, _ in sub]), min(z for z, _ in sub), max(z for z, _ in sub),
                     sum(p < .01 for _, p in sub), sum(p < .05 for _, p in sub)),
                 'control passes at full size; power at Voynich size given'))
    pages, keys, words, vis, conf, strata = voynich_setup()
    n = len(keys)
    V = vis_sims(vis, keys, FAMS); T = text_sims(text_profiles(words))
    part = Partial(list(conf.values()), n)
    v = mantel_table(V, T, part, NPERM, rng, strata)
    out['voynich'] = v
    A1 = [i for i, s in enumerate(strata) if s == 'A1']
    va = mantel_table({f: M[np.ix_(A1, A1)] for f, M in V.items()}, {m: M[np.ix_(A1, A1)] for m, M in T.items()},
                      Partial([C[np.ix_(A1, A1)] for C in conf.values()], len(A1)), NPERM, rng)
    out['voynich_A1'] = va
    B = [i for i, s in enumerate(strata) if s.startswith('B')]
    vb = mantel_table({f: M[np.ix_(B, B)] for f, M in V.items()}, {m: M[np.ix_(B, B)] for m, M in T.items()},
                      Partial([C[np.ix_(B, B)] for C in conf.values()], len(B)), NPERM, rng, [strata[i] for i in B])
    out['voynich_B'] = vb
    cells = lambda r: ', '.join('%s %+.1f' % (k, z) for k, z in r['cell_z'].items())
    rows.append(('V-38.12', 'NEURAL COMPOSITE on Voynich (same confounds, within-stratum permutation): all %d herbal pages; Currier A hand 1 only (%d); Currier B only (%d)' % (n, len(A1), len(B)),
                 'all: omni %+.4f z %+.1f p %.4f [cells %s]; A1: z %+.1f p %.4f; B: z %+.1f p %.4f' % (
                     v['omni'], v['omni_z'], v['omni_p'], cells(v), va['omni_z'], va['omni_p'], vb['omni_z'], vb['omni_p']), ''))
    allw = Counter(w for ws in words for w in ws)
    pool = [w for w, c in allw.items() if 5 <= c <= 40]
    Z = np.array([vis[k]['dino'] for k in keys], float)
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pl = []
    for frac in [0.03, 0.06, 0.1]:
        zz = []
        for rep in range(3):
            Tp = text_sims(text_profiles(plant_text(words, U[:, :10] * S[:10], frac, rng, pool)))
            zz.append(mantel_table(V, Tp, part, 400, rng, strata)['omni_z'])
        pl.append((frac, zz))
    out['planted'] = pl
    sh = [mantel_table(V, text_sims(text_profiles([words[i] for i in rng.permutation(n)])), part, 300, rng, strata)['omni_z'] for _ in range(10)]
    out['shuffled'] = sh
    rows.append(('V-38.12c', 'Controls for the neural composite on Voynich: planted DINO-driven vocabulary (3/6/10%, 3 reps); page-shuffled text (10)',
                 'planted ' + '; '.join('%d%%: z %s' % (round(f * 100), '/'.join('%.1f' % z for z in zz)) for f, zz in pl) +
                 '; shuffled z ' + ' '.join('%+.1f' % z for z in sh), ''))
    json.dump(out, open(os.path.join(CK, 'c2b.json'), 'w'), default=str)
    with open(os.path.join(LOOPS, 'v38_cycle2.txt'), 'a') as fh:
        for r in rows:
            fh.write('| %s | %s | %s | %s |\n' % r)
    for r in rows:
        print('| %s | %s | %s | %s |' % r)
