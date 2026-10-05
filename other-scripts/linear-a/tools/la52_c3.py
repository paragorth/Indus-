#!/usr/bin/env python3
"""LA-52 cycle 3: held-out replication. Chains run separately on two disjoint halves of the
documents (LA and LB). If 'dies under every learner' marks something real about a word, the
frequency-matched anchoring z of a word in half A must predict its z in half B.
usage: la52_c3.py [TAG]"""
import sys, os, json, collections
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la52_common as C
import la52_report as R

TAG = sys.argv[1] if len(sys.argv) > 1 else 'c3'
R.TAG = TAG
KINDS = ['ngram', 'template', 'topic']


def half(base):
    rk = {}
    for k in KINDS:
        R.SUF = '_' + k
        r = R.retention(base, base, base + 'S', emin=3.0, boot=20)
        if r is not None:
            rk[k] = r
    ks = list(rk)
    r0 = rk[ks[0]]
    M = np.array([rk[k]['d']['auc'] for k in ks])
    best = np.nanmax(M, 0)
    feat = {f: best[j] for j, f in enumerate(r0['F']) if r0['ok'][j]}
    agg = {}
    for k in ks:
        rk[k]['sites'] = set(f[1] for f in r0['F'] if f[0] == 'SW')
        agg[k] = R.word_agg(rk[k], mind=3.0, boot=5)
    words = set.intersection(*[set(x for x, v in agg[k].items() if 'long' in v) for k in ks])
    rows = {x: (max(agg[k][x]['long'][0] for k in ks), agg[ks[0]][x]['long'][2]) for x in words}
    xs = sorted(rows)
    lx = np.log([rows[x][1] for x in xs]); bv = np.array([rows[x][0] for x in xs])
    z = {}
    for i, x in enumerate(xs):
        nb = np.argsort(np.abs(lx - lx[i]))[1:13]
        m = np.median(bv[nb]); s = 1.4826 * np.median(np.abs(bv[nb] - m)) + 1e-3
        z[x] = float((bv[i] - m) / s)
    fam = collections.defaultdict(list)
    for f, v in feat.items():
        fam[f[0]].append(v)
    return dict(feat=feat, z=z, rows=rows, fam={k: float(np.median(v)) for k, v in fam.items()},
                n={k: (rk[k]['n'], rk[k]['nS']) for k in ks})


def compare(a, b, name, nperm=5000, seed=7):
    A, B = half(a), half(b)
    print(f'== {name}: halves {a} {A["n"]} / {b} {B["n"]}')
    fams = sorted(set(A['fam']) & set(B['fam']), key=lambda k: A['fam'][k])
    print('   family medians (best-learner AUC) A | B: ' + ', '.join(f'{k} {A["fam"][k]:.2f}|{B["fam"][k]:.2f}' for k in fams))
    if len(fams) > 3:
        print(f'   family order A vs B Spearman {spearmanr([A["fam"][k] for k in fams], [B["fam"][k] for k in fams]).correlation:.2f}')
    sf = sorted(set(A['feat']) & set(B['feat']))
    if len(sf) > 5:
        x = np.array([A['feat'][f] for f in sf]); y = np.array([B['feat'][f] for f in sf])
        rho = spearmanr(x, y).correlation
        # within-family permutation null (family alone predicts retention)
        rng = np.random.default_rng(seed)
        fams_f = np.array([f[0] for f in sf])
        null = []
        for _ in range(2000):
            yp = y.copy()
            for k in set(fams_f):
                m = fams_f == k
                yp[m] = rng.permutation(y[m])
            null.append(spearmanr(x, yp).correlation)
        null = np.array(null)
        print(f'   shared features {len(sf)}: Spearman A vs B {rho:.2f}; within-family shuffle {null.mean():.2f} +- {null.std():.2f}, P {((null >= rho).sum() + 1) / (len(null) + 1):.4f}')
    sw = sorted(set(A['z']) & set(B['z']))
    out = {'name': name, 'n_words': len(sw)}
    if len(sw) > 5:
        x = np.array([A['z'][w] for w in sw]); y = np.array([B['z'][w] for w in sw])
        rho = spearmanr(x, y).correlation
        rng = np.random.default_rng(seed)
        null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(nperm)])
        p = ((null >= rho).sum() + 1) / (nperm + 1)
        print(f'   shared words {len(sw)}: Spearman of frequency-matched z A vs B {rho:.2f} (permutation P {p:.4f})')
        both = sorted(sw, key=lambda w: A['z'][w] + B['z'][w])
        print('   most anchored in both halves (sum of z):')
        for w in both[:12]:
            print(f'     {w:16s} zA {A["z"][w]:6.2f} zB {B["z"][w]:6.2f}  bestA {A["rows"][w][0]:.3f} bestB {B["rows"][w][0]:.3f}')
        print('   most persistent in both halves:')
        for w in both[-6:]:
            print(f'     {w:16s} zA {A["z"][w]:6.2f} zB {B["z"][w]:6.2f}')
        out.update(rho=float(rho), p=float(p), words={w: (A['z'][w], B['z'][w]) for w in sw})
    json.dump(out, open(os.path.join(C.CK, f'{TAG}_{name}_halves.json'), 'w'))
    return out


if __name__ == '__main__':
    for a, b, name in [('LBA', 'LBB', 'LB'), ('LAA', 'LAB', 'LA'), ('URA', 'URB', 'UR')]:
        try:
            compare(a, b, name)
        except (FileNotFoundError, KeyError, IndexError) as e:
            print(name, 'not ready', e)
