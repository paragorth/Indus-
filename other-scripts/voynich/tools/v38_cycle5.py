"""v38 cycle 5: kill attempts on the neural link.
(a) FRESH networks never inspected (EfficientNet-B0, DINOv2), composite pre-registered.
(b) CONTENT-ONLY embeddings: DINO on the binary silhouette and on the grey plant (no palette).
(c) ADAPTIVE segmentation (faint drawings re-segmented).
(d) PRODUCTION DRIFT: do vellum/ink/paint/brightness similarities predict text? Re-run the
    neural composite with production similarity and text-style similarity (glyph unigram
    profile) added to the confounds.
(e) WHICH FEATURES: partial Mantel of each of the top 12 DINO principal components alone.
All on Voynich and on the Gerard control (where meaningful).
Writes data/v38_ckpt/c5.json and loops/v38_cycle5.txt.
"""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *
from v38_cycle1 import voynich_setup
from v38_cycle2 import gerard_setup

NPERM = int(os.environ.get('NPERM', 2000))


def load2(src, keys):
    v2 = json.load(open(os.path.join(DER, 'v38_vis2_%s.json' % src)))
    return {k: v2[k] for k in keys}


def emb_sim(vis, keys, f):
    F = np.array([vis[k][f] for k in keys], float)
    return cos_sim(F - F.mean(0))


def glyph_style(words):
    G = [Counter(g for w in ws for g in glyphs_of(w)) for ws in words]
    al = sorted(set().union(*G))
    M = np.array([[c[a] for a in al] for c in G], float)
    M = M / M.sum(1, keepdims=True)
    return cos_sim(M - M.mean(0))


def glyphs_of(w):
    for a, b in [('cth', 'T'), ('ckh', 'K'), ('cph', 'P'), ('cfh', 'F'), ('ch', 'C'), ('sh', 'S')]:
        w = w.replace(a, b)
    return list(w)


def line(tag, r):
    return '%s z %+.1f p %.4f' % (tag, r['omni_z'], r['omni_p'])


if __name__ == '__main__':
    rng = np.random.default_rng(3805)
    rows, out = [], {}
    pages, keys, words, vis, conf, strata = voynich_setup()
    n = len(keys)
    v2 = load2('voynich', keys)
    T = text_sims(text_profiles(words))
    part = Partial(list(conf.values()), n)
    gk, gw, gvis, gconf = gerard_setup()
    g2 = load2('gerard', gk)
    GT = text_sims(text_profiles(gw))
    gpart = Partial(list(gconf.values()), len(gk))
    res = {}
    for name, fams in [('fresh', ['effb0', 'dinov2']), ('sil', ['dino_sil']), ('gray', ['dino_gray'])]:
        rv = mantel_table({f: emb_sim(v2, keys, f) for f in fams}, T, part, NPERM, rng, strata)
        rg = mantel_table({f: emb_sim(g2, gk, f) for f in fams}, GT, gpart, NPERM, rng)
        res[name] = (rv, rg)
        out[name] = dict(voynich=rv, gerard=rg)
    rows.append(('V-38.16', 'FRESH-NETWORK replication, pre-registered: EfficientNet-B0 + DINOv2 composite x 5 text metrics, same confounds and permutations (%d)' % NPERM,
                 'Voynich %s (family z %s); Gerard %s' % (line('', res['fresh'][0]), ', '.join('%s %+.1f' % kv for kv in res['fresh'][0]['fam_z'].items()), line('', res['fresh'][1])),
                 'replicates' if res['fresh'][0]['omni_p'] < 0.01 else ('weak' if res['fresh'][0]['omni_p'] < 0.05 else 'does not replicate')))
    rows.append(('V-38.17', 'CONTENT-ONLY embeddings: DINO on the binary silhouette (form only), and on the grey plant image (no palette)',
                 'silhouette: Voynich %s, Gerard %s; grey: Voynich %s, Gerard %s' % (line('', res['sil'][0]), line('', res['sil'][1]), line('', res['gray'][0]), line('', res['gray'][1])), ''))
    rad = mantel_table({'dino_ad': emb_sim(v2, keys, 'dino_ad')}, T, part, NPERM, rng, strata)
    out['adaptive'] = rad
    rows.append(('V-38.18', 'ADAPTIVE segmentation (faint drawings re-segmented at a lower threshold), DINO', line('Voynich', rad), ''))
    # production drift
    P = np.array([v2[k]['prod'] for k in keys], float)
    Ps = feat_sim(P)
    rp = mantel_table({'prod': Ps}, T, part, NPERM, rng, strata)
    gst = glyph_style(words)
    part2 = Partial(list(conf.values()) + [Ps, gst], n)
    NV = {f: emb_sim(vis, keys, f) for f in ['r18', 'dino']}
    rk = mantel_table(NV, T, part2, NPERM, rng, strata)
    NV2 = {f: emb_sim(v2, keys, f) for f in ['effb0', 'dinov2']}
    rk2 = mantel_table(NV2, T, part2, NPERM, rng, strata)
    gP = feat_sim(np.array([g2[k]['prod'] for k in gk], float))
    grp = mantel_table({'prod': gP}, GT, gpart, NPERM, rng)
    out['prod'] = dict(voynich=rp, gerard=grp, neural_after=rk, fresh_after=rk2)
    rows.append(('V-38.19', 'PRODUCTION DRIFT: similarity of vellum, text-ink and paint colour and page brightness as a predictor of text similarity; then the neural composites with production similarity and glyph-unigram style similarity added to the confounds',
                 'production -> text: Voynich %s, Gerard %s; ResNet18+DINO after drift confounds: %s; EffB0+DINOv2 after: %s' % (line('', rp), line('', grp), line('', rk), line('', rk2)),
                 ''))
    # which components
    Z = np.array([vis[k]['dino'] for k in keys], float)
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pcs = {}
    for c in range(12):
        x = U[:, c]
        Sm = -np.abs(x[:, None] - x[None, :])
        r = mantel_table({'pc': Sm}, T, part, 1000, rng, strata)
        lo = [keys[i] for i in np.argsort(x)[:4]]; hi = [keys[i] for i in np.argsort(x)[-4:]]
        pcs[c] = dict(z=r['omni_z'], p=r['omni_p'], var=float(S[c] ** 2 / (S ** 2).sum()), lo=lo, hi=hi)
    out['pcs'] = pcs
    top = sorted(pcs.items(), key=lambda kv: -kv[1]['z'])[:3]
    rows.append(('V-38.20', 'WHICH FEATURES: each of the top 12 DINO principal components alone (|PC difference| as dissimilarity), partial Mantel (1000 perms; 12 tests)',
                 'z by PC: ' + ' '.join('%d:%+.1f' % (c, d['z']) for c, d in pcs.items()) + '; strongest: ' +
                 '; '.join('PC%d z %+.1f p %.3f (low end %s / high end %s)' % (c, d['z'], d['p'], ','.join(d['lo']), ','.join(d['hi'])) for c, d in top), ''))
    hc = {}
    for f in ['shape', 'colour', 'hog', 'edge']:
        F = np.array([vis[k][f] for k in keys], float)
        best = []
        for j in range(F.shape[1]):
            x = F[:, j]
            if x.std() == 0:
                continue
            r = mantel_table({'d': -np.abs(x[:, None] - x[None, :])}, {'tfidf': T['tfidf'], 'rare': T['rare']}, part, 200, rng, strata)
            best.append((r['omni_z'], j))
        best.sort(reverse=True)
        hc[f] = best[:3]
    out['handcrafted_dims'] = hc
    rows.append(('V-38.21', 'WHICH FEATURES, hand-crafted: every single descriptor dimension alone vs tf-idf + rare-word similarity (200 perms each; ~250 tests, so max z ~3 expected by chance)',
                 '; '.join('%s best dims %s' % (f, ', '.join('#%d z %+.1f' % (j, z) for z, j in b)) for f, b in hc.items()), ''))
    json.dump(out, open(os.path.join(CK, 'c5.json'), 'w'), default=str)
    with open(os.path.join(LOOPS, 'v38_cycle5.txt'), 'w') as fh:
        fh.write('# v38 cycle 5 - kill attempts on the neural link (fresh networks, content-only, segmentation, production drift, which features)\n')
        fh.write('| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            fh.write('| %s | %s | %s | %s |\n' % r)
    print(open(os.path.join(LOOPS, 'v38_cycle5.txt')).read())
