"""v38 cycle 7: two routes by which a picture could predict words without the words being
about the plant.
(a) LAYOUT: plant shape decides where text fits (short lines beside the stem, text above or
    below); line position changes Voynich word choice. Kill: text similarity from line-interior
    words only (first and last word of each line and the first line of each paragraph dropped),
    plus layout similarity (line-length histogram, line count, share of short lines) as a confound.
(b) LEAK: the plant crops may hold bits of handwriting. Probe: do embeddings of the page with the
    plant blanked (handwriting only) predict vocabulary? Then partial that similarity out of the
    plant link.
Neural composites (ResNet-18 + DINO; EfficientNet-B0 + DINOv2), same confounds, 2000 perms.
Writes data/v38_ckpt/c7.json and loops/v38_cycle7.txt.
"""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *
from v38_cycle1 import voynich_setup
from v38_lib import DER
from v8_lib import voynich_pages
import json as _j

NPERM = int(os.environ.get('NPERM', 2000))


def emb_sim(vis, keys, f):
    F = np.array([vis[k][f] for k in keys], float)
    return cos_sim(F - F.mean(0))


def interior_words(pid):
    recs = _j.load(open(os.path.join(os.path.dirname(DER), 'derived', 'ZL3b_lines.json')))
    out = {}
    for r in recs:
        if r['ltype'] not in ('P', 'C', 'R'):
            continue
        ws = [w for w in r['words'] if '?' not in w]
        if r['para_start'] or len(ws) < 3:
            continue
        out.setdefault(r['folio'], []).extend(ws[1:-1])
    return [out.get(p, []) for p in pid]


def layout(pages):
    L = []
    for p in pages:
        ll = np.array([len(l) for l in p['lines']])
        h, _ = np.histogram(ll, bins=[0, 3, 5, 7, 9, 11, 14, 40])
        L.append(np.concatenate([h / max(len(ll), 1), [np.log(len(ll)), ll.mean(), ll.std(), (ll < 5).mean()]]))
    return feat_sim(np.array(L))


if __name__ == '__main__':
    rng = np.random.default_rng(3807)
    pages, keys, words, vis, conf, strata = voynich_setup()
    n = len(keys)
    v2 = {k: v for k, v in json.load(open(os.path.join(DER, 'v38_vis2_voynich.json'))).items()}
    v3 = json.load(open(os.path.join(DER, 'v38_vis3_voynich.json')))
    NA = {f: emb_sim(vis, keys, f) for f in ['r18', 'dino']}
    NB = {f: emb_sim(v2, keys, f) for f in ['effb0', 'dinov2']}
    T = text_sims(text_profiles(words))
    iw = interior_words(keys)
    Ti = text_sims(text_profiles(iw))
    Ls = layout(pages)
    part = Partial(list(conf.values()), n)
    partL = Partial(list(conf.values()) + [Ls], n)
    rows, out = [], {}
    f = lambda r: 'z %+.1f p %.4f' % (r['omni_z'], r['omni_p'])
    rl = mantel_table({'layout': Ls}, T, part, NPERM, rng, strata)
    rlv = mantel_table({'layout': Ls}, {'d': NA['dino'], 'r': NA['r18']}, part, NPERM, rng, strata)
    a1 = mantel_table(NA, Ti, partL, NPERM, rng, strata)
    b1 = mantel_table(NB, Ti, partL, NPERM, rng, strata)
    out.update(layout_text=rl, layout_vis=rlv, interior_A=a1, interior_B=b1)
    rows.append(('V-38.26', 'LAYOUT route: layout similarity (line-length histogram, line count, short-line share) vs text and vs drawings; then neural composites on LINE-INTERIOR words only (line-first/last words and paragraph-first lines dropped; %d of %d tokens kept) with layout added to the confounds' % (sum(map(len, iw)), sum(map(len, words))),
                 'layout->text %s; layout->drawing %s; ResNet18+DINO interior+layout-partialled %s; EffB0+DINOv2 %s' % (f(rl), f(rlv), f(a1), f(b1)),
                 ''))
    LK = {f2: emb_sim(v3, keys, f2) for f2 in ['txt_r18', 'txt_dino']}
    rk = mantel_table(LK, T, part, NPERM, rng, strata)
    partK = Partial(list(conf.values()) + [Ls] + list(LK.values()), n)
    a2 = mantel_table(NA, T, partK, NPERM, rng, strata)
    b2 = mantel_table(NB, T, partK, NPERM, rng, strata)
    a3 = mantel_table(NA, Ti, partK, NPERM, rng, strata)
    out.update(leak_text=rk, after_leak_A=a2, after_leak_B=b2, after_all_interior=a3)
    rows.append(('V-38.27', 'LEAK route: embeddings of the page with the plant blanked (handwriting only) vs text; then plant composites with handwriting-image similarity and layout added to the confounds',
                 'handwriting-image -> text %s (family z %s); ResNet18+DINO after %s; EffB0+DINOv2 after %s; ResNet18+DINO, interior words, after everything %s' % (
                     f(rk), ', '.join('%s %+.1f' % kv for kv in rk['fam_z'].items()), f(a2), f(b2), f(a3)), ''))
    json.dump(out, open(os.path.join(CK, 'c7.json'), 'w'), default=str)
    with open(os.path.join(LOOPS, 'v38_cycle7.txt'), 'w') as fh:
        fh.write('# v38 cycle 7 - layout and handwriting-leak routes\n| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for x in rows:
            fh.write('| %s | %s | %s | %s |\n' % x)
    print(open(os.path.join(LOOPS, 'v38_cycle7.txt')).read())
