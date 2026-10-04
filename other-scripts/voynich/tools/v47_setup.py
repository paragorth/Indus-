"""v47: data set-up for the Voynich herbal (v38 pages + v47 embeddings), the pharmaceutical
register units, and the two real-herbal controls (Gerard 1636, Dodoens 1583)."""
import os, sys, json
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v47_lib import *

NETS4 = ['r18', 'dino', 'effb0', 'dinov2']      # the v38 composite (four networks)
NETS6 = NETS4 + ['clip', 'mae']


def eqm(a):
    n = len(a)
    return np.array([[float(a[i] == a[j] and a[i] is not None) for j in range(n)] for i in range(n)])


def vsetup(sub=None):
    """Herbal pages. sub: None (all 119) or 'A1' (Currier A, hand 1)."""
    from v38_cycle1 import voynich_setup
    pages, keys, words, vis, conf, strata = voynich_setup()
    v47 = json.load(open(os.path.join(CK47, 'vis_voynich.json')))
    v2 = json.load(open(os.path.join(DER, 'v38_vis2_voynich.json')))
    idx = [i for i, k in enumerate(keys) if k in v47 and (sub is None or strata[i] == sub)]
    pages = [pages[i] for i in idx]; keys = [keys[i] for i in idx]; words = [words[i] for i in idx]
    strata = [strata[i] for i in idx]
    conf = {k: C[np.ix_(idx, idx)] for k, C in conf.items()}
    emb = {k: v47[k] for k in keys}
    for k in keys:
        emb[k]['prod'] = v2[k]['prod']
    return dict(pages=pages, keys=keys, words=words, emb=emb, conf=conf, strata=strata,
                quire=[p['quire'] for p in pages])


def psetup():
    """Pharmaceutical register units (all Currier A, hand 1)."""
    v = json.load(open(os.path.join(CK47, 'vis_pharma.json')))
    keys = sorted(v, key=lambda k: (int(re.findall(r'\d+', k)[0]), k))
    words = [v[k]['words'] for k in keys]
    fol = [v[k]['folio'] for k in keys]
    n = len(keys)
    para = np.array([v[k]['para'] for k in keys])
    same_f = eqm(fol)
    adjp = same_f * (np.abs(para[:, None] - para[None, :]) == 1)
    leaf = [int(re.findall(r'\d+', f)[0]) for f in fol]
    L = np.array(leaf, float)
    ln = np.log([len(w) for w in words]); ar = np.log([v[k]['area'] + 1e-4 for k in keys])
    y = np.array([v[k]['y'] for k in keys])
    conf = dict(folio=same_f, adjp=adjp, leaf=eqm(leaf), logdist=np.log1p(np.abs(L[:, None] - L[None, :])),
                lensum=ln[:, None] + ln[None, :], lendiff=np.abs(ln[:, None] - ln[None, :]),
                areasum=ar[:, None] + ar[None, :], areadiff=np.abs(ar[:, None] - ar[None, :]),
                ydiff=np.abs(y[:, None] - y[None, :]))
    return dict(keys=keys, words=words, emb=v, conf=conf, strata=None, folio=fol,
                quire=['P%d' % (l // 10) for l in leaf])


def hsetup(src, adjw):
    """Real-herbal control: pages with a woodcut (area >= 0.06) and >= 30 OCR words."""
    v = json.load(open(os.path.join(CK47, 'vis_%s.json' % src)))
    keys = sorted([k for k in v if not v[k].get('skip') and 'r18' in v[k]], key=int)
    words = [v[k]['words'] for k in keys]
    order = np.array([int(k) for k in keys], float)
    D = np.abs(order[:, None] - order[None, :])
    ln = np.log([len(w) for w in words]); ar = np.log([v[k]['area'] for k in keys])
    conf = dict(adj=(D <= adjw).astype(float), logdist=np.log1p(D),
                lensum=ln[:, None] + ln[None, :], lendiff=np.abs(ln[:, None] - ln[None, :]),
                areasum=ar[:, None] + ar[None, :], areadiff=np.abs(ar[:, None] - ar[None, :]))
    # pseudo-quires: consecutive blocks of 8 scans' worth of pages, for held-out tests
    nb = 8
    q = ['Q%d' % int(i * nb / len(keys)) for i in range(len(keys))]
    return dict(keys=keys, words=words, emb=v, conf=conf, strata=None, quire=q)


def sims(S, nets):
    return {f: emb_sim_m([S['emb'][k][f] for k in S['keys']]) for f in nets}


def tsims(words):
    return text_sims(text_profiles(words))
