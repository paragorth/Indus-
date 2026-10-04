#!/usr/bin/env python3
"""LA-32 'the slips tell how they wrote': shared helpers.

Glyph rendering and shape similarity, sign maps (LA from lineara.xyz, LB from Unicode names),
word extraction for LA and LB, one-sign confusion extraction, degree-preserving partner null.
No sound values are used for Linear A; Linear B values only for the LB control and as an outside check.
"""
import os, sys, json, re, glob, unicodedata, subprocess, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
LA = os.path.join(HERE, '..')
DATA = os.path.join(LA, 'data')
CK = os.path.join(DATA, 'la32_ckpt')
LOOPS = os.path.join(LA, 'loops')
os.makedirs(CK, exist_ok=True)

# ---------------------------------------------------------------- Linear A corpus and sign map
def la_corpus():
    return json.load(open(os.path.join(DATA, 'corpus.json')))

def la_raw():
    p = os.path.join(CK, 'la_raw.json')
    if os.path.exists(p):
        return json.load(open(p))
    js = os.path.join(DATA, 'LinearAInscriptions.js')
    code = ("const fs=require('fs');eval(fs.readFileSync(%r,'utf8').replace('var inscriptions','global.inscriptions'));"
            "const o={};for(const [k,v] of inscriptions)o[k]=v;process.stdout.write(JSON.stringify(o));") % js
    raw = json.loads(subprocess.run(['node', '-e', code], capture_output=True, check=True).stdout)
    json.dump(raw, open(p, 'w'))
    return raw

def norm_la(s):
    return s.replace('₂', '2').replace('₃', '3')

def la_signmap():
    """transliteration sign -> Unicode char, by aligning words of equal sign count."""
    raw = la_raw()
    votes = collections.defaultdict(collections.Counter)
    for v in raw.values():
        for w, t in zip(v.get('words', []), v.get('transliteratedWords', [])):
            ch = [c for c in w if 0x10600 <= ord(c) <= 0x1077F and ord(c) != 0x1076B]
            comps = t.strip().split('-')
            if len(ch) == len(comps) and len(ch) >= 1:
                for c, s in zip(ch, comps):
                    votes[norm_la(s)][c] += 1
    return {s: cnt.most_common(1)[0][0] for s, cnt in votes.items()}

def la_words():
    """list of dicts: word tuple, doc, site, scribe, support."""
    out = []
    for d in la_corpus():
        for i, w in enumerate(d['words']):
            out.append(dict(w=tuple(norm_la(x) for x in w.split('-')), doc=d['id'], site=d['site'],
                            scribe=d.get('scribe') or '', support=d['support'], pos=i))
    return out

# ---------------------------------------------------------------- Linear B (DAMOS) words and sign map
LB_VAL = {}
for cp in range(0x10000, 0x1005E):
    try:
        n = unicodedata.name(chr(cp))
    except ValueError:
        continue
    if n.startswith('LINEAR B SYLLABLE'):
        LB_VAL[n.split()[-1].lower()] = chr(cp)
    elif n.startswith('LINEAR B SYMBOL'):
        LB_VAL['*' + str(int(n.split()[-1][1:]))] = chr(cp)

def lb_cv(s):
    """(consonant, vowel) of an LB value; pure vowels C=''; *-signs and odd ones None."""
    if s.startswith('*'):
        return None
    m = re.match(r'^([a-z]*?)([aeiou])(\d?)$', s)
    if not m:
        return None
    c, v, n = m.groups()
    if n and s not in ('a2', 'ra2', 'pu2', 'ro2', 'ta2', 'pa3', 'ra3', 'a3'):
        return None
    if n:  # a2 = ha, ra2 = rja/lja, pu2 = phu, ro2 = rjo, ta2 = tja, ra3 = rai, a3 = ai, pa3 = ?
        return dict(a2=('h', 'a'), ra2=('rj', 'a'), pu2=('ph', 'u'), ro2=('rj', 'o'), ta2=('tj', 'a'),
                    ra3=('r', 'ai'), a3=('', 'ai'), pa3=('p?', 'a')).get(s)
    return (c, v)

WRD = re.compile(r'^[a-z0-9*]+(?:-[a-z0-9*]+)+$')
def lb_words(sites=('KN', 'PY')):
    out = []
    for l in open(os.path.join(DATA, 'damos_items.jsonl')):
        d = json.loads(l)
        if not d.get('content'):
            continue
        h = d['heading']; site = h.split()[0]
        if sites and site not in sites:
            continue
        hand = re.findall(r'\(([^()]*)\)\s*$', h)
        hand = hand[0] if hand else ''
        series = h.split()[1] if len(h.split()) > 1 else ''
        txt = d['content']
        txt = re.sub(r'⟦[^⟧]*⟧', ' ', txt)
        for i, tok in enumerate(re.split(r'[\s,./|]+', txt)):
            if '̣' in tok:            # dotted (uncertain) letters
                continue
            if not WRD.match(tok):
                continue
            sg = tuple(tok.split('-'))
            if not all(s in LB_VAL for s in sg):
                continue
            out.append(dict(w=sg, doc=h, site=site, scribe=site + ':' + hand, series=site + series[:2], pos=i))
    return out

def lb_erasures():
    """⟦x⟧ y pairs: an erased word followed by a rewritten word."""
    out = []
    for l in open(os.path.join(DATA, 'damos_items.jsonl')):
        d = json.loads(l)
        if not d.get('content'):
            continue
        for m in re.finditer(r'⟦\s*([^⟧]*?)\s*⟧\s*([^\s,]+)', d['content']):
            out.append((d['heading'], m.group(1), m.group(2)))
    return out

# ---------------------------------------------------------------- glyph rendering and shape similarity
FONTS = dict(LA=os.path.join(CK, 'NotoSansLinearA-Regular.ttf'), LB=os.path.join(CK, 'NotoSansLinearB-Regular.ttf'))

def render(ch, font, size=96, box=64, scale_fit=True):
    f = ImageFont.truetype(font, size)
    im = Image.new('L', (size * 2, size * 2), 0)
    ImageDraw.Draw(im).text((size // 2, size // 2), ch, fill=255, font=f)
    a = np.array(im) > 100
    if not a.any():
        return None
    ys, xs = np.where(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = (box - 8) / max(h, w)
    im2 = Image.fromarray((a * 255).astype(np.uint8)).resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BILINEAR)
    b = np.zeros((box, box), np.float32)
    hh, ww = im2.size[1], im2.size[0]
    oy, ox = (box - hh) // 2, (box - ww) // 2
    b[oy:oy + hh, ox:ox + ww] = np.array(im2) / 255.0
    return b

def hog(b, cells=4, bins=8):
    gy, gx = np.gradient(ndimage.gaussian_filter(b, 1.0))
    mag = np.hypot(gx, gy); ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins
    n = b.shape[0] // cells; f = []
    for i in range(cells):
        for j in range(cells):
            m = mag[i * n:(i + 1) * n, j * n:(j + 1) * n].ravel(); a_ = ang[i * n:(i + 1) * n, j * n:(j + 1) * n].ravel()
            f.append(np.bincount(np.minimum(a_.astype(int), bins - 1), weights=m, minlength=bins))
    f = np.concatenate(f); return f / (np.linalg.norm(f) + 1e-9)

def shape_feats(imgs):
    F = {}
    for k, b in imgs.items():
        ink = b > 0.5
        dt = ndimage.distance_transform_edt(~ink)
        F[k] = dict(blur=ndimage.gaussian_filter(b, 3).ravel(), dt=dt, ink=ink, hog=hog(b),
                    prof=np.concatenate([b.mean(0), b.mean(1)]),
                    aspect=0.0)
    return F

def _corr(x, y):
    x = x - x.mean(); y = y - y.mean(); return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

def shape_matrices(signs, chars, font):
    """dict metric -> (n x n) similarity (higher = more alike)."""
    imgs = {}
    for s in signs:
        b = render(chars[s], font) if s in chars else None
        imgs[s] = b
    ok = [s for s in signs if imgs[s] is not None]
    F = shape_feats({s: imgs[s] for s in ok})
    n = len(signs); idx = {s: i for i, s in enumerate(signs)}
    M = {k: np.full((n, n), np.nan) for k in ('blur', 'chamfer', 'hog', 'prof')}
    for a in ok:
        for b in ok:
            i, j = idx[a], idx[b]
            if j < i:
                continue
            fa, fb = F[a], F[b]
            v = dict(blur=_corr(fa['blur'], fb['blur']),
                     chamfer=-0.5 * (fb['dt'][fa['ink']].mean() + fa['dt'][fb['ink']].mean()),
                     hog=float(fa['hog'] @ fb['hog']),
                     prof=_corr(fa['prof'], fb['prof']))
            for k in M:
                M[k][i, j] = M[k][j, i] = v[k]
    return M, imgs

def rank_norm(M):
    """off-diagonal percentile ranks in [0,1]; NaN kept."""
    R = np.full(M.shape, np.nan); iu = np.triu_indices(M.shape[0], 1)
    v = M[iu]; ok = ~np.isnan(v)
    r = np.empty(ok.sum()); r[np.argsort(v[ok], kind='stable')] = np.arange(ok.sum()) / max(1, ok.sum() - 1)
    vv = np.full(v.shape, np.nan); vv[ok] = r
    R[iu] = vv; R.T[iu] = vv
    return R

def combine(Ms, keys):
    return np.nanmean(np.stack([rank_norm(Ms[k]) for k in keys]), 0)

# ---------------------------------------------------------------- one-sign confusions
def one_sign_pairs(words, minlen=3, group='site', positions='all'):
    """Edges (a, b, w1, w2, pos, info) for word types differing in exactly one slot, both attested in the same group."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in words:
        if len(r['w']) >= minlen:
            by[r[group] if group else 'all'][r['w']].append(r)
    E = []
    for g, ws in by.items():
        keys = list(ws)
        idx = collections.defaultdict(list)
        for w in keys:
            for p in range(len(w)):
                idx[(len(w), p, w[:p] + ('_',) + w[p + 1:])].append(w)
        for (L, p, _), lst in idx.items():
            if len(lst) < 2:
                continue
            if positions == 'medial' and (p == 0 or p == L - 1):
                continue
            if positions == 'final' and p != L - 1:
                continue
            if positions == 'nonfinal' and p == L - 1:
                continue
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    w1, w2 = lst[i], lst[j]
                    n1, n2 = len(ws[w1]), len(ws[w2])
                    docs1 = {r['doc'] for r in ws[w1]}; docs2 = {r['doc'] for r in ws[w2]}
                    sc1 = {r['scribe'] for r in ws[w1]}; sc2 = {r['scribe'] for r in ws[w2]}
                    E.append(dict(a=w1[p], b=w2[p], w1=w1, w2=w2, pos=p, L=L, g=g, n1=n1, n2=n2,
                                  samedoc=bool(docs1 & docs2), samescribe=bool((sc1 & sc2) - {'', g + ':'}),
                                  rare=min(n1, n2) == 1 and max(n1, n2) >= 2))
    return E

# ---------------------------------------------------------------- scoring with a degree-preserving null
def score_edges(E, S, idx, nrep=4000, rng=None, weights=None):
    """mean similarity over edges vs a null that keeps each sign's confusion count (stub shuffle)."""
    rng = rng or np.random.default_rng(0)
    ed = [(idx[e['a']], idx[e['b']]) for e in E if e['a'] in idx and e['b'] in idx
          and not np.isnan(S[idx[e['a']], idx[e['b']]])]
    if len(ed) < 3:
        return dict(n=len(ed), obs=np.nan, null=np.nan, sd=np.nan, z=np.nan, p=np.nan)
    ed = np.array(ed)
    obs = float(np.nanmean(S[ed[:, 0], ed[:, 1]]))
    stubs = ed.ravel().copy(); m = len(ed)
    nul = np.empty(nrep)
    for r in range(nrep):
        for _ in range(20):
            rng.shuffle(stubs)
            a, b = stubs[:m], stubs[m:]
            if (a != b).all():
                break
        v = S[a, b]
        nul[r] = np.nanmean(np.where(a == b, np.nan, v))
    return dict(n=m, obs=obs, null=float(nul.mean()), sd=float(nul.std()), z=float((obs - nul.mean()) / (nul.std() + 1e-12)),
                p=float((1 + (nul >= obs).sum()) / (nrep + 1)))

def fmt(r):
    return f"n={r['n']} obs {r['obs']:.3f} null {r['null']:.3f} z {r['z']:.2f} P {r['p']:.4f}"

def la21_rows():
    """blind same-row co-assignment matrix from la21 (LA and LB runs), averaged over cycles."""
    out = {}
    for tag in ('LA', 'LB'):
        fs = [f for f in glob.glob(os.path.join(DATA, 'la21_ckpt', f'c*_{tag}.npz'))]
        acc = None; signs = None
        for f in fs:
            z = np.load(f)
            rows = z['rows']; signs = [str(s) for s in z['signs']]
            P = (rows[:, :, None] == rows[:, None, :]).mean(0)
            acc = P if acc is None else acc + P
        out[tag] = (signs, acc / len(fs), len(fs))
    return out
