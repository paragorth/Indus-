"""S-DARK-28 cycle 1: graphic-derivation graph of the Wells sign list.

For every sign D rendered from the lipi font (U+E000 + Wells number, as in S164) we ask
whether D = base + enclosure (box / oval / other ring), base + roof or baseline stroke,
base + 1-4 short added strokes, base doubled or mirrored, or base + another whole sign
(ligature).  Parts are found with connected components and hole filling, normalised
(bbox crop, square pad, 32x32) and matched to every sign by the symmetric dilated
overlap of tools/sign_similarity.py.

Controls: planted derivations (enclosure, roof, strokes, doubling, ligature built by
hand from random glyphs) must be found with the right base and type; the score of a
part against WRONG bases gives the false-positive rate and sets the threshold.
Outputs: data/derived/dark/loop28_edges.json, loop28_cycle1.txt
"""
import sys, json, random, collections, time
import numpy as np
sys.path.insert(0, 'tools')
from sign_glyphs import render
from scipy import ndimage as ndi
from PIL import Image, ImageDraw

random.seed(28); np.random.seed(28)
T0 = time.time()
OUT = 'data/derived/dark/'
SIGNS = json.load(open('data/derived/glyph_sim_signs.json'))
SIM_FINE = np.load('data/derived/glyph_sim_fine.npy')
IDX = {w: i for i, w in enumerate(SIGNS)}
NUMERAL = set(range(1, 8)) | set(range(12, 21)) | set(range(25, 30)) | set(range(31, 40))
R = 64   # render size

def norm(mask, out=32):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0: return None
    a = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape; s = max(h, w)
    pad = np.zeros((s, s), bool); pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return np.array(Image.fromarray(pad.astype(np.uint8) * 255).resize((out, out), Image.BILINEAR)) > 60

def nvec(m):
    return m.ravel().astype(np.float32), ndi.binary_dilation(m, iterations=1).ravel().astype(np.float32)

G = {}; NV = {}; ND = {}
for w in SIGNS:
    g = render(w, size=192, out=R)
    g = ndi.binary_opening(g, iterations=1) | g  # keep as is; opening only used to drop specks later
    G[w] = g
    n = norm(g); NV[w], ND[w] = nvec(n)
NVM = np.array([NV[w] for w in SIGNS]); NDM = np.array([ND[w] for w in SIGNS]); NN = NVM.sum(1)

def match(part):
    """symmetric dilated overlap of a normalised part with every sign -> (best sign, score, scores)"""
    n = norm(part)
    if n is None or n.sum() < 8: return None, 0.0, None
    v, d = nvec(n)
    inter = NDM @ v + NVM @ d
    s = inter / (NN + v.sum())
    return s

def components(mask, minpix=4):
    lab, k = ndi.label(mask, structure=np.ones((3, 3)))
    comps = []
    for i in range(1, k + 1):
        c = lab == i
        if c.sum() >= minpix: comps.append(c)
    return comps

def bbox(m):
    ys, xs = np.nonzero(m); return ys.min(), ys.max(), xs.min(), xs.max()

def elongation(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 3: return 1.0
    cov = np.cov(np.vstack([xs, ys]))
    ev = np.linalg.eigvalsh(cov)
    return float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))

def gaps(mask, axis):
    """indices of empty columns (axis=0) / rows (axis=1) strictly inside the bbox"""
    y0, y1, x0, x1 = bbox(mask)
    if axis == 0:
        prof = mask[y0:y1 + 1, :].sum(0); lo, hi = x0, x1
    else:
        prof = mask[:, x0:x1 + 1].sum(1); lo, hi = y0, y1
    return [i for i in range(lo + 1, hi) if prof[i] == 0], prof, lo, hi

def split_at(mask, axis, i):
    a = mask.copy(); b = mask.copy()
    if axis == 0: a[:, i:] = False; b[:, :i] = False
    else: a[i:, :] = False; b[:i, :] = False
    return a, b

def analyse(D, self_w=None):
    """return list of candidate derivations: dict(type, base_part, extra info)"""
    cands = []
    ink = D.sum(); H, W = D.shape
    y0, y1, x0, x1 = bbox(D); bh, bw = y1 - y0 + 1, x1 - x0 + 1
    comps = components(D)
    # --- enclosure: a component whose filled interior holds other ink
    for c in comps:
        f = ndi.binary_fill_holes(c); hole = f & ~c
        cy0, cy1, cx0, cx1 = bbox(c); carea = (cy1 - cy0 + 1) * (cx1 - cx0 + 1)
        if hole.sum() < 0.25 * carea: continue
        inner = D & hole & ~c
        if inner.sum() < 12: continue
        fillratio = f.sum() / carea
        shape = 'box' if fillratio > 0.9 else ('oval' if fillratio > 0.7 else 'ring')
        cands.append(dict(type='enclosure', sub=shape, part=inner, extra=c, ratio=inner.sum() / ink))
    # --- roof / baseline: a thin wide component at the very top or bottom
    if len(comps) >= 2:
        for c in comps:
            cy0, cy1, cx0, cx1 = bbox(c)
            if (cx1 - cx0 + 1) >= 0.7 * bw and (cy1 - cy0 + 1) <= 0.35 * bh:
                rest = D & ~c
                if rest.sum() < 0.4 * ink: continue
                ry0, ry1, _, _ = bbox(rest)
                if cy1 < ry0 + 0.1 * bh: cands.append(dict(type='roof', sub='roof', part=rest, extra=c, ratio=rest.sum() / ink))
                elif cy0 > ry1 - 0.1 * bh: cands.append(dict(type='roof', sub='baseline', part=rest, extra=c, ratio=rest.sum() / ink))
    # --- added strokes: 1-4 small components, remainder is the base
    small = [c for c in comps if c.sum() <= 0.18 * ink and (elongation(c) >= 2.2 or c.sum() <= 0.06 * ink)]
    if 1 <= len(small) <= 4 and len(small) < len(comps):
        strokes = np.zeros_like(D)
        for c in small: strokes |= c
        rest = D & ~strokes
        if rest.sum() >= 0.5 * ink:
            ry0, ry1, rx0, rx1 = bbox(rest); pos = []
            for c in small:
                cy, cx = ndi.center_of_mass(c)
                if cy < ry0: pos.append('above')
                elif cy > ry1: pos.append('below')
                elif cx < rx0: pos.append('left')
                elif cx > rx1: pos.append('right')
                else: pos.append('inside')
            cands.append(dict(type='strokes', sub=f'{len(small)}{collections.Counter(pos).most_common(1)[0][0]}', part=rest, extra=strokes, ratio=rest.sum() / ink, k=len(small), pos=pos))
    # --- doubling / ligature: split at an empty column or row (or the sparsest interior line)
    for axis in (0, 1):
        gp, prof, lo, hi = gaps(D, axis)
        cuts = []
        if gp:
            # group consecutive gap indices; take the middle of each gap run
            runs = []; cur = [gp[0]]
            for i in gp[1:]:
                if i == cur[-1] + 1: cur.append(i)
                else: runs.append(cur); cur = [i]
            runs.append(cur)
            cuts = [(r[len(r) // 2], True) for r in runs]
        else:
            seg = prof[lo + int(0.3 * (hi - lo)):hi - int(0.3 * (hi - lo))]
            if len(seg) and seg.min() <= 2:
                cuts = [(lo + int(0.3 * (hi - lo)) + int(np.argmin(seg)), False)]
        for i, clean in cuts:
            a, b = split_at(D, axis, i)
            if min(a.sum(), b.sum()) < 0.15 * ink: continue
            cands.append(dict(type='split', sub=('col' if axis == 0 else 'row') + ('' if clean else '~'), part=a, part2=b, ratio=a.sum() / ink, clean=clean))
    return cands

def part_sim(p, q):
    a = norm(p); b = norm(q)
    if a is None or b is None: return 0.0
    va, da = nvec(a); vb, db = nvec(b)
    return float((va @ db + vb @ da) / (va.sum() + vb.sum()))

def edges_for(D, self_w, thr, want_self=None):
    """evaluate candidates of D; return list of edges (base, type, sub, score, info)"""
    out = []
    for c in analyse(D):
        if c['type'] == 'split':
            s1 = match(c['part']); s2 = match(c['part2'])
            if s1 is None or s2 is None: continue
            # doubling: both halves the same sign; mirrored: half2 ~ flipped half1
            dsim = part_sim(c['part'], c['part2'])
            msim = part_sim(c['part'], c['part2'][:, ::-1] if 'col' in c['sub'] else c['part2'][::-1, :])
            b1 = int(np.argmax(s1)); b2 = int(np.argmax(s2))
            if dsim >= thr or msim >= thr:
                kind = 'doubled' if dsim >= msim else 'mirrored'
                base = SIGNS[b1] if s1[b1] >= s2[b2] else SIGNS[b2]
                sc = max(s1[b1], s2[b2])
                if base != self_w and sc >= thr:
                    out.append(dict(base=base, type='doubling', sub=kind + ('-' + c['sub']), score=float(sc), pair=float(max(dsim, msim))))
                continue
            w1, w2 = SIGNS[b1], SIGNS[b2]
            if s1[b1] >= thr and s2[b2] >= thr and self_w not in (w1, w2):
                if w1 in NUMERAL or w2 in NUMERAL:
                    base, num = (w2, w1) if w1 in NUMERAL else (w1, w2)
                    out.append(dict(base=base, type='strokes', sub='numeral-' + c['sub'], score=float(min(s1[b1], s2[b2])), partner=num))
                else:
                    # the ligature edge goes from each part to D; report the bigger part as base
                    big, sm = (w1, w2) if c['part'].sum() >= c['part2'].sum() else (w2, w1)
                    out.append(dict(base=big, type='ligature', sub=c['sub'], score=float(min(s1[b1], s2[b2])), partner=sm))
        else:
            s = match(c['part'])
            if s is None: continue
            order = np.argsort(-s)
            for bi in order[:3]:
                b = SIGNS[bi]
                if b == self_w: continue
                if s[bi] >= thr:
                    out.append(dict(base=b, type=c['type'], sub=c['sub'], score=float(s[bi]), ratio=float(c['ratio']), **({'k': c['k'], 'pos': c['pos']} if 'k' in c else {})))
                break
    return out

# ---------------- planted controls ----------------
def planted(w, kind, partner=None):
    g = G[w]; H = R
    canvas = np.zeros((H * 3, H * 3), bool)
    if kind in ('box', 'oval'):
        im = Image.new("L", (H * 3, H * 3), 0); d = ImageDraw.Draw(im)
        sm = np.array(Image.fromarray(g.astype(np.uint8) * 255).resize((int(H * 0.9), int(H * 0.9)))) > 100
        y0, y1, x0, x1 = bbox(sm); ph, pw = y1 - y0 + 1, x1 - x0 + 1
        oy, ox = H - ph // 2, H - pw // 2
        canvas[oy:oy + ph, ox:ox + pw] = sm[y0:y1 + 1, x0:x1 + 1]
        m = 8
        if kind == 'box': d.rectangle([ox - m, oy - m, ox + pw + m, oy + ph + m], outline=255, width=3)
        else: d.ellipse([ox - m - 4, oy - m - 4, ox + pw + m + 4, oy + ph + m + 4], outline=255, width=3)
        canvas |= np.array(im) > 100
    elif kind == 'roof':
        y0, y1, x0, x1 = bbox(g); pw = x1 - x0 + 1
        canvas[20:20 + H, H // 2:H // 2 + H] = g
        canvas[20 + y0 - 10:20 + y0 - 7, H // 2 + x0 - 3:H // 2 + x1 + 4] = True
    elif kind == 'strokes':
        y0, y1, x0, x1 = bbox(g); pw = x1 - x0 + 1
        canvas[20:20 + H, H // 2:H // 2 + H] = g
        for j in range(2):
            cx = H // 2 + x0 + pw // 3 * (j + 1) - 2
            canvas[20 + y0 - 12:20 + y0 - 3, cx:cx + 3] = True
    elif kind == 'doubled':
        y0, y1, x0, x1 = bbox(g); pw = x1 - x0 + 1
        canvas[10:10 + H, 4:4 + H] = g
        canvas[10:10 + H, 4 + pw + 8:4 + pw + 8 + H] |= g
    elif kind == 'mirrored':
        y0, y1, x0, x1 = bbox(g); pw = x1 - x0 + 1
        canvas[10:10 + H, 4:4 + H] = g
        canvas[10:10 + H, 4 + pw + 8:4 + pw + 8 + H] |= g[:, ::-1]
    elif kind == 'ligature':
        g2 = G[partner]
        y0, y1, x0, x1 = bbox(g); pw = x1 - x0 + 1
        canvas[10:10 + H, 4:4 + H] = g
        canvas[10:10 + H, 4 + pw + 8:4 + pw + 8 + H] |= g2
    # renormalise like a real glyph
    n = norm(canvas, out=R)
    return n

KINDMAP = {'box': 'enclosure', 'oval': 'enclosure', 'roof': 'roof', 'strokes': 'strokes', 'doubled': 'doubling', 'mirrored': 'doubling', 'ligature': 'ligature'}
NONNUM = [w for w in SIGNS if w not in NUMERAL and G[w].sum() > 60]
rep = []
rep.append(f'# LOOP 28 cycle 1: graphic-derivation graph from the lipi font ({time.strftime("%Y-%m-%dT%H:%M")})')
rep.append(f'signs rendered {len(SIGNS)}; non-numeral signs with ink {len(NONNUM)}; render {R}x{R}, parts normalised to 32x32, similarity = symmetric dilated overlap')

# score distributions: planted true base vs wrong bases (random pairs) -> threshold
true_scores = []; wrong_scores = []
plant = []
for kind in KINDMAP:
    for w in random.sample(NONNUM, 60):
        partner = random.choice([x for x in NONNUM if x != w]) if kind == 'ligature' else None
        D = planted(w, kind, partner)
        plant.append((w, kind, partner, D))
# raw score of the correct part against the true base and against random bases (threshold-free)
for w, kind, partner, D in plant:
    for c in analyse(D):
        if c['type'] == 'split':
            for p in (c['part'], c['part2']):
                s = match(p)
                if s is None: continue
                true_scores.append(float(s[IDX[w]])) if kind in ('doubled', 'mirrored', 'ligature') else None
                wrong_scores.extend(float(s[IDX[x]]) for x in random.sample(NONNUM, 20) if x != w)
        else:
            s = match(c['part'])
            if s is None: continue
            if KINDMAP[kind] == c['type']: true_scores.append(float(s[IDX[w]]))
            wrong_scores.extend(float(s[IDX[x]]) for x in random.sample(NONNUM, 20) if x != w)
wrong_scores = np.array(wrong_scores); true_scores = np.array(true_scores)
THR = float(np.quantile(wrong_scores, 0.995))
rep.append(f'score of planted part vs TRUE base: median {np.median(true_scores):.3f}, 10th pct {np.quantile(true_scores, .1):.3f} (n={len(true_scores)})')
rep.append(f'score of planted part vs WRONG bases (random pairs): median {np.median(wrong_scores):.3f}, 99th {np.quantile(wrong_scores, .99):.3f}, 99.5th {THR:.3f} (n={len(wrong_scores)})')
rep.append(f'threshold = 99.5th percentile of wrong-base scores = {THR:.3f}')

# recall with the threshold
rec = collections.Counter(); tot = collections.Counter(); wrongbase = collections.Counter()
for w, kind, partner, D in plant:
    tot[kind] += 1
    ed = edges_for(D, None, THR)
    ok = [e for e in ed if e['type'] == KINDMAP[kind] and (e['base'] == w or e.get('partner') == w)]
    if ok: rec[kind] += 1
    elif any(e['type'] == KINDMAP[kind] for e in ed): wrongbase[kind] += 1
rep.append('planted-derivation recall (correct type AND correct base) at this threshold:')
for kind in KINDMAP:
    rep.append(f'  {kind:9s} {rec[kind]:3d}/{tot[kind]} found; {wrongbase[kind]} with right type but wrong base')
# false positives: unmodified random glyph pairs: run the detector on real glyphs but score only edges to a random *other* base
# (the detector on real glyphs yields the real graph; here we count how often a planted part's best WRONG base beats THR)
fp = (wrong_scores >= THR).mean()
rep.append(f'false-positive rate of a wrong base passing the threshold: {fp*100:.2f}% per (part, wrong base) comparison')

# ---------------- real graph ----------------
EDGES = []
for w in SIGNS:
    if w in NUMERAL: continue
    D = G[w]
    for e in edges_for(D, w, THR):
        e['derived'] = w
        e['fine'] = float(SIM_FINE[IDX[w], IDX[e['base']]])
        EDGES.append(e)
# de-duplicate (derived, base, type) keeping the best score
best = {}
for e in EDGES:
    k = (e['derived'], e['base'], e['type'])
    if k not in best or e['score'] > best[k]['score']: best[k] = e
EDGES = sorted(best.values(), key=lambda e: (-e['score']))
# confidence: score relative to threshold band
for e in EDGES:
    e['conf'] = 'high' if e['score'] >= THR + 0.1 else ('mid' if e['score'] >= THR + 0.04 else 'low')
    if e['type'] in ('strokes', 'ligature') and e.get('sub', '').endswith('~'): e['conf'] = 'low'
by_type = collections.Counter(e['type'] for e in EDGES)
rep.append('')
rep.append(f'## derivation edges found among real signs: {len(EDGES)} (derived -> base), by type {dict(by_type)}')
rep.append(f'   confidence: {dict(collections.Counter(e["conf"] for e in EDGES))}')
# checks: Wells-number proximity and glyph_sim_fine vs random pairs
dist = np.array([abs(e['derived'] - e['base']) for e in EDGES])
rnd = np.array([abs(a - b) for a, b in zip(random.choices(SIGNS, k=5000), random.choices(SIGNS, k=5000))])
fine = np.array([e['fine'] for e in EDGES]); rfine = np.array([SIM_FINE[IDX[a], IDX[b]] for a, b in zip(random.choices(SIGNS, k=5000), random.choices(SIGNS, k=5000)) if a != b])
rep.append(f'   check 1 (Wells shape-family numbering): |W_derived - W_base| <= 20 in {np.mean(dist <= 20)*100:.0f}% of edges vs {np.mean(rnd <= 20)*100:.0f}% of random pairs; median distance {np.median(dist):.0f} vs {np.median(rnd):.0f}')
rep.append(f'   check 2 (glyph_sim_fine, independent whole-glyph measure): mean {fine.mean():.3f} for edges vs {rfine.mean():.3f} random pairs; edges above the random 95th pct ({np.quantile(rfine,.95):.3f}): {np.mean(fine > np.quantile(rfine,.95))*100:.0f}%')
for t in by_type:
    sub = [e for e in EDGES if e['type'] == t]
    rep.append(f'   {t}: n={len(sub)}, fine mean {np.mean([e["fine"] for e in sub]):.3f}, within-20 {np.mean([abs(e["derived"]-e["base"])<=20 for e in sub])*100:.0f}%')
rep.append('')
rep.append('## edge table (derived <- base, type/sub, score, conf, glyph_sim_fine, partner)')
for e in EDGES:
    rep.append(f"  W{e['derived']:<4d} <- W{e['base']:<4d} {e['type']:9s} {e.get('sub',''):14s} score {e['score']:.3f} {e['conf']:4s} fine {e['fine']:.2f}" + (f" partner W{e['partner']}" if 'partner' in e else '') + (f" pos {e['pos']}" if 'pos' in e else ''))
json.dump(EDGES, open(OUT + 'loop28_edges.json', 'w'), indent=0)
# render a sheet of the top edges for eyeballing
top = EDGES[:120]
cols = 6; rows = (len(top) + cols - 1) // cols
sheet = Image.new('L', (cols * 150, rows * 80), 255); d = ImageDraw.Draw(sheet)
for i, e in enumerate(top):
    for j, w in enumerate((e['derived'], e['base'])):
        im = Image.fromarray((~G[w]).astype(np.uint8) * 255)
        sheet.paste(im, ((i % cols) * 150 + j * 70, (i // cols) * 80))
    d.text(((i % cols) * 150, (i // cols) * 80 + 66), f"{e['derived']}<-{e['base']} {e['type'][:4]} {e['score']:.2f}", fill=0)
sheet.save(OUT + 'loop28_edges_sheet.png')
rep.append('')
rep.append(f'(sheet of the top 120 edges: {OUT}loop28_edges_sheet.png; elapsed {time.time()-T0:.0f}s)')
open(OUT + 'loop28_cycle1.txt', 'w').write('\n'.join(rep) + '\n')
print('\n'.join(rep[:40]))
