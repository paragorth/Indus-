"""S-DARK-50 cycle 1: two pictoriality measures per sign, built blind to slot.

(a) SHAPE proxy from the lipi font glyphs (same rendering as S164 / S-DARK-28):
    holes      = enclosed regions (filled - mask components)
    comps      = connected ink components
    strokes    = skeleton segments (Zhang-Suen thinning, junction pixels removed)
    curv       = orientation spread of the outline: 1 - share of outline-gradient mass in the
                 four dominant 10-degree bins (straight-line glyphs concentrate, curves spread)
    asym       = 1 - max(mirror IoU left/right, up/down)
    perim      = perimetric complexity P^2 / 4 pi A (data/derived/sign-complexity.json, else computed)
    far_stroke = 1 - max glyph_sim_fine to the pure stroke-numeral signs W1-W39
    PICT_SHAPE = mean z-score of (holes, curv, asym, perim, far_stroke, log strokes)
(b) LABEL proxy from the shape descriptions in data/derived/sign-dossiers-top200.json,
    keyword rule fixed in this file before any slot was looked at:
    1 = object/creature/tool word present, 0 = only stroke/geometry words, NaN = neither.
Also: a 3-way label (pictorial / geometric / unknown) and the agreement of (a) with (b):
AUC of PICT_SHAPE for label 1 vs 0, Spearman with frequency, and per-component AUCs.
Output: data/derived/dark/loop50_signs.json (per-sign table), loop50_cycle1.txt
"""
import sys, json, re, math, collections, time
import numpy as np
sys.path.insert(0, 'tools')
from sign_glyphs import render
from scipy import ndimage as ndi

OUT = 'data/derived/dark/'
T0 = time.time()
SIGNS = json.load(open('data/derived/glyph_sim_signs.json'))
SIM = np.load('data/derived/glyph_sim_fine.npy')
IDX = {w: i for i, w in enumerate(SIGNS)}
CPLX = {int(k): v for k, v in json.load(open('data/derived/sign-complexity.json')).items()}
DOSS = json.load(open('data/derived/sign-dossiers-top200.json'))
STROKE_NUMERALS = set(range(1, 8)) | set(range(12, 21)) | set(range(25, 30)) | set(range(31, 40))

# ---------------------------------------------------------------- label rule (written before slots were looked at)
PICT_WORDS = r"\b(fish|jar|man|men|person|people|figure|human|arrow|spear|tree|leaf|leaves|bird|animal|crab|scorpion|" \
             r"wheel|comb|pot|vessel|bull|ox|horn|head|heads|arm|arms|leg|legs|body|hand|hands|foot|feet|eye|face|" \
             r"pitchfork|bow|knife|blade|axe|trident|insect|flower|plant|branch|branches|seated|sitting|standing|" \
             r"carrying|carrier|yoke|boat|house|hut|ladder|cup|bowl|rake|brush|hair|tail|whisker|whiskers|wing|wings|" \
             r"feather|beetle|bee|pincer|pincers|tines|hat|cheeks|knees|lid|handle|handles|basket|cage|net|antenna|" \
             r"tongs|hook|flag|banner|drum|sun|star|moon|shield|mace|club|sickle|plough|plow|goat|deer|tiger|snake|" \
             r"serpent|frog|turtle|lizard|bud|seed|fruit|grain|ear|mouth|tooth|teeth|nose|beak|claw|paw|hoof|skull|" \
             r"spine|rib|ribs|bone|heart|lamp|torch|fire|water|wave|waves|river|mountain|hill|field|road|door|gate|" \
             r"window|roof|pillar|column|table|chair|stool|bed|cart|chariot|wheelbarrow|loom|spindle|spool|needle|" \
             r"thread|rope|knot|belt|garment|dress|skirt|crown|helmet|headdress|mask|bell|pipe|flute|harp|lyre|" \
             r"drum|anchor|oar|sail|paddle|fork|spoon|ladle|pan|cauldron|kettle|bottle|flask|jug|pitcher|urn|bin|" \
             r"sack|bag|box)\b"
GEOM_WORDS = r"\b(stroke|strokes|line|lines|bar|bars|vertical|horizontal|cross|circle|circles|oval|ovals|diamond|" \
             r"lozenge|triangle|triangles|square|squares|rectangle|rectangles|chevron|chevrons|hatch|hatched|hatching|" \
             r"grid|lattice|arc|arcs|curve|curves|dot|dots|zigzag|spiral|loop|loops|angle|angles|bracket|brackets|" \
             r"parenthesis|crescent|slash|slant|slanting|slanted|tick|ticks|notch|notches|u|v|x|y|t|w|m|n|shape|" \
             r"shaped|motif|symbol|sign|ligature|mirror|mirrored|doubled|double|rotated|inverted|half|full|height|" \
             r"wide|narrow|parallel|perpendicular|diagonal|concentric|nested|enclosed|enclosure|frame|border|outline|" \
             r"solid|filled|open|closed|segment|segments|stripe|stripes|band|bands|row|rows|column|columns)\b"
# note: "box" is in PICT (container word) and "square/rectangle" in GEOM; "bracket/crescent" GEOM. Fixed before slot use.

def label_of(desc):
    d = desc.lower()
    if not d.strip(): return float('nan'), 'unknown'
    if re.search(PICT_WORDS, d): return 1.0, 'pictorial'
    if re.search(GEOM_WORDS, d): return 0.0, 'geometric'
    return float('nan'), 'unknown'

# ---------------------------------------------------------------- glyph measures
R = 96

def thin(img):
    """Zhang-Suen thinning (vectorised)."""
    im = img.copy().astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(im, 1)
            p2 = P[:-2, 1:-1]; p3 = P[:-2, 2:]; p4 = P[1:-1, 2:]; p5 = P[2:, 2:]
            p6 = P[2:, 1:-1]; p7 = P[2:, :-2]; p8 = P[1:-1, :-2]; p9 = P[:-2, :-2]
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            m = (im == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if m.any():
                im[m] = 0; changed = True
    return im.astype(bool)

def skeleton_stats(sk):
    P = np.pad(sk.astype(np.uint8), 1)
    nb = sum(np.roll(np.roll(P, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))[1:-1, 1:-1]
    ends = int(((nb == 1) & sk).sum())
    junc = (nb >= 3) & sk
    segs = sk & ~ndi.binary_dilation(junc, structure=np.ones((3, 3)))
    _, nseg = ndi.label(segs, structure=np.ones((3, 3)))
    _, njunc = ndi.label(junc, structure=np.ones((3, 3)))
    return ends, njunc, max(nseg, 1)

def measures(w):
    g = render(w, size=192, out=R)
    if g is None or g.sum() < 10: return None
    g = ndi.binary_closing(g, iterations=1) | g
    # holes and components
    filled = ndi.binary_fill_holes(g)
    _, holes = ndi.label(filled & ~g, structure=np.ones((3, 3)))
    _, comps = ndi.label(g, structure=np.ones((3, 3)))
    # outline orientation spread
    sm = ndi.gaussian_filter(g.astype(float), 1.5)
    gy, gx = np.gradient(sm)
    edge = g ^ ndi.binary_erosion(g)
    ang = (np.degrees(np.arctan2(gy[edge], gx[edge])) % 180.0)
    mag = np.hypot(gx[edge], gy[edge])
    h, _ = np.histogram(ang, bins=18, range=(0, 180), weights=mag)
    h = h / max(h.sum(), 1e-9)
    curv = 1.0 - np.sort(h)[-4:].sum()
    # symmetry
    def iou(a, b): return (a & b).sum() / max((a | b).sum(), 1)
    sym = max(iou(g, g[:, ::-1]), iou(g, g[::-1, :]))
    # perimetric complexity
    perim = CPLX.get(w)
    if perim is None:
        Pm = float(edge.sum()); A = float(g.sum()); perim = Pm * Pm / (4 * math.pi * A)
    # skeleton
    sk = thin(g)
    ends, njunc, nseg = skeleton_stats(sk)
    # distance from the stroke / numeral cluster in glyph space
    i = IDX.get(w)
    far = float('nan')
    if i is not None:
        js = [IDX[s] for s in STROKE_NUMERALS if s in IDX and s != w]
        far = 1.0 - float(SIM[i, js].max()) if js else float('nan')
    return dict(holes=int(holes), comps=int(comps), curv=float(curv), asym=float(1 - sym), perim=float(perim),
                strokes=int(nseg), ends=int(ends), junctions=int(njunc), far_stroke=far, ink=float(g.mean()))

# ---------------------------------------------------------------- frequencies (seq_raw) for later matching
C = json.load(open('data/derived/merged-corpus-canonical.json'))
FREQ = {lv: collections.Counter() for lv in ('seq_raw', 'seq_strong', 'seq_all')}
for r in C:
    for lv in FREQ:
        for x in (r[lv] or []): FREQ[lv][x] += 1

rows = {}
desc = {d['glyph']: d.get('shape', '') for d in DOSS}
allsigns = sorted(set(SIGNS) | set(FREQ['seq_raw']) | set(CPLX))
nfail = 0
for w in allsigns:
    try:
        m = measures(w)
    except Exception as e:
        m = None
    if m is None: nfail += 1; continue
    lab, lab3 = label_of(desc.get(w, ''))
    m.update(w=w, label=lab, label3=lab3, desc=desc.get(w, ''), numeral=w in STROKE_NUMERALS,
             f_raw=FREQ['seq_raw'][w], f_strong=FREQ['seq_strong'][w], f_all=FREQ['seq_all'][w])
    rows[w] = m
print(f'rendered {len(rows)} signs, {nfail} without glyph, {time.time()-T0:.0f}s')

# composite shape index: z over all rendered non-numeral signs (numerals kept in the table, flagged)
keys = ['holes', 'curv', 'asym', 'perim', 'far_stroke', 'logstrokes']
for m in rows.values(): m['logstrokes'] = math.log(m['strokes'])
base = [m for m in rows.values() if not m['numeral'] and not math.isnan(m['far_stroke'])]
mu = {k: np.mean([m[k] for m in base]) for k in keys}
sd = {k: np.std([m[k] for m in base]) + 1e-9 for k in keys}
for m in rows.values():
    zs = [(m[k] - mu[k]) / sd[k] for k in keys if not math.isnan(m[k])]
    m['pict_shape'] = float(np.mean(zs))
    m['pict_shape_nocplx'] = float(np.mean([(m[k] - mu[k]) / sd[k] for k in ('holes', 'curv', 'asym', 'far_stroke') if not math.isnan(m[k])]))

json.dump(list(rows.values()), open(OUT + 'loop50_signs.json', 'w'), indent=0)

# ---------------------------------------------------------------- agreement
def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return (gt + 0.5 * eq) / (len(pos) * len(neg))

def spearman(a, b):
    a = np.argsort(np.argsort(a)); b = np.argsort(np.argsort(b))
    return float(np.corrcoef(a, b)[0, 1])

rep = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); rep.append(s)
P(f'# LOOP 50 cycle 1: pictoriality measures ({time.strftime("%Y-%m-%dT%H:%M")})')
P(f'signs rendered {len(rows)}; with a dossier description {sum(1 for m in rows.values() if m["desc"])}; '
  f'label pictorial {sum(1 for m in rows.values() if m["label3"]=="pictorial")}, geometric {sum(1 for m in rows.values() if m["label3"]=="geometric")}, '
  f'unknown-with-text {sum(1 for m in rows.values() if m["label3"]=="unknown" and m["desc"])}')
lab = [m for m in rows.values() if m['label3'] in ('pictorial', 'geometric')]
pos = [m for m in lab if m['label3'] == 'pictorial']; neg = [m for m in lab if m['label3'] == 'geometric']
P(f'label set: {len(pos)} pictorial vs {len(neg)} geometric (top-200 dossier signs only; numerals among them: {sum(m["numeral"] for m in lab)})')
for k in keys + ['pict_shape', 'pict_shape_nocplx', 'comps', 'ink']:
    a = auc([m[k] for m in pos if not math.isnan(m[k])], [m[k] for m in neg if not math.isnan(m[k])])
    P(f'  AUC(label pictorial > geometric) by {k:16s} = {a:.3f}   mean pict {np.nanmean([m[k] for m in pos]):.3f} geom {np.nanmean([m[k] for m in neg]):.3f}')
nn = [m for m in lab if not m['numeral']]
pos2 = [m['pict_shape'] for m in nn if m['label3'] == 'pictorial']; neg2 = [m['pict_shape'] for m in nn if m['label3'] == 'geometric']
P(f'  non-numeral signs only: AUC(pict_shape) = {auc(pos2, neg2):.3f} ({len(pos2)} vs {len(neg2)})')
# frequency dependence (S164): shape index vs log frequency
nz = [m for m in rows.values() if m['f_raw'] >= 2 and not m['numeral']]
P(f'frequency (seq_raw >= 2, non-numeral, n={len(nz)}): Spearman(pict_shape, log f) = {spearman([m["pict_shape"] for m in nz], [math.log(m["f_raw"]) for m in nz]):+.3f}; '
  f'perim {spearman([m["perim"] for m in nz], [math.log(m["f_raw"]) for m in nz]):+.3f}; holes {spearman([m["holes"] for m in nz], [math.log(m["f_raw"]) for m in nz]):+.3f}; '
  f'curv {spearman([m["curv"] for m in nz], [math.log(m["f_raw"]) for m in nz]):+.3f}; far_stroke {spearman([m["far_stroke"] for m in nz], [math.log(m["f_raw"]) for m in nz]):+.3f}')
# disagreements worth a look
P('label pictorial but lowest shape index:')
for m in sorted(pos, key=lambda m: m['pict_shape'])[:8]: P(f'   W{m["w"]:<4} {m["pict_shape"]:+.2f}  {m["desc"][:70]}')
P('label geometric but highest shape index:')
for m in sorted(neg, key=lambda m: -m['pict_shape'])[:8]: P(f'   W{m["w"]:<4} {m["pict_shape"]:+.2f}  {m["desc"][:70]}')
P('top 12 shape index overall:')
for m in sorted(rows.values(), key=lambda m: -m['pict_shape'])[:12]: P(f'   W{m["w"]:<4} {m["pict_shape"]:+.2f} holes {m["holes"]} curv {m["curv"]:.2f} asym {m["asym"]:.2f} strokes {m["strokes"]}  {m["desc"][:50]}')
P('bottom 12 shape index (non-numeral):')
for m in sorted([m for m in rows.values() if not m['numeral']], key=lambda m: m['pict_shape'])[:12]: P(f'   W{m["w"]:<4} {m["pict_shape"]:+.2f} holes {m["holes"]} curv {m["curv"]:.2f} asym {m["asym"]:.2f} strokes {m["strokes"]}  {m["desc"][:50]}')
P(f'done {time.time()-T0:.0f}s')
open(OUT + 'loop50_cycle1_log.txt', 'w').write('\n'.join(rep))
