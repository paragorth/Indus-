"""S-DARK-28 cycle 1b: ATTACHED modifications (strokes drawn through or onto the body,
cartouches touching the sign, fused ligatures), which the component detector of
dark_loop28_c1.py cannot see.

Method: subtractive template search.  For derived glyph D and candidate base B (non-numeral,
ink between 0.25 and 0.95 of D's), B's bbox crop is rescaled to 6 heights (0.55-1.0 of D's)
and slid over D by FFT correlation against the 1-px dilated D; coverage = share of B pixels
that land on D.  If coverage >= COV, the residual R = D minus the placed, dilated B is
classified: tiny -> 'same' (near-duplicate), 1-4 small components -> 'strokes+' (attached
strokes, with position), a component whose filled interior covers the base -> 'enclosure+',
a residual matching B again -> 'doubling+', matching another sign -> 'ligature+'.
Per derived sign the base that explains most ink is kept (ties by coverage).
Controls: planted attached strokes (drawn through the glyph), touching box, fused doubling;
wrong-base coverage distribution sets COV.  Output appended to loop28_edges.json as
loop28_edges_b.json and the union loop28_edges_all.json; report loop28_cycle1b.txt
"""
import sys, json, random, collections, time, itertools
import numpy as np
sys.path.insert(0, 'tools')
from sign_glyphs import render
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from PIL import Image
from multiprocessing import Pool
random.seed(281); np.random.seed(281)
T0 = time.time(); OUT = 'data/derived/dark/'; R = 64
SIGNS = json.load(open('data/derived/glyph_sim_signs.json')); IDX = {w: i for i, w in enumerate(SIGNS)}
SIM_FINE = np.load('data/derived/glyph_sim_fine.npy')
NUMERAL = set(range(1, 8)) | set(range(12, 21)) | set(range(25, 30)) | set(range(31, 40)) | {55, 56}
M = json.load(open('data/derived/merged-corpus-canonical.json'))
FREQ = collections.Counter(x for r in M for x in r['seq_raw'] if x not in (0, 999))
G = {w: render(w, size=192, out=R) for w in SIGNS}
BASES = [w for w in SIGNS if w not in NUMERAL and FREQ[w] >= 2 and G[w].sum() > 40]
SCALES = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]

def bbox(m):
    ys, xs = np.nonzero(m); return ys.min(), ys.max(), xs.min(), xs.max()
def crop(m):
    y0, y1, x0, x1 = bbox(m); return m[y0:y1 + 1, x0:x1 + 1]
def norm(mask, out=32):
    if mask.sum() == 0: return None
    a = crop(mask); h, w = a.shape; s = max(h, w)
    pad = np.zeros((s, s), bool); pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return np.array(Image.fromarray(pad.astype(np.uint8) * 255).resize((out, out), Image.BILINEAR)) > 60
def nvec(m): return m.ravel().astype(np.float32), ndi.binary_dilation(m, iterations=1).ravel().astype(np.float32)
NV = {}; ND = {}
for w in SIGNS:
    n = norm(G[w]); NV[w], ND[w] = nvec(n)
NVM = np.array([NV[w] for w in SIGNS]); NDM = np.array([ND[w] for w in SIGNS]); NN = NVM.sum(1)
def match(part):
    n = norm(part)
    if n is None or n.sum() < 8: return None
    v, d = nvec(n); s = (NDM @ v + NVM @ d) / (NN + v.sum())
    dens = v.sum() / 1024.0
    return np.where((NN / 1024.0 > 2.0 * dens) | (NN / 1024.0 < 0.5 * dens), 0.0, s)
# precomputed scaled base crops
BC = {}
for w in BASES:
    c = crop(G[w]); h, wd = c.shape
    BC[w] = []
    for s in SCALES:
        H = max(4, int(round(h * s))); W = max(2, int(round(wd * s * 1.0)))
        if H > R or W > R: continue
        sc = np.array(Image.fromarray(c.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)) > 100
        if sc.sum() >= 10: BC[w].append((s, sc))

def place(Dd, sc):
    """best position of template sc on dilated D: returns coverage and (y, x) offset"""
    corr = fftconvolve(Dd.astype(np.float32), sc[::-1, ::-1].astype(np.float32), mode='valid')
    k = int(np.argmax(corr)); y, x = np.unravel_index(k, corr.shape)
    return corr[y, x] / sc.sum(), (y, x)

def components(mask, minpix=4):
    lab, k = ndi.label(mask, structure=np.ones((3, 3)))
    return [lab == i for i in range(1, k + 1) if (lab == i).sum() >= minpix]

def analyse(D, self_w, COV, bases=None):
    ink = D.sum(); Dd = ndi.binary_dilation(D, iterations=1)
    hits = []
    for b in (bases or BASES):
        if b == self_w: continue
        ib = G[b].sum()
        if not (0.25 * ink <= ib * 1.0 and ib <= 1.3 * ink): continue  # base ink bound (scale can shrink it)
        best = None
        for s, sc in BC[b]:
            if sc.sum() < 0.2 * ink or sc.sum() > 0.97 * ink: continue
            cov, (y, x) = place(Dd, sc)
            if cov >= COV and (best is None or sc.sum() > best[1].sum()): best = (cov, sc, y, x, s)
        if best is None: continue
        cov, sc, y, x, s = best
        placed = np.zeros_like(D); placed[y:y + sc.shape[0], x:x + sc.shape[1]] = sc
        resid = D & ~ndi.binary_dilation(placed, iterations=1)
        explained = 1 - resid.sum() / ink
        comps = components(resid)
        rtot = resid.sum()
        if rtot < 0.06 * ink or not comps: kind, sub, extra = 'same', '', {}
        else:
            by0, by1, bx0, bx1 = bbox(placed)
            enc = None
            for c in comps:
                f = ndi.binary_fill_holes(c)
                if (f & placed).sum() >= 0.8 * placed.sum() and c.sum() >= 0.2 * ink: enc = c
            if enc is not None:
                fr = ndi.binary_fill_holes(enc).sum() / ((bbox(enc)[1] - bbox(enc)[0] + 1) * (bbox(enc)[3] - bbox(enc)[2] + 1))
                kind, sub, extra = 'enclosure+', ('box' if fr > 0.9 else 'oval' if fr > 0.7 else 'ring'), {}
            elif len(comps) <= 4 and max(c.sum() for c in comps) <= 0.22 * ink and rtot <= 0.5 * ink:
                pos = []
                for c in comps:
                    cy, cx = ndi.center_of_mass(c)
                    pos.append('above' if cy < by0 else 'below' if cy > by1 else 'left' if cx < bx0 else 'right' if cx > bx1 else 'inside')
                kind, sub, extra = 'strokes+', f'{len(comps)}{collections.Counter(pos).most_common(1)[0][0]}', dict(k=len(comps), pos=pos)
            else:
                sm = match(resid)
                if sm is None: continue
                j = int(np.argmax(sm))
                if sm[j] >= 0.889 and SIGNS[j] != self_w:
                    if SIGNS[j] == b: kind, sub, extra = 'doubling+', 'fused', {}
                    else: kind, sub, extra = 'ligature+', 'fused', dict(partner=SIGNS[j], pscore=float(sm[j]))
                else: continue
        hits.append(dict(base=b, type=kind, sub=sub, score=float(cov), explained=float(explained), scale=s, **extra))
    return hits

def best_hits(hits):
    """per derived: drop 'same'; keep the base explaining most ink, plus any other base of a different type within 0.05 explained"""
    hs = [h for h in hits if h['type'] != 'same']
    if not hs: return []
    hs.sort(key=lambda h: (-h['explained'], -h['score']))
    top = hs[0]; keep = [top]
    for h in hs[1:]:
        if h['explained'] >= top['explained'] - 0.05 and h['type'] != top['type'] and len(keep) < 3: keep.append(h)
    return keep

# ---------------- controls ----------------
def planted(w, kind, partner=None):
    g = G[w]; c = crop(g); h, wd = c.shape
    can = np.zeros((R * 3, R * 3), bool); oy, ox = R, R
    can[oy:oy + h, ox:ox + wd] = c
    if kind == 'through':      # one stroke drawn across the body plus one attached below
        cy = oy + h // 2; can[cy - 1:cy + 2, ox - 6:ox + wd + 6] = True
        cx = ox + wd // 2; can[oy + h - 2:oy + h + 10, cx - 1:cx + 2] = True
    elif kind == 'touchbox':   # box whose sides touch the glyph
        can[oy - 1:oy + h + 1, ox - 1:ox + 1] = True; can[oy - 1:oy + h + 1, ox + wd - 1:ox + wd + 1] = True
        can[oy - 1:oy + 1, ox - 1:ox + wd + 1] = True; can[oy + h - 1:oy + h + 1, ox - 1:ox + wd + 1] = True
    elif kind == 'fused':      # second copy touching the first
        can[oy:oy + h, ox + wd - 1:ox + 2 * wd - 1] |= c
    elif kind == 'fusedlig':
        c2 = crop(G[partner]); h2, w2 = c2.shape
        can[oy:oy + h2, ox + wd - 1:ox + wd - 1 + w2] |= c2
    return norm(can, out=R)
KM = {'through': 'strokes+', 'touchbox': 'enclosure+', 'fused': 'doubling+', 'fusedlig': 'ligature+'}

def work_plant(args):
    w, kind, partner, COV = args
    D = planted(w, kind, partner)
    return (w, kind, partner, analyse(D, None, COV))
def work_real(args):
    w, COV = args
    return (w, analyse(G[w], w, COV))
def work_cov(w):
    """coverage of WRONG bases on an unmodified glyph (random-pair control): max coverage per base"""
    D = G[w]; Dd = ndi.binary_dilation(D, iterations=1); ink = D.sum(); out = []
    for b in random.sample(BASES, 40):
        if b == w: continue
        ib = G[b].sum()
        if not (0.25 * ink <= ib and ib <= 1.3 * ink): continue
        m = 0
        for s, sc in BC[b]:
            if sc.sum() < 0.2 * ink or sc.sum() > 0.97 * ink: continue
            m = max(m, place(Dd, sc)[0])
        out.append((w, b, m))
    return out

if __name__ == '__main__':
    rep = [f'# LOOP 28 cycle 1b: attached modifications by subtractive template search ({time.strftime("%Y-%m-%dT%H:%M")})']
    rep.append(f'bases {len(BASES)} (non-numeral, freq >= 2); scales {SCALES}; derived = all {len([w for w in SIGNS if w not in NUMERAL])} non-numeral signs')
    with Pool(4) as pool:
        # 1. wrong-base coverage distribution on 150 random real glyphs
        cov = [x for lst in pool.map(work_cov, random.sample([w for w in SIGNS if w not in NUMERAL and G[w].sum() > 60], 150)) for x in lst]
        cv = np.array([c[2] for c in cov])
        COV = float(np.quantile(cv, 0.98))
        rep.append(f'random (derived, wrong base) best coverage: median {np.median(cv):.3f}, 95th {np.quantile(cv,.95):.3f}, 98th {COV:.3f} (n={len(cv)}); threshold COV = 98th pct = {COV:.3f}')
        rep.append('   note: some of these "random" pairs are real derivations, so the threshold is conservative')
        # 2. planted recall
        NONNUM = [w for w in BASES if G[w].sum() > 80]
        jobs = []
        for kind in KM:
            for w in random.sample(NONNUM, 40):
                jobs.append((w, kind, random.choice([x for x in NONNUM if x != w]) if kind == 'fusedlig' else None, COV))
        res = pool.map(work_plant, jobs)
        rec = collections.Counter(); tot = collections.Counter(); wrongb = collections.Counter(); anytype = collections.Counter()
        for w, kind, partner, hits in res:
            tot[kind] += 1; bh = best_hits(hits)
            if any(h['type'] == KM[kind] and (h['base'] == w or h.get('partner') == w) for h in bh): rec[kind] += 1
            elif any(h['base'] == w or h.get('partner') == w for h in bh): anytype[kind] += 1
            elif bh: wrongb[kind] += 1
        rep.append('planted recall (right base AND type) | right base, other type | wrong base:')
        for kind in KM: rep.append(f'  {kind:9s} {rec[kind]:2d}/{tot[kind]} | {anytype[kind]} | {wrongb[kind]}')
        # 3. real graph
        real = pool.map(work_real, [(w, COV) for w in SIGNS if w not in NUMERAL])
    EB = []
    for w, hits in real:
        for h in best_hits(hits):
            h['derived'] = w; h['fine'] = float(SIM_FINE[IDX[w], IDX[h['base']]]); h['conf'] = 'high' if h['score'] >= COV + 0.03 and h['explained'] >= 0.6 else 'mid'
            EB.append(h)
    same = collections.Counter()
    for w, hits in real:
        for h in hits:
            if h['type'] == 'same': same[(w, h['base'])] += 1
    by = collections.Counter(h['type'] for h in EB)
    rep.append(f'\n## attached-modification edges: {len(EB)} by type {dict(by)}; near-duplicate (same) pairs {len(same)}')
    dist = np.array([abs(e['derived'] - e['base']) for e in EB]); fine = np.array([e['fine'] for e in EB])
    rep.append(f'   |dW| <= 20 in {np.mean(dist<=20)*100:.0f}% of edges (random 4%); glyph_sim_fine mean {fine.mean():.3f} (random 0.40)')
    A = json.load(open(OUT + 'loop28_edges.json'))
    have = {(e['derived'], e['base']) for e in A}
    new = [e for e in EB if (e['derived'], e['base']) not in have]
    rep.append(f'   new (derived, base) pairs not found by the component detector: {len(new)}; overlap {len(EB) - len(new)}')
    rep.append('\n## edge table (derived <- base type/sub coverage explained scale conf fine)')
    for e in sorted(EB, key=lambda e: (e['type'], -e['explained'])):
        rep.append(f"  W{e['derived']:<4d} <- W{e['base']:<4d} {e['type']:11s} {e['sub']:8s} cov {e['score']:.3f} expl {e['explained']:.2f} s{e['scale']:.2f} {e['conf']} fine {e['fine']:.2f}" + (f" partner W{e['partner']}" if 'partner' in e else '') + (f" pos {e['pos']}" if 'pos' in e else ''))
    json.dump(EB, open(OUT + 'loop28_edges_b.json', 'w'), indent=0)
    # union: component edges + attached edges (type names unified: strokes+ -> strokes etc. kept with the + suffix in 'sub2')
    U = A + [dict(e, type=e['type'].rstrip('+'), method='attached') for e in new]
    json.dump(U, open(OUT + 'loop28_edges_all.json', 'w'), indent=0)
    rep.append(f'\nunion graph: {len(U)} edges -> loop28_edges_all.json; elapsed {time.time()-T0:.0f}s')
    open(OUT + 'loop28_cycle1b.txt', 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep[:30]))
