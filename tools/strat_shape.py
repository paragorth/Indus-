"""Does the graphic FORM of a sign encode its FUNCTION (designed code) or not (evolved script)?
Four tests with controls. Output: data/derived/strat_shape.txt.
 a. slot x shape-family contingency (families = Wells hundreds blocks, checked against the shape descriptions);
    control 1: slot labels shuffled over tokens (trivially significant, signs are slot-pure);
    control 2: family labels shuffled over SIGNS, size-matched (does the shape grouping add anything beyond the sign?).
 b. modifier rule: base -> base+mark pairs found by glyph containment (lipi font bitmaps, scale/translation search);
    offset = marked minus base in a slot/position profile; mean pairwise cosine of offsets vs random within-family pairs;
    the same, split by where the extra ink sits (inside / above / below / beside the base).
    Allograph merges (sign_allographs_levels.json) as a 'no shift expected' reference.
 c. ligatures: signs that contain two different simpler signs; does the ligature's left context match component A's
    left context and its right context component B's right context (reads as the sequence A B), or neither (new atom)?
 d. complexity (data/derived/sign-complexity.json, S164) vs frequency, vs slot, vs family.
Levels: seq_raw / seq_strong / seq (CLAUDE.md); slot labels from parsed_texts.json (seq level)."""
import json, collections, random, math, itertools, sys
import numpy as np
from scipy.ndimage import binary_dilation, zoom
from scipy.signal import fftconvolve
sys.path.insert(0, 'tools')
from sign_glyphs import render

OUT = open('data/derived/strat_shape.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.write(s + '\n')

C = json.load(open('data/derived/merged-corpus-canonical.json'))
PT = json.load(open('data/derived/parsed_texts.json'))
BR = json.load(open('data/derived/bridge_extended.json'))
CX = {int(k): v for k, v in json.load(open('data/derived/sign-complexity.json')).items()}
AL = json.load(open('data/derived/sign_allographs_levels.json'))
DOS = {x['glyph']: x for x in json.load(open('data/derived/sign-dossiers-top200.json'))}
rnd = random.Random(7)

def texts(level):
    seen = set(); T = []
    for r in C:
        s = r[level]
        if s and (r['site'], tuple(s)) not in seen:
            seen.add((r['site'], tuple(s))); T.append([int(a) for a in s])
    return T

# ---- shape families from the Wells numbering (hundreds), person block starts at 90 (W90, W91 are persons)
def fam(w):
    if w < 90: return 'stroke'
    if w < 200: return 'person'
    if w < 300: return 'fish/animal'
    if w < 400: return 'plant/misc3'
    if w < 500: return 'comb/fork/tri4'
    if w < 600: return 'arrow/box5'
    if w < 700: return 'rect/6'
    if w < 800: return 'U/jar/leaf7'
    if w < 900: return 'leaf/diamond8'
    return 'bracket9'
FAMS = ['stroke', 'person', 'fish/animal', 'plant/misc3', 'comb/fork/tri4', 'arrow/box5', 'rect/6', 'U/jar/leaf7', 'leaf/diamond8', 'bracket9']

P('# strat_shape: does form encode function?  (' + __import__('datetime').date.today().isoformat() + ')')
P('\n## Family check against the Wells shape descriptions (top-200 dossiers)')
kw = {'person': 'person', 'fish/animal': 'fish', 'plant/misc3': 'tree|plant|heart|garlic', 'U/jar/leaf7': 'jar|u with|u where|paw',
      'leaf/diamond8': 'leaf|diamond|wheel', 'bracket9': 'parenthes', 'stroke': 'stroke', 'arrow/box5': 'triangle|box|square|trapezoid|pincer|bowtie|x with',
      'comb/fork/tri4': 'pitch|fork|vertical line|rectangle|triangle', 'rect/6': 'rectangle|x but'}
import re
for f in FAMS:
    S = [w for w in DOS if fam(w) == f and DOS[w]['shape']]
    hit = [w for w in S if re.search(kw[f], DOS[w]['shape'].lower())]
    miss = [(w, DOS[w]['shape'][:40]) for w in S if w not in hit]
    P(f'{f:15s} described={len(S):3d} keyword-match={len(hit):3d} misses={miss[:6]}')

# ---- slot labels (parsed_texts, seq level)
slot_of = collections.defaultdict(collections.Counter)
tok_rows = []
for t in PT:
    for a, s in zip(t['seq'], t['slots']):
        slot_of[int(a)][s] += 1; tok_rows.append((int(a), s))
SLOTS = sorted({s for _, s in tok_rows})
freq = collections.Counter(a for a, _ in tok_rows)

def MI(pairs):
    n = len(pairs); cx = collections.Counter(a for a, _ in pairs); cy = collections.Counter(b for _, b in pairs); cxy = collections.Counter(pairs)
    return sum(v / n * math.log(v * n / (cx[a] * cy[b])) for (a, b), v in cxy.items())

P('\n## (a) slot x shape family')
rows = [(fam(a), s) for a, s in tok_rows if a != 0]
obs = MI(rows)
null = []
ys = [s for _, s in rows]
for _ in range(2000):
    rnd.shuffle(ys); null.append(MI([(f, s) for (f, _), s in zip(rows, ys)]))
P(f'MI(family, slot) = {obs:.4f} bits*ln; token-shuffle null mean {np.mean(null):.4f}, P = {(sum(x >= obs for x in null) + 1) / 2001:.4f} (trivial: signs are slot-pure)')
# control 2: reassign SIGNS to families keeping family sizes (in signs), 2000x
signs = sorted({a for a, _ in tok_rows if a != 0}); fam_of = {a: fam(a) for a in signs}
sign_slots = collections.defaultdict(list)
for a, s in tok_rows:
    if a != 0: sign_slots[a].append(s)
def MI_sign(assign): return MI([(assign[a], s) for a in signs for s in sign_slots[a]])
obs2 = MI_sign(fam_of); null2 = []
labels = [fam_of[a] for a in signs]
for _ in range(2000):
    rnd.shuffle(labels); null2.append(MI_sign(dict(zip(signs, labels))))
P(f'Sign-level control (shuffle family labels over the {len(signs)} signs): observed {obs2:.4f}, null mean {np.mean(null2):.4f}, 95th {np.percentile(null2, 95):.4f}, P = {(sum(x >= obs2 for x in null2) + 1) / 2001:.4f}')
# the same with frequency-weighting removed: each sign counted once with its dominant slot
dom = {a: collections.Counter(sign_slots[a]).most_common(1)[0][0] for a in signs}
obs3 = MI([(fam_of[a], dom[a]) for a in signs]); null3 = []
for _ in range(2000):
    rnd.shuffle(labels); null3.append(MI([(l, dom[a]) for a, l in zip(signs, labels)]))
P(f'Sign-level, one vote per sign (dominant slot): observed {obs3:.4f}, null mean {np.mean(null3):.4f}, P = {(sum(x >= obs3 for x in null3) + 1) / 2001:.4f}')
P('\nfamily        tokens  ' + '  '.join(f'{s[:6]:>6s}' for s in SLOTS) + '   purity  n_signs  signs>=10 with dominant slot')
famtok = collections.defaultdict(collections.Counter)
for f, s in rows: famtok[f][s] += 1
for f in FAMS:
    c = famtok[f]; n = sum(c.values())
    if not n: continue
    sg = [a for a in signs if fam_of[a] == f and freq[a] >= 10]
    dd = collections.Counter(dom[a] for a in sg)
    P(f'{f:14s} {n:6d}  ' + '  '.join(f'{c[s] / n:6.2f}' for s in SLOTS) + f'   {max(c.values()) / n:.2f}   {len([a for a in signs if fam_of[a] == f]):4d}    {dict(dd)}')
base = collections.Counter(s for _, s in rows); nb = sum(base.values())
P('baseline       ' + f'{nb:6d}  ' + '  '.join(f'{base[s] / nb:6.2f}' for s in SLOTS))
# per-family purity vs size-matched random sign sets
P('\nPer-family slot purity (max slot share of tokens) vs 2000 random sign sets of the same number of signs:')
for f in FAMS:
    S = [a for a in signs if fam_of[a] == f]
    if len(S) < 3: continue
    def pur(S):
        c = collections.Counter(s for a in S for s in sign_slots[a]); return max(c.values()) / sum(c.values())
    o = pur(S); nl = [pur(rnd.sample(signs, len(S))) for _ in range(2000)]
    P(f'  {f:14s} purity {o:.2f}  null {np.mean(nl):.2f}  P(>=) = {(sum(x >= o for x in nl) + 1) / 2001:.3f}')

# ---- glyphs and containment
P('\n## glyph containment (lipi font; base B inside marked M after scale/translation search)')
GS = json.load(open('data/derived/glyph_sim_signs.json'))
G = {}
for w in GS:
    g = render(w, out=40)
    if g is not None and g.sum() > 20: G[w] = g
ink = {w: int(G[w].sum()) for w in G}
def contain(b, m):
    """max over scales of the fraction of B's ink covered by M dilated once; returns (cover, offset, scale)."""
    M = binary_dilation(G[m], iterations=1).astype(float); best = (0, None, None)
    for sc in (1.0, 0.85, 0.7, 0.55, 0.45):
        B = G[b].astype(float)
        if sc < 1:
            B = zoom(B, sc, order=1) > 0.5; B = B.astype(float)
        if B.sum() < 8: continue
        cc = fftconvolve(M, B[::-1, ::-1], mode='valid')  # sum of B ink covered at each placement
        if cc.size == 0: continue
        i = np.unravel_index(cc.argmax(), cc.shape); cov = cc.max() / B.sum()
        if cov > best[0]: best = (cov, (i, B.shape), sc)
    return best
cands = sorted([w for w in G if freq[w] >= 5], key=lambda w: -freq[w])
bases = [w for w in cands if freq[w] >= 8]
import os, pickle
CACHE = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/contain.pkl'
if os.path.exists(CACHE): pairs = pickle.load(open(CACHE, 'rb'))
else:
    pairs = {}  # (b, m) -> (cover, place, scale)
    for m in cands:
        for b in bases:
            if b == m or ink[b] >= ink[m] * 0.95: continue
            cov, place, sc = contain(b, m)
            if cov >= 0.90:
                pairs[(b, m)] = (cov, place, sc)
    pickle.dump(pairs, open(CACHE, 'wb'))
P(f'{len(pairs)} candidate (base, marked) pairs with >= 90% of the base ink inside the marked sign (freq base >= 8, marked >= 5)')

def residual(b, m):
    """where the extra ink of M sits relative to the placed B: inside / above / below / side."""
    cov, ((i, j), (h, w)), sc = pairs[(b, m)]
    Bm = np.zeros_like(G[m], bool); Bm[i:i + h, j:j + w] = True  # bbox of base placement
    # extra ink = M ink outside the dilated placed base
    placedB = np.zeros_like(G[m], bool)
    Bs = (zoom(G[b].astype(float), sc, order=1) > 0.5) if sc < 1 else G[b]
    placedB[i:i + h, j:j + w] = Bs[:h, :w]
    extra = G[m] & ~binary_dilation(placedB, iterations=2)
    ys, xs = np.nonzero(extra); tot = len(ys)
    if tot < 6: return 'none', 0
    inside = ((ys >= i) & (ys < i + h) & (xs >= j) & (xs < j + w)).sum() / tot
    above = (ys < i).sum() / tot; below = (ys >= i + h).sum() / tot
    side = ((xs < j) | (xs >= j + w)).sum() / tot
    k = max([('inside', inside), ('above', above), ('below', below), ('side', side)], key=lambda x: x[1])
    return k[0], tot / ink[m]

# keep the most parsimonious pairs: for each marked sign the base with the most ink (largest component), plus any
# second base disjoint from it (for ligatures)
by_m = collections.defaultdict(list)
for (b, m), v in pairs.items(): by_m[m].append(b)
MODPAIRS = []  # (b, m, kind, extra_frac)
LIGS = []  # (m, A, B)
for m, bs in by_m.items():
    bs = sorted(bs, key=lambda b: -ink[b] * pairs[(b, m)][2] ** 2)
    b0 = bs[0]; kind, ef = residual(b0, m)
    sc0 = pairs[(b0, m)][2]
    if 0.08 <= ef <= 0.4 and sc0 >= 0.7 and ink[b0] * sc0 ** 2 >= 0.55 * ink[m] and (b0 >= 90 or m < 90):
        MODPAIRS.append((b0, m, kind, ef))
    # ligature: a second base covering mostly the residual
    for b1 in bs[1:]:
        if fam(b1) == fam(b0) and abs(ink[b1] - ink[b0]) < 0.2 * ink[b0]: continue
        cov1, ((i1, j1), (h1, w1)), sc1 = pairs[(b1, m)]
        cov0, ((i0, j0), (h0, w0)), sc0 = pairs[(b0, m)]
        # overlap of the two placements (bbox IoU)
        ix = max(0, min(i0 + h0, i1 + h1) - max(i0, i1)) * max(0, min(j0 + w0, j1 + w1) - max(j0, j1))
        if ix / min(h0 * w0, h1 * w1) < 0.5 and ink[b1] * sc1 ** 2 + ink[b0] * sc0 ** 2 > 0.6 * ink[m]:
            LIGS.append((m, b0, b1)); break
CURATED = [(220, 226), (220, 231), (220, 233), (220, 235), (220, 240), (740, 741), (740, 742), (740, 745), (790, 794), (790, 832),
           (700, 705), (700, 706), (850, 853), (850, 880), (455, 456), (555, 556), (590, 592), (520, 521), (803, 804), (845, 842),
           (390, 391), (405, 407), (95, 137), (95, 132), (90, 150), (702, 703), (625, 626), (315, 317), (900, 904), (900, 923), (817, 821)]
CURATED = [(b, m) for b, m in CURATED if freq[b] >= 5 and freq[m] >= 5]
auto_set = {(b, m) for b, m, _, _ in MODPAIRS}
P(f'curated Wells-variant list: {len(CURATED)} pairs, {len([p for p in CURATED if p in auto_set])} also found by containment')
P(f'{len(MODPAIRS)} base->marked pairs (extra ink 8-40%, scale>=0.7, base>=55% of ink); {len(LIGS)} ligature candidates (two disjoint components)')
P('base->marked pairs (W numbers, M via bridge), kind of mark:')
def Mno(w): return '/'.join('M%s' % x for x in BR.get(str(w), [])) or '-'
for b, m, kind, ef in sorted(MODPAIRS, key=lambda x: -freq[x[1]]):
    P(f'  {b:4d}({Mno(b)}) -> {m:4d}({Mno(m)})  mark={kind:6s} extra={ef:.2f} fam={fam(b)}->{fam(m)} n_base={freq[b]} n_marked={freq[m]}')

# ---- (b) modifier algebra
P('\n## (b) modifier rule: does the added mark shift use in one direction?')
POS = ['initial', 'medial', 'final']
def profile(a, T):
    v = np.zeros(len(SLOTS) + 3)
    for s, c in slot_of[a].items(): v[SLOTS.index(s)] += c
    v[:len(SLOTS)] /= max(1, v[:len(SLOTS)].sum())
    pos = np.zeros(3)
    for t in T:
        for i, x in enumerate(t):
            if x == a: pos[0 if i == 0 else 2 if i == len(t) - 1 else 1] += 1
    v[len(SLOTS):] = pos / max(1, pos.sum())
    return v
def cosv(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(np.dot(x, y) / nx / ny) if nx > 1e-9 and ny > 1e-9 else 0.0
def meancos(O):
    pr = list(itertools.combinations(O, 2)); return np.mean([cosv(x, y) for x, y in pr]) if pr else float('nan')
def kind_of(b, m):
    if (b, m) in pairs: return residual(b, m)[0]
    return 'n/a'
SETS = {'auto': MODPAIRS, 'curated': [(b, m, kind_of(b, m), 0) for b, m in CURATED]}
for setname, MODPAIRS in SETS.items():
  P(f'--- pair set: {setname} ({len(MODPAIRS)} pairs)')
  for level in ('seq_raw', 'seq_strong', 'seq'):
    T = texts(level)
    prof = {a: profile(a, T) for a in set([b for b, _, _, _ in MODPAIRS] + [m for _, m, _, _ in MODPAIRS] + cands)}
    offs = [prof[m] - prof[b] for b, m, _, _ in MODPAIRS]
    obs = meancos(offs)
    # control: for each base, a random partner from the same family with >= 5 tokens (not the real marked sign)
    null = []
    for _ in range(2000):
        O = []
        for b, m, _, _ in MODPAIRS:
            pool = [x for x in cands if fam(x) == fam(m) and x != b and x != m]
            if not pool: pool = [x for x in cands if x != b]
            O.append(prof[rnd.choice(pool)] - prof[b])
        null.append(meancos(O))
    P(f'[{level}] mean pairwise cosine of offsets over {len(offs)} pairs = {obs:.3f}; within-family random partner null mean {np.mean(null):.3f}, 95th {np.percentile(null, 95):.3f}, P = {(sum(x >= obs for x in null) + 1) / 2001:.4f}')
    if level == 'seq_raw':
        mean_off = np.mean(offs, axis=0)
        P('   mean offset (marked - base): ' + ', '.join(f'{k}={v:+.2f}' for k, v in zip(SLOTS + POS, mean_off)))
        sgn = np.sign(np.array(offs))
        P('   share of pairs moving the same way as the mean, per feature: ' + ', '.join(f'{k}={(np.sign(mean_off[i]) == sgn[:, i]).mean():.2f}' for i, k in enumerate(SLOTS + POS)))
        # by kind of mark
        for kind in ('inside', 'above', 'below', 'side'):
            K = [(b, m) for b, m, k, _ in MODPAIRS if k == kind]
            if len(K) < 3: P(f'   mark={kind}: only {len(K)} pairs'); continue
            o = meancos([prof[m] - prof[b] for b, m in K]); nl = []
            for _ in range(1000):
                nl.append(meancos([prof[rnd.choice([x for x in cands if fam(x) == fam(m) and x not in (b, m)] or [x for x in cands if x != b])] - prof[b] for b, m in K]))
            mo = np.mean([prof[m] - prof[b] for b, m in K], axis=0)
            P(f'   mark={kind:6s} n={len(K):2d} offset cosine {o:.3f} vs null {np.mean(nl):.3f} P={(sum(x >= o for x in nl) + 1) / 1001:.3f}; mean shift ' + ', '.join(f'{k}={v:+.2f}' for k, v in zip(SLOTS + POS, mo) if abs(v) >= 0.08))
        # cross-family consistency: pairs from different families only
        fams_in = collections.defaultdict(list)
        for b, m, k, _ in MODPAIRS: fams_in[fam(b)].append((b, m))
        P('   per family of the base: n pairs, mean final-share shift, mean NAME-share shift')
        for f, L in fams_in.items():
            o = np.mean([prof[m] - prof[b] for b, m in L], axis=0)
            P(f'     {f:14s} n={len(L):2d} final {o[len(SLOTS) + 2]:+.2f} NAME {o[SLOTS.index("NAME")]:+.2f} CLOSER {o[SLOTS.index("CLOSER")]:+.2f} COUNT {o[SLOTS.index("COUNT")]:+.2f}')
        # allograph reference
        AP = [(x['into'], x['form']) for x in AL['merges'] if x['level'] in ('strong', 'probable') and freq[x['into']] >= 5 and freq[x['form']] >= 5 and x['form'] in prof and x['into'] in prof]
        if len(AP) >= 3:
            ao = [prof[m] - prof[b] for b, m in AP]
            P(f'   allograph reference ({len(AP)} strong/probable merges, {level}): mean |offset| {np.mean([np.abs(o).sum() for o in ao]):.2f} vs modifier pairs {np.mean([np.abs(o).sum() for o in offs]):.2f}; offset cosine {meancos(ao):.3f}')

# ---- (c) ligatures
P('\n## (c) ligatures: sequence of components or new atom?')
DESC_LIGS = [(803, 790, 390), (806, 790, 390), (705, 700, 31), (706, 700, 31), (130, 90, 415), (772, 700, 415), (226, 220, 32), (880, 861, 820), (269, 906, 255)]
allL = {(m, a, b) for m, a, b in LIGS} | {(m, a, b) for m, a, b in DESC_LIGS if m in freq and freq[m] >= 5}
P(f'{len(LIGS)} glyph-detected + description list -> {len(allL)} ligatures tested: ' + ', '.join(f'{m}={a}+{b}' for m, a, b in sorted(allL, key=lambda x: -freq[x[0]])))
for level in ('seq_raw', 'seq'):
    T = texts(level)
    L = collections.defaultdict(collections.Counter); R = collections.defaultdict(collections.Counter); fr = collections.Counter()
    adj = collections.Counter()
    for t in T:
        for i, a in enumerate(t):
            fr[a] += 1
            L[a][t[i - 1] if i else '^'] += 1; R[a][t[i + 1] if i + 1 < len(t) else '$'] += 1
            if i + 1 < len(t): adj[(a, t[i + 1])] += 1
    def cosc(A, B):
        d = sum(A[k] * B[k] for k in A); n = math.sqrt(sum(v * v for v in A.values()) * sum(v * v for v in B.values()))
        return d / n if n else 0
    def seqscore(m, a, b): return (cosc(L[m], L[a]) + cosc(R[m], R[b])) / 2  # ligature reads as 'a b'
    def atomscore(m, a, b): return max((cosc(L[m], L[x]) + cosc(R[m], R[x])) / 2 for x in (a, b))
    pool = [x for x in fr if fr[x] >= 5]
    res = []
    P(f'[{level}] lig   comps   seq-score(best order)  seq-null(random comps)  P   full-context cos with best comp  null  P   A-B adjacent?')
    for m, a, b in sorted(allL, key=lambda x: -freq[x[0]]):
        if fr[m] < 5 or fr[a] < 5 or fr[b] < 5: continue
        s_ab, s_ba = seqscore(m, a, b), seqscore(m, b, a); s = max(s_ab, s_ba); order = f'{a}-{b}' if s_ab >= s_ba else f'{b}-{a}'
        nl = [max(seqscore(m, x, y), seqscore(m, y, x)) for x, y in (rnd.sample(pool, 2) for _ in range(1000))]
        at = atomscore(m, a, b); nla = [atomscore(m, x, y) for x, y in (rnd.sample(pool, 2) for _ in range(1000))]
        p1 = (sum(x >= s for x in nl) + 1) / 1001; p2 = (sum(x >= at for x in nla) + 1) / 1001
        res.append((p1, p2))
        P(f'  {m:4d}  {a}+{b}   {s:.2f} ({order})   {np.mean(nl):.2f}  {p1:.3f}      {at:.2f}   {np.mean(nla):.2f}  {p2:.3f}   {adj[(a, b)]}/{adj[(b, a)]}')
    if res:
        P(f'  summary [{level}]: {sum(p < 0.05 for p, _ in res)}/{len(res)} ligatures read as their component sequence (P<0.05); {sum(p < 0.05 for _, p in res)}/{len(res)} share context with a component at all')

# ---- (d) complexity vs frequency vs slot vs family
P('\n## (d) complexity vs frequency, slot, family')
from scipy.stats import spearmanr, kruskal
S2 = [a for a in signs if a in CX and freq[a] >= 2]
rho = spearmanr([math.log(freq[a]) for a in S2], [CX[a] for a in S2])
P(f'Spearman rho(log freq, complexity) = {rho.correlation:.3f} (n={len(S2)}, p={rho.pvalue:.3g}); S164 found -0.13')
# Zipf: rank-frequency slope, and the same for a designed code? report the slope and R^2 of log f vs log rank
fs = sorted([freq[a] for a in signs if a != 0], reverse=True); lr = np.log(np.arange(1, len(fs) + 1)); lf = np.log(fs)
A = np.vstack([lr, np.ones_like(lr)]).T; coef, res, *_ = np.linalg.lstsq(A, lf, rcond=None)
ss = ((lf - A @ coef) ** 2).sum(); r2 = 1 - ss / ((lf - lf.mean()) ** 2).sum()
P(f'Rank-frequency fit over {len(fs)} signs: slope {coef[0]:.2f}, R^2 {r2:.3f} (Zipf-like slope ~ -1 would be R^2 near 1 with slope -1)')
P('complexity by dominant slot (signs with >= 5 tokens):')
grp = collections.defaultdict(list)
for a in signs:
    if a in CX and freq[a] >= 5: grp[dom[a]].append(CX[a])
for s in SLOTS:
    if grp[s]: P(f'  {s:7s} n={len(grp[s]):3d} median {np.median(grp[s]):.1f} mean {np.mean(grp[s]):.1f}')
kw_ = kruskal(*[v for v in grp.values() if len(v) >= 3])
# permutation on the Kruskal H
vals = [(a, dom[a]) for a in signs if a in CX and freq[a] >= 5]; labs = [d for _, d in vals]; H0 = kw_.statistic; nh = []
for _ in range(2000):
    rnd.shuffle(labs); g = collections.defaultdict(list)
    for (a, _), l in zip(vals, labs): g[l].append(CX[a])
    nh.append(kruskal(*[v for v in g.values() if len(v) >= 3]).statistic)
P(f'  Kruskal H = {H0:.2f}, permutation P = {(sum(x >= H0 for x in nh) + 1) / 2001:.4f}')
# partial: complexity ~ slot after removing frequency (residual of complexity on log freq)
xs = np.array([math.log(freq[a]) for a, _ in vals]); ys_ = np.array([CX[a] for a, _ in vals]); bcoef = np.polyfit(xs, ys_, 1); resid = ys_ - np.polyval(bcoef, xs)
g = collections.defaultdict(list)
for (a, d), r in zip(vals, resid): g[d].append(r)
H1 = kruskal(*[v for v in g.values() if len(v) >= 3]).statistic; nh1 = []
labs = [d for _, d in vals]
for _ in range(2000):
    rnd.shuffle(labs); gg = collections.defaultdict(list)
    for r, l in zip(resid, labs): gg[l].append(r)
    nh1.append(kruskal(*[v for v in gg.values() if len(v) >= 3]).statistic)
P(f'  after regressing out log frequency: H = {H1:.2f}, P = {(sum(x >= H1 for x in nh1) + 1) / 2001:.4f}; residual medians ' + ', '.join(f'{s}={np.median(g[s]):+.1f}' for s in SLOTS if g[s]))
P('complexity by family (signs >= 5 tokens): ' + ', '.join(f'{f}={np.median([CX[a] for a in signs if fam(a) == f and a in CX and freq[a] >= 5]):.1f}' for f in FAMS if any(fam(a) == f and a in CX and freq[a] >= 5 for a in signs)))
OUT.close()

# ---- verdict
OUT = open('data/derived/strat_shape.txt', 'a')
P('\n## Verdict: does form encode function?')
P('(a) Shape family does carry slot information beyond chance grouping of signs (sign-level MI P = 0.0005; one vote per sign P = 0.0035),')
P('    but no family is slot-pure: purity 0.45-0.77 vs 0.58-0.62 for random sign sets, no family significant (best fish 0.75, P = 0.09).')
P('    The family signal is carried by a few frequent signs that happen to be frame words: the jar (closer) in the 700s, the')
P('    diamond/leaf openers in the 800s, stroke numerals in 1-89. Remove those three and the families are ordinary mixed vocabularies.')
P('(b) FAILS. The added mark does not do one job across families: offset-cosine 0.035 (auto, 41 pairs) and 0.109 (curated, 31 pairs),')
P('    P = 0.32-0.42 against random within-family partners, on all three merge levels. The one sub-result, mark above the base')
P('    (fish hat/whiskers, U + tall stroke, 455->456, 702->703; 6 pairs, P = 0.022), is uncorrected and is the known fish qualifier effect (S296).')
P('    The weak trend of S147/S323 (marked sign -> NAME element, 77% of curated pairs) repeats but is not stronger than drift to the baseline.')
P('(c) MOSTLY FAILS. 3 of 13 ligatures read as their component sequence (880 = 861+820 is an opener variant; 705/706 = U + tall stroke,')
P('    P = 0.05); 10 of 13, including leaf+tree 803/806, person+pitchfork 130, U+fork 772, share no context with either component (new atoms).')
P('    226 = fish + 2 strokes resembles the plain fish (cos 0.77, P = 0.014) but not the sequence 220-32 (S3 found 220-520 instead).')
P('(d) Script-like, not code-like: frequent signs are simpler (rho = -0.14, P = 0.003, as S164), rank-frequency is a straight log-log line')
P('    (slope -1.48, R^2 0.96). Complexity by slot is weak (P = 0.22 raw; P = 0.03 only after removing frequency, driven by 14 COUNT signs).')
P('Overall: form encodes function only where shape and frame word coincide (jar, diamond openers, numerals). The designed-code predictions')
P('(slot-pure families, one modifier = one job, ligature = component sequence) fail their controls. The inventory looks like an evolved')
P('pictographic script: shape families are semantic/iconic groups, not grammatical slots. Caveat: lipi font drawings are modern normalisations,')
P('and 76% of tokens are Mohenjo-daro + Harappa.')
OUT.close()
