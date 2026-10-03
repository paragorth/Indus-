#!/usr/bin/env python3
"""Loop 38: the space budget. Was the text planned to fit the seal face, and does
running out of room explain the texts that lack a closer?

Sign widths come from the lipi Indus font (U+E000 + Wells number, as S164 / loop 28):
each glyph is rendered, its ink bounding box measured, and its width expressed in units
of a reference sign height (median bbox height of the 100 commonest signs). The font is
a proxy for the cut glyphs; one CISI photo (M-6) is used as a proportion check.

Data: data/raw/inscriptions.csv (one row per inscribed line; 'text' in object order,
reversed = reading order; '/' = line break on the object; 000 = unread sign;
'[' / ']' = broken edge), merge levels from data/derived/sign_allographs_levels.json
(seq_raw = Wells numbers as given; seq_strong = strong merges; seq_all = strong+probable).

Usage: python3 tools/dark_loop38.py [cycle ...]   (default 1 2 3 4)
Outputs: data/derived/dark/loop38_c<N>.txt, loop38_glyph_widths.json
"""
import csv, json, re, sys, collections, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = '/home/user/Indus-'
OUT = ROOT + '/data/derived/dark/'
CSV = ROOT + '/data/raw/inscriptions.csv'
FONT = ROOT + '/tools/indus_font.ttf'
rng = np.random.default_rng(38)
LEVELS = ['seq_raw', 'seq_strong', 'seq_all']
NPERM = int(sys.argv[sys.argv.index('--nperm') + 1]) if '--nperm' in sys.argv else 1000

OPENERS = {817, 861, 820}
JAR = {740}
ARROW = {520}
OTHER_CLOSERS = {156, 527, 617, 226, 390, 405, 154, 158, 15, 254, 12}   # S289 paradigm, as loop 30
CLOSERS = JAR | ARROW | OTHER_CLOSERS
NUMERALS = {1, 2, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}
HOME = {'Mohenjo-daro', 'Harappa'}

# ------------------------------------------------------------------ glyph widths
def glyph_widths():
    f = ImageFont.truetype(FONT, 200)
    boxes = {}
    for w in range(1, 1000):
        im = Image.new('L', (800, 500), 0)
        ImageDraw.Draw(im).text((200, 80), chr(0xE000 + w), font=f, fill=255)
        a = np.array(im) > 128
        ys, xs = np.nonzero(a)
        if len(xs) == 0:
            continue
        boxes[w] = (int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    return boxes

def load_widths(freq):
    path = OUT + 'loop38_glyph_widths.json'
    boxes = glyph_widths()
    top = [w for w, _ in freq.most_common(100) if w in boxes]
    href = float(np.median([boxes[w][1] for w in top]))
    rel = {w: {'w_px': b[0], 'h_px': b[1], 'w_rel': b[0] / href, 'h_rel': b[1] / href,
               'aspect': b[0] / b[1]} for w, b in boxes.items()}
    json.dump({'href_px': href, 'note': 'lipi font U+E000+Wells, rendered at 200 px; w_rel = ink width / reference height (median bbox height of the 100 commonest signs)',
               'signs': {str(k): v for k, v in rel.items()}}, open(path, 'w'), indent=0)
    return {w: v['w_rel'] for w, v in rel.items()}, {w: v['h_rel'] for w, v in rel.items()}, href, boxes

# ------------------------------------------------------------------ corpus
def parse_line(t):
    t = (t or '').strip()
    broken = t.startswith(']') or t.endswith('[')
    core = t.strip('+[] ')
    lines = [l for l in core.split('/') if l.strip()]
    toks = [s for s in re.split(r'[-/]', core) if s.strip()]
    seq = [int(s) for s in toks if s.strip().isdigit()]
    return list(reversed(seq)), broken, len(lines)

def merge_maps():
    a = json.load(open(ROOT + '/data/derived/sign_allographs_levels.json'))
    strong, alll = {}, {}
    for m in a['merges']:
        if m['level'] == 'strong':
            strong[m['form']] = m['into']; alll[m['form']] = m['into']
        elif m['level'] == 'probable':
            alll[m['form']] = m['into']
    def close(mp):
        for k in list(mp):
            v = mp[k]; n = 0
            while v in mp and n < 10: v = mp[v]; n += 1
            mp[k] = v
        return mp
    return {'seq_raw': {}, 'seq_strong': close(strong), 'seq_all': close(alll)}, a['merges']

def mm(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except Exception:
        return None

def load_corpus():
    maps, merges = merge_maps()
    rows = []
    for r in csv.DictReader(open(CSV)):
        seq, broken, nlines = parse_line(r['text'])
        d = dict(cisi=r['cisi'], site=r['site'], type=r['type'], sub=r['type'].split(':')[0],
                 symbol=r['symbol'], material=r['material'], shape=r['shape'], dir=r['dir.'],
                 complete=r['complete'] == 'Y', broken=broken, nlines=nlines, sides=r['sides'],
                 H=mm(r['horizontal(mm)']), V=mm(r['vertical(mm)']), TH=mm(r['thickness(mm)']),
                 has000=(0 in seq), text=r['text'])
        for lv in LEVELS:
            d[lv] = [maps[lv].get(s, s) for s in seq]
        rows.append(d)
    return rows, merges

# ------------------------------------------------------------------ helpers
def ink(seq, W):
    return sum(W.get(s, np.nan) for s in seq)

def cv_r2(X, y, k=10, reps=10, seed=0):
    """Cross-validated R^2 of OLS y ~ 1 + X."""
    n = len(y); r = np.random.default_rng(seed); scores = []
    X1 = np.column_stack([np.ones(n), X]) if X.ndim == 2 else np.column_stack([np.ones(n), X])
    for _ in range(reps):
        idx = r.permutation(n); folds = np.array_split(idx, k); press = 0.0
        for f in folds:
            tr = np.setdiff1d(idx, f)
            b, *_ = np.linalg.lstsq(X1[tr], y[tr], rcond=None)
            press += ((y[f] - X1[f] @ b) ** 2).sum()
        scores.append(1 - press / ((y - y.mean()) ** 2).sum())
    return float(np.mean(scores))

def fit_score(Xtr, ytr, Xte, yte):
    X1 = np.column_stack([np.ones(len(ytr)), Xtr]); b, *_ = np.linalg.lstsq(X1, ytr, rcond=None)
    pred = np.column_stack([np.ones(len(yte)), Xte]) @ b
    return 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()

def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    # average ranks for ties
    def rk(x):
        s = np.sort(x); r = np.empty(len(x)); order = np.argsort(x); i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and s[j + 1] == s[i]: j += 1
            r[order[i:j + 1]] = (i + j) / 2.0; i = j + 1
        return r
    ra, rb = rk(a), rk(b)
    if ra.std() == 0 or rb.std() == 0: return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])

def elasticity(n, H, nboot=500):
    x = np.log(np.asarray(n, float)); y = np.log(np.asarray(H, float))
    b = np.polyfit(x, y, 1)[0]; bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(x), len(x)); bs.append(np.polyfit(x[i], y[i], 1)[0])
    return b, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def seal_sample(rows, lv, W, need_V=True, single=True):
    out = []
    for r in rows:
        if not r['type'].startswith('SEAL'): continue
        if r['dir'] in ('BUS', 'T/B', 'SYM'): continue
        if single and r['nlines'] != 1: continue
        if not r['complete'] or r['broken'] or r['has000']: continue
        s = r[lv]
        if len(s) < 2 or r['H'] is None or (need_V and r['V'] is None): continue
        if r['H'] < 5 or r['H'] > 70: continue
        if any(t not in W for t in s): continue
        out.append(r)
    return out

def strata_perm(labels, strata, nperm, stat_fn):
    """Permute a binary label within strata; return null distribution of stat_fn(labels)."""
    labels = np.asarray(labels); groups = collections.defaultdict(list)
    for i, s in enumerate(strata): groups[s].append(i)
    groups = [np.array(g) for g in groups.values()]
    null = []
    for _ in range(nperm):
        lab = labels.copy()
        for g in groups: lab[g] = labels[g][rng.permutation(len(g))]
        null.append(stat_fn(lab))
    return np.array(null)

def pval(null, obs, side):
    null = np.asarray(null)
    if side == 'greater': return float((np.sum(null >= obs) + 1) / (len(null) + 1))
    if side == 'less': return float((np.sum(null <= obs) + 1) / (len(null) + 1))
    return float((np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (len(null) + 1))

# ================================================================== cycle 1
def cycle1(rows, W, Hrel, href, boxes, merges):
    rep = [f'# LOOP 38 cycle 1: glyph widths and the ink-width vs face-width regression ({time.strftime("%Y-%m-%dT%H:%M")})']
    freq = collections.Counter(t for r in rows for t in r['seq_raw'] if t)
    rep.append(f'font glyphs rendered: {len(W)}; reference height {href:.0f} px (median bbox height of the 100 commonest signs)')
    ws = np.array([W[s] for s, _ in freq.most_common(200) if s in W])
    rep.append(f'w_rel over the 200 commonest signs: median {np.median(ws):.2f}, IQR {np.percentile(ws,25):.2f}-{np.percentile(ws,75):.2f}, min {ws.min():.2f}, max {ws.max():.2f}')
    wide = sorted([(W[s], s) for s, _ in freq.most_common(200) if s in W], reverse=True)[:12]
    narrow = sorted([(W[s], s) for s, _ in freq.most_common(200) if s in W])[:12]
    rep.append('widest common signs (w_rel): ' + ', '.join(f'W{s} {w:.2f}' for w, s in wide))
    rep.append('narrowest common signs: ' + ', '.join(f'W{s} {w:.2f}' for w, s in narrow))
    for s, name in [(740, 'jar'), (220, 'plain fish'), (240, 'whisker fish'), (520, 'arrow'), (2, 'tall 2'), (817, 'opener 817'), (861, 'opener 861'), (820, 'opener 820'), (33, 'tall 3'), (90, 'W90'), (400, 'W400'), (3, 'short 3')]:
        if s in W: rep.append(f'  W{s} {name}: w_rel {W[s]:.2f}, h_rel {Hrel[s]:.2f}, aspect {boxes[s][0]/boxes[s][1]:.2f}')
    # photo check M-6 (data/images/cisi-M-6a.png): text 244-65-880-820 reading order; measured on the photo
    photo = {820: 200 / 230, 880: 190 / 230, 65: 110 / 230, 244: 290 / 230}
    rep.append('photo check M-6 (CISI photo, signs 820 ring, 880 diamond, 65, 244 winged fish; width/height measured on the photo vs font aspect):')
    for s, a in photo.items():
        if s in boxes: rep.append(f'  W{s}: photo {a:.2f}  font {boxes[s][0]/boxes[s][1]:.2f}')
    rep.append('  M-6 face: text band = top 24% of the face; 4 signs span 85% of the width; sign height ~ 0.21 of face height')
    rows_out = {}
    for lv in LEVELS:
        S = seal_sample(rows, lv, W)
        home = [r for r in S if r['site'] in HOME]; other = [r for r in S if r['site'] not in HOME]
        rep.append(f'\n--- level {lv}: complete single-line seals with H,V and all signs rendered: {len(S)} (home {len(home)}, other sites {len(other)}, emblem seals {sum(1 for r in S if r["symbol"] not in ("None","-",""))})')
        def feats(G):
            n = np.array([len(r[lv]) for r in G], float)
            ik = np.array([ink(r[lv], W) for r in G]); H = np.array([r['H'] for r in G]); V = np.array([r['V'] for r in G])
            return n, ik, H, V
        n, ik, H, V = feats(home)
        rep.append(f'home: n signs mean {n.mean():.2f}; ink (sum w_rel) mean {ik.mean():.2f}; H mean {H.mean():.1f} mm; rho(n, ink) = {spearman(n, ik):.3f}')
        rep.append(f'  rho(H, n) = {spearman(H, n):.3f}; rho(H, ink) = {spearman(H, ik):.3f}; rho(V, n) = {spearman(V, n):.3f}; rho(V, ink) = {spearman(V, ik):.3f}')
        y = np.log(H)
        r2_n = cv_r2(np.log(n), y); r2_ink = cv_r2(np.log(ik), y); r2_both = cv_r2(np.column_stack([np.log(n), np.log(ik)]), y)
        rep.append(f'  10-fold CV R2 (log H): n only {r2_n:.4f}; ink only {r2_ink:.4f}; n + ink {r2_both:.4f}; gain {r2_both - r2_n:+.4f}')
        # held-out sites
        if len(other) >= 30:
            no, iko, Ho, Vo = feats(other)
            ho_n = fit_score(np.log(n)[:, None], y, np.log(no)[:, None], np.log(Ho))
            ho_b = fit_score(np.column_stack([np.log(n), np.log(ik)]), y, np.column_stack([np.log(no), np.log(iko)]), np.log(Ho))
            rep.append(f'  held-out sites ({len(other)}): R2 n only {ho_n:.4f}; n + ink {ho_b:.4f}; gain {ho_b - ho_n:+.4f}')
        # per-sign residual: does the sign's width predict its effect on H beyond n?
        # null: shuffle widths among signs
        signs = sorted({t for r in home for t in r[lv]})
        wv = np.array([W[s] for s in signs]); null_gain = []; null_ho = []
        seqs = [r[lv] for r in home]; seqs_o = [r[lv] for r in other]
        for p in range(NPERM):
            perm = dict(zip(signs, wv[rng.permutation(len(wv))]))
            ikp = np.array([sum(perm.get(t, W[t]) for t in s) for s in seqs])
            g = cv_r2(np.column_stack([np.log(n), np.log(ikp)]), y, reps=2) - cv_r2(np.log(n), y, reps=2)
            null_gain.append(g)
            if len(other) >= 30:
                ikpo = np.array([sum(perm.get(t, W[t]) for t in s) for s in seqs_o])
                null_ho.append(fit_score(np.column_stack([np.log(n), np.log(ikp)]), y, np.column_stack([np.log(no), np.log(ikpo)]), np.log(Ho)) - ho_n)
        ng = np.array(null_gain)
        rep.append(f'  width-shuffle null ({NPERM}x): CV gain median {np.median(ng):+.4f}, 95th {np.percentile(ng,95):+.4f}, max {ng.max():+.4f}; P(gain >= obs) = {pval(ng, r2_both - r2_n, "greater"):.3f}')
        if null_ho:
            nh = np.array(null_ho)
            rep.append(f'  held-out null: gain median {np.median(nh):+.4f}, 95th {np.percentile(nh,95):+.4f}; P = {pval(nh, ho_b - ho_n, "greater"):.3f}')
        # emblem seals only, and the implied line-height fraction
        E = [r for r in home if r['symbol'] not in ('None', '-', '')]
        ne, ike, He, Ve = feats(E)
        frac = He / (ike * Ve)             # line height as a fraction of V that would make the ink exactly fill H
        frac_n = He / (ne * Ve)
        rep.append(f'  emblem seals (home, {len(E)}): implied line-height fraction H/(ink*V): median {np.median(frac):.3f}, IQR {np.percentile(frac,25):.3f}-{np.percentile(frac,75):.3f}, CV {frac.std()/frac.mean():.3f}; with n instead of ink: median {np.median(frac_n):.3f}, CV {frac_n.std()/frac_n.mean():.3f}')
        rep.append(f'  share of emblem seals whose ink at 1/3 line height would exceed H: {np.mean(ike * Ve / 3 > He):.3f}; at 1/4: {np.mean(ike * Ve / 4 > He):.3f}; at 1/5: {np.mean(ike * Ve / 5 > He):.3f}')
        rep.append(f'  rho(frac, n) = {spearman(frac, ne):.3f} (negative = longer texts get a smaller line height, i.e. squeezed)')
        rows_out[lv] = dict(n=len(S), r2_n=r2_n, r2_both=r2_both, gain=r2_both - r2_n, p=pval(ng, r2_both - r2_n, 'greater'),
                            ho=(ho_n, ho_b) if len(other) >= 30 else None, frac=float(np.median(frac)), rho_frac=spearman(frac, ne),
                            rho_Hn=spearman(H, n), rho_Hink=spearman(H, ik), r2_ink=r2_ink)
    open(OUT + 'loop38_c1.txt', 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    return rows_out

# ================================================================== cycle 2
def cycle2(rows, W):
    rep = [f'# LOOP 38 cycle 2: density (signs per mm, ink per mm) against text length, by object class ({time.strftime("%Y-%m-%dT%H:%M")})']
    rep.append('elasticity b = slope of log H on log n (bootstrap 95% CI). b = 1: face width scales with the text (density flat, sized to text); b = 0: fixed face (density rises 1:1 with length, squeezed). Same for ink.')
    res = {}
    for lv in LEVELS:
        rep.append(f'\n--- level {lv}')
        groups = [('SEAL home', lambda r: r['type'].startswith('SEAL') and r['site'] in HOME),
                  ('SEAL Mohenjo-daro', lambda r: r['type'].startswith('SEAL') and r['site'] == 'Mohenjo-daro'),
                  ('SEAL Harappa', lambda r: r['type'].startswith('SEAL') and r['site'] == 'Harappa'),
                  ('SEAL other sites', lambda r: r['type'].startswith('SEAL') and r['site'] not in HOME),
                  ('SEAL emblem', lambda r: r['type'].startswith('SEAL') and r['symbol'] not in ('None', '-', '')),
                  ('SEAL script-only', lambda r: r['type'].startswith('SEAL') and r['symbol'] == 'None'),
                  ('TAB:B moulded', lambda r: r['type'].startswith('TAB:B')),
                  ('TAB:I incised', lambda r: r['type'].startswith('TAB:I')),
                  ('TAB:C copper', lambda r: r['type'].startswith('TAB:C')),
                  ('TAG sealings', lambda r: r['type'].startswith('TAG')),
                  ('POT graffiti', lambda r: r['type'].startswith('POT'))]
        for name, fn in groups:
            G = [r for r in rows if fn(r) and r['nlines'] == 1 and r['complete'] and not r['broken'] and not r['has000']
                 and len(r[lv]) >= 2 and r['H'] and 5 <= r['H'] <= 70 and r['dir'] not in ('BUS', 'T/B', 'SYM') and all(t in W for t in r[lv])]
            if len(G) < 25:
                rep.append(f'{name}: n = {len(G)} (too few)'); continue
            n = np.array([len(r[lv]) for r in G], float); ik = np.array([ink(r[lv], W) for r in G]); H = np.array([r['H'] for r in G])
            b, lo, hi = elasticity(n, H); bi, loi, hii = elasticity(ik, H)
            dens = n / H; dink = ik / H
            rep.append(f'{name}: n = {len(G)}; H median {np.median(H):.1f} mm (IQR {np.percentile(H,25):.1f}-{np.percentile(H,75):.1f}); signs/mm median {np.median(dens):.3f}; ink/mm median {np.median(dink):.3f}; '
                       f'b(log H ~ log n) = {b:.2f} [{lo:.2f},{hi:.2f}]; b(log H ~ log ink) = {bi:.2f} [{loi:.2f},{hii:.2f}]; rho(n, signs/mm) = {spearman(n, dens):.2f}; rho(n, ink/mm) = {spearman(n, dink):.2f}')
            # density by length bins
            bins = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 20)]
            parts = []
            for a, c in bins:
                m = (n >= a) & (n <= c)
                if m.sum() >= 8: parts.append(f'L{a}-{c}: H {np.median(H[m]):.1f} mm, ink/mm {np.median(dink[m]):.3f} (n {m.sum()})')
            rep.append('   ' + '; '.join(parts))
            res[(lv, name)] = (b, lo, hi, bi, loi, hii, len(G))
        # H vs V: do longer texts get wider (not taller) seals? aspect ratio H/V vs n
        G = [r for r in rows if r['type'].startswith('SEAL') and r['nlines'] == 1 and r['complete'] and not r['broken'] and not r['has000']
             and len(r[lv]) >= 2 and r['H'] and r['V'] and 5 <= r['H'] <= 70 and all(t in W for t in r[lv]) and r['shape'] in ('square', 'rectangular')]
        n = np.array([len(r[lv]) for r in G], float); H = np.array([r['H'] for r in G]); V = np.array([r['V'] for r in G]); ik = np.array([ink(r[lv], W) for r in G])
        rep.append(f'square/rectangular seals with H and V ({len(G)}): rho(n, H/V) = {spearman(n, H/V):.3f}; rho(ink, H/V) = {spearman(ik, H/V):.3f}; b(log V ~ log n) = {elasticity(n, V, 200)[0]:.2f}; b(log H ~ log n) = {elasticity(n, H, 200)[0]:.2f}; share H/V > 1.1: {np.mean(H/V > 1.1):.3f}')
    open(OUT + 'loop38_c2.txt', 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    return res

# ================================================================== cycle 3
def final_rates(rows, lv, min_tokens=5):
    """Per sign: share of its tokens that stand last, over complete seal texts >= 2 signs (all sites)."""
    tot = collections.Counter(); fin = collections.Counter()
    for r in rows:
        if not r['type'].startswith('SEAL') or not r['complete'] or r['broken'] or r['has000']: continue
        s = r[lv]
        if len(s) < 2: continue
        for i, t in enumerate(s):
            tot[t] += 1
            if i == len(s) - 1: fin[t] += 1
    return tot, fin

def cycle3(rows, W):
    rep = [f'# LOOP 38 cycle 3: closer-less texts as truncations? ({time.strftime("%Y-%m-%dT%H:%M")})']
    rep.append('Sample: complete single-line seals, no unread sign, L >= 2, H known. Closer = last sign in the S289 set (jar W740, arrow W520, W156/527/617/226/390/405/154/158/15/254/12); variant B also counts a final opener (S286).')
    rep.append(f'Label permuted within site x length strata ({NPERM}x). Space-driven omission predicts: closer-less texts denser (ink/H higher), on smaller faces, ending in a normally non-final sign.')
    out = {}
    for lv in LEVELS:
        S = seal_sample(rows, lv, W, need_V=False)
        tot, fin = final_rates(rows, lv)
        rep.append(f'\n--- level {lv}: n = {len(S)}')
        for variant, clos in [('A', CLOSERS), ('B', CLOSERS | OPENERS)]:
            lab = np.array([0 if r[lv][-1] in clos else 1 for r in S])   # 1 = closer-less
            strata = [(r['site'], min(len(r[lv]), 8)) for r in S]
            H = np.array([r['H'] for r in S]); n = np.array([len(r[lv]) for r in S], float)
            ik = np.array([ink(r[lv], W) for r in S]); dens = ik / H
            area = np.array([np.log(r['H'] * r['V']) if r['V'] else np.nan for r in S])
            # final rate of the last sign, leave-one-out
            def lo_rate(r):
                t = r[lv][-1]; T = tot[t] - 1; F = fin[t] - 1
                return F / T if T >= 5 else np.nan
            fr = np.array([lo_rate(r) for r in S])
            # width of the last sign and 'last sign is wide'
            wl = np.array([W[r[lv][-1]] for r in S])
            feats = [('ink/H density', dens), ('face width H (mm)', H), ('log area H*V', area), ('width of last sign (jar 0.67, arrow 0.51; a class effect, listed for completeness)', wl)]
            rep.append(f'variant {variant}: closer-less {lab.sum()} vs closer-bearing {(1-lab).sum()}')
            # the ending of closer-less texts: (i) within closer-less texts, do crowded faces end in less-final signs (space truncation)?
            m1 = (lab == 1) & ~np.isnan(fr)
            rho_cr = spearman(dens[m1], fr[m1])
            idx1 = np.nonzero(m1)[0]; st1 = [strata[i] for i in idx1]
            def perm_within(x, st):
                g = collections.defaultdict(list)
                for i, s in enumerate(st): g[s].append(i)
                xp = x.copy()
                for ii in g.values():
                    ii = np.array(ii); xp[ii] = x[ii][rng.permutation(len(ii))]
                return xp
            nullc = np.array([spearman(perm_within(dens[m1], st1), fr[m1]) for _ in range(NPERM)])
            rep.append(f'  within closer-less texts ({m1.sum()}): rho(ink/H crowding, LOO final rate of the last sign) = {rho_cr:+.3f}; crowding permuted within site x length: [{np.percentile(nullc,2.5):+.3f},{np.percentile(nullc,97.5):+.3f}], P (one-sided, negative = crowded faces end in non-final signs) {pval(nullc, rho_cr, "less"):.3f}')
            rep.append(f'    mean LOO final rate of the last sign: closer-less {np.nanmean(fr[lab==1]):.3f} (crowded top quartile {np.nanmean(fr[(lab==1) & (dens > np.nanpercentile(dens,75))]):.3f}, rest {np.nanmean(fr[(lab==1) & (dens <= np.nanpercentile(dens,75))]):.3f}) vs closer-bearing {np.nanmean(fr[lab==0]):.3f} (the latter is high by construction)')
            out[(lv, variant, 'crowd-final')] = (rho_cr, pval(nullc, rho_cr, 'less'))
            # (ii) does the last sign of a closer-less text look like the pre-closer sign of a closer-bearing text, or like a random sign of the text?
            last_cl = collections.Counter(r[lv][-1] for r, l in zip(S, lab) if l == 1)
            pre_cb = collections.Counter(r[lv][-2] for r, l in zip(S, lab) if l == 0 and len(r[lv]) >= 3)
            rand_cl = collections.Counter(t for r, l in zip(S, lab) if l == 1 for t in r[lv][:-1])
            def cos(a, b):
                ks = set(a) | set(b); va = np.array([a.get(k, 0) for k in ks], float); vb = np.array([b.get(k, 0) for k in ks], float)
                return float(va @ vb / np.sqrt((va @ va) * (vb @ vb)))
            rep.append(f'  last sign of closer-less texts vs pre-closer sign of closer-bearing texts: cosine {cos(last_cl, pre_cb):.3f}; vs the non-final signs of the closer-less texts themselves: {cos(last_cl, rand_cl):.3f}; vs the pre-closer sign of closer-bearing texts, non-final signs: {cos(pre_cb, rand_cl):.3f}')
            rep.append('    commonest last signs of closer-less texts: ' + ', '.join(f'W{s} {c}' for s, c in last_cl.most_common(10)) + ' | commonest pre-closer signs: ' + ', '.join(f'W{s} {c}' for s, c in pre_cb.most_common(10)))
            for name, x in feats:
                ok = ~np.isnan(x)
                def stat(l, x=x, ok=ok):
                    return np.nanmean(x[ok & (l == 1)]) - np.nanmean(x[ok & (l == 0)])
                obs = stat(lab); null = strata_perm(lab[ok], [strata[i] for i in np.nonzero(ok)[0]], NPERM, lambda l, x=x[ok]: x[l == 1].mean() - x[l == 0].mean())
                rep.append(f'  {name}: closer-less {np.nanmean(x[ok & (lab==1)]):.3f} vs closer-bearing {np.nanmean(x[ok & (lab==0)]):.3f}; diff {obs:+.3f}; null mean {null.mean():+.3f} [{np.percentile(null,2.5):+.3f},{np.percentile(null,97.5):+.3f}]; P two-sided {pval(null, obs, "two"):.3f}')
                out[(lv, variant, name)] = (obs, pval(null, obs, 'two'), null.mean())
            # among closer-less texts: does density exceed the closer-bearing texts of the same length AND site, in the crowded tail?
            q = np.nanpercentile(dens, 75)
            crowded = dens > q
            obs = lab[crowded].mean() - lab[~crowded].mean()
            null = strata_perm(lab, strata, NPERM, lambda l: l[crowded].mean() - l[~crowded].mean())
            rep.append(f'  closer-less share on crowded faces (ink/H above the 75th pct) {lab[crowded].mean():.3f} vs rest {lab[~crowded].mean():.3f}; diff {obs:+.3f}; P {pval(null, obs, "two"):.3f}')
        # wrapped (two-line) seals: the scribe who ran out of room and continued below
        Wr = [r for r in rows if r['type'].startswith('SEAL') and r['nlines'] >= 2 and r['complete'] and not r['broken'] and not r['has000'] and len(r[lv]) >= 2 and r['dir'] not in ('BUS', 'T/B', 'SYM')]
        Sg = [r for r in rows if r['type'].startswith('SEAL') and r['nlines'] == 1 and r['complete'] and not r['broken'] and not r['has000'] and len(r[lv]) >= 2 and r['dir'] not in ('BUS', 'T/B', 'SYM')]
        both = Wr + Sg; labw = np.array([1] * len(Wr) + [0] * len(Sg))
        strata = [(r['site'], min(len(r[lv]), 8)) for r in both]
        def seg_ends(r):
            """Last sign of each '/' segment in reading order (segment order itself unknown, S-DARK-8.2)."""
            core = r['text'].strip('+[] ').split('/')
            segs = [[int(t) for t in l.split('-') if t.strip().isdigit()] for l in core]
            segs = [[maps_lv.get(t, t) for t in reversed(s)] for s in segs if s]
            return [s[-1] for s in segs]
        maps_lv = {}
        for r in rows:   # recover this level's merge map from the rows (raw -> level)
            for a, b in zip(r['seq_raw'], r[lv]):
                if a != b: maps_lv[a] = b
        clos_rate = np.array([1.0 if r[lv][-1] in CLOSERS else 0.0 for r in both])
        clos_any = np.array([1.0 if any(e in CLOSERS for e in seg_ends(r)) else 0.0 for r in both])
        Lw = np.array([len(r[lv]) for r in Wr]); Ls = np.array([len(r[lv]) for r in Sg])
        rep.append(f'wrapped two-line seals: {len(Wr)} (L mean {Lw.mean():.2f}) vs single-line {len(Sg)} (L mean {Ls.mean():.2f})')
        for nm, cr in [('closer = last sign in the stored order', clos_rate), ('closer at the end of either segment', clos_any)]:
            obs = cr[labw == 1].mean() - cr[labw == 0].mean()
            null = strata_perm(labw, strata, NPERM, lambda l, cr=cr: cr[l == 1].mean() - cr[l == 0].mean())
            rep.append(f'  {nm}: wrapped {cr[labw==1].mean():.3f} vs single {cr[labw==0].mean():.3f}; length-and-site-matched diff {obs:+.3f} [null {np.percentile(null,2.5):+.3f},{np.percentile(null,97.5):+.3f}], P {pval(null, obs, "two"):.3f}')
        obs = clos_any[labw == 1].mean() - clos_any[labw == 0].mean()
        null = strata_perm(labw, strata, NPERM, lambda l: clos_any[l == 1].mean() - clos_any[l == 0].mean())
        hw = [r['H'] for r in Wr if r['H'] and 5 <= r['H'] <= 70]; hs = [r['H'] for r in Sg if r['H'] and 5 <= r['H'] <= 70]
        # H of wrapped vs single at matched length
        bothH = [r for r in both if r['H'] and 5 <= r['H'] <= 70]; labH = np.array([1 if r['nlines'] >= 2 else 0 for r in bothH]); Hb = np.array([r['H'] for r in bothH])
        stH = [(r['site'], min(len(r[lv]), 8)) for r in bothH]
        obsH = Hb[labH == 1].mean() - Hb[labH == 0].mean(); nullH = strata_perm(labH, stH, NPERM, lambda l: Hb[l == 1].mean() - Hb[l == 0].mean())
        rep.append(f'  face width H: wrapped {np.mean(hw):.1f} mm (n {len(hw)}) vs single {np.mean(hs):.1f}; matched diff {obsH:+.2f} mm, null [{np.percentile(nullH,2.5):+.2f},{np.percentile(nullH,97.5):+.2f}], P {pval(nullH, obsH, "two"):.3f}')
        # where does the break fall? share of wrapped seals whose first line ends in a closer-class sign / whose second line is a closer unit
        ends_first = collections.Counter()
        for r in Wr:
            e = seg_ends(r); k = sum(1 for x in e if x in CLOSERS)
            ends_first[f'{k} segment end(s) in a closer'] += 1
            core = [s for s in r['text'].strip('+[] ').split('/') if s]
            ends_first['segment sizes ' + '/'.join(str(len([t for t in s.split("-") if t.strip().isdigit()])) for s in core)] += 1
        rep.append('  wrapped seals, segment ends and sizes (stored order): ' + str(ends_first.most_common(14)))
        out[(lv, 'wrapped')] = (obs, pval(null, obs, 'two'), obsH, pval(nullH, obsH, 'two'), len(Wr))

        # wide signs avoided on narrow faces? per-sign mean residual H (log H on log n, within site) vs sign width
        Sv = seal_sample(rows, lv, W, need_V=False)
        n = np.array([len(r[lv]) for r in Sv], float); H = np.log(np.array([r['H'] for r in Sv]))
        res = H.copy()
        for site in set(r['site'] for r in Sv):
            m = np.array([r['site'] == site for r in Sv])
            if m.sum() >= 10:
                b = np.polyfit(np.log(n[m]), H[m], 1); res[m] = H[m] - np.polyval(b, np.log(n[m]))
            else: res[m] = H[m] - H.mean()
        cnt = collections.Counter(t for r in Sv for t in set(r[lv]))
        signs = [s for s, c in cnt.items() if c >= 20]
        eff = []
        for s in signs:
            m = np.array([s in r[lv] for r in Sv]); eff.append(res[m].mean() - res[~m].mean())
        eff = np.array(eff); wv = np.array([W[s] for s in signs])
        obs = spearman(wv, eff)
        null = np.array([spearman(wv[rng.permutation(len(wv))], eff) for _ in range(NPERM)])
        rep.append(f'wide signs on narrow faces: {len(signs)} signs with >= 20 texts; rho(sign width, mean residual log H of texts containing it) = {obs:+.3f}; width-shuffle null [{np.percentile(null,2.5):+.3f},{np.percentile(null,97.5):+.3f}], P (one-sided, positive = wide signs on wider faces) {pval(null, obs, "greater"):.3f}')
        top = sorted(zip(eff, signs, wv), reverse=True)[:6]; bot = sorted(zip(eff, signs, wv))[:6]
        rep.append('  signs on the widest faces (residual, width): ' + ', '.join(f'W{s} {e:+.3f} ({w:.2f})' for e, s, w in top) + ' | narrowest: ' + ', '.join(f'W{s} {e:+.3f} ({w:.2f})' for e, s, w in bot))
        for s, name in [(220, 'plain fish'), (240, 'whisker fish'), (235, 'hat fish'), (740, 'jar'), (2, 'tall 2'), (817, 'opener 817'), (861, 'opener 861'), (90, 'man W90 (M1)'), (33, 'tall 3'), (400, 'W400'), (125, 'W125 widest')]:
            if s in signs:
                i = signs.index(s); rep.append(f'  W{s} {name}: width {wv[i]:.2f}, residual log H {eff[i]:+.3f} ({cnt[s]} texts)')
        out[(lv, 'wide')] = (obs, pval(null, obs, 'greater'))
    open(OUT + 'loop38_c3.txt', 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    return out

# ================================================================== cycle 4
def cycle4(rows, W, merges):
    rep = [f'# LOOP 38 cycle 4: do narrow allographs cluster at the ends of crowded texts, and does the first sign\'s width predict the last sign\'s? ({time.strftime("%Y-%m-%dT%H:%M")})']
    out = {}
    # (a) allograph pairs: form vs into (seq_raw only, by construction)
    pairs = [(m['form'], m['into'], m['level']) for m in merges if m['form'] in W and m['into'] in W]
    rep.append(f'allograph pairs with both glyphs in the font: {len(pairs)} (levels: {collections.Counter(p[2] for p in pairs)})')
    S = [r for r in rows if r['type'].startswith('SEAL') and r['nlines'] == 1 and r['complete'] and not r['broken'] and not r['has000'] and len(r['seq_raw']) >= 2 and r['dir'] not in ('BUS', 'T/B', 'SYM') and all(t in W for t in r['seq_raw'])]
    SH = [r for r in S if r['H'] and 5 <= r['H'] <= 70]
    toks = []   # (pair idx, narrow?, final?, crowding, site, L)
    for pi, (f, into, lvl) in enumerate(pairs):
        wf, wi = W[f], W[into]
        if abs(wf - wi) < 0.05: continue
        narrow_sign = f if wf < wi else into
        for r in SH:
            s = r['seq_raw']
            for i, t in enumerate(s):
                if t in (f, into):
                    others = sum(W[u] for j, u in enumerate(s) if j != i)
                    toks.append((pi, int(t == narrow_sign), int(i == len(s) - 1), others / r['H'], r['site'], len(s), r['H']))
    rep.append(f'tokens of a variant pair on single-line seals with H: {len(toks)} over {len({t[0] for t in toks})} pairs (pairs where the two variants differ in width by >= 0.05)')
    pc = collections.Counter((t[0]) for t in toks)
    rep.append('pair token counts: ' + ', '.join(f'W{pairs[p][0]}/W{pairs[p][1]} {pairs[p][2][:4]} ({W[pairs[p][0]]:.2f} vs {W[pairs[p][1]]:.2f}) n={c}' for p, c in pc.most_common(12)))
    if toks:
        narrow = np.array([t[1] for t in toks]); final = np.array([t[2] for t in toks]); crowd = np.array([t[3] for t in toks]); pair = np.array([t[0] for t in toks])
        # test 1: narrow variant more often final?
        obs1 = narrow[final == 1].mean() - narrow[final == 0].mean() if final.sum() and (1 - final).sum() else np.nan
        null1 = strata_perm(narrow, pair, NPERM, lambda l: l[final == 1].mean() - l[final == 0].mean())
        rep.append(f'narrow-variant share at the end {narrow[final==1].mean():.3f} (n {final.sum()}) vs elsewhere {narrow[final==0].mean():.3f} (n {(1-final).sum()}); diff {obs1:+.3f}; within-pair shuffle null [{np.percentile(null1,2.5):+.3f},{np.percentile(null1,97.5):+.3f}]; P two-sided {pval(null1, obs1, "two"):.3f}')
        # test 2: narrow variant on more crowded faces?
        obs2 = crowd[narrow == 1].mean() - crowd[narrow == 0].mean()
        null2 = strata_perm(narrow, pair, NPERM, lambda l: crowd[l == 1].mean() - crowd[l == 0].mean())
        rep.append(f'crowding (ink of the other signs / H) for narrow variants {crowd[narrow==1].mean():.4f} vs wide {crowd[narrow==0].mean():.4f}; diff {obs2:+.4f}; null [{np.percentile(null2,2.5):+.4f},{np.percentile(null2,97.5):+.4f}]; P {pval(null2, obs2, "two"):.3f}')
        # test 3: interaction, final tokens on crowded faces
        m = final == 1
        if m.sum() >= 20:
            obs3 = crowd[m & (narrow == 1)].mean() - crowd[m & (narrow == 0)].mean() if (m & (narrow == 1)).sum() and (m & (narrow == 0)).sum() else np.nan
            null3 = strata_perm(narrow[m], pair[m], NPERM, lambda l, c=crowd[m]: c[l == 1].mean() - c[l == 0].mean() if l.sum() and (1 - l).sum() else 0.0)
            rep.append(f'final tokens only ({m.sum()}): crowding narrow {crowd[m & (narrow==1)].mean():.4f} vs wide {crowd[m & (narrow==0)].mean():.4f}; diff {obs3:+.4f}; P {pval(null3, obs3, "two"):.3f}')
        out['allo'] = (obs1, pval(null1, obs1, 'two'), obs2, pval(null2, obs2, 'two'), len(toks))
        # per-pair table for the common ones
        for p, c in pc.most_common(8):
            m = pair == p
            rep.append(f'  W{pairs[p][0]}/W{pairs[p][1]}: narrow = W{pairs[p][0] if W[pairs[p][0]] < W[pairs[p][1]] else pairs[p][1]}; narrow share final {narrow[m & (final==1)].mean() if (m&(final==1)).sum() else float("nan"):.2f} (n {(m&(final==1)).sum()}) vs non-final {narrow[m & (final==0)].mean() if (m&(final==0)).sum() else float("nan"):.2f} (n {(m&(final==0)).sum()}); crowding narrow {crowd[m&(narrow==1)].mean() if (m&(narrow==1)).sum() else float("nan"):.3f} vs wide {crowd[m&(narrow==0)].mean() if (m&(narrow==0)).sum() else float("nan"):.3f}')
    # (b) first-sign width vs last-sign width, all three levels
    for lv in LEVELS:
        G = seal_sample(rows, lv, W, need_V=False)
        G = [r for r in G if len(r[lv]) >= 3]
        wf = np.array([W[r[lv][0]] for r in G]); wl = np.array([W[r[lv][-1]] for r in G]); n = np.array([len(r[lv]) for r in G], float)
        H = np.array([r['H'] for r in G]); mid = np.array([sum(W[t] for t in r[lv][1:-1]) for r in G])
        # residualise last width on n, H, mid ink
        X = np.column_stack([np.ones(len(G)), np.log(n), np.log(H), mid]); b, *_ = np.linalg.lstsq(X, wl, rcond=None); rl = wl - X @ b
        b2, *_ = np.linalg.lstsq(X, wf, rcond=None); rf = wf - X @ b2
        obs = spearman(rf, rl)
        # null A: shuffle widths among signs (keeps sign identities and the frame; destroys width meaning)
        signs = sorted({t for r in G for t in r[lv]}); wv = np.array([W[s] for s in signs]); nullA = []
        for _ in range(NPERM):
            perm = dict(zip(signs, wv[rng.permutation(len(wv))]))
            pf = np.array([perm[r[lv][0]] for r in G]); pl = np.array([perm[r[lv][-1]] for r in G]); pm = np.array([sum(perm[t] for t in r[lv][1:-1]) for r in G])
            Xp = np.column_stack([np.ones(len(G)), np.log(n), np.log(H), pm])
            bb, *_ = np.linalg.lstsq(Xp, pl, rcond=None); bb2, *_ = np.linalg.lstsq(Xp, pf, rcond=None)
            nullA.append(spearman(pf - Xp @ bb2, pl - Xp @ bb))
        nullA = np.array(nullA)
        # also: last-sign width vs crowding by the preceding signs (ink before last / H)
        pre = np.array([sum(W[t] for t in r[lv][:-1]) for r in G]) / H
        rho_pre = spearman(pre, wl)
        nullB = []
        for _ in range(NPERM):
            perm = dict(zip(signs, wv[rng.permutation(len(wv))]))
            pl = np.array([perm[r[lv][-1]] for r in G]); pp = np.array([sum(perm[t] for t in r[lv][:-1]) for r in G]) / H
            nullB.append(spearman(pp, pl))
        nullB = np.array(nullB)
        rep.append(f'\n--- level {lv}: seals L >= 3 with H: {len(G)}. partial rho(first width, last width | n, H, middle ink) = {obs:+.3f}; width-shuffle null [{np.percentile(nullA,2.5):+.3f},{np.percentile(nullA,97.5):+.3f}], P two-sided {pval(nullA, obs, "two"):.3f}')
        rep.append(f'  rho(ink before the last sign / H, last-sign width) = {rho_pre:+.3f}; width-shuffle null [{np.percentile(nullB,2.5):+.3f},{np.percentile(nullB,97.5):+.3f}], P {pval(nullB, rho_pre, "two"):.3f} (squeezing predicts negative, beyond the frame null)')
        # last-sign width by closer class
        cl = np.array(['jar' if r[lv][-1] in JAR else 'arrow' if r[lv][-1] in ARROW else 'other' if r[lv][-1] in OTHER_CLOSERS else 'none' for r in G])
        rep.append('  last-sign width by class: ' + ', '.join(f'{c} {wl[cl==c].mean():.2f} (n {(cl==c).sum()})' for c in ('jar', 'arrow', 'other', 'none') if (cl == c).sum()))
        out[(lv, 'firstlast')] = (obs, pval(nullA, obs, 'two'), rho_pre, pval(nullB, rho_pre, 'two'), len(G))
    open(OUT + 'loop38_c4.txt', 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    return out

if __name__ == '__main__':
    cycles = [a for a in sys.argv[1:] if a.isdigit()] or ['1', '2', '3', '4']
    rows, merges = load_corpus()
    freq = collections.Counter(t for r in rows for t in r['seq_raw'] if t)
    W, Hrel, href, boxes = load_widths(freq)
    if '1' in cycles: cycle1(rows, W, Hrel, href, boxes, merges)
    if '2' in cycles: cycle2(rows, W)
    if '3' in cycles: cycle3(rows, W)
    if '4' in cycles: cycle4(rows, W, merges)
