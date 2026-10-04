#!/usr/bin/env python3
"""LA-32 cycle 2: controls that could kill the method.
(a) modern pair: English human misspellings (Wikipedia list) vs OCR errors (tesseract on degraded renders)
    scored against letter shape (rendered) and letter sound classes.
(b) planted dictation vs planted copying corpora, at LA size, on LB and on LA, with the generator's
    similarity different from the scorer's (sound: LB values -> scored by blind rows; shape: HOG -> scored by blur+chamfer+prof).
(c) how much of each slip kind could hide in real LA (slip fraction f where planted z reaches the real z)."""
import sys, os, json, pickle, difflib, subprocess, string
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la32_common import *
from PIL import ImageFilter

rng = np.random.default_rng(3202)
out = []
def log(s):
    print(s, flush=True); out.append(s)
M = pickle.load(open(os.path.join(CK, 'mats.pkl'), 'rb'))

# ============ (a) English
letters = list(string.ascii_lowercase); iEN = {c: i for i, c in enumerate(letters)}
DJ = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
SM_EN, _ = shape_matrices(letters, {c: c for c in letters}, DJ)
VIS_EN = combine(SM_EN, ['blur', 'chamfer', 'hog', 'prof'])
groups = ['aeiouy', 'ckqgx', 'szcx', 'td', 'pb', 'fvp', 'mn', 'lr', 'wuv', 'jg', 'h']
SND_EN = np.zeros((26, 26))
for g in groups:
    for a in g:
        for b in g:
            if a != b: SND_EN[iEN[a], iEN[b]] = 1
np.fill_diagonal(SND_EN, np.nan)
def onechar(a, b):
    if len(a) != len(b) or a == b: return None
    d = [(x, y) for x, y in zip(a, b) if x != y]
    if len(d) == 1 and d[0][0] in iEN and d[0][1] in iEN: return d[0]
    return None
mis = []
for l in open(os.path.join(CK, 'wiki_missp.txt'), encoding='utf8'):
    if '->' in l:
        w, r = l.strip().split('->', 1)
        for rr in r.split(','):
            d = onechar(w.strip().lower(), rr.strip().lower())
            if d: mis.append(dict(a=d[0], b=d[1]))
# OCR
ocr_p = os.path.join(CK, 'ocr_pairs.json')
if not os.path.exists(ocr_p):
    txt = open(os.path.join(CK, 'gutenberg.txt'), encoding='utf8').read()
    txt = ' '.join(txt.split()[3000:3000 + 9000])
    f = ImageFont.truetype(DJ, 13)
    pairs = []
    words = txt.split(); chunk = 300
    for k in range(0, len(words), chunk):
        seg = words[k:k + chunk]; lines = []; cur = ''
        for w in seg:
            if len(cur) + len(w) > 70: lines.append(cur); cur = w
            else: cur = (cur + ' ' + w).strip()
        lines.append(cur)
        im = Image.new('L', (640, 18 * len(lines) + 20), 255); dr = ImageDraw.Draw(im)
        for i, ln in enumerate(lines): dr.text((8, 8 + 18 * i), ln, fill=0, font=f)
        im = im.resize((im.width // 2, im.height // 2), Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.6))
        a = np.array(im, float) + rng.normal(0, 18, (im.height, im.width))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((im.width * 3, im.height * 3), Image.NEAREST)
        p = os.path.join(CK, 'ocr_tmp.png'); im.save(p)
        o = subprocess.run(['tesseract', p, '-', '--psm', '6'], capture_output=True, text=True,
                           env=dict(os.environ, OMP_THREAD_LIMIT='1')).stdout
        gt = [w.lower().strip(string.punctuation) for w in ' '.join(lines).split()]
        rd = [w.lower().strip(string.punctuation) for w in o.split()]
        sm = difflib.SequenceMatcher(a=gt, b=rd, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'replace' and i2 - i1 == j2 - j1:
                for x, y in zip(gt[i1:i2], rd[j1:j2]):
                    d = onechar(x, y)
                    if d: pairs.append(d)
    json.dump(pairs, open(ocr_p, 'w'))
ocr = [dict(a=a, b=b) for a, b in json.load(open(ocr_p))]
log('# LA-32 cycle 2: controls')
log(f'## (a) English: {len(mis)} one-letter human misspellings, {len(ocr)} one-letter OCR errors')
log(f'  shape vs sound class r = {np.corrcoef(VIS_EN[np.triu_indices(26,1)], SND_EN[np.triu_indices(26,1)])[0,1]:.3f}')
EN = {}
for tag, E in (('misspellings', mis), ('OCR', ocr)):
    for name, S in (('VIS', VIS_EN), ('SOUND', SND_EN)):
        r = score_edges(E, S, iEN, 3000, rng); EN[(tag, name)] = r
        log(f'  {tag:13s} {name:6s} {fmt(r)}')
    for n in (50, 163):
        acc = []
        for rep in range(100):
            sub = [E[i] for i in rng.choice(len(E), n, replace=False)]
            zv = score_edges(sub, VIS_EN, iEN, 300, rng)['z']; zs = score_edges(sub, SND_EN, iEN, 300, rng)['z']
            acc.append(zs > zv)
        log(f'  {tag:13s} subsample n={n}: called "sound" in {np.mean(acc):.2f} of 100')

# ============ (b) planted corpora
def plant(signs, src_words, n, kind, Sgen, idx, beta=12.0):
    """n slips: copy a random word token, replace one sign by a partner drawn from Sgen."""
    E = []
    tries = 0
    while len(E) < n and tries < 50 * n:
        tries += 1
        w = src_words[rng.integers(len(src_words))]
        p = rng.integers(len(w)); a = w[p]
        if a not in idx: continue
        row = Sgen[idx[a]].copy(); row[idx[a]] = np.nan
        ok = ~np.isnan(row)
        if ok.sum() < 5: continue
        if kind == 'sound':   # partner shares consonant (or vowel) under the generator's values
            cand = np.where(ok & (row > 0))[0]
            if len(cand) == 0: continue
            b = signs[cand[rng.integers(len(cand))]]
        else:
            pr = np.exp(beta * np.where(ok, row, -9)); pr[~ok] = 0; pr /= pr.sum()
            b = signs[rng.choice(len(signs), p=pr)]
        E.append(dict(a=a, b=b))
    return E

def rank_hog_only(SM):
    return rank_norm(SM['hog'])

def power(name, signs, idx, words_tok, bg, Gsnd, Gshp, Ssnd, Svis, sizes=(50, 163), fracs=(1.0, 0.5, 0.25), reps=60):
    res = {}
    for n in sizes:
        for f in fracs:
            for kind in ('sound', 'shape'):
                ns = int(round(n * f)); zs_, zv_ = [], []
                for r in range(reps):
                    E = plant(signs, words_tok, ns, kind, Gsnd if kind == 'sound' else Gshp, idx)
                    E = E + [bg[i] for i in rng.choice(len(bg), n - ns, replace=False)] if n > ns else E
                    zs_.append(score_edges(E, Ssnd, idx, 300, rng)['z']); zv_.append(score_edges(E, Svis, idx, 300, rng)['z'])
                zs_, zv_ = np.array(zs_), np.array(zv_)
                acc = np.mean(zs_ > zv_) if kind == 'sound' else np.mean(zv_ > zs_)
                res[(n, f, kind)] = (float(np.nanmean(zs_)), float(np.nanmean(zv_)), float(acc))
                log(f'  {name} n={n:3d} slip share {f:.2f} planted {kind:5s}: mean z sound {np.nanmean(zs_):5.2f}, shape {np.nanmean(zv_):5.2f}; correct call {acc:.2f}')
    return res

# LB: generator values CON|VOW (true), scorer blind rows; generator HOG, scorer blur+chamfer+prof
lb_signs = M['lb_signs']; iLB = {s: i for i, s in enumerate(lb_signs)}
GsLB = np.nan_to_num(M['CON_LB'], nan=0) + 0.0
GsLB[np.isnan(M['CON_LB'])] = np.nan
GhLB = rank_norm(M['SM_LB']['hog'])
SvLB = combine(M['SM_LB'], ['blur', 'chamfer', 'prof'])
B = lb_words(('KN',)); Btok = [r['w'] for r in B if len(r['w']) >= 3]
bgLB = one_sign_pairs(lb_words(), 3, 'site')
# background for a "no-slip" mix: pairs with random partners (pure noise), not real LB pairs (those are sound-laden)
def noise_bg(signs, toks, n):
    return [dict(a=t[rng.integers(len(t))], b=signs[rng.integers(len(signs))]) for t in (toks[i] for i in rng.integers(len(toks), size=n))]
log('\n## (b1) planted on Linear B words (background = random-partner pairs)')
PLB = power('LB', lb_signs, iLB, Btok, noise_bg(lb_signs, Btok, 400), GsLB, GhLB, M['SND_LB'], SvLB)

la_signs = M['la_signs']; iLA = {s: i for i, s in enumerate(la_signs)}
GsLA = M['CON_LA'].copy()
GhLA = rank_norm(M['SM_LA']['hog'])
SvLA = combine(M['SM_LA'], ['blur', 'chamfer', 'prof'])
WA = la_words(); Atok = [r['w'] for r in WA if len(r['w']) >= 2]
bgLA = one_sign_pairs(WA, 3, None)
log('\n## (b2) planted on Linear A words (background = the real LA one-sign pairs; generator LB-value consonants or HOG; scorer blind rows or other shape metrics)')
PLA = power('LA', la_signs, iLA, Atok, bgLA, GsLA, GhLA, M['SND_LA'], SvLA, sizes=(163,), fracs=(1.0, 0.5, 0.3, 0.2, 0.1))
# real LA under the same scorers
real = one_sign_pairs(WA, 3, None)
rz = (score_edges(real, M['SND_LA'], iLA, 3000, rng)['z'], score_edges(real, SvLA, iLA, 3000, rng)['z'])
log(f'  real LA (163 pairs) under the same scorers: z sound {rz[0]:.2f}, z shape(blur+chamfer+prof) {rz[1]:.2f}')
pickle.dump(dict(EN={f'{k[0]}|{k[1]}': v for k, v in EN.items()}, PLB={str(k): v for k, v in PLB.items()},
                 PLA={str(k): v for k, v in PLA.items()}, real=rz), open(os.path.join(CK, 'c2_res.pkl'), 'wb'))
open(os.path.join(CK, 'c2_report.txt'), 'w').write('\n'.join(out))
