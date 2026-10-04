#!/usr/bin/env python3
"""LA-32 cycle 2b: modern sanity pair, redone. More OCR errors (several fonts, harsher degradation) and
partial scores (shape residualised on sound class, sound residualised on shape decile) because in English
letters vowels both look and sound alike (r 0.14)."""
import sys, os, json, difflib, subprocess, string
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la32_common import *
from PIL import ImageFilter
rng = np.random.default_rng(32022)
out = []
def log(s):
    print(s, flush=True); out.append(s)
letters = list(string.ascii_lowercase); iEN = {c: i for i, c in enumerate(letters)}
DJ = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
SM_EN, _ = shape_matrices(letters, {c: c for c in letters}, DJ)
VIS = combine(SM_EN, ['blur', 'chamfer', 'hog', 'prof'])
groups = ['aeiouy', 'ckqgx', 'szcx', 'td', 'pb', 'fvp', 'mn', 'lr', 'wuv', 'jg', 'h']
SND = np.zeros((26, 26))
for g in groups:
    for a in g:
        for b in g:
            if a != b: SND[iEN[a], iEN[b]] = 1
np.fill_diagonal(SND, np.nan)

def partial(A, Bm, nb=5):
    """A residualised on bins of Bm (within-bin centring)."""
    R = np.full(A.shape, np.nan); iu = np.triu_indices(A.shape[0], 1)
    a, b = A[iu], Bm[iu]; ok = ~np.isnan(a) & ~np.isnan(b)
    qs = np.unique(np.nanquantile(b[ok], np.linspace(0, 1, nb + 1)))
    bins = np.digitize(b, qs[1:-1]) if len(qs) > 2 else np.zeros_like(b, int)
    r = np.full(a.shape, np.nan)
    for k in np.unique(bins[ok]):
        m = ok & (bins == k); r[m] = a[m] - a[m].mean()
    R[iu] = r; R.T[iu] = r
    return R

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
            d = onechar(rr.strip().lower(), w.strip().lower())   # (intended, written)
            if d: mis.append(dict(a=d[0], b=d[1]))
ocr_p = os.path.join(CK, 'ocr_pairs2.json')
if not os.path.exists(ocr_p):
    txt = open(os.path.join(CK, 'gutenberg.txt'), encoding='utf8').read().split()[3000:3000 + 10000]
    fonts = ['/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    pairs = []
    for k in range(0, len(txt), 500):
        seg = txt[k:k + 500]; lines = []; cur = ''
        for w in seg:
            if len(cur) + len(w) > 70: lines.append(cur); cur = w
            else: cur = (cur + ' ' + w).strip()
        lines.append(cur)
        f = ImageFont.truetype(fonts[(k // 500) % 2], 12)
        im = Image.new('L', (620, 16 * len(lines) + 20), 255); dr = ImageDraw.Draw(im)
        for i, ln in enumerate(lines): dr.text((8, 8 + 16 * i), ln, fill=0, font=f)
        sc, nz = [(1.5, 20), (1.8, 10), (1.8, 20)][(k // 500) % 3]
        im = im.resize((int(im.width / sc), int(im.height / sc)), Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.5))
        a = np.array(im, float) + rng.normal(0, nz, (im.height, im.width))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).resize((int(im.width * sc * 1.3), int(im.height * sc * 1.3)), Image.BILINEAR)
        p = os.path.join(CK, 'ocr_tmp.png'); im.save(p)
        o = subprocess.run(['tesseract', p, '-', '--psm', '6'], capture_output=True, text=True,
                           env=dict(os.environ, OMP_THREAD_LIMIT='1')).stdout
        gt = [w.lower().strip(string.punctuation) for w in ' '.join(lines).split()]
        rd = [w.lower().strip(string.punctuation) for w in o.split()]
        sm = difflib.SequenceMatcher(a=gt, b=rd, autojunk=False)
        nerr = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'replace' and i2 - i1 == j2 - j1:
                for x, y in zip(gt[i1:i2], rd[j1:j2]):
                    d = onechar(x, y)
                    if d: pairs.append(d); nerr += 1
        print('chunk', k, 'scale', sc, 'one-letter errors', nerr, flush=True)
    json.dump(pairs, open(ocr_p, 'w'))
ocr = [dict(a=a, b=b) for a, b in json.load(open(ocr_p))]
VISp = partial(VIS, SND, 2); SNDp = partial(SND, VIS, 5)
log('# LA-32 cycle 2b: English sanity pair (human misspellings vs OCR)')
log(f'misspellings (Wikipedia list, one-letter) {len(mis)}; OCR one-letter errors {len(ocr)}')
for tag, E in (('misspellings', mis), ('OCR', ocr)):
    for name, S in (('shape', VIS), ('sound', SND), ('shape|sound', VISp), ('sound|shape', SNDp)):
        log(f'  {tag:13s} {name:12s} {fmt(score_edges(E, S, iEN, 3000, rng))}')
    for n in (50, 163):
        if n > len(E): continue
        acc = []
        for rep in range(100):
            sub = [E[i] for i in rng.choice(len(E), n, replace=False)]
            acc.append(score_edges(sub, SNDp, iEN, 300, rng)['z'] > score_edges(sub, VISp, iEN, 300, rng)['z'])
        log(f'  {tag:13s} n={n}: partial sound z > partial shape z in {np.mean(acc):.2f} of 100 subsamples')
c = collections.Counter((e['a'], e['b']) for e in ocr).most_common(10); log('  top OCR: ' + ' '.join(f'{a}>{b} {n}' for (a, b), n in c))
c = collections.Counter((e['a'], e['b']) for e in mis).most_common(10); log('  top misspellings: ' + ' '.join(f'{a}>{b} {n}' for (a, b), n in c))
open(os.path.join(CK, 'c2b_report.txt'), 'w').write('\n'.join(out))
