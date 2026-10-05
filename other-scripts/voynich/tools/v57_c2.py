"""v57 cycle 2: WHERE THE PEN RAN DRY, at sub-word resolution.
Per text line, a column-by-column ink-darkness trace (mean contrast of ink pixels in a band
around the line track built from the v18 word boxes). Step detector: median darkness in the
next w px minus the previous w px. A re-dip is a sudden DARKENING; content changes make
darkening and lightening steps equally often, so the signal is the sign asymmetry
A = (dark - light) / (dark + light), separately at word gaps and inside words.
Null: each line's trace mirrored at random (sign flip of its dark-light balance).
Control: Latin (CREMMA) pages; also whether Latin darkenings sit at punctuation.
Images stay in the scratch cache; per-event rows go to data/v57_ckpt/c2_events.json."""
import sys, os, json, re, unicodedata
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v18_lib as V18
from v57_lib import DER, CK, vglyphs, lglyphs

SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v18'
NS = '{http://www.loc.gov/standards/alto/ns-v4#}'


def latin_punct(folio):
    """Per (li, k): does punctuation follow word k of line li? Same tokenisation as v18."""
    fo, f = folio.split(':', 1)
    root = ET.parse(os.path.join(SCR, 'lat', 'data', fo, f + '.xml')).getroot()
    tags = {t.get('ID'): t.get('LABEL') for t in root.iter(NS + 'OtherTag')}
    out, li = {}, 0
    for tb in root.iter(NS + 'TextBlock'):
        if 'Main' not in tags.get(tb.get('TAGREFS'), ''):
            continue
        for tl in tb.iter(NS + 'TextLine'):
            bl = [int(v) for v in tl.get('BASELINE', '').split()]
            s = tl.find(NS + 'String')
            if s is None or len(bl) < 4:
                continue
            raw = s.get('CONTENT', '')
            toks = re.findall(r'(\w+)(\W*)', ''.join(c if unicodedata.category(c)[0] in 'LNPZ' else '' for c in raw))
            words = [(w, bool(re.search(r'[^\s]', p))) for w, p in toks if w]
            if not words:
                continue
            for k, (w, pu) in enumerate(words):
                out[(li, k)] = pu
            li += 1
    return out


def img_path(which, folio):
    if which == 'V':
        return os.path.join(SCR, 'img', folio + '.jpg')
    fo, f = folio.split(':', 1)
    return os.path.join(SCR, 'lat', 'data', fo, f + '.jpg')


def page(args):
    which, pg = args
    fn = img_path(which, pg['folio'])
    if which == 'L':
        im = Image.open(fn); sc = 2000 / im.width
        tmp = os.path.join(CK, 'tmp_%d.png' % os.getpid())
        im.convert('RGB').resize((2000, int(im.height * sc))).save(tmp)
        gray, con, rb = V18.load(tmp); os.remove(tmp)
        pun = latin_punct(pg['folio'])
    else:
        gray, con, rb = V18.load(fn); pun = {}
    ink = con > 0.08
    H = con.shape[0]
    s = pg['pitch']; h = max(3, int(0.3 * s))
    lines = {}
    for w in pg['words']:
        lines.setdefault(w['li'], []).append(w)
    gl = vglyphs if which == 'V' else lglyphs
    rows = []
    for li, ws in lines.items():
        ws = sorted(ws, key=lambda w: w['k'])
        if len(ws) < 3:
            continue
        xa, xb = ws[0]['x0'], ws[-1]['x1']
        if xb - xa < 60:
            continue
        cx = np.array([(w['x0'] + w['x1']) / 2 for w in ws]); cy = np.array([w['y'] for w in ws])
        xs = np.arange(xa, xb)
        ys = np.interp(xs, cx, cy).astype(int)
        rr = np.clip(ys[None, :] + np.arange(-h, h + 1)[:, None], 0, H - 1)
        c = con[rr, xs[None, :]]; m = ink[rr, xs[None, :]]
        cnt = m.sum(0)
        dark = np.where(cnt >= 2, (c * m).sum(0) / np.maximum(cnt, 1), np.nan)
        # glyph unit (px per glyph) for this line
        ng = sum(len(gl(w['word'])) for w in ws)
        unit = (xb - xa) / max(1, ng + 0.6 * (len(ws) - 1))
        rows.append({'folio': pg['folio'], 'li': li, 'xa': int(xa), 'unit': float(unit),
                     'dark': [None if np.isnan(v) else round(float(v), 4) for v in dark],
                     'words': [[w['word'], w['x0'] - xa, w['x1'] - xa, w['k']] for w in ws],
                     'punct': [bool(pun.get((li, w['k']), False)) for w in ws]})
    return rows


def extract(which):
    fn = os.path.join(CK, f'c2_traces_{which}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    pages = json.load(open(os.path.join(DER, 'v18_words.json' if which == 'V' else 'v18_latin_words.json')))
    with ProcessPoolExecutor(int(os.environ.get("NW", "1"))) as ex:
        res = [r for rows in ex.map(page, [(which, p) for p in pages]) for r in rows]
    json.dump(res, open(fn, 'w'))
    return res


def events(row, wg, q):
    """Step events on one line trace. wg = window in glyph units. Returns list of
    (x, sign, strength). Darkness is z-scored within the line."""
    d = np.array([np.nan if v is None else v for v in row['dark']])
    ok = ~np.isnan(d)
    if ok.sum() < 30:
        return []
    z = (d - np.nanmean(d)) / (np.nanstd(d) + 1e-9)
    w = max(4, int(wg * row['unit']))
    idx = np.where(ok)[0]; zv = z[ok]
    n = len(idx)
    st = np.full(len(d), np.nan)
    cs = np.cumsum(np.r_[0, zv])
    # windows in ink-column count (pen travel on ink), centred on each ink column
    for j in range(w, n - w):
        st[idx[j]] = (cs[j + w] - cs[j]) / w - (cs[j] - cs[j - w]) / w
    a = np.abs(np.nan_to_num(st))
    thr = np.quantile(a[a > 0], q) if (a > 0).sum() > 10 else np.inf
    out = []
    order = np.argsort(-a)
    taken = np.zeros(len(d), bool)
    for i in order:
        if a[i] < thr:
            break
        lo, hi = max(0, i - w), min(len(d), i + w)
        if taken[lo:hi].any():
            continue
        taken[lo:hi] = True
        out.append((int(i), int(np.sign(st[i])), float(a[i])))
    return out


def classify(row, x, tol):
    """'gap' (+ index of word before) if x is within tol px of a word boundary, else
    ('in', word index, relative position)."""
    ws = row['words']
    for k in range(len(ws) - 1):
        b = (ws[k][2] + ws[k + 1][1]) / 2
        if abs(x - b) <= tol:
            return ('gap', k, None)
    for k, (wd, a, b, _) in enumerate(ws):
        if a <= x < b:
            return ('in', k, (x - a) / max(1, b - a))
    return ('edge', -1, None)


def asym(rows, wg, q, tolg, rng, nnull=4000):
    per_line = []  # (dark_gap, light_gap, dark_in, light_in, dark_punctgap, light_punctgap)
    evs = []
    for row in rows:
        ev = events(row, wg, q)
        c = np.zeros(6)
        for x, sg, a in ev:
            cl = classify(row, x, tolg * row['unit'])
            if cl[0] == 'gap':
                c[0 if sg > 0 else 1] += 1
                if row['punct'][cl[1]]:
                    c[4 if sg > 0 else 5] += 1
            elif cl[0] == 'in':
                c[2 if sg > 0 else 3] += 1
            evs.append((row['folio'], row['li'], x, sg, a, cl))
        per_line.append(c)
    P = np.array(per_line)
    tot = P.sum(0)
    def A(t):
        return np.array([(t[0] - t[1]) / max(1, t[0] + t[1]), (t[2] - t[3]) / max(1, t[2] + t[3]),
                         (t[4] - t[5]) / max(1, t[4] + t[5])])
    obs = A(tot)
    # mirror null: flipping a line swaps its dark and light counts
    sw = P[:, [1, 0, 3, 2, 5, 4]]
    nul = []
    for _ in range(nnull):
        f = rng.random(len(P)) < 0.5
        nul.append(A(np.where(f[:, None], sw, P).sum(0)))
    nul = np.array(nul)
    z = (obs - nul.mean(0)) / (nul.std(0) + 1e-9)
    # share of events at gaps vs expected share of trace length within tol of a gap
    gapfrac = []
    for row in rows:
        L = len(row['dark']); tol = tolg * row['unit']
        gapfrac.append(min(1.0, 2 * tol * (len(row['words']) - 1) / max(1, L)))
    return dict(wg=wg, q=q, tolg=tolg, counts=tot.tolist(), A_gap=float(obs[0]), A_in=float(obs[1]),
                A_punct=float(obs[2]), z_gap=float(z[0]), z_in=float(z[1]), z_punct=float(z[2]),
                gap_share_events=float((tot[0] + tot[1]) / max(1, tot[:4].sum())),
                gap_share_expected=float(np.mean(gapfrac))), evs


if __name__ == '__main__':
    rng = np.random.default_rng(5702)
    out = {}
    for which in ['L', 'V']:
        rows = extract(which)
        print(which, 'lines', len(rows), 'punct-gaps', sum(sum(r['punct'][:-1]) for r in rows), flush=True)
        res = []
        for wg in [1.5, 3, 6]:
            for q in [0.8, 0.9, 0.97]:
                for tolg in [0.5, 1.0]:
                    r, evs = asym(rows, wg, q, tolg, rng, nnull=1000)
                    res.append(r)
                    print(which, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        out[which] = res
    json.dump(out, open(os.path.join(CK, 'c2.json'), 'w'), indent=1)
