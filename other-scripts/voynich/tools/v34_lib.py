"""v34 CROWDED LINES ABBREVIATE: shared code.

Crowding of a text line = how much room is left between the end of its last word and
the right margin of its page/column (slack, in glyph units of that page; negative =
overshoot). Tight lines (slack small or negative) are the crowded ones.
Sources (numbers only, no images in the repo):
  Voynich: data/derived/v18_words.json (per-word x-extents measured on Yale IIIF images
           of Q20 + f58 by the v18 pipeline) + raw ZL3b lines (uncertain spaces).
  Latin:   HTR-United CREMMA-Medieval-Lat ALTO (scratch copy), line extents from the
           annotated baselines, raw transcription with abbreviation marks kept.
  Print:   Project Gutenberg plain texts (greedy machine wrapping, no abbreviation).
"""
import os, re, json, unicodedata, glob
import numpy as np
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CKPT = os.path.join(DATA, 'v34_ckpt')
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
LATDIR = os.path.join(SCR, 'v18', 'lat', 'data')
EVA_MULTI = ['cth', 'ckh', 'cph', 'cfh', 'ch', 'sh']
MAXSLACK = 0.25  # lines ending more than 25% of the column width short are dropped (para ends)


def glyphs(w):
    out, i = [], 0
    while i < len(w):
        for m in EVA_MULTI:
            if w.startswith(m, i):
                out.append(m); i += len(m); break
        else:
            out.append(w[i]); i += 1
    return out


# ---------------------------------------------------------------- Voynich
def raw_zl():
    """(folio, n) -> raw line text (with ',' uncertain spaces)."""
    out = {}
    for ln in open(os.path.join(DATA, 'ZL3b-n.txt'), encoding='utf-8', errors='replace'):
        m = re.match(r'<(f\w+)\.(\d+),([^>]*)>\s+(.*)', ln)
        if m:
            t = re.sub(r'<[^>]*>', '', m.group(4)).strip()
            out[(m.group(1), int(m.group(2)))] = t
    return out


def voynich_lines(src='v18_words.json', maxcost=None, ends=True, maxext=5.0):
    pages = json.load(open(os.path.join(DATA, 'derived', src)))
    raw = raw_zl()
    E = json.load(open(os.path.join(CKPT, 'lineends.json'))) if ends else {}
    lines = []
    for p in pages:
        byl = {}
        for w in p['words']:
            byl.setdefault(w['li'], []).append(w)
        pl = []
        for li, ws in sorted(byl.items()):
            ws = sorted(ws, key=lambda w: w['k'])
            if len(ws) != ws[0]['nw'] or len(ws) < 3:
                continue
            words = [w['word'] for w in ws]
            g = [max(1, len(glyphs(x))) for x in words]
            r = raw.get((p['folio'], ws[0]['n']), '')
            x1 = [w['x1'] for w in ws]
            e = E.get(p['folio'], {}).get(str(ws[0]['n'])) if ends else None
            if ends:
                if e is None or e.get('edge') or e['xend'] - x1[-1] > maxext * e['unit'] or e['xend'] <= ws[-1]['x0']:
                    continue
                x1[-1] = e['xend']
            pl.append({'page': p['folio'], 'n': ws[0]['n'], 'words': words, 'g': g,
                       'x0': [w['x0'] for w in ws], 'x1': x1,
                       'para_end': ws[0]['para_end'], 'para_start': ws[0]['para_start'],
                       'lcost': ws[0]['lcost'], 'raw': r})
        finish_page(pl)
        lines += pl
    return lines


def finish_page(pl, key_end='x1'):
    """Right margin R = 90th pct of line ends (non-para-end lines); unit = px per glyph."""
    if len(pl) < 5:
        for l in pl:
            l['ok'] = False
        return
    ends = np.array([l['x1'][-1] for l in pl if not l.get('para_end')])
    if len(ends) < 5:
        ends = np.array([l['x1'][-1] for l in pl])
    R = float(np.percentile(ends, 90))
    L = float(np.percentile([l['x0'][0] for l in pl], 50))
    us = [(l['x1'][-1] - l['x0'][0]) / (sum(l['g']) + 0.6 * (len(l['g']) - 1)) for l in pl if len(l['g']) >= 5]
    u = float(np.median(us)) if us else 1.0
    for l in pl:
        l['R'], l['L'], l['unit'] = R, L, u
        l['slack'] = (R - l['x1'][-1]) / u
        l['room'] = (R - l['x1'][-2]) / u if len(l['x1']) > 1 else np.nan  # room left for last word
        wl = l['x1'][-1] - l['x0'][-1]
        oth = [(b - a) / g for a, b, g in zip(l['x0'][:-1], l['x1'][:-1], l['g'][:-1])]
        l['comp'] = (wl / l['g'][-1]) / np.median(oth) if oth else np.nan  # last-word px/glyph vs rest
        l['ok'] = (not l.get('para_end')) and l['slack'] < MAXSLACK * (R - L) / u


# ---------------------------------------------------------------- Latin (CREMMA ALTO)
NS = '{http://www.loc.gov/standards/alto/ns-v4#}'
ABBR_LETTERS = set('ꝑꝓꝗꝙꝯꝰꝵẜħłđ⁊ꝝꝛꝫꝭꝱꝸꝶꝷ÷')


def abbr_count(tok):
    d = unicodedata.normalize('NFD', tok)
    return sum(1 for c in d if unicodedata.category(c) in ('Mn', 'Co') or c in ABBR_LETTERS)


def base_letters(tok):
    d = unicodedata.normalize('NFD', tok)
    return sum(1 for c in d if unicodedata.category(c)[0] in 'LN' and c not in ABBR_LETTERS)


def lat_tokens(raw):
    """split on spaces and on punctuation other than abbreviation signs."""
    t = ''.join(' ' if (unicodedata.category(c)[0] in 'Z' or (unicodedata.category(c)[0] == 'P' and c not in '⁊÷'))
                else c for c in raw)
    return [w for w in t.split() if w]


def latin_lines():
    lines = []
    for xmlf in sorted(glob.glob(os.path.join(LATDIR, '*', '*.xml'))):
        if 'chocomufin' in xmlf:
            continue
        root = ET.parse(xmlf).getroot()
        tags = {t.get('ID'): t.get('LABEL') for t in root.iter(NS + 'OtherTag')}
        ms = os.path.basename(os.path.dirname(xmlf))
        for bi, tb in enumerate(root.iter(NS + 'TextBlock')):
            if 'Main' not in tags.get(tb.get('TAGREFS'), ''):
                continue
            pl = []
            for tl in tb.iter(NS + 'TextLine'):
                if 'Default' not in tags.get(tl.get('TAGREFS'), 'Default'):
                    continue
                bl = [int(v) for v in tl.get('BASELINE', '').split()]
                s = tl.find(NS + 'String')
                if s is None or len(bl) < 4:
                    continue
                raw = s.get('CONTENT', '')
                toks = lat_tokens(raw)
                if len(toks) < 3:
                    continue
                xs = bl[::2]
                x0, x1 = min(xs), max(xs)
                g = [max(1, base_letters(t) + abbr_count(t)) for t in toks]
                pl.append({'page': ms + ':' + os.path.basename(xmlf)[:-4] + ':' + str(bi), 'ms': ms,
                           'words': toks, 'g': g, 'x0': [x0] * len(toks), 'x1': [x1] * len(toks),
                           'raw': raw, 'para_end': False})
            # crude per-word extents are not needed for slack; comp is undefined here
            finish_page(pl)
            for l in pl:
                l['comp'] = np.nan; l['room'] = np.nan
            lines += pl
    return lines


# ---------------------------------------------------------------- printed (Gutenberg)
def print_lines(fn, chunk=40):
    txt = open(os.path.join(DATA, fn), encoding='utf-8', errors='replace').read().split('\n')
    s = [i for i, l in enumerate(txt) if '*** START' in l]
    e = [i for i, l in enumerate(txt) if '*** END' in l]
    txt = txt[(s[0] + 1 if s else 0):(e[0] if e else len(txt))]
    W = int(np.percentile([len(l) for l in txt if len(l) > 20], 99))
    out = []
    for i in range(len(txt) - 1):
        l = txt[i].rstrip()
        if len(l) < 30 or not txt[i + 1].strip() or l.startswith(' '):
            continue
        toks = l.split()
        if len(toks) < 3:
            continue
        nxt = txt[i + 1].split()
        out.append({'page': fn + ':' + str(i // chunk), 'words': [re.sub(r'\W', '', t) or t for t in toks],
                    'g': [len(t) for t in toks], 'slack': float(W - len(l)), 'ok': True,
                    'next_first': len(nxt[0]) if nxt else 0, 'raw': l})
    return out


# ---------------------------------------------------------------- statistics
def rankz(x):
    from scipy.stats import rankdata
    return (rankdata(x) - (len(x) + 1) / 2) / len(x)


def strat_perm_corr(t, f, groups, nperm=2000, rng=None, covar=None):
    """Spearman-type corr of tightness t with feature f; null = permute t within groups.
    covar: optional matrix of covariates; f and t are residualised on them (ranks)."""
    rng = rng or np.random.default_rng(0)
    t = rankz(np.asarray(t, float)); f = rankz(np.asarray(f, float))
    if covar is not None:
        C = np.column_stack([np.ones(len(f))] + [rankz(c) for c in np.atleast_2d(covar)])
        f = f - C @ np.linalg.lstsq(C, f, rcond=None)[0]
    groups = np.asarray(groups)
    idx = [np.where(groups == gname)[0] for gname in np.unique(groups)]
    res = (lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]) if covar is not None else (lambda v: v)
    obs = float(np.corrcoef(res(t), f)[0, 1]) if f.std() > 0 else 0.0
    null = np.empty(nperm)
    tp = t.copy()
    for k in range(nperm):
        for ii in idx:
            tp[ii] = t[rng.permutation(ii)]
        null[k] = np.corrcoef(res(tp), f)[0, 1] if f.std() > 0 else 0
    z = (obs - null.mean()) / (null.std() + 1e-12)
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (nperm + 1)
    return obs, z, p, null


def multi_perm(t, feats, groups, nperm=2000, seed=0, covar=None):
    """Like strat_perm_corr for many features with the SAME within-group permutations of t,
    so a familywise (max |z|) p-value can be given. feats: dict name -> array."""
    rng = np.random.default_rng(seed)
    t = rankz(np.asarray(t, float))
    n = len(t)
    if covar is not None:
        C = np.column_stack([np.ones(n)] + [rankz(c) for c in np.atleast_2d(covar)])
        P = C @ np.linalg.pinv(C)
        res = lambda v: v - P @ v
    else:
        res = lambda v: v - v.mean()
    names = list(feats)
    Fm = np.column_stack([res(rankz(np.asarray(feats[k], float))) for k in names])
    sd = Fm.std(axis=0); sd[sd == 0] = np.inf
    Fm = Fm / sd
    groups = np.asarray(groups)
    idx = [np.where(groups == g)[0] for g in np.unique(groups)]

    def corr(tv):
        r = res(tv); r = r / (r.std() + 1e-12)
        return (r @ Fm) / n
    obs = corr(t)
    null = np.empty((nperm, len(names)))
    tp = t.copy()
    for k in range(nperm):
        for ii in idx:
            tp[ii] = t[rng.permutation(ii)]
        null[k] = corr(tp)
    mu, sdv = null.mean(0), null.std(0) + 1e-12
    z = (obs - mu) / sdv
    zn = np.abs((null - mu) / sdv).max(axis=1)
    out = {}
    for j, k in enumerate(names):
        p = (1 + np.sum(np.abs(null[:, j] - mu[j]) >= abs(obs[j] - mu[j]))) / (nperm + 1)
        pfw = (1 + np.sum(zn >= abs(z[j]))) / (nperm + 1)
        out[k] = {'r': float(obs[j]), 'z': float(z[j]), 'p': float(p), 'p_fw': float(pfw),
                  'mean': float(np.mean(feats[k]))}
    return out


def _v18_parse_raw(xmlf):
    """same line list as v18_latin_extract.parse, but keeping the raw transcription."""
    root = ET.parse(xmlf).getroot()
    tags = {t.get('ID'): t.get('LABEL') for t in root.iter(NS + 'OtherTag')}
    out = []
    for tb in root.iter(NS + 'TextBlock'):
        if 'Main' not in tags.get(tb.get('TAGREFS'), ''):
            continue
        for tl in tb.iter(NS + 'TextLine'):
            bl = [int(v) for v in tl.get('BASELINE', '').split()]
            s = tl.find(NS + 'String')
            if s is None or len(bl) < 4:
                continue
            raw = s.get('CONTENT', '')
            txt = ''.join(' ' if unicodedata.category(c)[0] in 'PZ' else (c if unicodedata.category(c)[0] in 'LN' else '') for c in raw)
            if [w for w in txt.split() if w]:
                out.append(raw)
    return out


def latin_img_lines(maxext=5.0):
    """CREMMA Latin through the Voynich image pipeline (v18 word extents + v34 line ends)."""
    pages = json.load(open(os.path.join(DATA, 'derived', 'v18_latin_words.json')))
    E = json.load(open(os.path.join(CKPT, 'lineends_lat.json')))
    lines = []
    for p in pages:
        ms, name = p['folio'].split(':', 1)
        raws = _v18_parse_raw(os.path.join(LATDIR, ms, name + '.xml'))
        byl = {}
        for w in p['words']:
            byl.setdefault(w['li'], []).append(w)
        pl = []
        for li, ws in sorted(byl.items()):
            ws = sorted(ws, key=lambda w: w['k'])
            e = E.get(p['folio'], {}).get(str(ws[0]['n']))
            if len(ws) != ws[0]['nw'] or len(ws) < 3 or e is None:
                continue
            x1 = [w['x1'] for w in ws]
            if e.get('edge') or e['xend'] - x1[-1] > maxext * e['unit'] or e['xend'] <= ws[-1]['x0']:
                continue
            x1[-1] = e['xend']
            raw = raws[ws[0]['n'] - 1] if ws[0]['n'] - 1 < len(raws) else ''
            toks = lat_tokens(raw)
            if not toks:
                continue
            pl.append({'page': p['folio'], 'ms': ms, 'n': ws[0]['n'], 'words': [w['word'] for w in ws],
                       'g': [max(1, len(w['word'])) for w in ws], 'x0': [w['x0'] for w in ws], 'x1': x1,
                       'raw': raw, 'rawtoks': toks, 'para_end': False, 'y': ws[0]['y']})
        # split into columns by line start x
        if not pl:
            continue
        xs = np.array([l['x0'][0] for l in pl]); o = np.sort(xs)
        cuts = [o[i + 1] for i in range(len(o) - 1) if o[i + 1] - o[i] > 300]
        for l in pl:
            c = sum(l['x0'][0] >= c for c in cuts)
            l['page'] = l['page'] + ':c%d' % c
        for c in sorted(set(l['page'] for l in pl)):
            sub = [l for l in pl if l['page'] == c]
            finish_page(sub)
            lines += sub
    return lines
