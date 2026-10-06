"""v76 VOYNICH PARAGRAPHS AS DOSSIERS: shared library.

Idea: if pages are independent records (v56), records may be written to a template (recipe entries, herbal
entries). Paragraphs copied from one plan are a controlled experiment: aligned, what varies alone is the item
(identifier), what varies together is linked attributes, what never varies is the template.

Corpus format = v72 (pages -> lines dict(w, ps)). Paragraph objects (built after the v72 E1c extraction):
  dict(pid, page, sec, half, pidx (paragraph index on page), npp (paragraphs on page), lines=[[tok..]..], toks)
Controls (data only; nobody's reading used):
  BRU  Brumati Italian herbal (page = species/genus entry, paragraphs = its paragraphs)
  HIL  Hildegard, Physica excerpts (page = chapter, paragraphs = verse groups of ~40 words)
  API  Apicius, De re coquinaria (page = Roman-numeral subsection, paragraph = numbered recipe incl. title)
All three go through a lossy merge payload code + the planted v72 surface machinery, then frozen E1c.
"""
import os, sys, re, json, math, random, html, glob, unicodedata, hashlib, collections
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v72_lib as L
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v76_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
SRC = os.path.join(ROOT, 'data', 'v58_ckpt', 'src')
RULE = 'E1c_keepd'
assert L.rule_hash(RULE) == '2803cbeb0deb51bf'


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def _norm(s):
    s = unicodedata.normalize('NFD', s.lower()); s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s.replace('j', 'i').replace('v', 'u')


def _W(s):
    return [w for w in re.findall(r'[a-z]+', _norm(s))]


# ------------------------------------------------------------------ control texts as (page_sec, [paragraph word lists])
def bru_records():
    import v49_lib
    ents = v49_lib.brumati_entries()
    order = {}
    for e in ents: order.setdefault(e['cls'], len(order))
    R = []
    for e in ents:
        paras = []
        for pi, p in enumerate(e['paras']):
            p = re.sub(r'\[p\. *\d+[^\]]*\]', ' ', p)
            if pi == 0: p = re.sub(r'^(\d+a?\.?|[IVXLC]+°?\.)\s', '', p)
            ws = v49_lib.brumati_words(p)
            if len(ws) >= 3: paras.append(ws)
        if paras and sum(map(len, paras)) >= 8: R.append(('G%d' % min(5, order[e['cls']] // 4), paras))
    return R


def hil_records():
    R = []
    for i in range(1, 6):
        t = open(os.path.join(SRC, 'hildegard_physica', 'hil_phy%d.html' % i), encoding='latin-1').read()
        t = re.sub(r'<[^>]+>', '', t); t = html.unescape(t)
        t = re.sub(r'\{[^}]*\}', ' ', t)
        chs = re.split(r'Capitulum [IVXLC]+\.', t)[1:]
        for ch in chs:
            vs = [v for v in re.split(r'\[\d+\]', ch) if v.strip()]
            if len(vs) < 2: continue
            title = _W(vs[0])
            paras, cur = [], title[:]
            for v in vs[1:]:
                cur += _W(v)
                if len(cur) >= 40: paras.append(cur); cur = []
            if cur: paras.append(cur)
            paras = [p for p in paras if len(p) >= 3]
            if paras: R.append(('H%d' % i, paras))
    return R


def cul_records(maxw=60, maxp=4):
    t = open(os.path.join(SRC, 'culpeper.txt'), encoding='utf-8', errors='replace').read()
    t = t.replace('\u2019', "'")
    blocks = re.split(r'\n\s*\n', t)
    R, cur, started = [], None, False
    for b in blocks:
        bs = b.strip()
        if re.fullmatch(r"[A-Z][A-Z ,'\-]{2,40}\.", bs) and b.startswith('    '):
            started = True
            if cur and cur[1]: R.append(cur)
            cur = ('C%d' % min(3, len(R) // 90), [_W(bs)])
            continue
        if not started or cur is None: continue
        ws = _W(bs)
        if len(ws) >= 4: cur[1].append(ws[:maxw])
    if cur and cur[1]: R.append(cur)
    # title joins the first paragraph
    out = []
    for sec, ps in R:
        if len(ps) >= 2: out.append((sec, ([ps[0] + ps[1]] + ps[2:])[:maxp]))
    return out


def api_records():
    R = []
    for i in range(1, 11):
        t = open(os.path.join(SRC, 'apicius_lat', 'apicius%d.shtml' % i), encoding='latin-1').read()
        t = re.sub(r'<[^>]+>', ' ', t); t = html.unescape(t)
        lines = [l.strip() for l in t.split('\n') if l.strip()]
        page, rec, body = None, None, []
        def flush():
            nonlocal rec
            if rec is not None and page is not None:
                ws = rec
                if len(ws) >= 4: page[1].append(ws)
            rec = None
        pages = []
        for ln in lines:
            if re.match(r'^[IVXL]+\.\s+\S', ln) and len(ln) < 120:
                flush(); page = ('A%d' % min(5, (i - 1) // 2), []); pages.append(page); continue
            if re.match(r'^\d+\.\s+\S', ln) and sum(c.isupper() for c in ln) >= 0.6 * max(1, sum(c.isalpha() for c in ln)):
                flush(); rec = _W(re.sub(r'^\d+\.', '', ln)); continue
            if rec is not None: rec += _W(ln)
        flush()
        R += [p for p in pages if p[1]]
    return R


def records_to_pages(R, seed=0, cap=36000, prefix='x'):
    """page = record; paragraph lines of 7-11 words (Voynich-like), first line ps."""
    rng = random.Random(seed); pages = []; n = 0
    for k, (sec, paras) in enumerate(R):
        if n >= cap: break
        lines = []
        for ws in paras:
            i = 0; first = True
            while i < len(ws):
                lw = rng.randint(7, 11)
                lines.append(dict(w=ws[i:i + lw], ps=first)); first = False; i += lw
            n += len(ws)
        pages.append(dict(id='%s%03d' % (prefix, k), sec=sec, lang='-', hand='-', quire='-', lines=lines))
    return pages


def planted_surface(plain_pages, seed=72):
    words = [w for p in plain_pages for l in p['lines'] for w in l['w']]
    code = L.payload_code(words, seed=seed, mode='merge')
    pay = L.encode_payload(plain_pages, code)
    surf = L.surface(pay, seed=seed + 1)
    return surf


def control_surface(name, seed=72):
    R = dict(BRU=bru_records, HIL=hil_records, API=api_records, CUL=cul_records)[name]()
    plain = records_to_pages(R, seed=seed, prefix=name.lower())
    return plain, planted_surface(plain, seed=seed)


# ------------------------------------------------------------------ generators on the surface
def gen_parcopy(pages, seed=1, c=0.3, m=0.4, idslot=False):
    """paragraph copy-and-vary: with prob c a paragraph is a copy of a random earlier paragraph of the same
    section (any page), token by token in place (lines re-broken to the target shape), each token mutated with
    prob m (half one-glyph change, half a fresh draw from the section word law); else drawn word by word from
    the section law. idslot=True plants a page-specific item: a page item word (fresh random glyph string)
    replaces token 1 of every paragraph and recurs at one random later position of the page (positive plant)."""
    rng = random.Random(seed); by = L._words_by_sec(pages)
    law = {k: L._cum(Counter(v)) for k, v in by.items()}
    gl = {k: L._cum(Counter(ch for w in v for ch in w)) for k, v in by.items()}
    pool = defaultdict(list); out = []
    for p in pages:
        sec = p['sec']; item = None
        if idslot:
            item = ''.join(L._draw(gl[sec], rng) for _ in range(rng.randint(4, 6)))
        # split page into paragraphs
        paras = []
        for l in p['lines']:
            if l['ps'] or not paras: paras.append([])
            paras[-1].append(l)
        nl = []
        for para in paras:
            n = sum(len(l['w']) for l in para)
            if pool[sec] and rng.random() < c:
                src = rng.choice(pool[sec]); toks = []
                for i in range(n):
                    if i < len(src):
                        w = src[i]
                        if rng.random() < m:
                            if rng.random() < 0.5 and len(w) > 0:
                                j = rng.randrange(len(w)); w = w[:j] + L._draw(gl[sec], rng) + w[j + 1:]
                            else:
                                w = L._draw(law[sec], rng)
                    else:
                        w = L._draw(law[sec], rng)
                    toks.append(w)
            else:
                toks = [L._draw(law[sec], rng) for _ in range(n)]
            if idslot and n >= 3:
                toks[1] = item
                if n > 3: toks[rng.randrange(3, n)] = item
            pool[sec].append(toks[:])
            k = 0
            for l in para:
                nl.append(dict(l, w=toks[k:k + len(l['w'])])); k += len(l['w'])
        out.append(dict(p, lines=nl))
    return out


GENS = dict(L.GENS)
GENS['PARCOPY'] = gen_parcopy


# ------------------------------------------------------------------ paragraphs from (extracted) pages
def paragraphs(pages):
    P = []
    for p in pages:
        paras = []
        for l in p['lines']:
            if l['ps'] or not paras: paras.append([])
            paras[-1].append(list(l['w']))
        paras = [x for x in paras if sum(map(len, x)) >= 4]
        for i, ls in enumerate(paras):
            P.append(dict(pid='%s.%d' % (p['id'], i), page=p['id'], sec=p['sec'], half=L.leaf_half(p['id']),
                          pidx=i, npp=len(paras), lines=ls, toks=[t for l in ls for t in l]))
    return P


def extracted_paragraphs(surf_pages):
    return paragraphs(L.extract(surf_pages, L.RULES[RULE]))


# ------------------------------------------------------------------ view / evaluation regions
def regions(par):
    """view positions: first-line positions 0-2, last 2 tokens of the paragraph, line-initial and line-final
    tokens; evaluation region = all other tokens (disjoint, so a view-defined group is tested on new tokens)."""
    view = set(); pos = 0; idx = []
    for li, l in enumerate(par['lines']):
        for j, t in enumerate(l):
            idx.append((li, j, len(l)))
    n = len(idx)
    for k, (li, j, ln) in enumerate(idx):
        if (li == 0 and j <= 2) or k >= n - 2 or j == 0 or j == ln - 1: view.add(k)
    ev = [par['toks'][k] for k in range(n) if k not in view]
    return ev


def sim_matrix(P, min_df=1):
    """idf-weighted cosine between paragraphs on evaluation-region token sets."""
    ev = [set(regions(p)) for p in P]
    df = Counter(t for s in ev for t in s)
    vocab = {t: i for i, t in enumerate(df)}
    N = len(P)
    X = np.zeros((N, len(vocab)), dtype=np.float32)
    for i, s in enumerate(ev):
        for t in s: X[i, vocab[t]] = math.log(N / df[t])
    nrm = np.linalg.norm(X, axis=1); nrm[nrm == 0] = 1
    X /= nrm[:, None]
    S = X @ X.T
    np.fill_diagonal(S, np.nan)
    return S


def baseline(P, S):
    """same-section cross-page mean and sd of S."""
    sec = np.array([p['sec'] for p in P]); page = np.array([p['page'] for p in P])
    B = np.zeros_like(S); SD = np.ones_like(S)
    for s in set(sec):
        ii = np.where(sec == s)[0]
        sub = S[np.ix_(ii, ii)]
        msk = page[ii][:, None] != page[ii][None, :]
        v = sub[msk & ~np.isnan(sub)]
        mu, sd = (float(v.mean()), float(v.std() + 1e-9)) if len(v) else (0.0, 1.0)
        B[np.ix_(ii, ii)] = mu; SD[np.ix_(ii, ii)] = sd
    Z = (S - B) / SD
    Z[page[:, None] == page[None, :]] = np.nan          # cross-page pairs only
    Z[sec[:, None] != sec[None, :]] = np.nan              # within section only
    return Z
