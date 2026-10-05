"""v62 THE VOYNICH IS VERSE: shared corpus builders, encoders, generators and scorers.

Every corpus is a list of pages; a page is a list of lines; a line is a list of words; a word is a
tuple of symbols (Voynich glyph units, or opaque verbose-code symbols for the controls).
Each line also carries flags (para_start, para_end) in a parallel structure.
"""
import os, re, sys, html, json, random, math
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib

ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v62_ckpt')
SRC58 = os.path.join(ROOT, 'data', 'v58_ckpt', 'src')
os.makedirs(CK, exist_ok=True)

VS_ALPHA = list('ABCDEFGHIJKLMNOPQR')


def verbose_table(seed=7):
    """Opaque verbose code: each letter -> fixed group of 1-3 symbols (same as v5_corpora)."""
    rng = random.Random(seed)
    tab, used = {}, set()
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        while True:
            k = rng.choice([1, 2, 2, 3])
            g = tuple(rng.choice(VS_ALPHA) for _ in range(k))
            if g not in used:
                used.add(g); tab[ch] = g; break
    return tab


def norm_latin(s):
    s = s.lower()
    s = s.replace('æ', 'ae').replace('œ', 'oe').replace('j', 'i').replace('v', 'u')
    s = re.sub(r'[àáâä]', 'a', s); s = re.sub(r'[èéêë]', 'e', s); s = re.sub(r'[ìíîï]', 'i', s)
    s = re.sub(r'[òóôö]', 'o', s); s = re.sub(r'[ùúûü]', 'u', s)
    return [w for w in re.split(r'[^a-z]+', s) if w]


def encode_lines(lines_words, tab):
    out = []
    for ws in lines_words:
        e = [tuple(s for c in w for s in tab.get(c, ())) for w in ws]
        e = [w for w in e if w]
        if e:
            out.append(e)
    return out


# ---------------- raw control texts (verse lines as lines) ----------------
def raw_regimen():
    t = open(os.path.join(SRC58, 'regimen_salernitanum', 'reg_sana.html'), encoding='utf-8', errors='replace').read()
    s = t[t.find('contentus'):]
    s = re.sub(r'<br\s*/?>', '\n', s); s = re.sub(r'</p>|</tr>', '\n', s)
    s = re.sub(r'<[^>]+>', '', s); s = html.unescape(s)
    L = [l.strip() for l in s.split('\n') if l.strip()]
    start = next(i for i, l in enumerate(L) if l.startswith('Anglorum regi'))
    out = []
    for l in L[start:]:
        if re.fullmatch(r'[IVXLC]+', l):
            out.append(None); continue          # chapter break
        ws = norm_latin(l)
        if len(ws) >= 4 and l[:1].isalpha():      # verse lines; headings are short
            out.append(ws)
    return out


def raw_macer():
    t = open(os.path.join(SRC58, 'macer_floridus.wiki'), encoding='utf-8', errors='replace').read()
    out = []
    for l in t.split('\n'):
        if l.startswith('=='):
            out.append(None); continue
        l = re.sub(r'\{\{[^}]*\}\}', '', l); l = re.sub(r'<[^>]+>', '', l)
        ws = norm_latin(l)
        if len(ws) >= 3:
            out.append(ws)
    return out


def raw_dante(n=4500):
    t = open(os.path.join(ROOT, 'data', 'pg1000.txt'), encoding='utf-8', errors='replace').read()
    m1 = re.search(r'\*\*\* ?START[^\n]*\n', t); m2 = re.search(r'\*\*\* ?END', t)
    t = t[m1.end():m2.start()]
    out = []
    for l in t.split('\n'):
        l = l.strip()
        if not l:
            continue
        if 'Canto' in l or 'CANTO' in l:
            out.append(None); continue
        ws = [w for w in re.split(r"[^a-zàèéìíòóùú]+", l.lower().replace("'", ' ').replace('’', ' ')) if w]
        ws = [norm_latin(w)[0] if norm_latin(w) else '' for w in ws]
        ws = [w for w in ws if w]
        if len(ws) >= 3:
            out.append(ws)
    lines = [x for x in out]
    # skip front matter: start at first canto break
    i = next(i for i, x in enumerate(lines) if x is None)
    return lines[i:i + n]


def raw_litany():
    t = open(os.path.join(CK, 'src', 'Litaniae_Sanctorum.wiki'), encoding='utf-8', errors='replace').read()
    t = re.sub(r'\{\{[^}]*\}\}', '', t); t = re.sub(r'<[^>]+>', '', t)
    out, pend = [], None
    for l in t.split('\n'):
        l = l.strip().lstrip(':').strip()
        if not l:
            continue
        if re.match(r'^[IVX]+ [A-Z]', l):
            if pend: out.append(pend); pend = None
            out.append(None); continue
        if l.startswith('R.') or l.startswith('V.'):
            ws = norm_latin(l[2:])
            if pend is not None:
                out.append(pend + ws); pend = None
            else:
                out.append(ws)
        else:
            if pend: out.append(pend)
            pend = norm_latin(l)
    if pend: out.append(pend)
    return [x for x in out if x is None or x]


def raw_hildegard():
    import glob
    words = []
    for f in sorted(glob.glob(os.path.join(SRC58, 'hildegard_physica', 'hil_phy*.html'))):
        t = open(f, encoding='utf-8', errors='replace').read()
        t = re.sub(r'<[^>]+>', ' ', t); t = html.unescape(t); t = re.sub(r'\[\d+\]', ' ', t)
        words += norm_latin(t)
    return words


def raw_caesar(n=20000):
    L = vlib.load_ref('Latin-Caesar', max_words=n, skip_frac=0.05)
    return [norm_latin(w)[0] for x in L for w in x['words'] if norm_latin(w)]


# ---------------- page structures ----------------
def pages_from_verse(lines, page_len=24):
    """verse lines (None = section break) -> pages of <= page_len lines (break also ends a page)."""
    pages, cur = [], []
    for x in lines:
        if x is None:
            if len(cur) >= page_len // 2:
                pages.append(cur); cur = []
            continue
        if len(cur) >= page_len:
            pages.append(cur); cur = []
        cur.append(x)
    if len(cur) >= 4: pages.append(cur)
    return pages


def reflow_words(words, rng, wpl_dist, page_len=24):
    """running prose -> lines with word counts drawn from a Voynich words-per-line distribution."""
    pages, cur, i = [], [], 0
    while i < len(words):
        k = rng.choice(wpl_dist)
        cur.append(words[i:i + k]); i += k
        if len(cur) >= page_len:
            pages.append(cur); cur = []
    if len(cur) >= 4: pages.append(cur)
    return pages


def voynich_pages(tr='ZL3b'):
    L = vlib.load_voynich(tr)
    pages, meta, cur, curf = [], [], [], None
    for r in L:
        if r['folio'] != curf:
            if cur: pages.append(cur)
            cur = []; curf = r['folio']
            meta.append(dict(folio=r['folio'], illus=r['illus'], lang=r['lang'], hand=r['hand'], quire=r['quire']))
        ws = [tuple(vlib.glyphs(w)) if '?' not in w else ('?' + w,) for w in r['words']]
        cur.append(ws)
    if cur: pages.append(cur)
    return pages, meta


def folio_num(f):
    m = re.match(r'f(\d+)', f)
    return int(m.group(1)) if m else 0


def build_all(seed=7):
    """Returns dict name -> dict(pages, meta, kind)."""
    rng = random.Random(seed)
    tab = verbose_table(seed)
    out = {}
    for tr in ('ZL3b', 'IT2a'):
        p, m = voynich_pages(tr)
        out['V-' + tr] = dict(pages=p, meta=m, kind='voynich')
    zl = out['V-ZL3b']['pages']
    wpl = [len(l) for pg in zl for l in pg]
    def enc_pages(pages):
        return [encode_lines(pg, tab) for pg in pages]
    out['Regimen(verse,rhymed)'] = dict(pages=enc_pages(pages_from_verse(raw_regimen())), kind='verse')
    out['Macer(verse,hexam)'] = dict(pages=enc_pages(pages_from_verse(raw_macer())), kind='verse')
    out['Dante(verse,terza)'] = dict(pages=enc_pages(pages_from_verse(raw_dante())), kind='verse')
    out['Litany(refrain)'] = dict(pages=enc_pages(pages_from_verse(raw_litany())), kind='litany')
    out['Hildegard(prose herbal)'] = dict(pages=enc_pages(reflow_words(raw_hildegard(), rng, wpl)), kind='prose')
    out['Caesar(prose)'] = dict(pages=enc_pages(reflow_words(raw_caesar(), rng, wpl)), kind='prose')
    # verse re-flowed: same words, line breaks moved (must lose rhyme and meter)
    def refl(name):
        ws = [w for pg in out[name]['pages'] for l in pg for w in l]
        return reflow_words(ws, random.Random(seed + 1), [len(l) for pg in out[name]['pages'] for l in pg])
    out['Regimen-reflowed'] = dict(pages=refl('Regimen(verse,rhymed)'), kind='null')
    out['Dante-reflowed'] = dict(pages=refl('Dante(verse,terza)'), kind='null')
    for k in out:
        out[k]['pages'] = [[l for l in pg if l] for pg in out[k]['pages']]
        out[k]['pages'] = [pg for pg in out[k]['pages'] if len(pg) >= 2]
    return out


# ---------------- generators / nulls ----------------
def shuffle_lines_within_page(pages, rng):
    out = []
    for pg in pages:
        q = pg[:]; rng.shuffle(q); out.append(q)
    return out


def reflow_page(pages, rng):
    """same words in page order, line breaks re-placed at random with the same number of lines."""
    out = []
    for pg in pages:
        ws = [w for l in pg for w in l]
        n = len(pg)
        if len(ws) <= n:
            out.append(pg); continue
        cuts = sorted(rng.sample(range(1, len(ws)), n - 1))
        b = [0] + cuts + [len(ws)]
        out.append([ws[b[i]:b[i + 1]] for i in range(n)])
    return out


def markov_line_generator(pages, rng, lam_page=0.4):
    """Word bigram resynthesis with planted line effects: line-initial distribution, line-final
    distribution conditioned on previous word (backoff to line-final unigram), page-specific unigram
    mixing (topic drift). Same page and line-length layout as the source. No rhyme, metre or refrain
    except what these line-position effects carry."""
    init, fin, uni = Counter(), Counter(), Counter()
    big, finb = defaultdict(Counter), defaultdict(Counter)
    for pg in pages:
        for l in pg:
            init[l[0]] += 1
            if len(l) > 1:
                fin[l[-1]] += 1; finb[l[-2]][l[-1]] += 1
            for a, b in zip(l[:-2], l[1:-1]):
                big[a][b] += 1
            for w in l: uni[w] += 1
    def draw(c):
        ks = list(c.keys()); ws = np.array(list(c.values()), float)
        return ks[rng.choices(range(len(ks)), weights=ws)[0]]
    cache = {}
    def samp(c, key):
        if key not in cache:
            ks = list(c.keys()); cum = np.cumsum(np.array(list(c.values()), float))
            cache[key] = (ks, cum)
        ks, cum = cache[key]
        return ks[int(np.searchsorted(cum, rng.random() * cum[-1], side='right'))]
    out = []
    for pi, pg in enumerate(pages):
        pu = Counter(w for l in pg for w in l)
        newpg = []
        for l in pg:
            n = len(l)
            line = []
            for j in range(n):
                if rng.random() < lam_page:
                    w = samp(pu, ('pg', pi))
                elif j == 0:
                    w = samp(init, 'init')
                elif j == n - 1:
                    prev = line[-1]
                    w = samp(finb[prev], ('fb', prev)) if prev in finb and rng.random() < 0.6 else samp(fin, 'fin')
                else:
                    prev = line[-1]
                    w = samp(big[prev], ('b', prev)) if prev in big and rng.random() < 0.7 else samp(uni, 'uni')
                line.append(w)
            newpg.append(line)
        out.append(newpg)
    return out


# ---------------- rhyme scoring ----------------
def alphabet(pages):
    return sorted({s for pg in pages for l in pg for w in l for s in w if not str(s).startswith('?')})


def random_defs(alpha, n, rng):
    """ending definitions: (class map dict or None, k, skip)."""
    defs = []
    for k in (1, 2, 3, 4):
        for s in (0, 1, 2):
            defs.append((None, k, s))
    while len(defs) < n:
        c = rng.choice([2, 3, 4, 5, 6, 8, 10, 14])
        m = {a: rng.randrange(c) for a in alpha}
        defs.append((m, rng.choice([1, 2, 3, 4, 5]), rng.choice([0, 0, 1, 2])))
    return defs


def ending(w, d):
    m, k, s = d
    if w and str(w[0]).startswith('?'):
        return None
    if len(w) < s + 1:
        return None
    seg = w[max(0, len(w) - s - k):len(w) - s]
    if m is not None:
        seg = tuple(m.get(x, -1) for x in seg)
    return seg


def prep_positions(pages, minw=3):
    """for each page: list of (final word, penultimate word, first word, second word) per line; lines with
    fewer than minw words get None entries (kept to preserve lag structure)."""
    P = []
    for pg in pages:
        rows = []
        for l in pg:
            if len(l) >= minw:
                rows.append((l[-1], l[-2], l[0], l[1]))
            else:
                rows.append(None)
        P.append(rows)
    return P


def lag_kappa(seqs, maxlag=4):
    """seqs: list per page of list of hashable-or-None labels. Returns array of kappa at lags 1..maxlag,
    kappa = (obs - exp) / (1 - exp), exp = within-page random pair match rate (exact shuffle expectation).
    Vectorised."""
    vocab = {}
    lab, pid = [], []
    for p, s in enumerate(seqs):
        for x in s:
            lab.append(-1 if x is None else vocab.setdefault(x, len(vocab))); pid.append(p)
    lab = np.array(lab, np.int64); pid = np.array(pid, np.int64)
    ok = lab >= 0
    V = len(vocab) + 1
    uk, cnt = np.unique(pid[ok] * V + lab[ok], return_counts=True)
    num = np.bincount(uk // V, weights=cnt * (cnt - 1.0), minlength=len(seqs))
    n = np.bincount(pid[ok], minlength=len(seqs)).astype(float)
    pe = np.where(n > 1, num / np.maximum(n * (n - 1), 1), 0.0)
    obs = np.zeros(maxlag); npair = np.zeros(maxlag); expn = np.zeros(maxlag)
    for d in range(1, maxlag + 1):
        a, b = lab[:-d], lab[d:]
        m = (pid[:-d] == pid[d:]) & (a >= 0) & (b >= 0)
        npair[d - 1] = m.sum(); obs[d - 1] = (a[m] == b[m]).sum(); expn[d - 1] = pe[pid[:-d][m]].sum()
    o = obs / np.maximum(npair, 1); e = expn / np.maximum(npair, 1)
    return (o - e) / np.maximum(1 - e, 1e-9), o, e, npair


def rhyme_score(P, d, maxlag=4):
    """kappa at each lag for line-final endings minus the same for the line's SECOND word (comparator
    away from the caesura of leonine verse; carries the same page drift)."""
    F = [[None if r is None else ending(r[0], d) for r in rows] for rows in P]
    M = [[None if r is None else ending(r[3], d) for r in rows] for rows in P]
    kF, oF, eF, nF = lag_kappa(F, maxlag)
    kM, *_ = lag_kappa(M, maxlag)
    return kF - kM, kF, kM, oF, eF


def def_str(d):
    m, k, s = d
    if m is None:
        return f'full k{k} s{s}'
    nc = len(set(m.values()))
    return f'{nc}cls k{k} s{s}'


def rhyme_z(P, d, maxlag=4):
    """z-scored difference: line-final kappa minus second-word kappa at each lag, SE from the binomial
    approximation kappa_se ~ sqrt(e / (n (1 - e))). Definitions with uninformative endings (exp > 0.5) get
    z = 0."""
    F = [[None if r is None else ending(r[0], d) for r in rows] for rows in P]
    M = [[None if r is None else ending(r[3], d) for r in rows] for rows in P]
    kF, oF, eF, nF = lag_kappa(F, maxlag)
    kM, oM, eM, nM = lag_kappa(M, maxlag)
    se = np.sqrt(eF / np.maximum(nF * (1 - eF), 1) + eM / np.maximum(nM * (1 - eM), 1)) + 1e-9
    z = (kF - kM) / se
    z[(eF > 0.5) | (eM > 0.5)] = 0.0
    return z, kF - kM, eF
