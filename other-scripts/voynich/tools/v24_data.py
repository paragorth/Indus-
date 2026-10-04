"""v24 data: correction sites for the Voynich (ZL markup), a real Latin control (SCTA / LombardPress
diplomatic transcriptions of Petrus Plaoul's commentary, 4 witnesses, TEI <del>/<add>/<subst>),
and two planted controls (template+junction generator; Latin with word-identity repair).
Writes pickles to data/v24_ckpt/.  Usage: python3 v24_data.py PLAOUL_REPO_DIR
"""
import os, sys, re, glob, json, pickle, random, difflib
import xml.etree.ElementTree as ET
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v24_lib as L
import vlib

TEI = '{http://www.tei-c.org/ns/1.0}'
SKIP = {'note', 'rdg', 'app', 'bibl', 'reg', 'expan', 'corr_x', 'fw', 'figure', 'head_x'}


# ------------------------------------------------------------------ Voynich ZL
def zl_lines(path=os.path.join(L.DATA, 'ZL3b-n.txt')):
    """raw ZL lines -> list of dicts with locus, ltype, raw text."""
    out = []
    for raw in open(path, encoding='latin-1'):
        m = re.match(r'^<(f\w+)\.(\d+),([@+*=&~])(\w)(\w*)>\s*(.*)$', raw.rstrip('\n'))
        if m:
            out.append(dict(folio=m.group(1), n=int(m.group(2)), ltype=m.group(4), code=m.group(3) + m.group(4) + m.group(5), raw=m.group(6)))
    return out


def clean_word(w):
    w = re.sub(r'<[^>]*>', '', w)
    w = w.replace('{', '').replace('}', '').replace("'", '')
    return w


def zl_variants(raw):
    """split a raw ZL line into word tokens, each keeping its markup; returns list of (markup_token)."""
    t = raw.replace('<->', '.').replace('<%>', '').replace('<$>', '')
    t = t.replace(',', '.')
    # keep comments attached to the preceding word
    toks, cur, depth = [], '', 0
    for ch in t:
        if ch == '<': depth += 1
        if ch == '>': depth -= 1
        if ch == '[': depth += 1
        if ch == ']': depth -= 1
        if ch == '.' and depth == 0:
            toks.append(cur); cur = ''
        else:
            cur += ch
    toks.append(cur)
    return [x for x in toks if x.strip()]


def reading(tok, k):
    """k-th alternative of every [a:b] group (k=0 first)."""
    def f(m):
        al = m.group(1).split(':')
        return al[k] if k < len(al) else al[0]
    return clean_word(re.sub(r'\[([^\]]*)\]', f, re.sub(r'<![^>]*>', '', tok)))


def voynich_corpus():
    """ZL paragraph-type lines as glyph tuples (first readings, uncertain words kept out of model only)."""
    lines = []
    for r in zl_lines():
        if r['ltype'] != 'P':
            continue
        ws = [reading(t, 0) for t in zl_variants(r['raw'])]
        ws = [L.U(w) for w in ws if w and '?' not in w and '@' not in w]
        lines.append([w for w in ws if w])
    return lines


def voynich_sites():
    """T1: ZL explicit correction notes; T2: ZL [a:b] single-unit alternatives (transcriber ambiguity).
    Context taken from the line's first readings."""
    T1, T1after, T2 = [], [], []
    prevfirst = None
    corr_re = re.compile(r'<!(corr\?|k on top of e|s above e|s above last a|erased\?)>')
    for r in zl_lines():
        toks = zl_variants(r['raw'])
        first = [L.U(reading(t, 0)) for t in toks]
        for i, t in enumerate(toks):
            ctx = dict(prev=first[i - 1] if i > 0 and first[i - 1] else None,
                       nxt=first[i + 1] if i + 1 < len(first) and first[i + 1] else None,
                       prev2=first[i - 2] if i > 1 and first[i - 2] else None,
                       li=(i == 0), pl=prevfirst)
            meta = dict(locus='%s.%d' % (r['folio'], r['n']), tok=t, ltype=r['ltype'])
            m = corr_re.search(t)
            if m:
                note = m.group(1)
                a0 = reading(t, 0)
                if note == 'k on top of e':           # k written over an e
                    T1.append(L.Site(L.U(a0[:-1] + 'e'), L.U(a0), meta=dict(meta, note=note), **ctx))
                elif note == 's above e':              # s added above the line
                    T1.append(L.Site(L.U(reading(t, 1)), L.U(a0), meta=dict(meta, note=note), **ctx))
                elif '[' in t and note == 'corr?':     # both states recorded as alternatives
                    a1 = reading(t, 1)
                    if a1 and '?' not in a1 and '?' not in a0:
                        T1.append(L.Site(L.U(a1), L.U(a0), meta=dict(meta, note=note + ' (direction unknown)'), **ctx))
                    else:
                        T1after.append(dict(meta, note=note, after=a0))
                else:
                    T1after.append(dict(meta, note=note, after=a0))
            elif '[' in t and r['ltype'] == 'P':
                a0, a1 = reading(t, 0), reading(t, 1)
                if a0 and a1 and '?' not in a0 + a1 and '@' not in a0 + a1 and a0 != a1:
                    s = L.Site(L.U(a1), L.U(a0), meta=dict(meta, note='alt'), **ctx)
                    if s.kind in ('sub', 'ins', 'del'):
                        T2.append(s)
        if r['ltype'] == 'P' and first and first[0]:
            prevfirst = first[0][0]
    return T1, T1after, T2


# ------------------------------------------------------------------ Latin (SCTA Plaoul)
def walk(el, mode, out):
    tag = el.tag.replace(TEI, '')
    if tag in SKIP:
        pass
    elif tag == 'del' and mode == 'after':
        pass
    elif tag == 'add' and mode == 'before':
        pass
    elif tag == 'choice':
        kids = {k.tag.replace(TEI, ''): k for k in el}
        k = kids.get('orig') or kids.get('abbr') or kids.get('sic')
        if k is not None:
            walk(k, mode, out)
    elif tag == 'g':
        pass
    elif tag == 'lb':
        if el.get('break') != 'no':
            out.append(' ')
    else:
        if el.text:
            out.append(el.text)
        for k in el:
            walk(k, mode, out)
            if k.tail:
                out.append(k.tail)
        return
    # skipped / special elements: only their tail continues the text (handled by parent)


def toks(s):
    s = s.lower()
    return [w for w in re.sub(r'[^a-z]+', ' ', s).split() if w]


def plaoul(repo):
    files = sorted(f for f in glob.glob(os.path.join(repo, 'lectio*', '*_lectio*.xml'))
                   if os.path.basename(f).split('_')[0] in ('reims', 'sorb', 'svict', 'vat', 'cod-mmppdi'))
    lines, sites, wordlev = [], [], Counter()
    for f in files:
        try:
            txt = open(f, encoding='utf-8').read()
        except Exception:
            continue
        txt = re.sub(r'<g ref="#dbdash"/>\s*<lb([^>]*)/>', r'<lb break="no"\1/>', txt)
        try:
            root = ET.fromstring(txt.encode('utf-8'))
        except ET.ParseError:
            continue
        body = root.find('.//' + TEI + 'body')
        if body is None:
            continue
        for p in body.iter(TEI + 'p'):
            ob, oa = [], []
            walk(p, 'before', ob); walk(p, 'after', oa)
            tb, ta = toks(''.join(ob)), toks(''.join(oa))
            if not ta:
                continue
            lines.append([tuple(w) for w in ta])
            if tb == ta:
                continue
            sm = difflib.SequenceMatcher(a=tb, b=ta, autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == 'equal':
                    continue
                wordlev[op] += 1
                if op == 'replace' and (i2 - i1) == (j2 - j1):
                    for k in range(i2 - i1):
                        bw, aw = tb[i1 + k], ta[j1 + k]
                        j = j1 + k
                        s = L.Site(tuple(bw), tuple(aw), prev=tuple(ta[j - 1]) if j > 0 else None,
                                   nxt=tuple(ta[j + 1]) if j + 1 < len(ta) else None,
                                   prev2=tuple(ta[j - 2]) if j > 1 else None, li=False, pl=None,
                                   meta=dict(file=os.path.basename(f), before=bw, after=aw))
                        sites.append(s)
    return lines, sites, wordlev


# ------------------------------------------------------------------ planted generator
def planted_generator(seed=11, nwords=36000, err=0.05, p_fix_bad=0.9, p_fix_ok=0.03):
    rng = random.Random(seed)
    sm = json.load(open(os.path.join(L.DATA, 'derived', 'v7_slotmodels.json')))['planted_clean_K4']
    fill = sm['fillers']
    wts = [[1.0 / (r + 1) ** 1.1 for r in range(len(f))] for f in fill]
    first_gl = sorted({f[0] for f in fill[0] + fill[1] + fill[2] + fill[3] if f})
    # junction table: each last glyph allows a random ~half of first glyphs
    last_gl = sorted({f[-1] for s in fill for f in s if f})
    allow = {g: set(rng.sample(first_gl, max(2, len(first_gl) // 2))) for g in last_gl}
    fillsets = [set(f) for f in fill]

    def parse_ok(w):
        s = ''.join(w)
        def rec(i, k):
            if k == 4:
                return i == len(s)
            return any(s.startswith(f, i) and rec(i + len(f), k + 1) for f in fill[k])
        return len(s) > 0 and rec(0, 0)

    def gen_word():
        while True:
            w = ''.join(rng.choices(fill[k], wts[k])[0] for k in range(4))
            if w:
                return tuple(w)
    lines, cur, prev = [], [], None
    alpha = sorted({g for s in fill for f in s for g in f})
    clean = []
    for t in range(nwords):
        while True:
            w = gen_word()
            if prev is None or w[0] in allow.get(prev[-1], set(first_gl)):
                break
        cur.append(w); prev = w
        if len(cur) == 10:
            clean.append(cur); cur = []; prev = None
    # inject errors and correct only rule-breaking ones
    gf = Counter(g for Lw in clean for w in Lw for g in w)
    final, sites_raw = [], []
    for li, Lw in enumerate(clean):
        out = []
        for i, w in enumerate(Lw):
            if rng.random() < err:
                j = rng.randrange(len(w)); cand = [x for x in alpha if x != w[j]]
                g = rng.choices(cand, [gf[x] for x in cand])[0]
                bad = w[:j] + (g,) + w[j + 1:]
                broke = (not parse_ok(bad)) or (i > 0 and bad[0] not in allow.get(out[i - 1][-1], set(first_gl)))
                if rng.random() < (p_fix_bad if broke else p_fix_ok):
                    out.append(w); sites_raw.append((li, i, bad, w, broke))
                else:
                    out.append(bad)
            else:
                out.append(w)
        final.append(out)
    sites = []
    for li, i, bad, w, broke in sites_raw:
        Lw = final[li]
        sites.append(L.Site(bad, w, prev=Lw[i - 1] if i > 0 else None, nxt=Lw[i + 1] if i + 1 < len(Lw) else None,
                            prev2=Lw[i - 2] if i > 1 else None, li=(i == 0),
                            pl=final[li - 1][0][0] if li > 0 else None, meta=dict(broke=broke)))
    order = sm['order']; cuts = sm['cuts']
    return final, sites, (order, cuts)


# ------------------------------------------------------------------ planted language (Latin word-identity repair)
def planted_language(seed=12, err=0.05, p_fix_nonword=0.85, p_fix_word=0.15, key='Latin-Caesar'):
    rng = random.Random(seed)
    lines = vlib.load_ref(key, max_words=40000)
    lines = [[tuple(w) for w in Lr['words'] if re.fullmatch('[a-z]+', w)] for Lr in lines]
    lines = [Lw for Lw in lines if Lw]
    vocab = Counter(w for Lw in lines for w in Lw)
    alpha = sorted({g for w in vocab for g in w})
    gf = Counter(g for w, c in vocab.items() for g in w for _ in range(c))
    final, raw = [], []
    for li, Lw in enumerate(lines):
        out = []
        for i, w in enumerate(Lw):
            if rng.random() < err and len(w) > 1:
                j = rng.randrange(len(w)); cand = [x for x in alpha if x != w[j]]
                g = rng.choices(cand, [gf[x] for x in cand])[0]
                bad = w[:j] + (g,) + w[j + 1:]
                nonword = vocab[bad] == 0
                if rng.random() < (p_fix_nonword if nonword else p_fix_word):
                    out.append(w); raw.append((li, i, bad, w, nonword))
                else:
                    out.append(bad)
            else:
                out.append(w)
        final.append(out)
    sites = []
    for li, i, bad, w, nonword in raw:
        Lw = final[li]
        sites.append(L.Site(bad, w, prev=Lw[i - 1] if i > 0 else None, nxt=Lw[i + 1] if i + 1 < len(Lw) else None,
                            prev2=Lw[i - 2] if i > 1 else None, li=False, pl=None, meta=dict(nonword=nonword)))
    return final, sites



# ------------------------------------------------------------------ planted copying (generator copied from an exemplar)
def planted_copy(seed=13, err=0.05, p_fix=0.9, p_neigh=0.7):
    """clean generator text = the exemplar; copying slips take the glyph at the same position of the next
    (anticipation) or previous (perseveration) word, else a frequency-weighted random glyph; the scribe
    restores the exemplar (p_fix) whatever rule the slip breaks."""
    rng = random.Random(seed)
    clean, _, slot = planted_generator(seed=seed, err=0.0)
    gf = Counter(g for Lw in clean for w in Lw for g in w); alpha = sorted(gf)
    final, raw = [], []
    for li, Lw in enumerate(clean):
        out = []
        for i, w in enumerate(Lw):
            if rng.random() < err:
                j = rng.randrange(len(w)); g = None
                if rng.random() < p_neigh:
                    nb = [Lw[i + d] for d in (1, -1) if 0 <= i + d < len(Lw)]
                    nb = [x for x in nb if len(x) > j and x[j] != w[j]]
                    if nb:
                        g = rng.choice(nb)[j]
                if g is None:
                    cand = [x for x in alpha if x != w[j]]; g = rng.choices(cand, [gf[x] for x in cand])[0]
                bad = w[:j] + (g,) + w[j + 1:]
                if rng.random() < p_fix:
                    out.append(w); raw.append((li, i, bad, w))
                else:
                    out.append(bad)
            else:
                out.append(w)
        final.append(out)
    sites = []
    for li, i, bad, w in raw:
        Lw = final[li]
        sites.append(L.Site(bad, w, prev=Lw[i - 1] if i > 0 else None, nxt=Lw[i + 1] if i + 1 < len(Lw) else None,
                            prev2=Lw[i - 2] if i > 1 else None, li=(i == 0),
                            pl=final[li - 1][0][0] if li > 0 else None))
    return final, sites, slot

if __name__ == '__main__':
    repo = sys.argv[1]
    T1, T1a, T2 = voynich_sites()
    vc = voynich_corpus()
    pl_lines, pl_sites, wl = plaoul(repo)
    pg_lines, pg_sites, slot = planted_generator()
    pk_lines, pk_sites = planted_language()
    pickle.dump(dict(T1=T1, T1after=T1a, T2=T2, corpus=vc), open(os.path.join(L.CK, 'voy.pkl'), 'wb'))
    pickle.dump(dict(lines=pl_lines, sites=pl_sites, wordlev=wl), open(os.path.join(L.CK, 'plaoul.pkl'), 'wb'))
    pickle.dump(dict(lines=pg_lines, sites=pg_sites, slot=slot), open(os.path.join(L.CK, 'pgen.pkl'), 'wb'))
    pickle.dump(dict(lines=pk_lines, sites=pk_sites), open(os.path.join(L.CK, 'plang.pkl'), 'wb'))
    print('Voynich T1 (before known)', len(T1), [(''.join(s.before), ''.join(s.after), s.kind, s.meta['locus'], s.meta['note']) for s in T1])
    print('Voynich T1 after-only', len(T1a), [(d['locus'], d['after'], d['note']) for d in T1a])
    print('Voynich T2 alternatives (single-unit)', len(T2), Counter(s.kind for s in T2))
    print('Voynich corpus lines', len(vc), sum(map(len, vc)))
    print('Plaoul lines', len(pl_lines), sum(map(len, pl_lines)), 'word-level ops', dict(wl), '1-1 replaced words', len(pl_sites),
          Counter(s.kind for s in pl_sites))
    print('planted gen', sum(map(len, pg_lines)), 'sites', len(pg_sites), Counter(s.meta['broke'] for s in pg_sites))
    print('planted lang', sum(map(len, pk_lines)), 'sites', len(pk_sites), Counter(s.meta['nonword'] for s in pk_sites))
