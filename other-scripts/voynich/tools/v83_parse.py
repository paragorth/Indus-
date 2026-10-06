#!/usr/bin/env python3
"""v83 uncertainty-aware IVTFF parser (ZL3b, IT2a in EVA; GC2a in v101, converted to EVA).

Every word keeps its first reading (as the old parser did) plus a set of flags:
  alt   the word contains an alternative reading [a:b] (any position)
  ill   an illegible glyph '?' (or '???') in the first reading
  rare  a high-ascii / rare-glyph code @nnn; (EVA files) or a v101 glyph with no EVA equivalent (GC2a)
  lig   a ligature brace {..} (ZL only)
  mark  a glyph-variant apostrophe ' (ZL only: c'y, qo'ky, e'a'iin)
  usp   an uncertain word space ',' on either side of the word (word boundary uncertain)
  cmt   an inline comment on the word that questions its reading (corr?, unclear, funny d, faint, erased?, small e,
        mark above o, separated ch, bar over o, long tail m, s above e, k on top of e, subscript, raised, rare 'ed',
        spots near dom, bar interrupted)
  dmg   the word touches physical damage noted inline (tear, hole, gap, long gap, wide gap, next line intrudes)
  ldmg  the line or page is marked damaged in the file's '#' comments (f37r torn and stitched; f101v creases badly
        damaged; f72r3 C1 start could be missing; fRos left margin unreadable)
Separators: '.' certain space, ',' uncertain space, '<->' drawing intrusion, '<~>' intrusion with bad alignment
(IVTFF 2.0: implies a word space; the legacy parser deleted it and merged 6 ZL word pairs).
Modes for load(): 'legacy' (= tools/parse_ivtff.py words), 'all' (every word, '<~>' fixed), 'clean' (words with no
flag), 'glyph' (words with no glyph-level flag: alt/ill/rare/lig/mark/cmt/dmg/ldmg; uncertain spaces kept),
'agree' (words read identically by ZL, IT2a and GC2a on the same locus).
Dropped words are replaced by None in 'words_masked' so adjacency code can skip pairs instead of joining neighbours.
"""
import json, os, re, sys, collections, difflib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v83_ckpt'); os.makedirs(CK, exist_ok=True)
SRC = {'ZL3b': os.path.join(DATA, 'ZL3b-n.txt'), 'IT2a': os.path.join(DATA, 'IT2a-n.txt'),
       'GC2a': os.path.join(DATA, 'v24_ckpt', 'GC2a-n.txt')}
FLAGS = ['alt', 'ill', 'rare', 'lig', 'mark', 'usp', 'cmt', 'dmg', 'ldmg']
GLYPH_FLAGS = set(FLAGS) - {'usp'}

UNC_COMMENT = re.compile(r"corr|unclear|funny|faint|erased|small e|mark above|separated|bar over|bar interr|long tail|"
                         r"s above|k on top|subscript|raised|rare 'ed'|spots near")
DMG_COMMENT = re.compile(r'tear|hole|gap|intrud')
# '#' comment lines that mark following lines (block) or the whole page as damaged (curated from ZL3b, see v83_cycle1)
DMG_PAGE = re.compile(r'torn and stiched|badly damaged')
DMG_BLOCK = re.compile(r'could be missing|margin unreadable')


def _v101_map():
    """v101 -> EVA via the published bitrans tables (as tools/v74_gc.py), but keeping '!' and '%' (v101 glyphs Le, Lc:
    sh-family), which v74_gc.py deleted as if they were IVTFF fillers."""
    sys.path.insert(0, HERE)
    import v74_gc
    m, how = v74_gc.mapping()
    return m, how


_V101 = None


def v101_to_eva(word):
    """Returns (eva, rare_flag). Longest match; unknown -> '?' with rare flag."""
    global _V101
    if _V101 is None:
        m, how = _v101_map(); _V101 = (m, sorted(m, key=len, reverse=True))
    m, keys = _V101
    out, i, rare = [], 0, False
    while i < len(word):
        if word[i] == '?':
            out.append('?'); i += 1; continue
        for k in keys:
            if word.startswith(k, i):
                e = m[k]
                if e == '?': rare = True
                out.append(e); i += len(k); break
        else:
            out.append('?'); rare = True; i += 1
    return ''.join(out), rare


def _split_alt(s):
    """Expand '[a:b]' groups -> (first reading, list of all readings up to 32), has_alt."""
    parts = re.split(r'(\[[^\]]*\])', s)
    opts = [[]]
    for p in parts:
        if p.startswith('[') and p.endswith(']'):
            ch = p[1:-1].split(':')
            opts = [o + [c] for o in opts for c in ch][:32]
        else:
            opts = [o + [p] for o in opts]
    readings = [''.join(o) for o in opts]
    return readings[0], readings, ('[' in s)


def parse_word(raw, alphabet='eva'):
    """raw word text (comments already removed) -> dict(w, readings, flags)."""
    fl = set()
    first, readings, has_alt = _split_alt(raw)
    if has_alt: fl.add('alt')
    if '{' in first: fl.add('lig')
    if "'" in first: fl.add('mark')
    if alphabet != 'eva' and re.search(r'@\d+;', first): fl.add('rare')     # v101 high-ascii 'weirdo' glyph

    def norm(s):
        s = s.replace('{', '').replace('}', '').replace("'", '')
        if alphabet == 'eva':
            if re.search(r'@\d+;', s): fl.add('rare')
            s = re.sub(r'@\d+;', '?', s)
            return s
        e, rare = v101_to_eva(s)
        if rare: fl.add('rare')
        return e
    w = norm(first)
    rd = [norm(r) for r in readings]
    if '?' in first: fl.add('ill')
    if alphabet == 'eva' and '?' in w and 'rare' not in fl: fl.add('ill')
    return dict(w=w, readings=sorted(set(rd)), flags=fl)


def parse_line_text(text, alphabet='eva'):
    """IVTFF line body -> (words, seps, para_start, para_end). words: dicts with w, readings, flags, comments."""
    ps = '<%>' in text; pe = '<$>' in text
    text = text.replace('<%>', '').replace('<$>', '')
    comments = []

    def keep(m):
        comments.append(m.group(1)); return '\x01%d\x02' % (len(comments) - 1)
    text = re.sub(r'<!([^>]*)>', keep, text)
    text = text.replace('<->', '\x03').replace('<~>', '\x04')
    text = re.sub(r'<[^>]*>', '', text)     # any other marker
    if alphabet == 'eva':
        toks = re.split(r'([.,\x03\x04])', text)
    else:
        toks = re.split(r'([.,\x03\x04])', text)
    words, seps = [], []
    pend_cmt = []      # comments standing alone between separators attach to the previous word (or the next)
    for i, tk in enumerate(toks):
        if i % 2 == 1:
            if words and len(seps) < len(words):
                seps.append({'.': '.', ',': ',', '\x03': '-', '\x04': '~'}[tk])
            continue
        cm = [comments[int(k)] for k in re.findall(r'\x01(\d+)\x02', tk)]
        body = re.sub(r'\x01\d+\x02', '', tk).strip()
        if not body:
            if cm:
                if words: words[-1]['comments'] += cm
                else: pend_cmt += cm
            continue
        d = parse_word(body, alphabet)
        d['comments'] = pend_cmt + cm; pend_cmt = []
        if words and len(seps) < len(words):
            seps.append('.')      # two words with no separator cannot happen; guard
        words.append(d)
    seps = seps[:max(0, len(words) - 1)]
    for j, d in enumerate(words):
        for c in d['comments']:
            if UNC_COMMENT.search(c): d['flags'].add('cmt')
            if DMG_COMMENT.search(c): d['flags'].add('dmg')
        if (j > 0 and seps[j - 1] == ',') or (j < len(seps) and seps[j] == ','):
            d['flags'].add('usp')
    return words, seps, ps, pe


def parse(name):
    path = SRC[name]
    alphabet = 'eva' if name in ('ZL3b', 'IT2a') else 'v101'
    enc = 'utf-8' if alphabet == 'eva' else 'latin-1'
    recs, page, block_dmg, page_dmg = [], {}, False, set()
    after_header = False
    for raw in open(path, encoding=enc, errors='replace'):
        raw = raw.rstrip('\n')
        if not raw: continue
        if raw.startswith('#'):
            if DMG_PAGE.search(raw):
                page_dmg.add(page.get('folio'))
            elif DMG_BLOCK.search(raw):
                block_dmg = True
            else:
                block_dmg = False      # a damage block ends at the next comment line
            continue
        m = re.match(r'^<(f\w+)>\s*<!(.*)>', raw)
        if m:
            meta = dict(re.findall(r'\$(\w)=(\S)', m.group(2)))
            page = {'folio': m.group(1), 'quire': meta.get('Q'), 'panel': meta.get('P'), 'illus': meta.get('I'),
                    'lang': meta.get('L'), 'hand': meta.get('H')}
            block_dmg = False; after_header = True
            continue
        m = re.match(r'^<(f\w+)\.(\d+),([@+*=&~])(\w)(\w*)>\s*(.*)$', raw)
        if not m: continue
        after_header = False
        folio, n, pre, lt, sub, text = m.groups()
        words, seps, ps, pe = parse_line_text(text, alphabet)
        if not words: continue
        ld = block_dmg or folio in page_dmg
        if ld:
            for d in words: d['flags'].add('ldmg')
        r = dict(page)
        r.update(folio=folio, n=int(n), ltype=lt, ltcode=pre + lt + sub, para_start=ps, para_end=pe,
                 words=[d['w'] for d in words], readings=[d['readings'] for d in words],
                 flags=[sorted(d['flags']) for d in words], comments=[d['comments'] for d in words], seps=seps,
                 line_dmg=ld)
        recs.append(r)
    # page-level damage notes that come after some lines were already read (none in ZL3b, kept for safety)
    for r in recs:
        if r['folio'] in page_dmg and not r['line_dmg']:
            r['line_dmg'] = True; r['flags'] = [sorted(set(f) | {'ldmg'}) for f in r['flags']]
    return recs


# ------------------------------------------------------------------ cross-transcription agreement
def agreement(Z, I, G):
    """Mark, for each word of each transcription, whether the same locus in the other two has the identical word in an
    aligned position (difflib on word sequences). Adds key 'agree' (list of bools) to every record in place.
    Agreement compares first readings."""
    def idx(R): return {(r['folio'], r['n']): r for r in R}
    iz, ii, ig = idx(Z), idx(I), idx(G)

    def match_set(a, b):
        s = set()
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for blk in sm.get_matching_blocks():
            for k in range(blk.size): s.add((blk.a + k, blk.b + k))
        return s
    for R, others in ((Z, (ii, ig)), (I, (iz, ig)), (G, (iz, ii))):
        for r in R:
            key = (r['folio'], r['n'])
            ok = [True] * len(r['words'])
            for o in others:
                ro = o.get(key)
                if ro is None:
                    ok = [False] * len(ok); break
                ms = match_set(r['words'], ro['words'])
                got = {a for a, b in ms}
                ok = [x and (j in got) for j, x in enumerate(ok)]
            r['agree'] = ok
    return Z, I, G


def build(save=True):
    Z, I, G = parse('ZL3b'), parse('IT2a'), parse('GC2a')
    agreement(Z, I, G)
    if save:
        for nm, R in (('ZL3b', Z), ('IT2a', I), ('GC2a', G)):
            json.dump(R, open(os.path.join(CK, nm + '_v83.json'), 'w'))
    return dict(ZL3b=Z, IT2a=I, GC2a=G)


_CACHE = {}


DERIVED = os.path.join(DATA, 'derived')


def write_derived():
    """compact committed copy (data/derived/<name>_v83.json): words, flags, separators, agreement; no readings or
    comments (those stay in the git-ignored checkpoint). Lets a fresh clone use the default loader without GC2a."""
    for nm in ('ZL3b', 'IT2a'):
        R = json.load(open(os.path.join(CK, nm + '_v83.json')))
        out = []
        for r in R:
            rr = {k: r[k] for k in ('folio', 'quire', 'panel', 'illus', 'lang', 'hand', 'n', 'ltype', 'ltcode',
                                    'para_start', 'para_end', 'words', 'seps', 'line_dmg')}
            rr['flags'] = [','.join(f) for f in r['flags']]
            rr['agree'] = ''.join('1' if a else '0' for a in r['agree'])
            out.append(rr)
        json.dump(out, open(os.path.join(DERIVED, nm + '_v83.json'), 'w'), separators=(',', ':'))


def _from_derived(nm):
    R = json.load(open(os.path.join(DERIVED, nm + '_v83.json')))
    for r in R:
        r['flags'] = [f.split(',') if f else [] for f in r['flags']]
        r['agree'] = [c == '1' for c in r['agree']]
    return R


def records(name):
    if name not in _CACHE:
        p = os.path.join(CK, name + '_v83.json')
        if os.path.exists(p):
            _CACHE[name] = json.load(open(p))
        elif os.path.exists(os.path.join(DERIVED, name + '_v83.json')):
            _CACHE[name] = _from_derived(name)
        else:
            build(); _CACHE[name] = json.load(open(p))
    return _CACHE[name]


def keep_word(r, j, mode):
    f = r['flags'][j]
    if mode == 'all': return True
    if mode == 'clean': return not f
    if mode == 'glyph': return not (set(f) & GLYPH_FLAGS)
    if mode == 'agree': return r['agree'][j]
    if mode == 'agreeclean': return r['agree'][j] and not f
    raise ValueError(mode)


def legacy_lines(name):
    return json.load(open(os.path.join(DATA, 'derived', name + '_lines.json')))


def load(name='ZL3b', mode='all', ltypes=None, masked=False, thin=None):
    """Line records like data/derived/<name>_lines.json ('words', 'uncertain', ...), filtered by mode.
    masked=True keeps dropped words as None (for adjacency-aware code); else they are removed.
    thin=(rng, keep_prob_by_line) is used by the random-thinning control (see v83_harness)."""
    if mode == 'legacy':
        out = legacy_lines(name)
        return [r for r in out if not ltypes or r['ltype'] in ltypes]
    out = []
    for r in records(name):
        if ltypes and r['ltype'] not in ltypes: continue
        ws = [w if keep_word(r, j, mode) else None for j, w in enumerate(r['words'])]
        if not masked:
            ws = [w for w in ws if w is not None]
        if not [w for w in ws if w]: continue
        rr = {k: r[k] for k in ('folio', 'quire', 'panel', 'illus', 'lang', 'hand', 'n', 'ltype', 'ltcode',
                                'para_start', 'para_end')}
        rr['words'] = ws
        rr['uncertain'] = [(w is None) or ('?' in w) for w in ws]
        out.append(rr)
    return out


if __name__ == '__main__':
    D = build()
    for nm, R in D.items():
        n = sum(len(r['words']) for r in R)
        c = collections.Counter(f for r in R for fl in r['flags'] for f in fl)
        anyf = sum(1 for r in R for fl in r['flags'] if fl)
        agr = sum(1 for r in R for a in r['agree'] if a)
        print(nm, 'lines', len(R), 'words', n, 'any flag', anyf, round(anyf / n, 4), 'agree', agr, round(agr / n, 4), dict(c))
    write_derived()
