"""v28 corpora: graphic-unit transcriptions of handwritten scripts, other alphabets and Voynich segmentations.
Each corpus = (words as tuples of units, alphabet, fonts, x-height reference).  Cached in data/v28_ckpt/."""
import os, sys, re, glob, unicodedata
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v28_lib as X, v25_lib as L, v25_shapes as S
from v18_lib import glyphs as vglyphs

SCR = X.SCR
ALLO = os.path.join(SCR, 'allo')
MED_FONTS = ('junicode', 'freeserif', 'unifont')

# allograph families (graphic variants of one letter) used for collapsing and for the allograph-pair test
COLLAPSE = {'ſ': 's', 'ꝛ': 'r', 'ɼ': 'r', 'ꝺ': 'd', 'ꞇ': 't', 'ʒ': 'z', 'j': 'i', 'v': 'u', 'Ꝯ': 'ꝯ', 'Ꝺ': 'd'}


def family(g):
    g = g.replace(X.VS_SAME, '').replace(X.VS_ITAL, '')
    b = g.lower() if len(g) == 1 else g
    return COLLAPSE.get(b, COLLAPSE.get(g, b))


# ---------------------------------------------------------------- page XML (PAGE / ALTO)
def xml_lines(files):
    import xml.etree.ElementTree as ET
    out = []
    for f in files:
        t = ET.parse(f)
        for u in t.iter():
            tag = u.tag.split('}')[-1]
            if tag == 'TextLine':
                tes = [te for te in u if te.tag.split('}')[-1] == 'TextEquiv']
                for te in tes[:1]:
                    for un in te:
                        if un.tag.split('}')[-1] == 'Unicode' and un.text:
                            out.append(un.text)
            if tag == 'String' and u.get('CONTENT') is not None:
                out.append(u.get('CONTENT'))
    return out


def cast_lines(hand=None):
    fs = [f for f in sorted(glob.glob(os.path.join(ALLO, 'cast', '**', '*.xml'), recursive=True)) if 'normalized' not in f]
    if hand:
        fs = [f for f in fs if f'/{hand}/' in f]
    return xml_lines(fs)


def enhg_lines():
    return xml_lines(sorted(glob.glob(os.path.join(ALLO, 'enhg', '**', '*.xml'), recursive=True)))


def catmus_lines(lang='Lat'):
    import pyarrow.parquet as pq
    out = []
    for f in sorted(glob.glob(os.path.join(SCR, 'catmus', f'L-{lang}*.parquet'))):
        t = pq.read_table(f, columns=['text', 'region']).to_pylist()
        out += [r['text'] for r in t if r['region'] in ('MainZone', None)]
    return out


# unit functions
def u_allo(w):
    return X.units_nfd(w)


def u_graph(w):  # allographs collapsed, marks kept
    return tuple(family(c) if not unicodedata.combining(c) else c for c in X.units_nfd(w))


def u_plain(w):  # allographs collapsed, marks dropped
    return tuple(family(c) for c in X.units_nfd(w) if not unicodedata.combining(c))


MINIM = {'m': 'ııı', 'n': 'ıı', 'u': 'ıı', 'i': 'ı'}


def u_minim(w):  # allographic + minim letters split into minim strokes (like EVA i-strings)
    out = []
    for c in X.units_nfd(w):
        out.extend(MINIM.get(c, c))
    return tuple(out)


def bpe(words, nmerge):
    """greedy byte-pair merges on units (rendered later as the concatenated string)."""
    ws = [list(w) for w in words]
    merges = []
    for _ in range(nmerge):
        c = Counter()
        for w in ws:
            for a, b in zip(w, w[1:]):
                c[(a, b)] += 1
        if not c:
            break
        (a, b), _ = c.most_common(1)[0]
        merges.append(a + b)
        for i, w in enumerate(ws):
            j, o = 0, []
            while j < len(w):
                if j + 1 < len(w) and w[j] == a and w[j + 1] == b:
                    o.append(a + b); j += 2
                else:
                    o.append(w[j]); j += 1
            ws[i] = o
    return [tuple(w) for w in ws], merges


# ---------------------------------------------------------------- wikipedia scripts
def wiki(lang, nchar=1_500_000):
    import pyarrow.parquet as pq
    f = glob.glob(os.path.join(SCR, 'v28txt', f'20231101.{lang}_*.parquet'))[0]
    pf = pq.ParquetFile(f)
    txt, n = [], 0
    for b in pf.iter_batches(columns=['text'], batch_size=500):
        for t in b.column('text').to_pylist():
            txt.append(t); n += len(t)
            if n > nchar:
                return '\n'.join(txt)
    return '\n'.join(txt)


ETH = (0x1200, 0x137F)


def ethiopic_words(lang):
    t = wiki(lang)
    return [tuple(w) for w in re.findall('[ሀ-፿]+', t)]


def eth_hand(alph):
    """featural decomposition from the Unicode layout (consonant row x vowel order), not hand-drawn."""
    sh = {}
    for g in alph:
        o = ord(g) - 0x1200
        c, v = divmod(o, 8)
        d = {f'C{c}': 1}
        if v:
            d[f'V{v}'] = 1
        sh[g] = d
    return sh


def cyr_words(lang):
    t = wiki(lang).lower()
    if lang == 'ru':
        t = t.replace('ё', 'е')
        return [tuple(w) for w in re.findall('[а-я]+', t)]
    t = unicodedata.normalize('NFC', t)
    t = ''.join(c for c in t if not unicodedata.combining(c) and c not in '҃҄҅҆҇')
    return [tuple(w) for w in re.findall('[Ѐ-ӿꙀ-ꚟ]+', t)]


GLAG = {'а': 'ⰰ', 'б': 'ⰱ', 'в': 'ⰲ', 'г': 'ⰳ', 'д': 'ⰴ', 'є': 'ⰵ', 'е': 'ⰵ', 'ѥ': 'ⰵ', 'ж': 'ⰶ', 'ѕ': 'ⰷ', 'ꙃ': 'ⰷ', 'з': 'ⰸ', 'ꙁ': 'ⰸ',
        'ї': 'ⰹ', 'і': 'ⰹ', 'и': 'ⰻ', 'й': 'ⰻ', 'ћ': 'ⰼ', 'ђ': 'ⰼ', 'к': 'ⰽ', 'л': 'ⰾ', 'м': 'ⰿ', 'н': 'ⱀ', 'о': 'ⱁ', 'ѻ': 'ⱁ', 'п': 'ⱂ',
        'р': 'ⱃ', 'с': 'ⱄ', 'т': 'ⱅ', 'у': 'ⱆ', 'ꙋ': 'ⱆ', 'ѹ': 'ⱆ', 'ф': 'ⱇ', 'х': 'ⱈ', 'ѡ': 'ⱉ', 'ѿ': 'ⱉ', 'щ': 'ⱋ', 'ц': 'ⱌ', 'ч': 'ⱍ',
        'ш': 'ⱎ', 'ъ': 'ⱏ', 'ь': 'ⱐ', 'ѣ': 'ⱑ', 'ꙗ': 'ⱑ', 'я': 'ⱑ', 'ю': 'ⱓ', 'ѧ': 'ⱔ', 'ѩ': 'ⱗ', 'ѫ': 'ⱘ', 'ѭ': 'ⱙ', 'ѳ': 'ⱚ', 'ѵ': 'ⱛ',
        'ꙑ': 'ⱏⰹ', 'ы': 'ⱏⰹ', 'ѯ': 'ⰽⱄ', 'ѱ': 'ⱂⱄ', 'ꙙ': 'ⱔ', 'ꙛ': 'ⱘ'}


def glag_words():
    out = []
    for w in cyr_words('cu'):
        if all(c in GLAG for c in w):
            out.append(tuple(x for c in w for x in GLAG[c]))
    return out


def armenian_words():
    t = wiki('hy').lower()
    return [tuple(w) for w in re.findall('[ա-և]+', t)]


# ---------------------------------------------------------------- Voynich segmentations
def voy_raw(src='ZL3b'):
    import json
    d = json.load(open(os.path.join(L.DATA, 'derived', f'{src}_lines.json')))
    out = []
    for Ln in d:
        if Ln['ltype'] != 'P':
            continue
        for w, u in zip(Ln['words'], Ln['uncertain']):
            if u or '?' in w or not re.fullmatch('[a-z]+', w):
                continue
            out.append(w)
    return out


def seg_v25(w):
    return tuple(vglyphs(w))


def seg_split(w):  # every EVA character its own unit (benches split into c + h, sh = s + h)
    return tuple(w)


def seg_istr(w):  # v25 units, but maximal i-runs (+ following n/r/l/m) and e-runs as single units
    g = list(vglyphs(w)); out = []; k = 0
    while k < len(g):
        if g[k] in ('i', 'e'):
            j = k
            while j < len(g) and g[j] == g[k]:
                j += 1
            u = ''.join(g[k:j])
            if g[k] == 'i' and j < len(g) and g[j] in ('n', 'r', 'l', 'm'):
                u += g[j]; j += 1
            out.append(u); k = j
        else:
            out.append(g[k]); k += 1
    return tuple(out)


def seg_benchsplit_only(w):  # benched gallows split into ch + gallows (cth -> ch t), ch/sh kept
    out = []
    for g in vglyphs(w):
        if g in ('cth', 'ckh', 'cph', 'cfh'):
            out += ['ch', g[1]]
        else:
            out.append(g)
    return tuple(out)


VOY_SEG = dict(v25=seg_v25, split=seg_split, istr=seg_istr, benchsplit=seg_benchsplit_only)


def voy_hand(alph):
    """hand decomposition for multi-unit strings = sum of v25 primitives of their parts (only if all parts known)."""
    sh = {}
    for g in alph:
        parts = vglyphs(g) if g not in S.VOYNICH else [g]
        if all(p in S.VOYNICH for p in parts):
            c = Counter()
            for p in parts:
                c.update(S.VOYNICH[p])
            sh[g] = dict(c)
    return sh


# ---------------------------------------------------------------- registry
def build(name):
    """returns dict(words, alph, fonts, xref, hand(optional))"""
    cached = X.load(f'corp_{name}.pkl')
    if cached is not None:
        return cached
    hand = None
    if name.startswith('cast'):
        _, h, lev = name.split(':')
        lines = cast_lines(None if h == 'pool' else h.replace('-', '/'))
        fn = dict(allo=u_allo, graph=u_graph, plain=u_plain, minim=u_minim)[lev.split('+')[0]]
        ws = X.words_from_lines(lines, fn)
        fonts, xref = MED_FONTS, 'x'
    elif name.startswith('enhg'):
        lev = name.split(':')[1]
        ws = X.words_from_lines(enhg_lines(), dict(allo=u_allo, graph=u_graph, plain=u_plain, minim=u_minim)[lev.split('+')[0]])
        fonts, xref = MED_FONTS, 'x'
    elif name.startswith('catmus'):
        lev = name.split(':')[1]
        ws = X.words_from_lines(catmus_lines('Lat'), dict(allo=u_allo, plain=u_plain, minim=u_minim)[lev.split('+')[0]])
        fonts, xref = MED_FONTS, 'x'
    elif name.startswith('latinprint'):
        ws = L.latin(10 ** 9)
        fonts, xref = ('freeserif', 'junicode', 'unifont'), 'x'
    elif name in ('am', 'ti'):
        ws = ethiopic_words(name)
        fonts, xref = ('notoeth', 'notoserifeth', 'freeserif'), 'ሀ'
    elif name == 'ru':
        ws = cyr_words('ru'); fonts, xref = ('freeserif', 'dejavuserif', 'unifont'), 'х'
    elif name == 'cu':
        ws = cyr_words('cu'); fonts, xref = ('freeserif', 'unifont'), 'х'
    elif name == 'glag':
        ws = glag_words(); fonts, xref = ('notoglag', 'freeserif', 'unifont'), 'ⱁ'
    elif name == 'hy':
        ws = armenian_words(); fonts, xref = ('notoarm', 'notoserifarm', 'freeserif'), 'ա'
    elif name.startswith('voy'):
        _, src, seg = name.split(':')
        ws = [VOY_SEG[seg.split('+')[0]](w) for w in voy_raw(src)]
        if seg == 'v25':
            ws = [w for w in ws if all(g in S.VOYNICH for g in w)]
        fonts, xref = ('eva',), 'o'
    else:
        raise KeyError(name)
    # optional planted positional allographs "+pallo6" (identical drawing) / "+pallo6i" (italic drawing):
    # the word-final form of the 6 most frequent letters becomes a separate unit
    if '+pallo' in name:
        tag = name.split('+pallo')[1]
        k = int(tag.rstrip('i')); mark = X.VS_ITAL if tag.endswith('i') else X.VS_SAME
        cnt = Counter(g for w in ws for g in w)
        tops = {g for g, _ in cnt.most_common(k)}
        ws = [w[:-1] + ((w[-1] + mark,) if w[-1] in tops else (w[-1],)) for w in ws]
    # optional BPE suffix "+bpeN"
    if '+bpe' in name:
        n = int(name.split('+bpe')[1])
        ws, merges = bpe(L._truncate(ws, X.TOK * 2), n)
    ws, A = X.finish(ws)
    if name in ('am', 'ti'):
        hand = eth_hand(A)
    if name.startswith('voy'):
        hand = voy_hand(A)
    # drop units without ink or font coverage in the first font
    A2 = [g for g in A if all(X.covered(g, X.FONT[f]) for f in fonts) and X.has_ink(g, X.FONT[fonts[0]])]
    if len(A2) < len(A):
        As = set(A2)
        ws = [w for w in ws if all(g in As for g in w)]
        ws, A2 = X.finish(ws)
    out = dict(words=ws, alph=A2, fonts=fonts, xref=xref, hand=hand, dropped=sorted(set(A) - set(A2)))
    X.save(f'corp_{name}.pkl', out)
    return out


if __name__ == '__main__':
    for n in sys.argv[1:]:
        c = build(n)
        print(n, len(c['alph']), sum(map(len, c['words'])), ''.join(c['alph'])[:200], 'dropped', c['dropped'])
