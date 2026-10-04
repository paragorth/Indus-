#!/usr/bin/env python3
"""X-4 build: one battery for three scripts. Document corpora for the language-vs-generator battery.

Every corpus = list of documents; document = list of lines; line = list of words; word = list of sign labels.
Two segmentations for list-genre corpora:
  W  the scribes' own word division (LA, LB, Ur III, Voynich, prose)
  E  'entry as word': all non-numeral signs of an entry line joined into one unit (the only option for PE and
     proto-cuneiform, which write no word divider; LA_E, LB_E, UR3_E are its calibration)
Numerals are dropped everywhere (the Voynich and prose have none); broken signs (x, ...) dropped.

Corpora
  LA, LA_E    Linear A (lineara.xyz corpus.json), syllabograms + logograms as single-sign words
  PE_E        Proto-Elamite (CDLI, data/pe_corpus.json), entry sign groups
  PC_E        proto-cuneiform Uruk IV/III admin (data/pe2_pc_corpus.json), entry sign groups
  LB, LB_E    Linear B commodity accounts (DAMOS via data/x2/corpus_LB.json): designation words + logogram
  UR3, UR3_E  Ur III Sumerian admin tablets (CDLI ATF, scratch), sign readings, ATF words
  GRC         Iliad respelled in a Linear-B-like CV syllabary, words kept (syllabic PROSE: script-type control)
  VOY         Voynich ZL3b pages, EVA glyph units
  LAT ITA DEU CES   prose (letters), from the Voynich v33 loaders, pseudo-documents of 6 lines
Output: data/x4_ckpt/corpora.json (git-ignored).
"""
import os, sys, re, json, random, unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PE = os.path.dirname(HERE)
OS = os.path.dirname(PE)
CK = os.path.join(PE, 'data', 'x4_ckpt'); os.makedirs(CK, exist_ok=True)
sys.path.insert(0, os.path.join(OS, 'voynich', 'tools'))
sys.path.insert(0, HERE)
X3SCR = '/tmp/claude-0/x3'
BAD = {'x', 'X', '...', '', 'n', 'N'}


def norm(s):
    s = s.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def load_la():
    d = json.load(open(os.path.join(OS, 'linear-a', 'data', 'corpus.json')))
    W, E = [], []
    for ins in d:
        doc, docE, cur = [], [], []
        def flush():
            if cur:
                doc.append(list(cur)); docE.append([[s for w in cur for s in w]])
        for t in ins['tokens']:
            if t['t'] == 'nl':
                flush(); cur = []
            elif t['t'] == 'word':
                w = [norm(x).lower() for x in t['s'] if x not in BAD]
                if w: cur.append(w)
            elif t['t'] == 'logo':
                cur.append(['L:' + norm(t['v'])])
        flush()
        if doc: W.append(doc); E.append(docE)
    return W, E


def load_tab(fn, keepnum=False):
    out = []
    for tab in json.load(open(fn)):
        doc = []
        for L in tab['lines']:
            s = [x for x in L['signs'] if x.lower() not in BAD and not x.startswith('[')]
            if s: doc.append([s])
        if doc: out.append(doc)
    return out


def load_lb():
    W, E = [], []
    for tab in json.load(open(os.path.join(PE, 'data', 'x2', 'corpus_LB.json'))):
        doc = []
        for e in tab['entries']:
            ws = [[norm(s) for s in w.split('-') if s and s not in BAD] for w in e['des']]
            ws = [w for w in ws if w]
            if e.get('com'): ws.append(['L:' + e['com']])
            if ws: doc.append(ws)
        if doc:
            W.append(doc); E.append([[[s for w in l for s in w]] for l in doc])
    return W, E


NUMRE = re.compile(r'^(\d+(/\d+)?|n)\(.*\)$|^\d+(/\d+)?$|^n$')


def ur3_words(line):
    line = re.sub(r'^\S+\.\s*', '', line)
    line = re.sub(r'\$.*', '', line).replace('_', ' ')
    line = re.sub(r'[#?!*\[\]<>⸢⸣«»]', '', line)
    ws = []
    for w in line.split():
        if w.startswith('($') or w == '...': continue
        w = re.sub(r'\{([^}]*)\}', r'-\1-', w)
        sg = []
        for s in re.split(r'[-.:+]', w):
            s = s.strip().lower()
            if not s: continue
            if NUMRE.match(s): sg = None; break           # the numeral word is dropped
            s = re.sub(r'\(.*\)$', '', s).strip('|()')
            if not s or s in ('x', '...', 'n') or re.search(r'[^a-z0-9šṣṭḫŋ]', s): continue
            sg.append(s)
        if sg: ws.append(sg)
    return ws


def load_ur3(max_tok=45000, seed=3):
    import csv
    csv.field_size_limit(10 ** 9)
    per = {}
    for r in csv.DictReader(open(os.path.join(X3SCR, 'cdli_cat.csv'), encoding='utf-8', errors='replace')):
        per[r['id_text'].zfill(6)] = (r.get('period', ''), r.get('genre', ''))
    docs, pid, lang = {}, None, None
    for L in open(os.path.join(X3SCR, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if L.startswith('&'):
            m = re.match(r'&P(\d+)', L); pid = m.group(1) if m else None; lang = None
        elif L.startswith('#atf: lang'):
            p = L.split(); lang = p[2] if len(p) > 2 else None
        elif pid and lang == 'sux' and re.match(r"^\d+'?\.", L):
            pr, ge = per.get(pid, ('', ''))
            if pr.startswith('Ur III') and 'Administrative' in ge:
                ws = ur3_words(L.strip())
                if ws: docs.setdefault(pid, []).append(ws)
    ids = sorted(docs); random.Random(seed).shuffle(ids)
    W, n = [], 0
    for i in ids:
        W.append(docs[i]); n += sum(len(l) for l in docs[i])
        if n >= max_tok: break
    E = [[[[s for w in l for s in w]] for l in d] for d in W]
    return W, E


def load_grc(max_tok=45000):
    from x3_build import grc_syll
    txt = open(os.path.join(X3SCR, 'il1.xml'), encoding='utf-8').read()
    lines = []
    for L in re.findall(r'<l[^>]*>(.*?)</l>', txt, flags=re.S):
        L = re.sub(r'<[^>]+>', ' ', L)
        ws = [grc_syll(w) for w in re.findall(r'[^\W\d_]+', L)]
        ws = [w for w in ws if w]
        if ws: lines.append(ws)
    return chunk(lines, max_tok)


def chunk(lines, max_tok, per=6):
    docs, n = [], 0
    for i in range(0, len(lines), per):
        d = lines[i:i + per]; docs.append(d); n += sum(len(l) for l in d)
        if n >= max_tok: break
    return docs


def load_voy():
    import v21_lib as V
    return [[[list(w) for w in l] for pa in p['paras'] for l in pa] for p in V.voynich_pages('ZL3b', minw=20)]


def load_prose(code, max_tok=45000):
    import v33_lib as V3
    lines = V3.lang_lines(code)
    return chunk([[list(w) for w in l] for l in lines[200:]], max_tok)


def main():
    C = {}
    C['LA'], C['LA_E'] = load_la()
    C['PE_E'] = load_tab(os.path.join(PE, 'data', 'pe_corpus.json'))
    C['PC_E'] = load_tab(os.path.join(PE, 'data', 'pe2_pc_corpus.json'))
    C['LB'], C['LB_E'] = load_lb()
    C['UR3'], C['UR3_E'] = load_ur3()
    C['GRC'] = load_grc()
    C['VOY'] = load_voy()
    for k, code in (('LAT', 'la'), ('ITA', 'it'), ('DEU', 'de'), ('CES', 'cs')):
        C[k] = load_prose(code)
    for k, v in C.items():
        toks = [w for d in v for l in d for w in l]
        sg = Counter(s for w in toks for s in w)
        print(f'{k:6s} docs {len(v):5d} words {len(toks):6d} types {len(set(map(tuple, toks))):6d} '
              f'signs {sum(sg.values()):6d} sign-types {len(sg):5d} mean wlen {sum(map(len, toks)) / len(toks):.2f}')
    json.dump(C, open(os.path.join(CK, 'corpora.json'), 'w'))


if __name__ == '__main__':
    main()
