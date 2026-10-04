#!/usr/bin/env python3
"""X-3 build: sign-level line corpora for the 'which script teaches which' transfer test.

A corpus = list of lines; a line = list of sign labels (str). Labels are kept as each
transliteration convention writes them, lower-cased for syllabograms, so that corpora which
share a convention (Linear A / Linear B / syllabified Greek; Ur III / Akkadian / Elamite
cuneiform readings; PE / proto-cuneiform numerals) share labels and the others do not.

Corpora (ids):
  LA   Linear A lines (lineara.xyz corpus.json via the X-1 loader); syllabograms lower-case,
       logograms 'L:OLE' -> 'OLE', numbers NUM, fractions FR:x
  PE   Proto-Elamite entry lines (CDLI, X-1 library)
  VOY  Voynich ZL3b paragraph lines, EVA glyphs (X-1 loader)
  LB   Linear B lines split into syllabic signs (DAMOS, X-1 library)
  PC   proto-cuneiform entry lines (CDLI qpc, X-1 library); numerals 'k(N01)' -> 'N01'
  UR3  Ur III administrative tablets (CDLI ATF lang sux, catalogue period Ur III), sign readings
  AKK  Akkadian (CDLI ATF lang akk, any period), sign readings
  ELX  Elamite (CDLI ATF lang elx), sign readings (small, ~LA size)
  LAT  Latin letters (Caesar, Gutenberg, X-1 reference loader)
  GRC  Iliad (Perseus canonical-greekLit TEI) respelled in a Linear-B-like CV syllabary
       (positive control for LB; rules in grc_syll)
Numbers in cuneiform corpora -> NUM; broken signs x / ... dropped; determinatives kept as
their own sign ('d', 'ki', 'gesz').
Sources (scratch, not committed): $X3_SCR/cdli.atf (github cdli-gh/data cdliatf_unblocked.atf,
LFS media URL), $X3_SCR/cdli_cat.csv, $X3_SCR/il1.xml
(raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/tlg0012/tlg001/
tlg0012.tlg001.perseus-grc2.xml). Output: $X3_SCR/x3_corpora.json
"""
import os, sys, re, json, csv, random, unicodedata
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OS = os.path.abspath(os.path.join(HERE, '..', '..'))
SCR = os.environ.get('X3_SCR', '/tmp/claude-0/x3')
sys.path.insert(0, os.path.join(OS, 'voynich', 'tools'))
csv.field_size_limit(10 ** 9)


def norm_sub(s):
    return s.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))


def load_la():
    import x1_battery as B
    out = []
    for t in B.load_linear_a():
        s = []
        for x in t:
            x = norm_sub(x)
            if x.startswith('L:'):
                s.append(x[2:])
            elif x == 'NUM' or x.startswith('FR:') or x.startswith('*'):
                s.append(x)
            else:
                s.append(x.lower())
        out.append(s)
    return out


def load_lib(name):
    import x1_battery as B
    return B._jsonl(name)


def load_pc():
    out = []
    for t in load_lib('proto_cuneiform'):
        out.append([re.sub(r'^\d+\((N\d+)\)$', r'\1', x) for x in t])
    return out


def load_voy():
    import x1_battery as B
    return B.load_voynich_glyph()


def load_lat():
    import x1_battery as B
    return B.load_gutenberg_chars('Latin-Caesar', max_tokens=120000)


# ------------------------------------------------------------------ CDLI ATF
NUMRE = re.compile(r'^(\d+(/\d+)?|n)\(.*\)$|^\d+(/\d+)?$|^n$')


def atf_tokens(line):
    line = re.sub(r'^\S+\.\s*', '', line)               # line number
    line = re.sub(r'\$.*', '', line)
    line = line.replace('_', ' ')
    line = re.sub(r'[#?!*\[\]<>⸢⸣«»]', '', line)
    toks = []
    for w in line.split():
        if w.startswith('($') or w in ('...',):
            continue
        w = re.sub(r'\{([^}]*)\}', r'-\1-', w)           # determinatives as separate signs
        for s in re.split(r'[-.:+]', w):
            s = s.strip().lower()
            if not s:
                continue
            if NUMRE.match(s):
                toks.append('NUM'); continue
            s = re.sub(r'\(.*\)$', '', s)                  # sign-name qualifier
            s = s.strip('|()')
            if not s or s in ('x', '...', 'n') or re.search(r'[^a-z0-9šṣṭḫŋ]', s):
                continue
            toks.append(s)
    return toks


def load_cdli():
    per = {}
    for r in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8', errors='replace')):
        per[r['id_text'].zfill(6)] = r.get('period', '')
    out = {'UR3': [], 'AKK': [], 'ELX': []}
    pid, lang = None, None
    for L in open(os.path.join(SCR, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if L.startswith('&'):
            m = re.match(r'&P(\d+)', L); pid = m.group(1) if m else None; lang = None
        elif L.startswith('#atf: lang'):
            p = L.split(); lang = p[2] if len(p) > 2 else None
        elif re.match(r"^\d+'?\.", L) and pid:
            key = None
            if lang == 'sux' and per.get(pid, '').startswith('Ur III'):
                key = 'UR3'
            elif lang == 'akk':
                key = 'AKK'
            elif lang == 'elx':
                key = 'ELX'
            if key:
                t = atf_tokens(L.strip())
                if len(t) >= 1:
                    out[key].append((pid, t))
    return out


# ------------------------------------------------------------------ Greek -> CV syllabary
GMAP = {'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'ζ': 'z', 'η': 'e', 'θ': 't', 'ι': 'i',
        'κ': 'k', 'λ': 'r', 'μ': 'm', 'ν': 'n', 'ξ': 'X', 'ο': 'o', 'π': 'p', 'ρ': 'r', 'σ': 's',
        'ς': 's', 'τ': 't', 'υ': 'u', 'φ': 'p', 'χ': 'k', 'ψ': 'P', 'ω': 'o'}
STOP = {'b': 'p', 'g': 'k'}
V = set('aeiou')


def grc_syll(word):
    """Approximate Linear B spelling: l->r, aspirates and voiced labial/velar -> plain stops,
    final consonants and pre-consonantal liquids/nasals/s dropped, second vowel of a diphthong
    dropped (ai ei oi au eu ou -> a e o a e o), other clusters with the following vowel echoed."""
    w = ''.join(c for c in unicodedata.normalize('NFD', word.lower()) if unicodedata.category(c) != 'Mn')
    s = ''.join(GMAP.get(c, '') for c in w).replace('X', 'ks').replace('P', 'ps')
    s = ''.join(STOP.get(c, c) for c in s)
    s = re.sub(r'([aeo])[iu]', r'\1', s)
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c in V:
            out.append(c); i += 1; continue
        # consonant: find next vowel
        j = i
        while j < len(s) and s[j] not in V:
            j += 1
        if j >= len(s):
            break                                          # final consonants dropped
        cl = s[i:j]
        vow = s[j]
        if len(cl) > 1 and cl[0] in 'rmns':
            cl = cl[1:]                                    # pre-consonantal r m n s dropped
        if len(cl) > 1 and cl[0] == 's':
            cl = cl[1:]
        for k, cc in enumerate(cl):
            if cc == 'w':
                cc = 'w'
            out.append(cc + vow)
        i = j + 1
    return out


def load_grc():
    txt = open(os.path.join(SCR, 'il1.xml'), encoding='utf-8').read()
    lines = re.findall(r'<l[^>]*>(.*?)</l>', txt, flags=re.S)
    out = []
    for L in lines:
        L = re.sub(r'<[^>]+>', ' ', L)
        s = []
        for w in re.findall(r'[^\W\d_]+', L):
            s += grc_syll(w)
        if s:
            out.append(s)
    return out


LB_DROP = {'mut.', 'inf.', 'sup.', 'deest', '•', ':', '.a', '.b', 'vacat', 'v.', 'lat.', 'α', 'β', 'γ'}


def clean_lb(lines):
    out = []
    for t in lines:
        s = []
        for x in t:
            x = ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn')
            if not x or x in LB_DROP or re.fullmatch(r'[α-ω]+', x):
                continue
            if re.fullmatch(r'\d+', x):
                x = 'NUM'
            s.append(x)
        if s:
            out.append(s)
    return out


def main():
    C = {}
    C['LA'] = load_la(); C['PE'] = load_lib('proto_elamite'); C['VOY'] = load_voy()
    C['LB'] = clean_lb(load_lib('linb_syll')); C['PC'] = load_pc(); C['LAT'] = load_lat()
    C['GRC'] = load_grc()
    cd = load_cdli()
    rng = random.Random(3)
    for k, v in cd.items():
        # random documents until 120k tokens (keeps document lines together)
        docs = {}
        for pid, t in v:
            docs.setdefault(pid, []).append(t)
        ids = sorted(docs); rng.shuffle(ids)
        lines, n = [], 0
        for i in ids:
            for t in docs[i]:
                lines.append(t); n += len(t)
            if n >= 120000:
                break
        C[k] = lines
    for k, v in C.items():
        c = Counter(x for t in v for x in t)
        print(k, len(v), 'lines', sum(map(len, v)), 'tokens', len(c), 'types', c.most_common(8))
    json.dump(C, open(os.path.join(SCR, 'x3_corpora.json'), 'w'))


if __name__ == '__main__':
    main()
