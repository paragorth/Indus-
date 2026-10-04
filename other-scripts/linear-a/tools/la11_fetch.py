#!/usr/bin/env python3
"""LA-11 data: open texts in ~40 real languages -> CV-syllable tables (the raw texts are NOT kept).

For each language: download a small open corpus (Universal Dependencies treebank, eBible.org Bible, or CDLI ATF),
romanize to a phoneme string (language-specific digraph tables; vowels collapsed to a e i o u),
segment every word type into open syllables "as a syllabary would write them" under three schemes:
  DROP   only the consonant right before a vowel is written (cluster-initial and final consonants dropped)
  ECHO   every consonant of a cluster is written with the following vowel (ko-no-so style); final consonants dropped
  MIXED  s / sonorants before another consonant dropped, other cluster consonants echoed; final consonants dropped
and store, per scheme: syllable unigram + bigram counts over the word TYPES of a training half (hash split),
and a held-out list of up to 1500 test types whose syllables are re-coded to opaque integers (identity hidden).
Output: data/la11/lang/<code>.json, data/la11/sources.json.  Raw downloads live in a scratch dir and are deleted.
"""
import os, re, io, sys, json, zipfile, hashlib, random, unicodedata, collections, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'la11')
LANGD = os.path.join(OUT, 'lang')
SCR = os.environ.get('LA11_SCRATCH', '/tmp/la11_raw')
UD = 'https://raw.githubusercontent.com/UniversalDependencies/{repo}/master/{pre}-ud-{part}.conllu'
MAXBYTES = 6_000_000      # per UD file (range request)
MAXTOK = 250_000          # word tokens read per language

# code, name, family, source, args
LANGS = [
    ('grc', 'Ancient Greek', 'IE-Greek', 'ud', ('UD_Ancient_Greek-Perseus', 'grc_perseus', 'form')),
    ('ell', 'Modern Greek', 'IE-Greek', 'ud', ('UD_Greek-GDT', 'el_gdt', 'form')),
    ('lat', 'Latin', 'IE-Italic', 'ud', ('UD_Latin-Perseus', 'la_perseus', 'form')),
    ('ita', 'Italian', 'IE-Italic', 'ud', ('UD_Italian-ISDT', 'it_isdt', 'form')),
    ('hit', 'Hittite', 'IE-Anatolian', 'ud', ('UD_Hittite-HitTB', 'hit_hittb', 'form')),
    ('san', 'Sanskrit (Vedic)', 'IE-Indo-Aryan', 'ud', ('UD_Sanskrit-Vedic', 'sa_vedic', 'form')),
    ('hin', 'Hindi', 'IE-Indo-Aryan', 'ud', ('UD_Hindi-HDTB', 'hi_hdtb', 'Translit')),
    ('lit', 'Lithuanian', 'IE-Baltic', 'ud', ('UD_Lithuanian-ALKSNIS', 'lt_alksnis', 'form')),
    ('hye', 'Armenian', 'IE-Armenian', 'ud', ('UD_Armenian-ArmTDP', 'hy_armtdp', 'Translit')),
    ('rus', 'Russian', 'IE-Slavic', 'ud', ('UD_Russian-GSD', 'ru_gsd', 'form')),
    ('cym', 'Welsh', 'IE-Celtic', 'ud', ('UD_Welsh-CCG', 'cy_ccg', 'form')),
    ('fin', 'Finnish', 'Uralic', 'ud', ('UD_Finnish-TDT', 'fi_tdt', 'form')),
    ('est', 'Estonian', 'Uralic', 'ud', ('UD_Estonian-EDT', 'et_edt', 'form')),
    ('hun', 'Hungarian', 'Uralic', 'ud', ('UD_Hungarian-Szeged', 'hu_szeged', 'form')),
    ('tur', 'Turkish', 'Turkic', 'ud', ('UD_Turkish-IMST', 'tr_imst', 'form')),
    ('uig', 'Uyghur', 'Turkic', 'ud', ('UD_Uyghur-UDT', 'ug_udt', 'Translit')),
    ('eus', 'Basque', 'Isolate-Basque', 'ud', ('UD_Basque-BDT', 'eu_bdt', 'form')),
    ('kat', 'Georgian', 'Kartvelian', 'ud', ('UD_Georgian-GLC', 'ka_glc', 'Translit')),
    ('akk', 'Akkadian', 'AfroAsiatic-Semitic', 'ud', ('UD_Akkadian-RIAO', 'akk_riao', 'form')),
    ('arb', 'Arabic', 'AfroAsiatic-Semitic', 'ud', ('UD_Arabic-PADT', 'ar_padt', 'Translit')),
    ('hbo', 'Ancient Hebrew', 'AfroAsiatic-Semitic', 'ud', ('UD_Ancient_Hebrew-PTNK', 'hbo_ptnk', 'Translit')),
    ('mlt', 'Maltese', 'AfroAsiatic-Semitic', 'ud', ('UD_Maltese-MUDT', 'mt_mudt', 'form')),
    ('amh', 'Amharic', 'AfroAsiatic-Semitic', 'ud', ('UD_Amharic-ATT', 'am_att', 'Translit')),
    ('cop', 'Coptic', 'AfroAsiatic-Egyptian', 'ud', ('UD_Coptic-Scriptorium', 'cop_scriptorium', 'mwt')),
    ('hau', 'Hausa', 'AfroAsiatic-Chadic', 'bible', ('hauulb',)),
    ('tam', 'Tamil', 'Dravidian', 'ud', ('UD_Tamil-TTB', 'ta_ttb', 'Translit')),
    ('tel', 'Telugu', 'Dravidian', 'ud', ('UD_Telugu-MTG', 'te_mtg', 'Translit')),
    ('kor', 'Korean', 'Koreanic', 'ud', ('UD_Korean-Kaist', 'ko_kaist', 'form')),
    ('jpn', 'Japanese', 'Japonic', 'ud', ('UD_Japanese-GSD', 'ja_gsd', 'form')),
    ('cmn', 'Mandarin', 'Sino-Tibetan', 'ud', ('UD_Chinese-GSD', 'zh_gsd', 'form')),
    ('ind', 'Indonesian', 'Austronesian', 'ud', ('UD_Indonesian-GSD', 'id_gsd', 'form')),
    ('tgl', 'Tagalog', 'Austronesian', 'bible', ('tglulb',)),
    ('haw', 'Hawaiian', 'Austronesian', 'bible', ('haw1868',)),
    ('yor', 'Yoruba', 'NigerCongo', 'ud', ('UD_Yoruba-YTB', 'yo_ytb', 'form')),
    ('wol', 'Wolof', 'NigerCongo', 'ud', ('UD_Wolof-WTB', 'wo_wtb', 'form')),
    ('swh', 'Swahili', 'NigerCongo', 'bible', ('swhulb',)),
    ('quh', 'Quechua (S. Bolivian)', 'Quechuan', 'bible', ('quhNT',)),
    ('nhe', 'Nahuatl (Huasteca)', 'UtoAztecan', 'bible', ('nheBl',)),
    ('sux', 'Sumerian', 'Isolate-Sumerian', 'cdli', ('sux',)),
]

VOW = set('aeiou')
VCOLL = {'ä': 'e', 'æ': 'e', 'ö': 'o', 'ø': 'o', 'ü': 'u', 'ı': 'i', 'ə': 'e', 'ɛ': 'e', 'ɔ': 'o', 'ɨ': 'i', 'å': 'o',
         'õ': 'o', 'ë': 'e', 'ɐ': 'a', 'ɑ': 'a', 'ʊ': 'u', 'ɪ': 'i', 'ẹ': 'e', 'ọ': 'o', 'ụ': 'u', 'ị': 'i', 'ơ': 'o', 'ư': 'u'}
APOS = set("'’ʼʾʿ`ʻʼ")
SON = set(['s', 'š', 'z', 'n', 'm', 'r', 'l', 'ŋ', 'ɲ', 'ʎ', 'j', 'w', 'ɬ', 'ś', 'ṣ', 'ṇ', 'ṃ', 'ṁ', 'ṅ', 'ñ', 'ṛ', 'ḷ'])

# language-specific digraph / letter tables (applied to lowercase NFC text, longest match first)
DIG = {
    'default': {},
    'lat': {'qu': 'kw', 'ph': 'f', 'th': 't', 'ch': 'k', 'c': 'k', 'x': 'ks', 'y': 'i', 'v': 'w', 'j': 'j'},
    'ita': {'gli': 'ʎi', 'gn': 'ɲ', 'sci': 'ši', 'sce': 'še', 'sc': 'sk', 'chi': 'ki', 'che': 'ke', 'ghi': 'gi', 'ghe': 'ge',
            'ci': 'či', 'ce': 'če', 'gi': 'ǰi', 'ge': 'ǰe', 'c': 'k', 'qu': 'kw', 'h': '', 'z': 'ʦ'},
    'lit': {'ch': 'x', 'dž': 'ǰ', 'dz': 'ʣ', 'c': 'ʦ', 'y': 'i'},
    'cym': {'ch': 'x', 'dd': 'ð', 'ff': 'f', 'f': 'v', 'll': 'ɬ', 'ng': 'ŋ', 'ph': 'f', 'rh': 'r', 'th': 'θ', 'y': 'e', 'c': 'k', 'w': 'u'},
    'fin': {'y': 'ü', 'ng': 'ŋ'},
    'est': {'y': 'ü', 'õ': 'o', 'š': 'š'},
    'hun': {'sz': 'S', 'cs': 'č', 'zs': 'ž', 'gy': 'ɟ', 'ny': 'ɲ', 'ty': 'ć', 'ly': 'j', 'dzs': 'ǰ', 's': 'š', 'c': 'ʦ', 'S': 's'},
    'tur': {'ç': 'č', 'ş': 'š', 'ğ': '', 'c': 'ǰ', 'y': 'j'},
    'uig': {'ch': 'č', 'sh': 'š', 'zh': 'ž', 'gh': 'ɣ', 'ng': 'ŋ', 'j': 'ǰ', 'y': 'j', 'c': 'ǰ'},
    'eus': {'tx': 'č', 'tz': 'ʦ', 'ts': 'ʦ', 'x': 'š', 'rr': 'r', 'll': 'ʎ', 'ñ': 'ɲ', 'h': '', 'j': 'x', 'y': 'j'},
    'kat': {'ch': 'č', 'sh': 'š', 'zh': 'ž', 'dz': 'ʣ', 'ts': 'ʦ', 'gh': 'ɣ', 'kh': 'x'},
    'mlt': {'għ': '', 'ċ': 'č', 'ġ': 'ǰ', 'ħ': 'ħ', 'x': 'š', 'ż': 'z', 'z': 'ʦ', 'q': 'ʔ', 'ie': 'i', 'j': 'j', 'h': ''},
    'hau': {'sh': 'š', 'ts': 'ʦ', 'ƙ': 'q', 'ɓ': 'ƀ', 'ɗ': 'đ', 'c': 'č', 'j': 'ǰ', 'y': 'j', "'y": 'j'},
    'ind': {'ng': 'ŋ', 'ny': 'ɲ', 'sy': 'š', 'kh': 'x', 'c': 'č', 'j': 'ǰ', 'y': 'j'},
    'tgl': {'ng': 'ŋ', 'y': 'j', 'c': 'k', 'ch': 'č', 'qu': 'k', 'j': 'h', 'v': 'b', 'f': 'p'},
    'haw': {"ʻ": 'ʔ', "'": 'ʔ', '‘': 'ʔ'},
    'yor': {'gb': 'ƀ', 'ṣ': 'š', 'j': 'ǰ', 'y': 'j'},
    'wol': {'ñ': 'ɲ', 'c': 'č', 'j': 'ǰ', 'y': 'j'},
    'swh': {'ch': 'č', 'sh': 'š', "ng'": 'ŋ', 'ny': 'ɲ', 'dh': 'ð', 'th': 'θ', 'gh': 'ɣ', 'j': 'ǰ', 'y': 'j'},
    'quh': {'chh': 'č', "ch'": 'č', 'ch': 'č', 'sh': 'š', 'll': 'ʎ', 'ñ': 'ɲ', "k'": 'k', "q'": 'q', "p'": 'p', "t'": 't',
            'kh': 'k', 'qh': 'q', 'ph': 'p', 'th': 't', 'j': 'h', 'y': 'j', 'hu': 'w', 'c': 'k'},
    'nhe': {'tl': 'ƛ', 'tz': 'ʦ', 'ch': 'č', 'x': 'š', 'hu': 'w', 'uh': 'w', 'qui': 'ki', 'que': 'ke', 'qu': 'k', 'c': 'k', 'j': 'h', 'y': 'j'},
    'hit': {'š': 's', 'ḫ': 'h'},
    'akk': {},
    'sux': {},
    'san': {'kh': 'kʰ', 'gh': 'gʰ', 'ch': 'cʰ', 'jh': 'jʰ', 'ṭh': 'ṭʰ', 'ḍh': 'ḍʰ', 'th': 'tʰ', 'dh': 'dʰ', 'ph': 'pʰ', 'bh': 'bʰ',
            'ai': 'ai', 'au': 'au', 'y': 'j', 'ṃ': 'ṁ'},
    'hin': {'kh': 'kʰ', 'gh': 'gʰ', 'ch': 'cʰ', 'jh': 'jʰ', 'ṭh': 'ṭʰ', 'ḍh': 'ḍʰ', 'th': 'tʰ', 'dh': 'dʰ', 'ph': 'pʰ', 'bh': 'bʰ', 'y': 'j'},
    'tam': {'y': 'j'}, 'tel': {'kh': 'kʰ', 'gh': 'gʰ', 'ch': 'cʰ', 'th': 'tʰ', 'dh': 'dʰ', 'ph': 'pʰ', 'bh': 'bʰ', 'y': 'j'},
    'hye': {'ch': 'čʰ', 'sh': 'š', 'zh': 'ž', 'gh': 'ɣ', 'kh': 'x', 'ts': 'ʦ', 'dz': 'ʣ', 'y': 'j', 'w': 'v'},
    'arb': {'y': 'j', 'ẗ': 't'},
    'hbo': {'y': 'j', 'sh': 'š', 'ts': 'ʦ', 'kh': 'x', 'ch': 'x'},
    'amh': {'y': 'j', 'sh': 'š', 'ch': 'č'},
    'jpn': {'shi': 'ši', 'sh': 'š', 'chi': 'či', 'ch': 'č', 'tsu': 'ʦu', 'ts': 'ʦ', 'fu': 'ɸu', 'j': 'ǰ', 'y': 'j'},
    'cmn': {'zh': 'ẑ', 'ch': 'ĉ', 'sh': 'ŝ', 'ng': 'ŋ', 'y': 'j', 'v': 'ü', 'c': 'ʦ', 'z': 'ʣ', 'q': 'ć', 'x': 'ś', 'j': 'ʥ'},
    'kor': {}, 'rus': {}, 'grc': {}, 'ell': {}, 'cop': {},
}
VOWEL_Y = {'fin', 'est'}  # 'y' handled via table already

CYR = dict(zip('абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
               ['a', 'b', 'v', 'g', 'd', 'je', 'jo', 'ž', 'z', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'f', 'x', 'ʦ', 'č', 'š', 'šč', '', 'i', '', 'e', 'ju', 'ja']))
GRC = {'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'ζ': 'z', 'η': 'e', 'θ': 'tʰ', 'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm', 'ν': 'n',
       'ξ': 'ks', 'ο': 'o', 'π': 'p', 'ρ': 'r', 'σ': 's', 'ς': 's', 'τ': 't', 'υ': 'u', 'φ': 'pʰ', 'χ': 'kʰ', 'ψ': 'ps', 'ω': 'o'}
ELL = {'α': 'a', 'β': 'v', 'γ': 'ɣ', 'δ': 'ð', 'ε': 'e', 'ζ': 'z', 'η': 'i', 'θ': 'θ', 'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm', 'ν': 'n',
       'ξ': 'ks', 'ο': 'o', 'π': 'p', 'ρ': 'r', 'σ': 's', 'ς': 's', 'τ': 't', 'υ': 'i', 'φ': 'f', 'χ': 'x', 'ψ': 'ps', 'ω': 'o'}
ELLDIG = [('ου', 'u'), ('αι', 'e'), ('ει', 'i'), ('οι', 'i'), ('υι', 'i'), ('μπ', 'b'), ('ντ', 'd'), ('γκ', 'g'), ('γγ', 'g'), ('τσ', 'ʦ'), ('τζ', 'ʣ'), ('αυ', 'av'), ('ευ', 'ev')]
COP = {'ⲁ': 'a', 'ⲃ': 'b', 'ⲅ': 'g', 'ⲇ': 'd', 'ⲉ': 'e', 'ⲍ': 'z', 'ⲏ': 'e', 'ⲑ': 'tʰ', 'ⲓ': 'i', 'ⲕ': 'k', 'ⲗ': 'l', 'ⲙ': 'm', 'ⲛ': 'n',
       'ⲝ': 'ks', 'ⲟ': 'o', 'ⲡ': 'p', 'ⲣ': 'r', 'ⲥ': 's', 'ⲧ': 't', 'ⲩ': 'u', 'ⲫ': 'pʰ', 'ⲭ': 'kʰ', 'ⲯ': 'ps', 'ⲱ': 'o', 'ϣ': 'š',
       'ϥ': 'f', 'ϧ': 'x', 'ϩ': 'h', 'ϫ': 'ǰ', 'ϭ': 'č', 'ϯ': 'ti'}
# Korean jamo
KI = ['g', 'kk', 'n', 'd', 'tt', 'r', 'm', 'b', 'pp', 's', 'ss', '', 'ǰ', 'ǰǰ', 'č', 'kʰ', 'tʰ', 'pʰ', 'h']
KV = ['a', 'e', 'ja', 'je', 'o', 'je', 'jo', 'je', 'o', 'wa', 'we', 'we', 'jo', 'u', 'wo', 'we', 'wi', 'ju', 'u', 'ui', 'i']
KF = ['', 'k', 'k', 'ks', 'n', 'nǰ', 'nh', 't', 'l', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh', 'm', 'p', 'ps', 't', 't', 'ŋ', 't', 't', 'k', 't', 'p', 't']

def kor_roman(w):
    out = []
    for ch in w:
        o = ord(ch) - 0xAC00
        if 0 <= o < 11172:
            out.append(KI[o // 588] + KV[(o % 588) // 28] + KF[o % 28])
        else:
            return None
    return ''.join(out)

_kks = None
def jpn_roman(w):
    global _kks
    if _kks is None:
        import pykakasi; _kks = pykakasi.kakasi()
    r = ''.join(x['hepburn'] for x in _kks.convert(w))
    return r if re.fullmatch(r'[a-z]+', r or '') else None

def cmn_roman(w):
    import pypinyin
    if not all('一' <= c <= '鿿' for c in w): return None
    return ''.join(pypinyin.lazy_pinyin(w, style=pypinyin.Style.NORMAL, v_to_u=True))

def grc_roman(w, table, digs):
    w = unicodedata.normalize('NFD', w.lower())
    rough = '̔' in w[:3]
    w = ''.join(c for c in w if unicodedata.category(c) != 'Mn')
    w = unicodedata.normalize('NFC', w)
    for a, b in digs: w = w.replace(a, b)
    if any(c not in table and c not in 'abcdefghijklmnopqrstuvwxyzʰðɣθʦʣ' for c in w): return None
    r = ''.join(table.get(c, c) for c in w)
    return ('h' + r) if rough else r

def sux_word(t):
    t = re.sub(r'\{[^}]*\}', '', t)
    t = re.sub(r'[\[\]#?!<>*]', '', t)
    if not t or re.search(r'[A-Z(]|\.\.\.|^x$|-x-|-x$|^x-', t): return None
    t = t.replace('sz', 'š').replace("s,", 'ṣ').replace("t,", 'ṭ').replace('j', 'ŋ')
    t = re.sub(r'\d+', '', t).replace('-', '').replace('_', '')
    return t if re.fullmatch(r'[a-zšṣṭŋ]+', t) else None

def hit_word(t):
    t = re.sub(r'[\[\]#?!<>*⸢⸣]', '', t)
    if not t or re.search(r'[A-Z0-9]', t) or '...' in t: return None
    t = t.replace('-', '').replace('=', '')
    return t if re.fullmatch(r'[a-zšḫṣṭâêîûáéíúàèìù]+', t) else None

def romanize(code, w):
    """-> lowercase roman string or None (word rejected)."""
    if code == 'kor': return kor_roman(w)
    if code == 'jpn': return jpn_roman(w)
    if code == 'cmn': return cmn_roman(w)
    if code == 'grc': return grc_roman(w, GRC, [('ου', 'u'), ('ει', 'e'), ('γγ', 'ŋg'), ('γκ', 'ŋk')])
    if code == 'ell': return grc_roman(w, ELL, ELLDIG)
    if code == 'rus':
        w = w.lower()
        if not all(c in CYR for c in w): return None
        return ''.join(CYR[c] for c in w)
    if code == 'cop':
        w = unicodedata.normalize('NFD', w.lower()); w = ''.join(c for c in w if unicodedata.category(c) != 'Mn')
        w = w.replace('ⲟⲩ', 'u')
        if not all(c in COP for c in w): return None
        return ''.join(COP[c] for c in w)
    if code == 'sux': return sux_word(w)
    if code == 'hit': return hit_word(w)
    w = unicodedata.normalize('NFC', w).lower()
    if code == 'akk': w = w.replace('-', '')
    if re.search(r'[0-9_@#/\\]', w): return None
    return w

def phonemes(code, r):
    """roman string -> list of phoneme symbols (strings), or None."""
    tab = DIG.get(code, {})
    keys = sorted(tab, key=len, reverse=True)
    out = []; i = 0
    while i < len(r):
        for k in keys:
            if r.startswith(k, i):
                rep = tab[k]; i += len(k)
                for ch in split_sym(rep): out.append(ch)
                break
        else:
            out.append(r[i]); i += 1
    # normalise symbols
    res = []
    for p in out:
        if p in APOS:
            if res and res[-1] not in VOW and res[-1] != 'ʔ': res[-1] = res[-1] + "'"   # ejective
            else: res.append('ʔ')
            continue
        if p in VCOLL: res.append(VCOLL[p]); continue
        if len(p) == 1:
            b = unicodedata.normalize('NFD', p)[0]
            if not (p.isalpha() or p in 'ʔʕʦʣ'): continue          # punctuation, digits, hyphens dropped
            if b in 'aeiouy' and p not in 'yý':
                res.append(VCOLL.get(p, b) if b != 'y' else 'i'); continue
            if p in 'yý': res.append('i' if code in VOWEL_Y else 'j'); continue
            res.append(p)
        else:
            res.append(p)
    # collapse identical neighbours (long vowels, geminates)
    col = []
    for p in res:
        if col and col[-1] == p: continue
        col.append(p)
    return col if any(p in VOW for p in col) else None

def split_sym(rep):
    """table outputs: split into symbols; a base letter followed by ʰ or ' stays one symbol."""
    out = []
    for ch in rep:
        if ch in "ʰ'" and out: out[-1] += ch
        else: out.append(ch)
    return out

def syllabify(ph, scheme):
    """phoneme list -> tuple of CV / V syllables."""
    syl = []; cl = []
    for i, p in enumerate(ph):
        if p in VOW:
            if scheme == 'DROP':
                syl.append((cl[-1] if cl else '') + p)
            elif scheme == 'ECHO':
                for c in cl: syl.append(c + p)
                if not cl: syl.append(p)
            else:  # MIXED: sonorant / sibilant before another consonant is dropped, other consonants echoed
                keep = [c for j, c in enumerate(cl) if not (j < len(cl) - 1 and c in SON)]
                for c in keep: syl.append(c + p)
                if not keep: syl.append(p)
            cl = []
        else:
            cl.append(p)
    return tuple(syl)

SCHEMES = ('DROP', 'ECHO', 'MIXED')

# ---------------------------------------------------------------- sources
def get(url, path, rng=None):
    if os.path.exists(path) and os.path.getsize(path) > 0: return True
    cmd = ['curl', '-sS', '-f', '-L', '-o', path, url]
    if rng: cmd[1:1] = ['-r', rng]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0

def ud_words(code, repo, pre, field):
    os.makedirs(SCR, exist_ok=True)
    cnt = collections.Counter(); files = []; lic = None
    rd = os.path.join(SCR, pre + '-README.md')
    if get(f'https://raw.githubusercontent.com/UniversalDependencies/{repo}/master/README.md', rd):
        m = re.search(r'License:\s*([^\n]+)', open(rd, errors='ignore').read())
        lic = m.group(1).strip() if m else None
    ntok = 0
    for part in ('train', 'test', 'dev'):
        if ntok >= MAXTOK: break
        fn = os.path.join(SCR, f'{pre}-{part}.conllu')
        if not get(UD.format(repo=repo, pre=pre, part=part), fn, f'0-{MAXBYTES}'): continue
        files.append(UD.format(repo=repo, pre=pre, part=part))
        skip_to = 0
        for line in open(fn, encoding='utf-8', errors='ignore'):
            if not line or line[0] == '#' or '\t' not in line: continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 10: continue
            idx = f[0]
            if field == 'mwt':
                if '-' in idx:
                    a, b = idx.split('-'); skip_to = int(b); cnt[f[1]] += 1; ntok += 1; continue
                if '.' in idx: continue
                if int(idx) <= skip_to: continue
                if f[3] == 'PUNCT': continue
                cnt[f[1]] += 1; ntok += 1; continue
            if '-' in idx or '.' in idx: continue
            if f[3] in ('PUNCT', 'NUM', 'SYM', 'X'): continue
            if field == 'form': w = f[1]
            else:
                m = re.search(r'(?:^|\|)' + field + r'=([^|]+)', f[9])
                if not m: continue
                w = m.group(1)
            cnt[w] += 1; ntok += 1
            if ntok >= MAXTOK: break
    return cnt, files, lic

def bible_words(code, bid):
    os.makedirs(SCR, exist_ok=True)
    url = f'https://ebible.org/Scriptures/{bid}_vpl.zip'
    zp = os.path.join(SCR, bid + '_vpl.zip')
    if not get(url, zp): return collections.Counter(), [], None
    z = zipfile.ZipFile(zp)
    txt = [n for n in z.namelist() if n.endswith('_vpl.txt')][0]
    cnt = collections.Counter(); ntok = 0
    for line in io.TextIOWrapper(z.open(txt), encoding='utf-8', errors='ignore'):
        parts = line.split(' ', 2)
        if len(parts) < 3: continue
        for w in re.findall(r"[^\W\d_]+(?:['’ʻ‘][^\W\d_]+)*", parts[2]):
            if w[0].isupper() and ntok > 0: pass
            cnt[w] += 1; ntok += 1
        if ntok >= MAXTOK: break
    lic = None
    about = [n for n in z.namelist() if n.endswith('about.htm')]
    if about:
        t = re.sub(r'<[^>]+>', ' ', z.read(about[0]).decode('utf-8', 'ignore'))
        m = re.search(r'(public domain|Creative Commons[^.]*|CC BY[^.]*)', t, re.I)
        lic = m.group(1).strip() if m else 'see about page'
    return cnt, [url], lic

def cdli_words():
    os.makedirs(SCR, exist_ok=True)
    url = 'https://media.githubusercontent.com/media/cdli-gh/data/master/cdliatf_unblocked.atf'
    fn = os.path.join(SCR, 'cdli_part.atf')
    get(url, fn, '20000000-32000000')
    cnt = collections.Counter(); ntok = 0; lang = None
    for line in open(fn, encoding='utf-8', errors='ignore'):
        if line.startswith('&'): lang = None; continue
        if line.startswith('#atf: lang'): lang = line.split()[-1]; continue
        if lang != 'sux': continue
        m = re.match(r'^\d+\'?\.\s+(.*)', line)
        if not m: continue
        for t in m.group(1).split():
            cnt[t] += 1; ntok += 1
        if ntok >= MAXTOK: break
    return cnt, [url + ' (bytes 20,000,000-32,000,000)'], 'CDLI bulk data (cdli-gh/data), CDLI terms of use'

# ---------------------------------------------------------------- build
def build(entry):
    code, name, fam, src, args = entry
    if src == 'ud': cnt, files, lic = ud_words(code, *args)
    elif src == 'bible': cnt, files, lic = bible_words(code, *args)
    else: cnt, files, lic = cdli_words()
    roman = {}
    for w, n in cnt.items():
        r = romanize(code, w)
        if not r: continue
        ph = phonemes(code, r)
        if not ph: continue
        roman.setdefault(tuple(ph), 0); roman[tuple(ph)] += n
    rec = dict(code=code, name=name, family=fam, source=src, files=files, licence=lic,
               n_tokens=sum(cnt.values()), n_word_types=len(cnt), n_phon_types=len(roman), schemes={})
    for sc in SCHEMES:
        types = {}
        for ph in roman:
            s = syllabify(ph, sc)
            if len(s) >= 2: types[s] = types.get(s, 0) + roman[ph]
        tr, te = [], []
        for s in sorted(types):
            h = int(hashlib.md5(('|'.join(s)).encode()).hexdigest(), 16)
            (tr if h % 2 == 0 else te).append(s)
        uni = collections.Counter(); bi = collections.Counter()
        for s in tr:
            for a in s: uni[a] += 1
            seq = ('^',) + s + ('$',)
            for a, b in zip(seq, seq[1:]): bi[a + '\t' + b] += 1
        rnd = random.Random(hash(code + sc) & 0xffffffff)
        rnd.shuffle(te); te = te[:1500]
        inv = sorted(set(a for s in te for a in s)); rnd.shuffle(inv); ix = {a: i for i, a in enumerate(inv)}
        rec['schemes'][sc] = dict(n_train_types=len(tr), n_test_types=len(te), n_syll=len(uni),
                                  uni=dict(uni), bi=dict(bi), test=[[ix[a] for a in s] for s in te],
                                  sample=[('-'.join(s)) for s in tr[:0]])
    rec['example'] = [('-'.join(syllabify(ph, 'MIXED')), ''.join(ph)) for ph in list(roman)[:8]]
    return rec

def main():
    os.makedirs(LANGD, exist_ok=True)
    want = set(sys.argv[1:])
    srcs = {}
    sp = os.path.join(OUT, 'sources.json')
    if os.path.exists(sp): srcs = json.load(open(sp))
    for e in LANGS:
        if want and e[0] not in want: continue
        fn = os.path.join(LANGD, e[0] + '.json')
        if os.path.exists(fn) and not want: continue
        try:
            rec = build(e)
        except Exception as ex:
            print(e[0], 'FAILED', ex, flush=True); continue
        json.dump(rec, open(fn, 'w'))
        srcs[e[0]] = {k: rec[k] for k in ('name', 'family', 'source', 'files', 'licence', 'n_tokens', 'n_word_types', 'n_phon_types')}
        srcs[e[0]]['n_train_types'] = {s: rec['schemes'][s]['n_train_types'] for s in SCHEMES}
        srcs[e[0]]['n_syll'] = {s: rec['schemes'][s]['n_syll'] for s in SCHEMES}
        json.dump(srcs, open(sp, 'w'), indent=1, ensure_ascii=False)
        print(e[0], rec['n_tokens'], rec['n_phon_types'], srcs[e[0]]['n_train_types'], srcs[e[0]]['n_syll'], rec['licence'], rec['example'][:3], flush=True)

if __name__ == '__main__':
    main()
