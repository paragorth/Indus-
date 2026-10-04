"""v35 CLOSE THE GAP: does the v25 shape-predicts-behaviour link appear in INVENTED scripts and CIPHER alphabets?

Exact v25/v28 pipeline (v25_lib behaviour ppmi/svd/potts on L1 R1 L2 R2 + edges; Mantel Spearman; 5,000 random
glyph-to-shape permutations; v25_image zone/frame/topo/combo descriptors via v28_lib.image_sims; hand stroke
decompositions in v35_shapes.py written from the drawn forms before any statistic was computed).
Only the corpus and the glyph source change.

Corpora (all kept in the scratchpad, never committed):
  copiale   Copiale cipher (1760s, 75k symbols), line transcription of the learnable-typewriter HTR set
            (HF learnable-typewriter/copiale, annotation.json).  No word spaces: a line is a 'word'; rare or
            unmapped symbols split the line.  Glyphs: the Copiale TrueType font embedded in Knight, Megyesi &
            Schaefer 2011 (ACL W11-1202), Figure 2 maps keyboard characters to transcription names.
  tengwar   English (Pride and Prejudice) written in an orthographic Tengwar tehta mode; tehtar are separate units
            in reading order (like v28's combining marks).  FreeMonoTengwar (CSUR PUA).  POSITIVE control (featural).
  shavian   same English text in Shavian (ReadLex spellings).  Noto Sans Shavian.
  deseret   same text, Shavian phonemes mapped one-to-one to Deseret letters.  Noto Sans Deseret.
  cherokee  Cherokee syllabary (ChrEn monolingual + parallel Cherokee side).  Noto Sans Cherokee.
  cree      Northern East Cree New Testament in Canadian syllabics (eBible crl).  Noto Sans Canadian Aboriginal.
            POSITIVE control (featural by rotation).
  voy:*     Voynich ZL (v28 'voy:ZL3b:v25' corpus) and size/format-matched variants (lines without spaces, 73k tokens).
"""
import os, sys, re, json, unicodedata, random, glob
from collections import Counter
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v25_lib as L, v28_lib as X

SCR = L.SCR
V35 = os.path.join(SCR, 'v35')
CK = os.path.join(L.DATA, 'v35_ckpt'); os.makedirs(CK, exist_ok=True)
X.CK = CK  # v28 save/load go to v35_ckpt
TOK = X.TOK

X.FONT.update(
    copiale=os.path.join(V35, 'pdffont_25.ttf'),
    tengwar=os.path.join(V35, 'ftw/FreeMonoTengwar.2013-07-21/FreeMonoTengwar.ttf'),
    shavian=os.path.join(V35, 'NotoSansShavian.ttf'),
    deseret=os.path.join(V35, 'NotoSansDeseret.ttf'),
    cherokee=os.path.join(V35, 'NotoSansCherokee.ttf'),
    ucas=os.path.join(V35, 'NotoSansCanadianAboriginal.ttf'),
    freesans='/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    dejavusans='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
)

# ------------------------------------------------------------------ Copiale
# keyboard char of the embedded Copiale font -> transcription name (Knight et al. 2011, Figure 2)
COP_KEY = {
    'a': 'a', 'A': 'ah', '6': 'del', 'b': 'b', '<': 'tri', 'c': 'c', 'C': 'c.', '5': 'gam', 'd': 'd', '!': 'iot',
    'e': 'e', 'E': 'eh', '^': 'lam', 'f': 'f', '>': 'pi', 'g': 'g', '/': 'arr', 'h': 'h', 'H': 'h.', '-': 'hd',
    '?': 'bas', 'i': 'i', 'I': 'ih', '4': 'car', 'j': 'j', '+': 'plus', 'k': 'k', 'T': 'cross', 'l': 'l', '0': 'fem',
    'm': 'm', 'M': 'm.', 'B': 'mu', '1': 'mal', 'n': 'n', 'N': 'n.', 'D': 'nu', '\\': 'ft', 'o': 'o', 'O': 'oh',
    '&': 'o.', 'W': 'no', 'p': 'p', 'P': 'p.', 'Q': 'sqp', 'q': 'q', 'Z': 'zzz', 'r': 'r', 'R': 'r.', 'F': 'ru',
    '_': 'pipe', 's': 's', 'S': 's.', '`': 'longs', 't': 't', ')': 'grr', 'u': 'u', 'U': 'uh', 'G': 'uu', ']': 'grl',
    'v': 'v', '[': 'grc', 'w': 'w', '#': 'tri..', '7': 'hk', 'x': 'x', 'X': 'x.', '2': 'lip', '~': 'sqi', '(': 'y',
    'y': 'y..', '9': 'nee', ':': ':', 'z': 'z', '@': 'o..', '.': '.', 'L': 'ds', '=': 'ni', '*': 'star', ',': '...',
    'K': 'gs', '"': 'ki', '%': 'bigx', '|': 'bar', 'J': 'zs', '$': 'smil', '¬': 'gat', '3': 'three',
    'Y': 'ns', '¢': 'smir', '±': 'toe', '8': 'inf'}
COP_NAME = {v: k for k, v in COP_KEY.items()}


def copiale_lines():
    d = json.load(open(os.path.join(SCR, 'copiale', 'ann.json')))
    def key(k):
        m = re.match(r'(\d+)(B?)_(\d+)', k)
        return (int(m.group(1)), m.group(2), int(m.group(3)))
    return [d[k]['label'].split() for k in sorted(d, key=key)]


def split_rare(lines, ok, minc=20):
    """lines of units -> 'words' = maximal runs of units that are mappable (ok) and have >= minc tokens."""
    c = Counter(g for ln in lines for g in ln)
    out = []
    for ln in lines:
        cur = []
        for g in ln:
            if ok(g) and c[g] >= minc:
                cur.append(g)
            else:
                if cur:
                    out.append(tuple(cur))
                cur = []
        if cur:
            out.append(tuple(cur))
    return out


def copiale_words(variant='full'):
    lines = copiale_lines()
    if variant == 'merge':   # kill control: diacritic variants merged into their base letter
        mp = {'ah': 'a', 'eh': 'e', 'ih': 'i', 'oh': 'o', 'uh': 'u', 'c.': 'c', 'h.': 'h', 'm.': 'm', 'n.': 'n',
              'o.': 'o', 'p.': 'p', 'r.': 'r', 's.': 's', 'x.': 'x', 'mu': 'm', 'nu': 'n', 'ru': 'r', 'uu': 'u',
              'y..': 'y', 'tri..': 'tri', 'o..': 'o', 'hd': 'h'}
        lines = [[mp.get(g, g) for g in ln] for ln in lines]
    ws = split_rare(lines, lambda g: g in COP_NAME)
    return [tuple(COP_NAME[g] for g in w) for w in ws]


# ------------------------------------------------------------------ English text
def english_words(n_tok=TOK * 2):
    t = ''
    for f in ('pg1342.txt', 'pg84.txt'):
        x = open(os.path.join(V35, f), encoding='utf-8').read()
        a = x.find('*** START'); b = x.find('*** END')
        t += x[a:b] + '\n'
    t = t.replace('’', "'").replace('‘', "'")
    return re.findall(r"[A-Za-z]+(?:'[a-z]+)?", t)


# Tengwar (orthographic tehta mode for English; tehta emitted before the consonant it is written on)
TW = {n: chr(c) for c, n in [(0xE000, 'tinco'), (0xE001, 'parma'), (0xE002, 'calma'), (0xE003, 'quesse'), (0xE004, 'ando'),
                             (0xE005, 'umbar'), (0xE006, 'anga'), (0xE007, 'ungwe'), (0xE008, 'thule'), (0xE009, 'formen'),
                             (0xE00A, 'harma'), (0xE00B, 'hwesta'), (0xE00D, 'ampa'), (0xE00F, 'unque'), (0xE010, 'numen'),
                             (0xE011, 'malta'), (0xE012, 'noldo'), (0xE014, 'ore'), (0xE015, 'vala'), (0xE016, 'anna'),
                             (0xE020, 'romen'), (0xE022, 'lambe'), (0xE024, 'silme'), (0xE026, 'esse'), (0xE028, 'hyarmen'),
                             (0xE029, 'hwestaS'), (0xE02C, 'longCarrier'), (0xE02E, 'shortCarrier'),
                             (0xE040, 'tehtaA'), (0xE046, 'tehtaE'), (0xE044, 'tehtaI'), (0xE04A, 'tehtaO'),
                             (0xE04C, 'tehtaU'), (0xE042, 'tehtaY'), (0xE051, 'tehtaBar')]}
TW_NAME = {v: k for k, v in TW.items()}
_TW_DI = {'ch': 'calma', 'th': 'thule', 'sh': 'harma', 'gh': 'hwesta', 'ph': 'formen', 'ng': 'noldo', 'wh': 'hwestaS',
          'qu': 'unque'}
_TW_C = {'t': 'tinco', 'p': 'parma', 'c': 'quesse', 'k': 'quesse', 'd': 'ando', 'b': 'umbar', 'j': 'anga', 'g': 'ungwe',
         'f': 'formen', 'v': 'ampa', 'n': 'numen', 'm': 'malta', 'w': 'vala', 'l': 'lambe', 's': 'silme', 'z': 'esse',
         'h': 'hyarmen', 'q': 'unque'}
_TW_V = {'a': 'tehtaA', 'e': 'tehtaE', 'i': 'tehtaI', 'o': 'tehtaO', 'u': 'tehtaU'}


def tengwar_word(w):
    w = w.lower().replace("'", '')
    toks = []  # ('C', name) / ('V', tehta)
    i = 0
    while i < len(w):
        two = w[i:i + 2]
        ch = w[i]
        if two in _TW_DI:
            toks.append(('C', _TW_DI[two])); i += 2; continue
        if ch == 'x':
            toks += [('C', 'quesse'), ('C', 'silme')]; i += 1; continue
        if ch == 'y':
            nxt = w[i + 1] if i + 1 < len(w) else ''
            if i == 0 and nxt in 'aeiou' and nxt:
                toks.append(('C', 'anna'))
            else:
                toks.append(('V', 'tehtaY'))
            i += 1; continue
        if ch == 'r':
            nxt = w[i + 1] if i + 1 < len(w) else ''
            toks.append(('C', 'romen' if nxt and nxt in 'aeiouy' else 'ore')); i += 1; continue
        if ch in _TW_V:
            toks.append(('V', _TW_V[ch])); i += 1; continue
        if ch in _TW_C:
            # doubled consonant: one tengwa + bar below
            if i + 1 < len(w) and w[i + 1] == ch:
                toks.append(('C', _TW_C[ch])); toks.append(('B', 'tehtaBar')); i += 2; continue
            toks.append(('C', _TW_C[ch])); i += 1; continue
        return None
    out = []
    for k, (typ, nm) in enumerate(toks):
        if typ == 'V':
            nxt = toks[k + 1][0] if k + 1 < len(toks) else None
            out.append(TW[nm])
            if nxt != 'C':
                out.append(TW['shortCarrier'])
        else:
            out.append(TW[nm])
    return tuple(out)


def tengwar_words():
    ws = [tengwar_word(w) for w in english_words()]
    return [w for w in ws if w]


# Shavian via ReadLex; Deseret by one-to-one phoneme mapping from Shavian
_RL = None


def readlex():
    global _RL
    if _RL is None:
        d = json.load(open(os.path.join(V35, 'readlex.json'), encoding='utf-8'))
        best = {}
        for k, lst in d.items():
            for e in lst:
                w = e['Latn'].lower()
                if w not in best or e.get('freq', 0) > best[w][1]:
                    best[w] = (e['Shaw'], e.get('freq', 0))
        _RL = {w: s for w, (s, _) in best.items()}
    return _RL


def shavian_words():
    rl = readlex()
    out = []
    for w in english_words():
        s = rl.get(w.lower())
        if s:
            u = tuple(c for c in s if 0x10450 <= ord(c) <= 0x1047F)
            if u:
                out.append(u)
    return out


_SH2DS = {  # Shavian letter -> Deseret lowercase letters (U+10428 + index)
    '𐑐': [0x11], '𐑚': [0x12], '𐑑': [0x13], '𐑛': [0x14], '𐑒': [0x17], '𐑜': [0x18], '𐑓': [0x19], '𐑝': [0x1A],
    '𐑔': [0x1B], '𐑞': [0x1C], '𐑕': [0x1D], '𐑟': [0x1E], '𐑖': [0x1F], '𐑠': [0x20], '𐑗': [0x15], '𐑡': [0x16],
    '𐑘': [0x0F], '𐑢': [0x0E], '𐑙': [0x25], '𐑣': [0x10], '𐑤': [0x22], '𐑮': [0x21], '𐑥': [0x23], '𐑯': [0x24],
    '𐑦': [0x06], '𐑰': [0x00], '𐑧': [0x07], '𐑱': [0x01], '𐑨': [0x08], '𐑲': [0x0C], '𐑩': [0x0A], '𐑳': [0x0A],
    '𐑪': [0x09], '𐑴': [0x04], '𐑫': [0x0B], '𐑵': [0x05], '𐑬': [0x0D], '𐑶': [0x26], '𐑭': [0x02], '𐑷': [0x03],
    '𐑸': [0x02, 0x21], '𐑹': [0x03, 0x21], '𐑺': [0x01, 0x21], '𐑻': [0x0A, 0x21], '𐑼': [0x0A, 0x21],
    '𐑽': [0x00, 0x21], '𐑾': [0x00, 0x0A], '𐑿': [0x27]}


def deseret_words():
    out = []
    for w in shavian_words():
        u = []
        for c in w:
            if c not in _SH2DS:
                u = None; break
            u += [chr(0x10428 + i) for i in _SH2DS[c]]
        if u:
            out.append(tuple(u))
    return out


# ------------------------------------------------------------------ Cherokee, Cree syllabics
def cherokee_words():
    t = open(os.path.join(V35, 'chr_mono.txt'), encoding='utf-8').read() + '\n' + \
        open(os.path.join(V35, 'chr_train.txt'), encoding='utf-8').read()
    t = t.upper()  # Cherokee lowercase block -> standard syllabary
    return [tuple(w) for w in re.findall(r'[Ꭰ-Ᏽ]+', t)]


def cree_words():
    t = ''.join(open(f, encoding='utf-8-sig').read() + '\n' for f in sorted(glob.glob(os.path.join(V35, 'crl', 'crl_*_read.txt'))))
    return [tuple(w) for w in re.findall(r'[ᐁ-ᙬᙯ-ᙿ]+', t)]


# ------------------------------------------------------------------ Voynich controls
def voy_words():
    import v28_corpora as C
    return C.build('voy:ZL3b:v25')['words']


def voy_lines_nospace():
    """ZL lines with word spaces removed (Copiale format): runs of clean v25 units between bad words."""
    import v25_shapes as S
    from v18_lib import glyphs as vglyphs
    out = []
    for ln in L.voynich_lines('ZL3b'):
        cur = []
        for w in ln:
            if w is None:
                if cur:
                    out.append(tuple(cur))
                cur = []
            else:
                cur += list(w)
        if cur:
            out.append(tuple(cur))
    return out



# ------------------------------------------------------------------ Borg cipher (17th c.; Uppsala/SU transcription 0001r-0204v)
# transcription letter -> cipher symbol as drawn in the project's key image (key.png), rendered from Unicode fonts
BORG_KEY = {'a': 'α', 'b': 'D', 'c': 'δ', 'd': 'Δ', 'h': 'H', 'i': '♊', 'k': 'κ', 'm': '♏', 'n': '♎', 'o': '□', 'q': '☿',
            'v': '♈', 'w': 'ꝏ', 'x': '*', 'y': '♒', '0': '♀', '1': '♂', '4': '4', '5': '5', '6': '6', '8': '8', '9': '♋',
            'O': '¤', 'M': '♍', 'H': 'ff', 'T': 'Ħ', 'W': '~', 'Z': 'ʒ', 'I': 'i', 'Y': 'y'}


def borg_words():
    t = open(os.path.join(V35, 'borg.txt'), encoding='utf-8').read()
    lines = [l for l in t.split('\n') if l and l[0] not in '#<']
    lines = [re.sub(r'\[[^\]]*\]?', ' ', l) for l in lines]   # cleartext in brackets
    out = []
    for l in lines:
        for w in re.split(r'[\s,.:;?!()\[\]/<>\-]+', l):
            if not w:
                continue
            for part in re.split(r'[^' + re.escape(''.join(BORG_KEY)) + r']+', w):
                if part:
                    out.append(tuple(BORG_KEY[c] for c in part))
    return out


# ------------------------------------------------------------------ planted homophonic ciphers (calibration)
def planted_cipher(design, seed=3):
    """Printed Latin enciphered as a homophonic cipher onto the 84 Copiale glyphs.  Each letter gets homophones in
    proportion to its frequency (>= 1); each token picks one uniformly.  design='rand': glyphs dealt at random;
    'family': each letter's homophones are a cluster of shape-similar glyphs (v35 hand Jaccard, greedy)."""
    import v35_shapes as SH
    rng = random.Random(seed)
    ws = L.latin(10 ** 9)
    G = list(build('copiale')['alph'])
    cnt = Counter(g for w in ws for g in w)
    letters = [a for a, _ in cnt.most_common()]
    tot = sum(cnt.values())
    k = {a: max(1, int(round(len(G) * cnt[a] / tot))) for a in letters}
    while sum(k.values()) > len(G):
        a = max(k, key=k.get); k[a] -= 1
    while sum(k.values()) < len(G):
        a = max(letters, key=lambda x: cnt[x] / k[x]); k[a] += 1
    pool = G[:]
    rng.shuffle(pool)
    hom = {}
    if design == 'rand':
        i = 0
        for a in letters:
            hom[a] = pool[i:i + k[a]]; i += k[a]
    else:
        F, _ = L.shape_matrix({g: SH.COPIALE[COP_KEY[g]] for g in G}, G)
        J = L.jaccard_sim(F); gi = {g: i for i, g in enumerate(G)}
        free = set(G)
        for a in letters:
            seed_g = rng.choice(sorted(free))
            grp = [seed_g]; free.discard(seed_g)
            while len(grp) < k[a]:
                best = max(sorted(free), key=lambda g: (np.mean([J[gi[g], gi[h]] for h in grp]), rng.random()))
                grp.append(best); free.discard(best)
            hom[a] = grp
    return [tuple(rng.choice(hom[c]) for c in w) for w in ws]


# ------------------------------------------------------------------ registry
REG = dict(
    copiale=(lambda: copiale_words('full'), ('copiale',), 'o'),
    copiale_merge=(lambda: copiale_words('merge'), ('copiale',), 'o'),
    tengwar=(tengwar_words, ('tengwar',), TW['ore']),
    shavian=(shavian_words, ('shavian',), '𐑪'),
    deseret=(deseret_words, ('deseret', 'freeserif'), chr(0x10428 + 0x09)),
    cherokee=(cherokee_words, ('cherokee', 'freeserif'), 'Ꭴ'),
    cree=(cree_words, ('ucas', 'freesans'), 'ᐊ'),
    borg=(borg_words, ('dejavusans', 'freeserif'), 'α'),
    plant_rand=(lambda: planted_cipher('rand'), ('copiale',), 'o'),
    plant_family=(lambda: planted_cipher('family'), ('copiale',), 'o'),
    voy=(voy_words, ('eva',), 'o'),
    voy_lines=(voy_lines_nospace, ('eva',), 'o'),
)


def build(name, tok=TOK):
    """name may carry '@<tok>' to truncate (e.g. voy@73000)."""
    base, _, t = name.partition('@')
    tok = int(t) if t else tok
    c = X.load(f'corp_{name}.pkl')
    if c is not None:
        return c
    fn, fonts, xref = REG[base]
    ws = fn()
    fonts = tuple(f for f in fonts if all(True for _ in [0]))
    ws2, A = X.finish(ws, tok=tok)
    A2 = [g for g in A if X.covered(g, X.FONT[fonts[0]]) and X.has_ink(g, X.FONT[fonts[0]])]
    if len(A2) < len(A):
        As = set(A2)
        ws2 = [w for w in ws2 if all(g in As for g in w)]
        ws2, A2 = X.finish(ws2, tok=tok)
    import v35_shapes as SH
    hand = SH.hand(base, A2)
    out = dict(words=ws2, alph=A2, fonts=fonts, xref=xref, hand=hand, dropped=sorted(set(A) - set(A2)))
    X.save(f'corp_{name}.pkl', out)
    return out


def montage(chars, font, fn, labels=None, size=64, cols=12):
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(font, size)
    lf = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    cw, chh = int(size * 1.8), int(size * 2.2)
    rows = (len(chars) + cols - 1) // cols
    im = Image.new('L', (cols * cw, rows * chh), 255)
    d = ImageDraw.Draw(im)
    for k, ch in enumerate(chars):
        x, y = (k % cols) * cw, (k // cols) * chh
        d.text((x + size // 3, y + int(size * 1.3)), X.disp(ch), font=f, fill=0, anchor='ls')
        d.text((x + 2, y + chh - 18), (labels[k] if labels else ch)[:12], font=lf, fill=0)
        d.rectangle([x, y, x + cw - 1, y + chh - 1], outline=200)
    im.save(fn)
