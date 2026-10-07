"""v88: ACROSS THE GUTTER.

Pivot from the brief (page-order continuity search = v65, already done): instead of asking
whether page p continues into page p+1 in reading order, ask whether a LINE continues
across the physical gutter of an opening (verso row k -> facing recto row k), through the
vellum (recto row k end lies behind verso row k start), or across the flat unfolded sheet
(conjugate halves of one face). If the text was written row by row across a spread, the
within-line word-junction coupling (0.16 bits, absent across ordinary line breaks) and the
line 'mood' should reappear across the gutter at the same row and not at offset rows.

Physical model from the ZL3b IVTFF headers via v65_lib.structure (quire, bifolio, leaf).
Only single-panel sides (fNr / fNv) are used so that rows are physical rows.
"""
import os, re, sys, json, math, random, hashlib
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib
import v65_lib as V

ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CK = os.path.join(DATA, 'v88_ckpt')
LOOPS = os.path.join(ROOT, 'loops')
os.makedirs(CK, exist_ok=True)
GAP = V.GAP


def vpages(name='ZL3b'):
    """single-panel sides only: (n,'r'/'v') -> list of lines (lists of glyph-unit words)."""
    L = vlib.load_voynich(name, ltypes=('P',))
    pages = defaultdict(list); meta = {}
    for r in L:
        if not re.match(r'^f\d+[rv]$', r['folio']):
            continue
        ws = [''.join(vlib.glyphs(w)) for w in r['words']]
        ws = [w for w in ws if w and '?' not in w and '*' not in w]
        if not ws:
            continue
        k = V.side_of(r['folio'])
        pages[k].append(ws)
        meta.setdefault(k, {'sec': r['illus'], 'lang': r['lang'], 'hand': r['hand'], 'quire': r['quire']})
    return {k: v for k, v in pages.items() if len(v) >= 3}, meta


def phys_seq(quires):
    """binding page sequence with GAPs for lost leaves (and at quire joins: no GAP)."""
    seq = []
    for qn, bifs in quires:
        nb = len(bifs)
        seq += V.seq_for(bifs, tuple(range(nb)), (0,) * nb)
    return seq


def pair_classes(quires, pages):
    """-> dict class -> list of (P, Q) side-key pairs (line rows of P end -> rows of Q start)."""
    seq = phys_seq(quires)
    out = defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        if a == GAP or b == GAP:
            continue
        if a[1] == 'v' and b[1] == 'r' and a in pages and b in pages:
            out['GUT'].append((a, b))          # facing pages of an opening, left -> right
            out['GUTR'].append((b, a))         # right -> left (control direction)
    for n in sorted({k[0] for k in pages}):
        if (n, 'r') in pages and (n, 'v') in pages:
            out['LEAF'].append(((n, 'r'), (n, 'v')))   # through the vellum: recto row k end backs verso row k start
    for qn, bifs in quires:
        for B, A, Bl in bifs:
            if A is None or Bl is None:
                continue
            for P, Q in (((A, 'v'), (Bl, 'r')), ((Bl, 'v'), (A, 'r'))):
                if P in pages and Q in pages and (P, Q) not in out['GUT']:
                    out['FLAT'].append((P, Q))  # same face of the unfolded sheet, left -> right
    return dict(out)


# ---------------------------------------------------------------- junction model
def wclass(w, salt='v88'):
    return hashlib.md5((salt + w).encode()).digest()[0] & 1


class Junction:
    """PMI(last glyph of word a, first glyph of word b) from within-line adjacent pairs.
    Trained only on pairs whose LEFT word is in vocabulary half `half` (None = all)."""
    def __init__(self, pages, keys=None, half=None, salt='v88', n=1):
        self.n = n
        c = Counter(); ca = Counter(); cb = Counter(); N = 0
        for k in (keys or pages):
            for L in pages[k]:
                for a, b in zip(L, L[1:]):
                    if half is not None and wclass(a, salt) != half:
                        continue
                    x, y = a[-n:], b[:n]
                    c[(x, y)] += 1; ca[x] += 1; cb[y] += 1; N += 1
        self.c, self.ca, self.cb, self.N = c, ca, cb, max(N, 1)
        self.Vx, self.Vy = len(ca) + 1, len(cb) + 1

    def pmi(self, a, b):
        x, y = a[-self.n:], b[:self.n]
        pxy = (self.c.get((x, y), 0) + 0.5) / (self.N + 0.5 * self.Vx * self.Vy)
        px = (self.ca.get(x, 0) + 0.5) / (self.N + 0.5 * self.Vx)
        py = (self.cb.get(y, 0) + 0.5) / (self.N + 0.5 * self.Vy)
        return math.log2(pxy / (px * py))


def within_line_mi(pages, J, half=None, salt='v88'):
    v = []
    for L in pages.values():
        for a, b in zip(L, L[1:]):
            if half is not None and wclass(a, salt) != half:
                continue
            v.append(J.pmi(a, b))
    return float(np.mean(v)) if v else float('nan')


def glyph_vec(line, alpha):
    v = np.zeros(len(alpha))
    for w in line:
        for g in w:
            if g in alpha:
                v[alpha[g]] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


def row_profile(pages, pairs, f, D=(-2, -1, 0, 1, 2)):
    """mean f(P row k, Q row k+d) over pairs and rows, per offset d."""
    out = {}
    for d in D:
        v = []
        for P, Q in pairs:
            A, B = pages[P], pages[Q]
            for k in range(len(A)):
                j = k + d
                if 0 <= j < len(B):
                    v.append(f(A[k], B[j]))
        out[d] = float(np.mean(v)) if v else float('nan')
    return out


def contrast(prof):
    off = [prof[d] for d in prof if d != 0 and not math.isnan(prof[d])]
    return prof[0] - float(np.mean(off))


# ---------------------------------------------------------------- controls
def pour_spread(template, words, seed, quires, encode=True):
    """Write a continuous text into the template layout, but across every opening whose two
    sides are both in the template: verso row k, then facing recto row k, then verso row k+1...
    (a text written row by row across the spread). Other pages are filled in normal flow."""
    enc = V.opaque_encoder(seed) if encode else (lambda w: w)
    seq = [k for k in phys_seq(quires)]
    pages = {}; i = 0; t = 0
    done = set()
    while t < len(seq):
        a = seq[t]
        b = seq[t + 1] if t + 1 < len(seq) else GAP
        if a != GAP and b != GAP and a in template and b in template and a[1] == 'v' and b[1] == 'r':
            LA, LB = template[a], template[b]
            outA, outB = [], []
            for k in range(max(len(LA), len(LB))):
                if k < len(LA):
                    n = len(LA[k]); outA.append([enc(w) for w in words[i:i + n]]); i += n
                if k < len(LB):
                    n = len(LB[k]); outB.append([enc(w) for w in words[i:i + n]]); i += n
            pages[a], pages[b] = outA, outB
            t += 2; continue
        if a != GAP and a in template:
            lines = []
            for L in template[a]:
                lines.append([enc(w) for w in words[i:i + len(L)]]); i += len(L)
            pages[a] = lines
        t += 1
    return pages


def pour_flow(template, words, seed, quires, encode=True):
    enc = V.opaque_encoder(seed) if encode else (lambda w: w)
    pages = {}; i = 0
    for a in phys_seq(quires):
        if a == GAP or a not in template:
            continue
        lines = []
        for L in template[a]:
            lines.append([enc(w) for w in words[i:i + len(L)]]); i += len(L)
        pages[a] = lines
    return pages


def cury_words():
    t = open(os.path.join(DATA, 'v58_ckpt', 'src', 'forme_of_cury.txt'), encoding='utf-8', errors='replace').read()
    return V.norm_words(t)


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
