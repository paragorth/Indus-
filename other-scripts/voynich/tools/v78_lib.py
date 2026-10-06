"""v78 WHY IS THE SAME WORD WRITTEN TWICE? mechanism fingerprint of adjacent doubled tokens.

Corpus format = v72 pages (list of dict(id, sec, lines=[dict(w=[raw surface words], ps)])). A 'token' for the
double test is the frozen v72 extraction E1c (hash 2803cbeb0deb51bf) of each raw word; E1c is a per-word function,
so a double is two adjacent words in one line with the same E1c form; its raw forms may or may not be identical.

Mechanisms (operators that insert copies into a base text from which every double has been collapsed):
  CHANCE  a token is followed by an independently drawn token of the same type with prob ~ its frequency
          (copy re-spelled from the corpus's raw forms of that type)
  REDUP   grammatical reduplication: a lexical class of types (a random share of types, frequency preference a)
          is doubled at rate r, rarely tripled; copy re-spelled (meaning-level repeat)
  LIST    ditto / 'the same again' at record positions (line-first, second, line-final, paragraph-first line);
          runs continue with prob g; copy re-spelled
  DITTOG  copying slip: any token repeated at rate r, part of them across the line break (last word of a line
          written again at the start of the next); copy = identical raw form (visual copy)
  TALLY   counting by repetition: 1-3 tally types (any of the 80 commonest) expanded into runs of 2,3,4,... (geometric); identical raw
  GEN     generator self-copying: a token replaced by a copy of one of the last W raw words on the page,
          one glyph changed with prob m (Timm-style copy and vary)
Features (fingerprint of the doubles) are computed on E1c lines; see feats().
"""
import os, sys, json, math, random, re, collections, pickle, zlib
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v72_lib as V
RULE = V.RULES['E1c_keepd']
assert V.rule_hash('E1c_keepd') == '2803cbeb0deb51bf'
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v78_ckpt'); os.makedirs(CK, exist_ok=True)
SRC = os.path.join(CK, 'src')
LOOPS = V.LOOPS
MECHS = ['CHANCE', 'REDUP', 'LIST', 'DITTOG', 'TALLY', 'GEN']

_E = {}


def e1c(w):
    x = _E.get(w)
    if x is None:
        x = V.extract_word(w, False, False, RULE); _E[w] = x
    return x


def row(fn, rid, method, result, verdict):
    V.row(fn, rid, method, result, verdict)


def leaf(pid):
    return V.leaf_half(pid)


def half(pages, h):
    return [p for p in pages if leaf(p['id']) == h]


# ------------------------------------------------------------------ helpers
def ed1(a, b):
    """True if edit distance exactly 1."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or a == b: return False
    if la == lb: return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb: a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]: i += 1
    return a[i:] == b[i + 1:]


def collapse(pages):
    """base text: runs of identical E1c tokens inside a line collapsed to their first word."""
    out = []
    for p in pages:
        nl = []
        for l in p['lines']:
            ws = []
            for w in l['w']:
                if ws and e1c(w) == e1c(ws[-1]): continue
                ws.append(w)
            nl.append(dict(w=ws, ps=l['ps']))
        out.append(dict(id=p['id'], sec=p['sec'], lines=nl))
    return out


# ------------------------------------------------------------------ fingerprint
FEATS = ['d1', 'lagspike', 'near_spike', 'near_ratio', 'respell', 'run3', 'pos0', 'pos1', 'posend', 'xline',
         'freq', 'conc', 'dlen', 'pagedisp', 'psline', 'typerec', 'jsd12', 'resp12']


def feats(pages, detail=False):
    T = [[[e1c(w) for w in l['w']] for l in p['lines']] for p in pages]
    tokc = Counter(x for P in T for L in P for x in L)
    ntok = sum(tokc.values())
    logf = {k: math.log(v) for k, v in tokc.items()}
    raw_of = defaultdict(Counter)
    for p in pages:
        for l in p['lines']:
            for w in l['w']: raw_of[e1c(w)][w] += 1
    npair = d = 0; lag = [0, 0, 0, 0]; nlag = [0, 0, 0, 0]; e1 = [0, 0, 0, 0]
    dbl = []  # (type, page, line, pos, linelen, ps, rawsame)
    runs = Counter(); xl = 0; nxl = 0
    pos0p = pos1p = endp = psp = 0
    per_page_pairs = []; per_page_d = []
    for pi, (p, P) in enumerate(zip(pages, T)):
        pp = pd = 0
        prev_last = None
        for li, (l, L) in enumerate(zip(p['lines'], P)):
            n = len(L)
            if prev_last is not None and n:
                nxl += 1; xl += (L[0] == prev_last)
            prev_last = L[-1] if n else None
            for k in (1, 2, 3):
                for i in range(n - k):
                    nlag[k] += 1
                    if L[i] == L[i + k]: lag[k] += 1
                    elif len(L[i]) >= 2 and len(L[i + k]) >= 2 and ed1(L[i], L[i + k]): e1[k] += 1
            for i in range(n - 1):
                npair += 1; pp += 1
                pos0p += (i == 0); pos1p += (i == 1); endp += (i + 1 == n - 1); psp += l['ps']
                if L[i] == L[i + 1]:
                    d += 1; pd += 1
                    dbl.append((L[i], pi, li, i, n, l['ps'], l['w'][i] == l['w'][i + 1]))
            i = 0
            while i < n:
                j = i
                while j + 1 < n and L[j + 1] == L[i]: j += 1
                if j > i: runs[j - i + 1] += 1
                i = j + 1
        per_page_pairs.append(pp); per_page_d.append(pd)
    eps = 0.5
    f = {}
    f['d1'] = math.log10((d + eps) / max(1, npair))
    l23 = (lag[2] + lag[3]) / max(1, nlag[2] + nlag[3])
    f['lagspike'] = math.log((d + eps) / max(1, npair)) - math.log(l23 + eps / max(1, npair))
    n23 = (e1[2] + e1[3]) / max(1, nlag[2] + nlag[3])
    f['near_spike'] = math.log((e1[1] + eps) / max(1, nlag[1])) - math.log(n23 + eps / max(1, nlag[1]))
    f['near_ratio'] = math.log((e1[1] + eps) / (d + eps))
    # re-spelling: share of doubles whose raw forms differ, relative to the chance that two raw draws differ
    if dbl:
        diff = 1 - np.mean([x[6] for x in dbl])
        base = []
        for x in dbl:
            c = raw_of[x[0]]; s = sum(c.values())
            base.append(1 - sum(v * v for v in c.values()) / (s * s))
        b = np.mean(base)
        f['respell'] = (diff + 0.02) / (b + 0.02)
    else:
        f['respell'] = 1.0
    r2 = sum(runs.values()); r3 = sum(v for k, v in runs.items() if k >= 3)
    f['run3'] = (r3 + 0.25) / (r2 + 1)
    nd = len(dbl) + 1
    f['pos0'] = math.log(((sum(x[3] == 0 for x in dbl) + 0.5) / nd) / (pos0p / max(1, npair)))
    f['pos1'] = math.log(((sum(x[3] == 1 for x in dbl) + 0.5) / nd) / (pos1p / max(1, npair)))
    f['posend'] = math.log(((sum(x[3] + 2 == x[4] for x in dbl) + 0.5) / nd) / (endp / max(1, npair)))
    f['psline'] = math.log(((sum(bool(x[5]) for x in dbl) + 0.5) / nd) / (max(1, psp) / max(1, npair)))
    f['xline'] = math.log((xl + eps) / max(1, nxl)) - math.log((d + eps) / max(1, npair))
    mlf = sum(v * logf[k] for k, v in tokc.items()) / ntok
    f['freq'] = (np.mean([logf[x[0]] for x in dbl]) - mlf) if dbl else 0.0
    dc = Counter(x[0] for x in dbl)
    simp_d = sum(v * v for v in dc.values()) / max(1, len(dbl)) ** 2
    simp_t = sum(v * v for v in tokc.values()) / ntok ** 2
    f['conc'] = math.log((simp_d + 1e-4) / (simp_t + 1e-4))
    mlen = sum(v * len(k) for k, v in tokc.items()) / ntok
    f['dlen'] = (np.mean([len(x[0]) for x in dbl]) - mlen) if dbl else 0.0
    pp = np.array(per_page_pairs, float); pdd = np.array(per_page_d, float)
    ex = pp * d / max(1, pp.sum()); m = ex > 0
    f['pagedisp'] = math.log(((pdd[m] - ex[m]) ** 2 / ex[m]).sum() / max(1, m.sum() - 1) + 0.1) if d else 0.0
    # type recurrence: share of doubles whose type is doubled on at least one other page
    tp = defaultdict(set)
    for x in dbl: tp[x[0]].add(x[1])
    f['typerec'] = np.mean([len(tp[x[0]]) > 1 for x in dbl]) if dbl else 0.0
    # lag-1 doubles vs lag-2/3 repeats: type divergence and re-spelling difference
    c1 = Counter(); c2 = Counter(); r1 = []; r2 = []
    for p in pages:
        for l in p['lines']:
            W = l['w']; Tl = [e1c(w) for w in W]
            for k in (1, 2, 3):
                for i in range(len(Tl) - k):
                    if Tl[i] == Tl[i + k]:
                        (c1 if k == 1 else c2)[Tl[i]] += 1; (r1 if k == 1 else r2).append(W[i] != W[i + k])
    f['jsd12'] = _jsd(c1, c2) if c1 and c2 else 1.0
    f['resp12'] = (np.mean(r1) if r1 else 0.5) - (np.mean(r2) if r2 else 0.5)
    if detail:
        return f, dict(npair=npair, d=d, lag=lag, nlag=nlag, e1=e1, runs=dict(runs), xl=xl, nxl=nxl, dbl=dbl,
                       top=dc.most_common(25), rawsame=float(np.mean([x[6] for x in dbl])) if dbl else None)
    return f


def _jsd(a, b):
    na = sum(a.values()); nb = sum(b.values()); s = 0.0
    for k in set(a) | set(b):
        p = a[k] / na; q = b[k] / nb; m = (p + q) / 2
        if p: s += 0.5 * p * math.log2(p / m)
        if q: s += 0.5 * q * math.log2(q / m)
    return s


def fvec(f):
    return np.array([f[k] for k in FEATS], float)


# ------------------------------------------------------------------ mechanism operators
class Ctx:
    def __init__(self, base, seed):
        self.rng = random.Random(seed)
        self.tc = Counter(); self.raw = defaultdict(Counter)
        self.glyph = Counter()
        for p in base:
            for l in p['lines']:
                for w in l['w']:
                    t = e1c(w); self.tc[t] += 1; self.raw[t][w] += 1; self.glyph.update(w)
        self.N = sum(self.tc.values())
        self.rawcum = {t: V._cum(c) for t, c in self.raw.items()}
        self.gcum = V._cum(self.glyph)
        self.types = [t for t, c in self.tc.most_common()]

    def respell(self, w):
        c = self.rawcum.get(e1c(w))
        return V._draw(c, self.rng) if c else w

    def mutate(self, w):
        j = self.rng.randrange(len(w)); return w[:j] + V._draw(self.gcum, self.rng) + w[j + 1:]


def sample_params(rng):
    """random mixture: each mechanism on with prob 0.5 (at least one on), random log-uniform parameters."""
    def lu(a, b): return math.exp(rng.uniform(math.log(a), math.log(b)))
    P = {}
    while not P:
        if rng.random() < 0.5: P['CHANCE'] = dict(r=lu(0.3, 30))
        if rng.random() < 0.5: P['REDUP'] = dict(k=lu(0.002, 0.2), a=rng.uniform(-1, 1), r=lu(0.02, 0.8), t=rng.uniform(0, 0.15))
        if rng.random() < 0.5: P['LIST'] = dict(pos=rng.choice(['first', 'second', 'last', 'ps', 'first+second']),
                                                r=lu(0.01, 0.4), g=rng.uniform(0, 0.4))
        if rng.random() < 0.5: P['DITTOG'] = dict(r=lu(0.001, 0.05), x=rng.uniform(0, 0.6))
        if rng.random() < 0.5: P['TALLY'] = dict(m=rng.choice([1, 2, 3]), r=lu(0.02, 0.6), g=rng.uniform(0.15, 0.75))
        if rng.random() < 0.5: P['GEN'] = dict(p=lu(0.01, 0.4), W=rng.choice([1, 2, 3, 5, 10, 20, 40, 60]), m=rng.uniform(0, 0.8))
    return P


def apply(base, P, seed):
    """apply a mixture to the base; returns (pages, labels) where labels[(pi,li,i)] = mechanism that wrote word i
    as a copy of word i-1 (for counting each mechanism's share of the doubles)."""
    C = Ctx(base, seed); rng = C.rng
    # pre-draw lexical choices
    red = set()
    if 'REDUP' in P:
        q = P['REDUP']; k = max(1, int(round(q['k'] * len(C.types))))
        cand = [t for t in C.types if C.tc[t] >= 2] or C.types
        wts = np.array([C.tc[t] ** q['a'] for t in cand], float); wts /= wts.sum()
        idx = np.random.default_rng(seed).choice(len(cand), size=min(k, len(cand)), replace=False, p=wts)
        red = {cand[i] for i in idx}
    tal = []
    if 'TALLY' in P:
        top = C.types[:80]
        tal = rng.sample(top, min(P['TALLY']['m'], len(top)))
    out = []; lab = {}
    for pi, p in enumerate(base):
        hist = []; nl = []
        carry = None
        for li, l in enumerate(p['lines']):
            ws = []; src = l['w']; n = len(src)
            if carry is not None:
                ws.append(carry); lab[(pi, li, 0)] = 'DITTOG_X'; carry = None
            for i, w in enumerate(src):
                if 'GEN' in P and len(hist) >= 2 and rng.random() < P['GEN']['p']:
                    q = P['GEN']; w0 = rng.choice(hist[-q['W']:])
                    w = C.mutate(w0) if rng.random() < q['m'] else w0
                    if ws and e1c(w) == e1c(ws[-1]): lab[(pi, li, len(ws))] = 'GEN'
                ws.append(w); hist.append(w)
                t = e1c(w); extra = []
                if 'CHANCE' in P and rng.random() < min(0.9, P['CHANCE']['r'] * C.tc[t] / C.N):
                    extra.append(('CHANCE', C.respell(w)))
                if red and t in red and rng.random() < P['REDUP']['r']:
                    extra.append(('REDUP', C.respell(w)))
                    if rng.random() < P['REDUP']['t']: extra.append(('REDUP', C.respell(w)))
                if 'LIST' in P:
                    q = P['LIST']; hit = ((q['pos'] in ('first', 'first+second') and i == 0) or
                                          (q['pos'] in ('second', 'first+second') and i == 1) or
                                          (q['pos'] == 'last' and i == n - 1) or (q['pos'] == 'ps' and l['ps']))
                    if hit and rng.random() < q['r']:
                        extra.append(('LIST', C.respell(w)))
                        while rng.random() < q['g']: extra.append(('LIST', C.respell(w)))
                if tal and t in tal and rng.random() < P['TALLY']['r']:
                    extra.append(('TALLY', w))
                    while rng.random() < P['TALLY']['g'] and len(extra) < 8: extra.append(('TALLY', w))
                if 'DITTOG' in P:
                    if i == n - 1 and rng.random() < min(0.9, P['DITTOG']['r'] * P['DITTOG']['x'] * 8): carry = w
                    elif rng.random() < P['DITTOG']['r']: extra.append(('DITTOG', w))
                for m, x in extra:
                    lab[(pi, li, len(ws))] = m; ws.append(x); hist.append(x)
            nl.append(dict(w=ws, ps=l['ps']))
        out.append(dict(id=p['id'], sec=p['sec'], lines=nl))
    return out, lab


def shares(pages, lab):
    """share of the within-line E1c doubles written by each mechanism (unlabelled = BASE)."""
    c = Counter()
    for pi, p in enumerate(pages):
        for li, l in enumerate(p['lines']):
            L = [e1c(w) for w in l['w']]
            for i in range(1, len(L)):
                if L[i] == L[i - 1]:
                    m = lab.get((pi, li, i), 'BASE'); c['DITTOG' if m == 'DITTOG_X' else m] += 1
    n = sum(c.values())
    return {m: c[m] / n for m in MECHS + ['BASE']} if n else {m: 0.0 for m in MECHS + ['BASE']}, n


# ------------------------------------------------------------------ corpora
def plain_pages(paras, prefix, nsec=6, cap=40000, line_w=8):
    """paras: list of word lists -> v72 pages with ordinal sections."""
    n = len(paras)
    ents = [('b%d' % min(nsec - 1, i * nsec // max(1, n)), ws) for i, ws in enumerate(paras)]
    return V._pages_from_entries(ents, None, line_w=line_w, page_tok=160, cap=cap, prefix=prefix)


def through_surface(pages, seed):
    words = [w for p in pages for l in p['lines'] for w in l['w']]
    code = V.payload_code(words, seed=seed, mode='merge')
    pay = V.encode_payload(pages, code)
    surf = V.surface(pay, seed=seed + 1)
    for p in surf:
        for l in p['lines']: l.pop('orig', None)
    return surf
