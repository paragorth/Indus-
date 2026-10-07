"""v85 DIFFERENT SCRIBES, ONE DICTIONARY?  Shared helpers.

Idea: if several scribes wrote meaningful text about the same subjects, they share a content lexicon (same words for
same things) beyond what their private glyph habits produce; if each wrote by a private habit, mid/low-frequency
vocabulary splits by hand.

Statistic (frozen 7 Oct 2026 before any Voynich number was looked at):
  level    : 'canon' = v72 extraction rule E2_line applied to every word (removes q-, e, d, ch~sh, k~t, p~f, benches,
             line-initial marker glyph, paragraph p/f, final m->n: the known spelling mood); 'surf' = words as written.
  sample   : N tokens from a set of pages (pages shuffled, taken whole until N, last page cut).
  band     : tokens whose type is NOT among the T=30 most frequent types of the two samples pooled.
  COV(X,Y) : mean of [share of Y band tokens whose type occurs in X] and the reverse.
  within   : COV between two disjoint page halves of one hand (same section), each N tokens.
  cross    : COV between N tokens of hand X and N tokens of hand Y.
  null     : per-hand glyph-unit Markov generator (order 2; order 3 as a stronger variant) fitted on that hand's
             own words at the measured level, per section and line slot (first word / other); it has the hand's
             glyph mood and word shapes but no lexicon.  E = COV(real) - COV(generated, same page/line shapes).
  DSR      : dictionary sharing ratio = E_cross / E_within (1: one shared dictionary; 0: private dictionaries).
"""
import os, sys, re, json, math, random, pickle, hashlib
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
CK = os.path.join(ROOT, 'data', 'v85_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v72_lib as V

TOPT = 30
FROZEN = 'v85 COV: band = not top30 pooled; COV mean both directions; E = real - glyph Markov per hand x sec x slot; ' \
         'DSR = E_cross/E_within; canon = E2_line'
FROZEN_HASH = hashlib.sha256(FROZEN.encode()).hexdigest()[:16]


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def psave(name, obj): pickle.dump(obj, open(os.path.join(CK, name), 'wb'))


def pload(name):
    p = os.path.join(CK, name)
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else None


# ------------------------------------------------------------------ levels
def canon(pages):
    return V.extract(pages, V.RULES['E2_line'])


def level(pages, lev):
    return canon(pages) if lev == 'canon' else pages


def ntok(pages): return sum(len(l['w']) for p in pages for l in p['lines'])


# ------------------------------------------------------------------ Voynich groups
def voy_groups(name='ZL3b'):
    P = V.voynich(name)
    G = defaultdict(list)
    for p in P:
        G[(p['sec'], p['hand'], p['lang'])].append(p)
    return P, G


# ------------------------------------------------------------------ control texts
def _clean(ws):
    out = []
    for w in ws:
        w = re.sub(r'[^a-z]', '', w.lower())
        if w: out.append(w)
    return out


def ctrl_entries():
    C = pload('ctrl_entries.pkl')
    if C: return C
    import v37_lib as L37
    S = json.load(open(os.path.join(ROOT, 'data', 'v75_ckpt', 'systems.json')))
    def v75(k): return [_clean(ws) for s, ws in S[k]['entries']]
    g = json.load(open(os.path.join(ROOT, 'data', 'derived', 'v13_gerard.json')))['pages']
    ger = [_clean([w for w in g[k]['ocr'] if not re.search(r'\d', w)]) for k in sorted(g, key=int)]
    C = dict(CULP=[_clean(w) for w in L37.culpeper()], GERARD=ger, MACER=v75('LANG_macer'),
             HILDE=v75('LANG_hildegard'), KONRAD=v75('LANG_konrad_plants'), CURY=[_clean(r) for r in L37.cury()])
    C = {k: [e for e in v if len(e) >= 3] for k, v in C.items()}
    psave('ctrl_entries.pkl', C)
    return C


PROF = {
    'A': dict(pad_mid=[('', 0.35), ('e', 0.30), ('ee', 0.15), ('d', 0.10), ('ed', 0.10)],
              pad_y=[('', 0.40), ('e', 0.20), ('d', 0.25), ('ed', 0.15)], q_y=0.55, q_o=0.08, pS=0.75, pS2=(0.2, 0.1),
              tG=0.5, tH=0.5, flip=0.25, bench=0.3, mark=0.5, mfin=0.3),
    'B': dict(pad_mid=[('', 0.15), ('e', 0.15), ('ee', 0.40), ('d', 0.25), ('ed', 0.05)],
              pad_y=[('', 0.15), ('e', 0.45), ('d', 0.10), ('ed', 0.30)], q_y=0.25, q_o=0.25, pS=0.35, pS2=(0.6, 0.3),
              tG=0.15, tH=0.8, flip=0.10, bench=0.6, mark=0.25, mfin=0.6),
}


def surface_prof(pages, seed, prof):
    """v72 surface machinery with a scribe profile (different spelling habits, same script)."""
    pr = PROF[prof]; rng = random.Random(seed); out = []
    for p in pages:
        nl = []; prev_mark = None
        for l in p['lines']:
            hab = dict(G=rng.random() < pr['tG'], H=rng.random() < pr['tH'], C=rng.random() < 0.4)
            ws = []
            for wi, pw in enumerate(l['w']):
                s = ''; syms = list(pw)
                for j, c in enumerate(syms):
                    if c in 'GHC':
                        if rng.random() < pr['flip']: hab[c] = not hab[c]
                        if c == 'C':
                            pS = pr['pS'] if (l['ps'] or wi == 0) else (pr['pS2'][0] if hab['C'] else pr['pS2'][1])
                            g = 'S' if rng.random() < pS else 'C'
                        elif c == 'G':
                            g = 't' if hab['G'] else 'k'
                        else:
                            g = 'f' if hab['H'] else 'p'
                        if g in 'kt' and s.endswith('C') and rng.random() < pr['bench']:
                            s = s[:-1] + ('T' if g == 't' else 'K')
                        elif g in 'pf' and s.endswith('C') and rng.random() < pr['bench']:
                            s = s[:-1] + ('F' if g == 'f' else 'P')
                        else:
                            s += g
                        if j < len(syms) - 1: s += V._pick(rng, pr['pad_mid'])
                    elif c == 'y' and j == len(syms) - 1 and j > 0:
                        s += V._pick(rng, pr['pad_y']) + 'y'
                    else:
                        s += c
                if wi > 0 and rng.random() < (pr['q_y'] if ws[-1].endswith('y') else pr['q_o']): s = 'q' + s
                if wi == len(l['w']) - 1 and s.endswith('n') and rng.random() < pr['mfin']: s = s[:-1] + 'm'
                ws.append(s)
            if ws and rng.random() < pr['mark']:
                m = rng.choice([x for x in V.MARK if x != prev_mark]); prev_mark = m; ws[0] = m + ws[0]
            if ws and l['ps'] and rng.random() < 0.5:
                ws[0] = rng.choice('pf') + ws[0]
            nl.append(dict(l, w=ws))
        out.append(dict(p, lines=nl))
    return out


def make_pages(entries, prefix, nsec=4, cap=60000):
    n = len(entries)
    E = [('%s%d' % (prefix, min(nsec - 1, i * nsec // n)), ws) for i, ws in enumerate(entries)]
    return V._pages_from_entries(E, None, line_w=8, page_tok=160, cap=cap, prefix=prefix)


def ctrl_pair(kind, seed=850):
    """returns (pages_X, pages_Y) as two 'scribes' in glyph surface form.
    kind: SAME_<TEXT> = one text, entries dealt to two scribes in alternating blocks of 3;
          TWO_<A>_<B> = two different texts; PROFSAME_* = same as SAME but both with profile A (no machinery change)."""
    C = ctrl_entries()
    parts = kind.split('_')
    if parts[0] in ('SAME', 'PROFSAME'):
        E = C[parts[1]]
        EX = [e for i, e in enumerate(E) if (i // 3) % 2 == 0]; EY = [e for i, e in enumerate(E) if (i // 3) % 2 == 1]
    else:
        EX, EY = C[parts[1]], C[parts[2]]
    code = V.payload_code([w for e in EX + EY for w in e], seed=seed, mode='merge')
    px = V.encode_payload(make_pages(EX, 'x'), code); py = V.encode_payload(make_pages(EY, 'y'), code)
    for P in (px, py):
        for p in P:
            for l in p['lines']: l.pop('orig', None)
    py_prof = 'A' if parts[0] == 'PROFSAME' else 'B'
    return surface_prof(px, seed + 1, 'A'), surface_prof(py, seed + 2, py_prof)


# ------------------------------------------------------------------ generators (fitted per hand)
class MK:
    """glyph-unit Markov of order k per (section, slot) with back-off to the pooled hand model."""
    def __init__(self, pages, k=2):
        self.k = k; self.T = defaultdict(lambda: defaultdict(Counter)); self.TG = defaultdict(Counter)
        for p in pages:
            for l in p['lines']:
                for i, w in enumerate(l['w']):
                    key = (p['sec'], i == 0); x = '^' * k + w + '$'
                    for j in range(k, len(x)):
                        self.T[key][x[j - k:j]][x[j]] += 1; self.TG[x[j - k:j]][x[j]] += 1
        self.cum = {}

    def _c(self, key, ctx):
        kk = (key, ctx)
        if kk not in self.cum:
            c = self.T[key].get(ctx) or self.TG.get(ctx)
            ks = list(c); vs = np.cumsum([c[x] for x in ks]); self.cum[kk] = (ks, vs / vs[-1])
        return self.cum[kk]

    def word(self, key, rng):
        x = '^' * self.k
        while len(x) < 25:
            ks, cp = self._c(key, x[-self.k:])
            c = ks[int(np.searchsorted(cp, rng.random()))]
            if c == '$': break
            x += c
        return x[self.k:] or 'o'

    def gen(self, pages, seed):
        rng = random.Random(seed)
        return [dict(p, lines=[dict(l, w=[self.word((p['sec'], i == 0), rng) for i in range(len(l['w']))])
                               for l in p['lines']]) for p in pages]


def priv_gen(pages, seed, k=2, shared_dict=None, p_cite=0.25):
    """PRIVATE-HABIT control: a finite dictionary made by the hand's own glyph Markov (as many types as the hand has),
    used with the hand's own rank-frequency law (likeliest shapes get the top ranks) plus page self-citation.
    shared_dict: use this dictionary (list of types in rank order) instead (SHARED-DICTIONARY control)."""
    rng = random.Random(seed)
    words = [w for p in pages for l in p['lines'] for w in l['w']]
    cnt = sorted(Counter(words).values(), reverse=True)
    if shared_dict is None:
        m = MK(pages, k); seen = Counter(); tries = 0
        while len(seen) < len(cnt) and tries < 200 * len(cnt):
            seen[m.word((pages[0]['sec'], False), rng)] += 1; tries += 1
        dic = [w for w, _ in seen.most_common()]
    else:
        dic = shared_dict
    dic = dic[:len(cnt)]
    # shuffle rank order a little inside the dictionary so two hands with a shared dictionary use it differently
    cum = np.cumsum(cnt[:len(dic)]); cum = cum / cum[-1]
    out = []
    for p in pages:
        hist = []; nl = []
        for l in p['lines']:
            ws = []
            for _ in l['w']:
                if hist and rng.random() < p_cite: w = rng.choice(hist[-60:])
                else: w = dic[int(np.searchsorted(cum, rng.random()))]
                ws.append(w); hist.append(w)
            nl.append(dict(l, w=ws))
        out.append(dict(p, lines=nl))
    return out, dic


# ------------------------------------------------------------------ sampling and the statistic
def sample(pages, N, rng, exclude=None):
    idx = [i for i in range(len(pages)) if not exclude or i not in exclude]
    rng.shuffle(idx); out = []; n = 0; used = set()
    for i in idx:
        if n >= N: break
        used.add(i)
        for l in pages[i]['lines']:
            for w in l['w']:
                if n < N: out.append(w); n += 1
    return out, used


def cov(x, y, T=TOPT):
    top = {w for w, _ in Counter(x + y).most_common(T)}
    sx, sy = set(x), set(y)
    by = [w for w in y if w not in top]; bx = [w for w in x if w not in top]
    a = sum(w in sx for w in by) / max(1, len(by)); b = sum(w in sy for w in bx) / max(1, len(bx))
    return 0.5 * (a + b)


def within(pages, N, rng):
    x, used = sample(pages, N, rng)
    y, _ = sample(pages, N, rng, exclude=used)
    return cov(x, y) if len(y) >= 0.9 * N else None


def cross(px, py, N, rng):
    x, _ = sample(px, N, rng); y, _ = sample(py, N, rng)
    return cov(x, y)


def dsr_pair(px, py, N, R=30, seed=851, k=2, gens=None):
    """px, py at the measured level.  Returns dict with real and generated COVs and DSR.
    gens: optional (list of generated px versions, list of generated py versions) to use as the null."""
    rng = random.Random(seed)
    canw_x = ntok(px) >= 2 * N * 0.95; canw_y = ntok(py) >= 2 * N * 0.95
    def stats(PX, PY, rr):
        c = [cross(PX, PY, N, rr) for _ in range(R)]
        w = []
        if canw_x: w += [v for v in (within(PX, N, rr) for _ in range(R)) if v is not None]
        if canw_y: w += [v for v in (within(PY, N, rr) for _ in range(R)) if v is not None]
        return float(np.mean(c)), (float(np.mean(w)) if w else float('nan')), float(np.std(c)), (float(np.std(w)) if w else float('nan'))
    cr, wr, sc, sw = stats(px, py, rng)
    if gens is None:
        mx, my = MK(px, k), MK(py, k)
        gens = ([mx.gen(px, seed + 10 + i) for i in range(3)], [my.gen(py, seed + 20 + i) for i in range(3)])
    cg, wg = [], []
    for gx, gy in zip(*gens):
        a, b, _, _ = stats(gx, gy, rng); cg.append(a); wg.append(b)
    cg, wg = float(np.mean(cg)), float(np.nanmean(wg)) if not all(math.isnan(v) for v in wg) else float('nan')
    Ec, Ew = cr - cg, wr - wg
    return dict(N=N, cross=cr, within=wr, cross_gen=cg, within_gen=wg, Ec=Ec, Ew=Ew, sd_c=sc, sd_w=sw,
                DSR=Ec / Ew if Ew and not math.isnan(Ew) and Ew > 0 else float('nan'),
                ratio_c=cr / cg if cg else float('nan'), ratio_w=wr / wg if wg else float('nan'))


def fmt(d):
    return 'N %d: cross %.3f (gen %.3f, x%.2f), within %.3f (gen %.3f, x%.2f), E_c %.3f E_w %.3f, DSR %.2f' % (
        d['N'], d['cross'], d['cross_gen'], d['ratio_c'], d['within'], d['within_gen'], d['ratio_w'], d['Ec'], d['Ew'], d['DSR'])
