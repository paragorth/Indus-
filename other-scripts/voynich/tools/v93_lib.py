"""v93 THE PAGE TRANSLATES ITSELF (interlinear / self-parallel hypothesis) and later cycles.

Corpus format (as v72): list of pages; page = dict(id, sec, lines=[dict(w=[words], ps=bool)]).
A pairing rule R maps a source unit to a partner unit on the same page. A word-translation model (IBM-1, vectorised
EM) is fitted on training-folio pairs; it is scored by held-out per-token bits gained over a unigram model.
Every rule has a TWIN R' at the same distance but the other phase/offset; score = gain(R) - gain(R').
An interlinear text (verse + translation) has a large phase asymmetry; monolingual text, generators and
line-shuffled Voynich should have none.
"""
import os, sys, json, math, random, re, zlib, hashlib
from collections import Counter, defaultdict
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
CK = os.path.join(ROOT, 'data', 'v93_ckpt'); os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(ROOT, 'loops')
import v72_lib as V


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------ corpora
def texts():
    return json.load(open(os.path.join(ROOT, 'data', 'v89_ckpt', 'texts.json')))


def _clean(t):
    import unicodedata
    t = unicodedata.normalize('NFD', t.lower()); t = ''.join(c for c in t if unicodedata.category(c) != 'Mn')
    return ''.join(c for c in t if c.isalpha())


def _units_words(u):
    out, i = [], 0
    tok = u['tok']
    for n in u['p']:
        seg = [_clean(t) for t in tok[i:i + n] if t != 'thinsp']
        out.append([t for t in seg if t]); i += n
    return out


def _paginate(paras, prefix, page_tok=160, groups=None):
    """paras: list of list-of-lines (each line a word list). Fill pages to ~page_tok words; a page break
    falls only where groups[i] changes (keeps a text paragraph and its translation on one page)."""
    pages, cur, n = [], None, 0
    for pi, para in enumerate(paras):
        newgrp = groups is None or pi == 0 or groups[pi] != groups[pi - 1]
        if cur is None or (n >= page_tok and newgrp):
            cur = dict(id='%s%03d' % (prefix, len(pages)), sec='S', lang='-', hand='-', lines=[]); pages.append(cur); n = 0
        for i, l in enumerate(para):
            if l:
                cur['lines'].append(dict(w=l, ps=(i == 0))); n += len(l)
    return pages


def _encode(pages, langs_of_line, seed):
    """encode each line with the payload code of its language, then the v72 surface machinery."""
    words = defaultdict(list)
    for p in pages:
        for l in p['lines']: words[l['lang']] += l['w']
    codes = {lg: V.payload_code(ws, seed=seed + i, mode='verbose') for i, (lg, ws) in enumerate(sorted(words.items()))}
    enc = []
    for p in pages:
        enc.append(dict(p, lines=[dict(l, w=[''.join(codes[l['lang']][c] for c in w) for w in l['w']], orig=l['w'])
                                  for l in p['lines']]))
    return V.surface(enc, seed=seed)


def plant(kind, seed=93):
    T = texts()
    if kind.startswith('PS'):
        A = [_units_words(u) for u in T['psalms_he']['units']]; B = [_units_words(u) for u in T['psalms_en']['units']]
        paras = []
        for a, b in zip(A, B):
            if kind == 'PS_LINE':        # verse in Hebrew, next line its English translation
                para = []
                for x, y in zip(a, b): para += [('he', x[:12]), ('en', y[:12])]
            elif kind == 'PS_INLINE':    # one line = Hebrew verse half + English verse half
                para = [('mix', x[:6] + y[:6]) for x, y in zip(a, b)]
            elif kind == 'PS_EN':        # monolingual control: English verses as lines
                para = [('en', y[:12]) for y in b]
            elif kind == 'PS_HE':
                para = [('he', x[:12]) for x in a]
            paras.append(para)
        pages = _paginate([[l for _, l in pa] for pa in paras], kind)
        # attach languages
        k = 0; flat = [lg for pa in paras for lg, l in pa if l]
        for p in pages:
            for l in p['lines']: l['lang'] = flat[k]; k += 1
        if kind == 'PS_INLINE':
            for p in pages:
                for l in p['lines']: l['lang'] = 'mix'
        return _encode(pages, None, seed)
    if kind.startswith('CE'):
        A = [_units_words(u) for u in T['celsus_lat']['units']]; B = [_units_words(u) for u in T['celsus_eng']['units']]
        items = []
        for a, b in zip(A, B):
            if len(a) != len(b): continue
            for x, y in zip(a, b):
                if len(x) >= 12 and len(y) >= 12: items.append((x[:48], y[:64]))
        paras, grp = [], []
        for gi, (x, y) in enumerate(items[:420]):
            if kind == 'CE_PARA':
                paras.append([('la', x[i:i + 8]) for i in range(0, len(x), 8)])
                paras.append([('en', y[i:i + 8]) for i in range(0, len(y), 8)]); grp += [gi, gi]
            elif kind == 'CE_LA':
                paras.append([('la', x[i:i + 8]) for i in range(0, len(x), 8)]); grp.append(gi)
        pages = _paginate([[l for _, l in pa] for pa in paras], kind, groups=grp)
        flat = [lg for pa in paras for lg, l in pa if l]; k = 0
        for p in pages:
            for l in p['lines']: l['lang'] = flat[k]; k += 1
        return _encode(pages, None, seed)
    raise KeyError(kind)


def voynich(name):
    return V.voynich(name)


def line_shuffle(pages, seed=0):
    """within each paragraph keep the first line, shuffle the others (destroys interlinear order)."""
    rng = random.Random(seed); out = []
    for p in pages:
        paras, cur = [], []
        for l in p['lines']:
            if l['ps'] and cur: paras.append(cur); cur = []
            cur.append(l)
        if cur: paras.append(cur)
        nl = []
        rng.shuffle(paras)
        for pa in paras:
            rest = pa[1:]; rng.shuffle(rest); nl += [pa[0]] + rest
        out.append(dict(p, lines=nl))
    return out


def corpus(name):
    if name.endswith('~shuf'): return line_shuffle(corpus(name[:-5]), seed=7)
    if name in ('ZL3b', 'IT2a', 'GC2a'): return voynich(name)
    if name.startswith('ZLshuf'): return line_shuffle(voynich('ZL3b'), seed=int(name[-1]))
    if name.startswith('ITshuf'): return line_shuffle(voynich('IT2a'), seed=int(name[-1]))
    if name in ('SELFCIT', 'MK2', 'JUNC', 'WSHUF'): return V.GENS[name](voynich('ZL3b'), seed=93)
    return plant(name)


def is_voy(name):
    return name[:2] in ('ZL', 'IT', 'GC', 'SE', 'MK', 'JU', 'WS')


def split_of(pid, is_voy):
    """0 = train, 1 = selection, 2 = test."""
    h = V.leaf_half(pid) if is_voy else zlib.crc32(pid.encode()) % 2
    if h == 0: return 0
    return 1 + zlib.crc32((pid + 's93').encode()) % 2


# ------------------------------------------------------------------ representations
def rep(w, kind):
    if kind == 'whole': return w
    if kind == 'core':
        if len(w) > 1 and w[0] == 'q': w = w[1:]
        w = w.replace('S', 'C').replace('T', 'Ck').replace('K', 'Ck').replace('P', 'Cp').replace('F', 'Cp')
        w = w.replace('t', 'k').replace('f', 'p').replace('e', '')
        return w or 'o'
    if kind.startswith('pre'): return w[:int(kind[3:])]
    if kind.startswith('suf'): return w[-int(kind[3:]):]
    if kind == 'frame': return w[:2] + '|' + w[-2:] if len(w) > 4 else w
    raise KeyError(kind)


# ------------------------------------------------------------------ pairing rules
def paragraphs(page):
    paras, cur = [], []
    for l in page['lines']:
        if l['ps'] and cur: paras.append(cur); cur = []
        cur.append(l['w'])
    if cur: paras.append(cur)
    return paras


def _alt(seq):
    """seq of units -> (ph0 pairs, ph1 pairs) with equal counts: odd-length run, (0,1),(2,3).. vs (1,2),(3,4).."""
    if len(seq) % 2 == 0: seq = seq[:-1]
    if len(seq) < 3: return [], []
    return ([(seq[i], seq[i + 1]) for i in range(0, len(seq) - 1, 2)],
            [(seq[i], seq[i + 1]) for i in range(1, len(seq) - 1, 2)])


def pairs_both(page, rule):
    """rule = (fam, param) -> (pairs under phase/offset A, pairs under its twin B), same distance, equal counts
    per group. Paragraph-first lines are left out of line-level rules (they are known to be special)."""
    fam = rule[0]; A, B = [], []
    paras = paragraphs(page)
    if fam == 'ALT':
        for pa in paras:
            a, b = _alt(pa[1:]); A += a; B += b
    elif fam == 'PARA':
        a, b = _alt([[w for l in pa for w in l] for pa in paras]); A += a; B += b
    elif fam in ('HALF', 'PAGEH'):
        seqs = [pa[1:] for pa in paras] if fam == 'HALF' else [[l['w'] for l in page['lines']]]
        for L in seqs:
            n = len(L)
            if n < 5: continue
            h = n // 2 + rule[1]
            for i in range(0, n):
                if i + h + 1 < n and i < h: A.append((L[i], L[i + h])); B.append((L[i], L[i + h + 1]))
    elif fam == 'HLINE':
        f = rule[1]
        for pa in paras:
            L = [w for w in pa[1:] if len(w) >= 4]
            cut = [max(1, min(len(w) - 1, int(round(len(w) * f)))) for w in L]
            for i in range(len(L) - 1):
                A.append((L[i][:cut[i]], L[i][cut[i]:])); B.append((L[i][cut[i]:], L[i + 1][:cut[i + 1]]))
    return A, B


# ------------------------------------------------------------------ IBM-1 (vectorised)
class Ibm1:
    def __init__(self, train_pairs, N=1000, iters=6, rev=False):
        if rev: train_pairs = [(b, a) for a, b in train_pairs]
        cs = Counter(w for a, _ in train_pairs for w in a); ct = Counter(w for _, b in train_pairs for w in b)
        self.sv = {w: i + 1 for i, (w, _) in enumerate(cs.most_common(N))}   # 0 = NULL, len+1 = UNK
        self.tv = {w: i for i, (w, _) in enumerate(ct.most_common(N))}
        self.S = len(self.sv) + 2; self.T = len(self.tv) + 1
        self.rev = rev
        uni = np.ones(self.T)
        for w, c in ct.items(): uni[self.tv.get(w, self.T - 1)] += c
        self.uni = uni / uni.sum()
        xs, ys, gid = self._triples(train_pairs)
        t = np.full((self.S, self.T), 1.0 / self.T)
        flat = xs * self.T + ys
        for _ in range(iters):
            v = t[xs, ys]
            z = np.bincount(gid, weights=v); post = v / z[gid]
            c = np.bincount(flat, weights=post, minlength=self.S * self.T).reshape(self.S, self.T)
            c += 1e-3
            t = c / c.sum(1, keepdims=True)
        self.t = t

    def _sid(self, w): return self.sv.get(w, self.S - 1)
    def _tid(self, w): return self.tv.get(w, self.T - 1)

    def _triples(self, pairs):
        xs, ys, gid = [], [], []; g = 0
        for a, b in pairs:
            sa = [0] + [self._sid(w) for w in a]
            for w in b:
                y = self._tid(w)
                xs += sa; ys += [y] * len(sa); gid += [g] * len(sa); g += 1
        return np.array(xs, int), np.array(ys, int), np.array(gid, int)

    def gain(self, pairs, lam=0.5):
        """bits per target token gained over the unigram (mixture lam)."""
        if self.rev: pairs = [(b, a) for a, b in pairs]
        if not pairs: return float('nan'), 0
        xs, ys, gid = self._triples(pairs)
        if len(xs) == 0: return float('nan'), 0
        cnt = np.bincount(gid); s = np.bincount(gid, weights=self.t[xs, ys]) / cnt
        yfirst = ys[np.r_[0, np.cumsum(cnt)[:-1]]]
        pu = self.uni[yfirst]
        g = np.log2(lam * s + (1 - lam) * pu) - np.log2(pu)
        return float(g.mean()), int(len(g))

    def top_pairs(self, k=20, min_t=0.05):
        inv_s = {i: w for w, i in self.sv.items()}; inv_t = {i: w for w, i in self.tv.items()}
        out = []
        for i in range(1, self.S - 1):
            j = int(self.t[i].argmax()); p = float(self.t[i, j])
            if j in inv_t and p >= min_t: out.append((inv_s[i], inv_t[j], p))
        return out


def rep_pairs(pairs, kind):
    return [([rep(w, kind) for w in a], [rep(w, kind) for w in b]) for a, b in pairs]


def score_hyp(pages_split, hyp, eval_splits=(1, 2), return_model=False):
    """hyp = dict(rule, rep, N, lam). Fit on split 0 once; returns {split: score} plus gains."""
    PB = {id(p): pairs_both(p, tuple(hyp['rule'])) for p, s in pages_split}
    res = {}
    for k, tag in enumerate(('R', 'T')):
        tr = rep_pairs([pr for p, s in pages_split if s == 0 for pr in PB[id(p)][k]], hyp['rep'])
        evs = {e: rep_pairs([pr for p, s in pages_split if s == e for pr in PB[id(p)][k]], hyp['rep']) for e in eval_splits}
        if len(tr) < 30 or min(len(v) for v in evs.values()) < 10: return None
        gs = {e: [] for e in eval_splits}
        for rev in (False, True):
            m = Ibm1(tr, N=hyp['N'], rev=rev)
            for e in eval_splits: gs[e].append(m.gain(evs[e], lam=hyp.get('lam', 0.5))[0])
            if return_model and tag == 'R': res['model_%d' % rev] = m
        res[tag] = {e: float(np.mean(gs[e])) for e in eval_splits}
        res['n_' + tag] = {e: len(evs[e]) for e in eval_splits}
    res['score'] = {e: res['R'][e] - res['T'][e] for e in eval_splits}
    return res


def random_hyp(rng):
    fam = rng.choice(['ALT', 'ALT', 'HALF', 'PARA', 'HLINE', 'HLINE', 'PAGEH'])
    if fam in ('ALT', 'PARA'): rule = (fam, 0)
    elif fam == 'HLINE': rule = (fam, rng.choice([0.34, 0.4, 0.5, 0.6, 0.66]))
    else: rule = (fam, rng.choice([0, -1, -2]))
    return dict(rule=list(rule), rep=rng.choice(['whole', 'whole', 'core', 'core', 'frame', 'pre2', 'pre3', 'suf2', 'suf3']),
                N=rng.choice([150, 300, 600, 1000, 2000]), lam=rng.choice([0.2, 0.5, 0.8]))
