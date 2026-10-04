"""v15 shared code: "the glyphs are chaff, the message is in the lengths".

* Voynich paragraph lines with gap types kept ('.' clear space, ',' uncertain /
  narrow space, '-' drawing interruption), from the IVTFF files.
* Length channels: glyph units (vlib.glyphs), EVA characters, pen strokes.
* Letter trigram models for la, it, de, cs, he, oc, grc.
* Null generators (word shuffles within line / page / section).
* Homophonic annealing solver for symbol stream -> letters, scored by a
  trigram model, with held-out scoring.
"""
import os, re, sys, json, math, random, unicodedata
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib  # noqa: E402

ROOT = vlib.ROOT
DATA = vlib.DATA
SCR = os.environ.get('V15_SCRATCH', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v15')
RES = os.path.join(DATA, 'results', 'v15')
os.makedirs(RES, exist_ok=True)

# ------------------------------------------------------------------ Voynich

def _clean_keep_gaps(text):
    ps = '<%>' in text
    pe = '<$>' in text
    text = text.replace('<->', '-')
    text = re.sub(r'<![^>]*>', '', text)
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\[([^:\]]*)(:[^\]]*)?\]', r'\1', text)
    text = text.replace('{', '').replace('}', '').replace("'", '')
    text = re.sub(r'@\d+;', '?', text)
    toks = re.split(r'([.,\-])', text)
    words, gaps, pend = [], [], None
    for t in toks:
        if t in ('.', ',', '-'):
            if words:
                # strongest separator wins if several in a row
                rank = {'-': 3, '.': 2, ',': 1}
                if pend is None or rank[t] > rank[pend]:
                    pend = t
        elif t:
            if words:
                gaps.append(pend or '.')
            words.append(t)
            pend = None
    return words, gaps, ps, pe


def load_v(name='ZL3b', ltypes=('P',)):
    path = os.path.join(DATA, name + '-n.txt')
    recs, page = [], {}
    for raw in open(path, encoding='utf-8', errors='replace'):
        raw = raw.rstrip('\n')
        if not raw or raw.startswith('#'):
            continue
        m = re.match(r'^<(f\w+)>\s*<!(.*)>', raw)
        if m:
            meta = dict(re.findall(r'\$(\w)=(\S)', m.group(2)))
            page = {'folio': m.group(1), 'illus': meta.get('I'), 'lang': meta.get('L'), 'hand': meta.get('H'),
                    'quire': meta.get('Q')}
            continue
        m = re.match(r'^<(f\w+)\.(\d+),([@+*=&~])(\w)(\w*)>\s*(.*)$', raw)
        if not m:
            continue
        folio, n, pre, lt, sub, text = m.groups()
        if ltypes and lt not in ltypes:
            continue
        words, gaps, ps, pe = _clean_keep_gaps(text)
        if not words:
            continue
        r = dict(page); r.update({'folio': folio, 'n': int(n), 'words': words, 'gaps': gaps,
                                  'para_start': ps, 'para_end': pe})
        recs.append(r)
    return recs


STROKES = dict(a=2, b=2, c=1, d=2, e=1, f=4, g=2, h=1, i=1, j=2, k=3, l=2, m=3, n=2, o=1, p=4, q=2,
               r=2, s=2, t=3, u=2, v=2, x=2, y=2, z=2)


def glen(w):
    return len(vlib.glyphs(w))


def clen(w):
    return len(w)


def slen(w):
    return sum(STROKES.get(ch, 2) for ch in w)


# ------------------------------------------------------------------ natural texts as "lines/pages"

def ref_lines(key, n_words=35000, skip_frac=0.05, page_lines=None):
    """Gutenberg reference text -> pseudo-manuscript lines with folio/illus labels.
    Pages = blocks of 9 lines (Voynich median), sections = blocks of 20 pages."""
    L = vlib.load_ref(key, max_words=n_words, skip_frac=skip_frac)
    out = []
    for i, l in enumerate(L):
        out.append({'folio': 'p%d' % (i // 9), 'illus': 's%d' % (i // 180), 'words': l['words'],
                    'gaps': ['.'] * (len(l['words']) - 1), 'para_start': l['para_start'],
                    'para_end': l['para_end'], 'lang': 'X'})
    return out


# ------------------------------------------------------------------ shuffles (nulls keep every word)

def shuffle_lines(lines, level, rng):
    """level: 'line' (words within line), 'page', 'section'. Line word counts are kept."""
    if level == 'line':
        out = []
        for L in lines:
            w = L['words'][:]; rng.shuffle(w)
            d = dict(L); d['words'] = w; out.append(d)
        return out
    key = 'folio' if level == 'page' else 'illus'
    groups = defaultdict(list)
    for i, L in enumerate(lines):
        groups[L[key]].append(i)
    out = [None] * len(lines)
    for g, idx in groups.items():
        pool = [w for i in idx for w in lines[i]['words']]
        rng.shuffle(pool)
        k = 0
        for i in idx:
            n = len(lines[i]['words'])
            d = dict(lines[i]); d['words'] = pool[k:k + n]; k += n
            out[i] = d
    return out


# ------------------------------------------------------------------ letter models

def _strip(s):
    s = unicodedata.normalize('NFD', s)
    return ''.join(ch for ch in s if not unicodedata.combining(ch))


def _text_for(lang):
    if lang == 'la':
        return open(os.path.join(DATA, 'plain', 'la.txt'), encoding='utf-8').read()
    if lang == 'cs':
        return open(os.path.join(DATA, 'plain', 'cs.txt'), encoding='utf-8').read()
    if lang == 'it':
        return ' '.join(' '.join(l['words']) for l in vlib.load_ref('Italian-Dante'))
    if lang == 'de':
        return ' '.join(' '.join(l['words']) for l in vlib.load_ref('German-Kafka'))
    f = {'he': 'he_raw.txt', 'oc': 'oc_raw.txt', 'grc': 'grc_raw.txt'}[lang]
    return open(os.path.join(SCR, f), encoding='utf-8').read()


HEB_FINAL = {'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}


def letters(lang, text=None, max_chars=600000):
    """Normalised letter string (no spaces) for a language."""
    t = text if text is not None else _text_for(lang)
    if lang == 'he':
        t = ''.join(HEB_FINAL.get(c, c) for c in t)
        s = ''.join(c for c in t if 'א' <= c <= 'ת')
        return s[:max_chars]
    t = _strip(t.lower())
    if lang == 'grc':
        t = t.replace('ς', 'σ')
        s = ''.join(c for c in t if 'α' <= c <= 'ω')
        return s[:max_chars]
    t = t.replace('ß', 'ss')
    s = ''.join(c for c in t if 'a' <= c <= 'z')
    if lang == 'la':
        s = s.replace('j', 'i').replace('v', 'u').replace('w', 'uu')
    return s[:max_chars]


class LM:
    """Character trigram model, add-k smoothing, log2 conditional probs."""

    def __init__(self, s, k=0.1, min_share=0.0005):
        c = Counter(s)
        n = len(s)
        alpha = sorted(ch for ch, v in c.items() if v / n >= min_share)
        self.alpha = alpha
        self.idx = {ch: i for i, ch in enumerate(alpha)}
        A = len(alpha)
        x = np.array([self.idx[ch] for ch in s if ch in self.idx], dtype=np.int64)
        self.x = x
        cnt3 = np.zeros((A, A, A))
        np.add.at(cnt3, (x[:-2], x[1:-1], x[2:]), 1)
        p = (cnt3 + k) / (cnt3.sum(2, keepdims=True) + k * A)
        self.LP = np.log2(p)
        u = np.bincount(x, minlength=A) + 1.0
        self.uni = u / u.sum()
        self.A = A

    def score_idx(self, y):
        y = np.asarray(y)
        if len(y) < 3:
            return float('nan')
        return float(self.LP[y[:-2], y[1:-1], y[2:]].mean())

    def encode(self, s):
        return np.array([self.idx[ch] for ch in s if ch in self.idx], dtype=np.int64)


_LMS = {}


def get_lm(lang):
    if lang not in _LMS:
        _LMS[lang] = LM(letters(lang))
    return _LMS[lang]


LANGS = ['la', 'it', 'de', 'cs', 'he', 'oc', 'grc']


# ------------------------------------------------------------------ symbol-stream solver

def tri_table(sym):
    sym = np.asarray(sym, dtype=np.int64)
    S = int(sym.max()) + 1 if len(sym) else 1
    code = (sym[:-2] * S + sym[1:-1]) * S + sym[2:]
    u, c = np.unique(code, return_counts=True)
    a = u // (S * S); b = (u // S) % S; cc = u % S
    return a, b, cc, c.astype(float), S


class Solver:
    """Fit symbol->letter key on a training stream by random guessing + annealing,
    report train and held-out mean log2 P(letter | 2 previous letters)."""

    def __init__(self, lm, train, test, S):
        self.lm = lm
        self.S = S
        a, b, c, w, _ = tri_table(train)
        self.a, self.b, self.c, self.w = a, b, c, w
        self.W = w.sum()
        self.test = test
        self.inv = [np.where((a == s) | (b == s) | (c == s))[0] for s in range(S)]

    def score_keys(self, K):
        """K: (n, S) keys -> train mean log prob for each."""
        LP = self.lm.LP
        return (LP[K[:, self.a], K[:, self.b], K[:, self.c]] * self.w).sum(1) / self.W

    def heldout(self, key):
        return self.lm.score_idx(key[self.test]) if len(self.test) > 2 else float('nan')

    def random_stage(self, n, rng):
        A = self.lm.A
        K = rng.choice(A, size=(n, self.S), p=self.lm.uni)
        sc = np.concatenate([self.score_keys(K[i:i + 250]) for i in range(0, n, 250)])
        return K, sc

    def anneal(self, key, rng, steps=4000, T0=0.02, T1=0.0005):
        key = key.copy()
        LP = self.lm.LP
        a, b, c, w = self.a, self.b, self.c, self.w
        cur = float((LP[key[a], key[b], key[c]] * w).sum())
        A = self.lm.A
        props_s = rng.integers(0, self.S, steps)
        props_l = rng.choice(A, size=steps, p=self.lm.uni)
        us = rng.random(steps)
        for t in range(steps):
            s = props_s[t]; l = props_l[t]
            old = key[s]
            if l == old:
                continue
            ix = self.inv[s]
            if len(ix) == 0:
                continue
            ka, kb, kc = key[a[ix]], key[b[ix]], key[c[ix]]
            before = (LP[ka, kb, kc] * w[ix]).sum()
            key[s] = l
            ka, kb, kc = key[a[ix]], key[b[ix]], key[c[ix]]
            after = (LP[ka, kb, kc] * w[ix]).sum()
            d = (after - before) / self.W
            T = T0 * (T1 / T0) ** (t / steps)
            if d >= 0 or us[t] < math.exp(d / T):
                cur += after - before
            else:
                key[s] = old
        return key, cur / self.W


def solve(lm, stream, S, rng, n_random=2000, n_starts=3, steps=None, split=0.5):
    stream = np.asarray(stream, dtype=np.int64)
    cut = int(len(stream) * split)
    tr, te = stream[:cut], stream[cut:]
    sv = Solver(lm, tr, te, S)
    K, sc = sv.random_stage(n_random, rng)
    order = np.argsort(-sc)
    best_rand_train = float(sc[order[0]])
    best_rand_test = sv.heldout(K[order[0]])
    steps = steps or max(3000, 60 * S)
    best = None
    for j in range(n_starts):
        k, s = sv.anneal(K[order[j]], rng, steps=steps)
        if best is None or s > best[1]:
            best = (k, s)
    key, tr_score = best
    return {'rand_train': best_rand_train, 'rand_test': best_rand_test,
            'train': tr_score, 'test': sv.heldout(key), 'key': key.tolist(),
            'sample': ''.join(lm.alpha[i] for i in key[te[:80]])}


# ------------------------------------------------------------------ length streams and schemes

def length_stream(lines, fn=glen):
    return [fn(w) for L in lines for w in L['words']]


def scheme_stream(lens, scheme, per_line=None):
    """Map a raw length stream to symbols. Returns (symbols, S)."""
    L = np.asarray(lens)
    if scheme == 'len':        # single length, clipped 1..10
        s = np.clip(L, 1, 10) - 1; return s, 10
    if scheme == 'pair':       # non-overlapping pairs of lengths clipped 1..9
        x = np.clip(L, 1, 9) - 1
        x = x[:len(x) // 2 * 2]
        return x[0::2] * 9 + x[1::2], 81
    if scheme == 'bacon2':     # parity bits in groups of 5
        bits = L % 2
        bits = bits[:len(bits) // 5 * 5].reshape(-1, 5)
        return bits @ np.array([16, 8, 4, 2, 1]), 32
    if scheme == 'baconT':     # short/long (<= 5 vs >= 6) bits in groups of 5
        bits = (L >= 6).astype(int)
        bits = bits[:len(bits) // 5 * 5].reshape(-1, 5)
        return bits @ np.array([16, 8, 4, 2, 1]), 32
    if scheme == 'tri3':       # length mod 3, groups of 3 trits
        t = L % 3
        t = t[:len(t) // 3 * 3].reshape(-1, 3)
        return t @ np.array([9, 3, 1]), 27
    if scheme == 'mod4pair':   # (len mod 4, next len mod 4) pairs
        t = L % 4
        t = t[:len(t) // 2 * 2]
        return t[0::2] * 4 + t[1::2], 16
    raise ValueError(scheme)


SCHEMES = ['len', 'pair', 'bacon2', 'baconT', 'tri3', 'mod4pair']


# ------------------------------------------------------------------ positive control: plant a text in the lengths

def plant(template, message, scheme, rng, fn=glen, src_lines=None):
    """Return lines with the same words-per-line template as `template`, whose
    word lengths encode `message` (a letter string) under `scheme`, with words
    drawn from the Voynich vocabulary of the required length (token frequency).
    The letter->length tables are built to reproduce the Voynich length mix."""
    src_lines = src_lines or template
    toks = [w for L in src_lines for w in L['words']]
    by_len = defaultdict(list)
    for w in toks:
        by_len[min(fn(w), 10)].append(w)
    lens = np.array([min(fn(w), 10) for w in toks])
    alpha = sorted(set(message))
    li = {ch: i for i, ch in enumerate(alpha)}
    p = Counter(message); n = len(message)
    pm = {ch: p[ch] / n for ch in alpha}
    if scheme == 'len':
        f = Counter(lens.tolist()); fl = {k: v / len(lens) for k, v in f.items()}
        rem = dict(fl); table = {}
        for ch in sorted(alpha, key=lambda c: -pm[c]):
            L = max(rem, key=lambda k: rem[k]); table[ch] = [(L, 1.0)]; rem[L] -= pm[ch]
        emit = lambda ch: [table[ch][0][0]]
    elif scheme == 'pair':
        x = np.clip(lens, 1, 9); pr = Counter(zip(x[0::2].tolist(), x[1::2].tolist()))
        tot = sum(pr.values())
        rem = dict(pm); table = defaultdict(list)
        items = sorted(pr.items(), key=lambda kv: -kv[1])
        need_one = sorted(alpha, key=lambda c: -pm[c])
        for pair, v in items:
            if need_one and len(items) - len(sum(table.values(), [])) <= len(need_one):
                ch = need_one.pop(0)
            else:
                ch = max(rem, key=lambda c: rem[c])
                if ch in need_one:
                    need_one.remove(ch)
            table[ch].append((pair, v / tot)); rem[ch] -= v / tot

        def emit(ch):
            opts = table[ch]; w = np.array([o[1] for o in opts]); k = rng.choice(len(opts), p=w / w.sum())
            return list(opts[k][0])
    elif scheme == 'bacon2':
        f = Counter(lens.tolist())
        odd = [k for k in f if k % 2]; even = [k for k in f if k % 2 == 0]
        po = np.array([f[k] for k in odd], float); pe = np.array([f[k] for k in even], float)

        def emit(ch):
            v = li[ch]; out = []
            for bit in ((v >> 4) & 1, (v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1):
                out.append(int(rng.choice(odd, p=po / po.sum())) if bit else int(rng.choice(even, p=pe / pe.sum())))
            return out
    else:
        raise ValueError(scheme)
    need = sum(len(L['words']) for L in template)
    seq, k = [], 0
    while len(seq) < need:
        seq += emit(message[k % n]); k += 1
    out, j = [], 0
    for L in template:
        ws = []
        for _ in L['words']:
            Lw = seq[j]; j += 1
            pool = by_len.get(Lw) or by_len[max(by_len)]
            ws.append(pool[rng.integers(len(pool))])
        d = dict(L); d['words'] = ws; d['gaps'] = ['.'] * (len(ws) - 1); out.append(d)
    return out


def caesar_letters():
    t = ' '.join(' '.join(l['words']) for l in vlib.load_ref('Latin-Caesar', skip_frac=0.02))
    return letters('la', t)


def gibbs(sv, key, rng, sweeps=12, T0=0.03, T1=0.0005):
    """Heat-bath annealing: for each symbol in random order, score every letter
    at once and sample from softmax(score / T). Returns (key, train score)."""
    key = key.copy()
    LP = sv.lm.LP
    A = sv.lm.A
    letters_ = np.arange(A)[:, None]
    for sw in range(sweeps):
        T = T0 * (T1 / T0) ** (sw / max(1, sweeps - 1))
        for s in rng.permutation(sv.S):
            ix = sv.inv[s]
            if len(ix) == 0:
                continue
            a, b, c, w = sv.a[ix], sv.b[ix], sv.c[ix], sv.w[ix]
            ka = np.where(a == s, letters_, key[a]); kb = np.where(b == s, letters_, key[b])
            kc = np.where(c == s, letters_, key[c])
            sc = (LP[ka, kb, kc] * w).sum(1) / sv.W
            z = (sc - sc.max()) / T
            p = np.exp(z); p /= p.sum()
            key[s] = rng.choice(A, p=p)
    # final greedy sweep
    for s in range(sv.S):
        ix = sv.inv[s]
        if len(ix) == 0:
            continue
        a, b, c, w = sv.a[ix], sv.b[ix], sv.c[ix], sv.w[ix]
        ka = np.where(a == s, letters_, key[a]); kb = np.where(b == s, letters_, key[b])
        kc = np.where(c == s, letters_, key[c])
        key[s] = int(np.argmax((LP[ka, kb, kc] * w).sum(1)))
    return key, float(sv.score_keys(key[None, :])[0])


def solve2(lm, stream, S, rng, n_random=2000, n_starts=3, sweeps=12, split=0.5):
    stream = np.asarray(stream, dtype=np.int64)
    cut = int(len(stream) * split)
    tr, te = stream[:cut], stream[cut:]
    sv = Solver(lm, tr, te, S)
    K, sc = sv.random_stage(n_random, rng)
    order = np.argsort(-sc)
    out = {'rand_train': float(sc[order[0]]), 'rand_test': sv.heldout(K[order[0]]),
           'rand_median': float(np.median(sc))}
    best = None
    for j in range(n_starts):
        k, s = gibbs(sv, K[order[j]], rng, sweeps=sweeps)
        if best is None or s > best[1]:
            best = (k, s)
    key, trs = best
    out.update({'train': trs, 'test': sv.heldout(key), 'key': key.tolist(),
                'sample': ''.join(lm.alpha[i] for i in key[te[:80]])})
    return out
