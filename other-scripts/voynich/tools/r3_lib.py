#!/usr/bin/env python3
"""R-3: grow the script in a box.

Small societies keep records of a world (ledger entries: people, goods, numbers; or procedure/text
entries: concepts, function words, topical pages) and evolve a writing system for it by iterated
learning (scribe lineages, learning bottleneck, memory cap on logograms, abbreviation of frequent
forms, spelling drift, optional secrecy: homophones and nulls; optional word dividers).
Each evolved corpus is scored on a fixed blind panel and compared with real corpora (ABC).

Unified corpus format: list of documents; document = list of entries;
entry = (words, q) with words a list of tuples of sign strings and q an int or None.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json, math, random, re, gzip, pickle
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
VD = os.path.join(HERE, '..', 'data')
CK = os.path.join(VD, 'r3_ckpt')
OS = os.path.join(HERE, '..', '..')
X2 = os.path.join(OS, 'proto-elamite', 'data', 'x2')
SCR_UR3 = os.environ.get('R3_UR3', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/x2/corpus_UR3.json')
os.makedirs(CK, exist_ok=True)

NTOK = 2500          # sign tokens per panel sample (all corpora, real and simulated)

# ----------------------------------------------------------------------------- real corpora

def _x2_doc(t, split_words=True, numeral_re=None):
    doc = []
    for e in t['entries']:
        ws = []
        for w in e.get('des') or []:
            if w is None or '...' in w or w in ('x', 'X', '[...]'):
                continue
            if numeral_re is not None and numeral_re.search(w):
                continue
            if split_words:
                s = tuple(x for x in re.split(r'[-.]', w) if x and x not in ('x', '...'))
            else:
                s = (w,)
            if s:
                ws.append(s)
        if e.get('com'):
            ws.append((str(e['com']),))
        q = e.get('q')
        if q is not None and e.get('frac') and q == 0:
            q = None
        if ws or q is not None:
            doc.append((ws, q))
    return doc


def load_x2(name, max_docs=None, seed=7):
    if name == 'UR3':
        f = os.path.join(CK, 'ur3_sub.pkl')
        if os.path.exists(f):
            return pickle.load(open(f, 'rb'))
        D = json.load(open(SCR_UR3))
        random.Random(seed).shuffle(D)
        D = D[:4000]
        out = [d for d in (_x2_doc(t, True, re.compile(r'\(|^\d')) for t in D) if d]
        pickle.dump(out, open(f, 'wb'))
        return out
    D = json.load(open(os.path.join(X2, 'corpus_%s.json' % name)))
    if name == 'PE':
        # PE designations are sign sequences with no word divider: the whole designation is one 'word'
        out = []
        for t in D:
            doc = []
            for e in t['entries']:
                ws = []
                s = tuple(x for x in (e.get('des') or []) if x and 'X' != x and '...' not in x)
                if s:
                    ws.append(s)
                if e.get('com'):
                    ws.append((e['com'],))
                q = e.get('q')
                if q is not None and e.get('frac') and q == 0:
                    q = None
                if ws or q is not None:
                    doc.append((ws, q))
            if doc:
                out.append(doc)
        return out
    return [d for d in (_x2_doc(t, True) for t in D) if d]


EVA_MULTI = ['cth', 'ckh', 'cph', 'cfh', 'ch', 'sh']


def eva_signs(w):
    out, i = [], 0
    while i < len(w):
        for m in EVA_MULTI:
            if w.startswith(m, i):
                out.append(m); i += len(m); break
        else:
            out.append(w[i]); i += 1
    return tuple(out)


def load_voynich(fn='ZL3b-n.txt', lang=None):
    pages, cur, curlang = [], None, None
    for line in open(os.path.join(VD, fn), encoding='utf-8', errors='replace'):
        line = line.rstrip('\n')
        if not line or line.startswith('#'):
            continue
        m = re.match(r'<(f\w+)>\s*(.*)', line)
        if m:
            if cur:
                pages.append((curlang, cur))
            cur = []
            lm = re.search(r'\$L=(\w)', m.group(2))
            curlang = lm.group(1) if lm else None
            continue
        m = re.match(r'<(f\w+)\.(\d+)[^>]*>\s*(.*)', line)
        if not m or cur is None:
            continue
        txt = re.sub(r'<![^>]*>', '', m.group(3))
        txt = txt.replace('<->', '.').replace('<~>', '.')
        txt = re.sub(r'<[^>]*>', '', txt)
        txt = re.sub(r'\{[^}]*\}', '', txt)
        txt = re.sub(r'\[([^:\]]*):[^\]]*\]', r'\1', txt)
        ws = []
        for w in re.split(r'[.,\s]+', txt):
            if not w or '?' in w or not re.fullmatch(r'[a-z]+', w):
                continue
            ws.append(eva_signs(w))
        if ws:
            cur.append((ws, None))
    if cur:
        pages.append((curlang, cur))
    return [p for l, p in pages if p and (lang is None or l == lang)]


def load_latin(fn='pg218.txt'):
    txt = open(os.path.join(VD, fn), encoding='utf-8', errors='replace').read()
    a = txt.find('*** START'); b = txt.find('*** END')
    txt = txt[txt.find('\n', a) + 1:b]
    docs = []
    for para in re.split(r'\n\s*\n', txt):
        doc = []
        for line in para.split('\n'):
            ws = [tuple(w) for w in re.findall(r'[a-z]+', line.lower())]
            if ws:
                doc.append((ws, None))
        if len(doc) >= 2:
            docs.append(doc)
    return docs


def load_real(name):
    if name in ('PE', 'LA', 'LB', 'UR3'):
        return load_x2(name)
    if name == 'VMS':
        return load_voynich()
    if name == 'VMS_A':
        return load_voynich(lang='A')
    if name == 'VMS_B':
        return load_voynich(lang='B')
    if name == 'LAT':
        return load_latin('pg218.txt')
    if name == 'ITA':
        return load_latin('pg1000.txt')
    raise KeyError(name)


def shuffle_corpus(C, seed=0):
    """Control: every sign token redealt over the corpus (word lengths, entries, docs, numbers kept)."""
    rng = random.Random(seed)
    pool = [s for d in C for ws, q in d for w in ws for s in w]
    rng.shuffle(pool)
    it = iter(pool)
    return [[([tuple(next(it) for _ in w) for w in ws], q) for ws, q in d] for d in C]


# ----------------------------------------------------------------------------- panel

FIT_NAMES = ['sign_types', 'sign_zipf', 'sign_top10', 'sign_hapax', 'wlen_mean', 'wlen_cv', 'p_wlen1',
             'wpe_mean', 'epd_log', 'word_ttr', 'word_hapax', 'dup_entry', 'kl_first', 'kl_last',
             'mi_junc', 'mi_inner', 'kl_entry_first', 'p_num', 'q_log', 'p_round']
NUM_STATS = ['p_num', 'q_log', 'p_round']
HELD_NAMES = ['burst', 'adj_rep']


def sample_docs(C, rng, ntok=NTOK):
    idx = list(range(len(C)))
    rng.shuffle(idx)
    out, n = [], 0
    for i in idx:
        d = C[i]
        out.append(d)
        n += sum(len(w) for ws, q in d for w in ws)
        if n >= ntok:
            break
    return out


def _kl(p_counter, q_counter, keys):
    tp = sum(p_counter.values()); tq = sum(q_counter.values())
    if tp == 0 or tq == 0:
        return 0.0
    k = len(keys)
    s = 0.0
    for x in keys:
        p = (p_counter.get(x, 0) + 0.5) / (tp + 0.5 * k)
        q = (q_counter.get(x, 0) + 0.5) / (tq + 0.5 * k)
        s += p * math.log2(p / q)
    return s


def _mi(pairs, rng):
    if len(pairs) < 20:
        return 0.0
    def mi(ps):
        n = len(ps)
        cxy = Counter(ps); cx = Counter(a for a, b in ps); cy = Counter(b for a, b in ps)
        return sum(c / n * math.log2(c * n / (cx[a] * cy[b])) for (a, b), c in cxy.items())
    a = [x for x, y in pairs]; b = [y for x, y in pairs]
    bb = b[:]; rng.shuffle(bb)
    return mi(pairs) - mi(list(zip(a, bb)))


def panel(docs, rng):
    """Blind statistics on one sample (list of docs)."""
    signs = Counter(); words = Counter(); wl = []
    first = Counter(); last = Counter(); efirst = Counter()
    junc = []; inner = []; nent = 0; nnum = 0; qs = []
    entries = Counter(); wpe = []
    doc_of = defaultdict(set); wcount_doc = []
    adj_hits = 0; adj_n = 0
    for di, d in enumerate(docs):
        prev = None
        for ws, q in d:
            nent += 1
            if q is not None:
                nnum += 1; qs.append(q)
            wpe.append(len(ws))
            entries[tuple(ws)] += 1
            cur = set()
            for j, w in enumerate(ws):
                signs.update(w); words[w] += 1; wl.append(len(w))
                first[w[0]] += 1; last[w[-1]] += 1
                if j == 0:
                    efirst[w[0]] += 1
                if j > 0:
                    junc.append((ws[j - 1][-1], w[0]))
                for k in range(1, len(w)):
                    inner.append((w[k - 1], w[k]))
                doc_of[w].add(di)
                cur.add(w)
                if prev is not None:
                    adj_n += 1; adj_hits += (w in prev)
            if cur:
                prev = cur
    ntok = sum(signs.values())
    if ntok < 50 or not words:
        return None
    fr = np.array(sorted(signs.values(), reverse=True), float)
    r = np.arange(1, len(fr) + 1)
    k = min(30, len(fr))
    zipf = float(np.polyfit(np.log(r[:k]), np.log(fr[:k]), 1)[0]) if k >= 3 else 0.0
    wl = np.array(wl, float)
    nw = sum(words.values())
    keys = list(signs.keys())
    qpos = [x for x in qs if x and x > 0]
    S = {
        'sign_types': math.log(len(signs)),
        'sign_zipf': zipf,
        'sign_top10': float(fr[:10].sum() / ntok),
        'sign_hapax': float(np.mean(fr == 1)),
        'wlen_mean': float(wl.mean()),
        'wlen_cv': float(wl.std() / wl.mean()),
        'p_wlen1': float(np.mean(wl == 1)),
        'wpe_mean': float(np.mean(wpe)),
        'epd_log': math.log(nent / len(docs)),
        'word_ttr': len(words) / nw,
        'word_hapax': float(np.mean(np.array(list(words.values())) == 1)),
        'dup_entry': sum(c for c in entries.values() if c > 1) / nent,
        'kl_first': _kl(first, signs, keys),
        'kl_last': _kl(last, signs, keys),
        'mi_junc': _mi(junc, rng),
        'mi_inner': _mi(inner, rng),
        'kl_entry_first': _kl(efirst, first, keys),
        'p_num': nnum / nent,
        'q_log': float(np.mean(np.log10(np.array(qpos) + 1))) if qpos else 0.0,
        'p_round': float(np.mean([x % 10 == 0 for x in qpos if x >= 10])) if any(x >= 10 for x in qpos) else 0.0,
    }
    # held-out statistics (never used in fitting)
    nd = len(docs)
    dsize = np.array([sum(len(ws) for ws, q in d) for d in docs], float)
    pd = dsize / dsize.sum()
    rat = []
    for w, c in words.items():
        if c >= 3:
            exp = float(np.sum(1 - (1 - pd) ** c))
            rat.append(len(doc_of[w]) / exp)
    S['burst'] = float(np.mean(rat)) if rat else 1.0
    S['adj_rep'] = adj_hits / adj_n if adj_n else 0.0
    return S


def panel_mean(C, n=8, seed=0, ntok=NTOK):
    rng = random.Random(seed)
    rows = [panel(sample_docs(C, rng, ntok), rng) for _ in range(n)]
    rows = [r for r in rows if r]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, rows


# ----------------------------------------------------------------------------- simulator

PRIOR = {  # name: (lo, hi, kind)
    # world
    'p_ledger': (0, 1, 'u'),
    'lg_people': (0.7, 4.0, 'u'),
    'name_a': (0.3, 1.6, 'u'),
    'lam_name': (0, 1.5, 'u'),
    'lg_goods': (0, 2.5, 'u'),
    'goods_a': (0.3, 2.0, 'u'),
    'p_num': (0, 1, 'u'),
    'q_mu': (0, 2.5, 'u'),
    'q_sd': (0.1, 1.2, 'u'),
    'p_round': (0, 1, 'u'),
    'lg_vocab': (1.3, 4.0, 'u'),
    'vocab_a': (0.6, 1.6, 'u'),
    'text_len': (1.5, 12, 'u'),
    'p_func': (0, 0.6, 'u'),
    'lg_func': (0.5, 1.7, 'u'),
    'topical': (0, 0.9, 'u'),
    'lg_epd': (0.3, 1.6, 'u'),
    'p_header': (0, 1, 'u'),
    # writing system
    'lg_syl': (1.1, 2.6, 'u'),
    'syl_alpha': (-1.3, 0.7, 'u'),
    'wlen': (1.2, 5.0, 'u'),
    'lg_logo': (0, 2.7, 'u'),
    'n_gen': (1, 20, 'i'),
    'bottleneck': (0.02, 1.0, 'u'),
    'abbrev': (0, 0.5, 'u'),
    'p_var': (0, 0.3, 'u'),
    'n_scribes': (1, 10, 'i'),
    'shared': (0, 1, 'u'),
    'secrecy': (0, 1, 'u'),
    'p_div': (0.3, 1.0, 'u'),
}
PNAMES = list(PRIOR)


def draw_prior(rng):
    th = {}
    for k, (lo, hi, kind) in PRIOR.items():
        th[k] = rng.randint(lo, hi) if kind == 'i' else rng.uniform(lo, hi)
    return th


def _zipf(n, a):
    w = 1.0 / np.arange(1, n + 1) ** a
    return w / w.sum()


class World:
    def __init__(self, th, seed):
        self.th = th
        self.rng = np.random.default_rng(seed)
        self.r = random.Random(seed)
        R = self.rng
        self.n_people = int(10 ** th['lg_people'])
        self.n_goods = max(1, int(round(10 ** th['lg_goods'])))
        self.n_vocab = int(10 ** th['lg_vocab'])
        self.n_func = max(1, int(round(10 ** th['lg_func'])))
        # concept id ranges
        self.off_p = 0
        self.off_g = self.n_people
        self.off_v = self.off_g + self.n_goods
        self.off_f = self.off_v + self.n_vocab
        self.nc = self.off_f + self.n_func
        self.pp = _zipf(self.n_people, th['name_a'])
        self.pg = _zipf(self.n_goods, th['goods_a'])
        self.pv = _zipf(self.n_vocab, th['vocab_a'])
        self.pf = _zipf(self.n_func, 1.0)
        # phonology for spelled forms
        ns = max(5, int(round(10 ** th['lg_syl'])))
        self.ns = ns
        base = _zipf(ns, 0.8)
        a = 10 ** th['syl_alpha']
        self.init = R.dirichlet(a * ns * base + 1e-3)
        self.trans = R.dirichlet(a * ns * base + 1e-3, size=ns)
        self.trans_c = np.cumsum(self.trans, 1)
        self.init_c = np.cumsum(self.init)
        self.logo_next = 0
        self.bank = []

    # concept usage marginal (approximate, for the learning bottleneck)
    def marginal(self, K=4000):
        th = self.th
        pl = th['p_ledger']
        names_per = 1 + th['lam_name']
        led = pl * (names_per + 1)
        txt = (1 - pl) * th['text_len']
        tot = led + txt + 1e-9
        parts = [(self.off_p, self.pp, pl * names_per / tot), (self.off_g, self.pg, pl / tot),
                 (self.off_v, self.pv, txt * (1 - th['p_func']) / tot), (self.off_f, self.pf, txt * th['p_func'] / tot)]
        ids, ps = [], []
        for off, p, w in parts:
            k = min(len(p), K)
            ids.append(np.arange(off, off + k)); ps.append(p[:k] * w)
        return np.concatenate(ids), np.concatenate(ps)

    def _fill(self, n=3000):
        R = self.rng
        L = np.minimum(1 + R.poisson(max(0.0, self.th['wlen'] - 1), n), 10)
        S = np.zeros((n, 10), np.int64)
        u = R.random((n, 10))
        S[:, 0] = np.minimum((self.init_c[None, :] < (u[:, :1] * self.init_c[-1])).sum(1), self.ns - 1)
        for i in range(1, int(L.max())):
            c = self.trans_c[S[:, i - 1]]
            S[:, i] = np.minimum((c < u[:, i:i + 1] * c[:, -1:]).sum(1), self.ns - 1)
        self.bank = [tuple(row[:l]) for row, l in zip(S.tolist(), L.tolist())]

    def spell(self):
        if not self.bank:
            self._fill()
        return self.bank.pop()

    def new_logo(self):
        self.logo_next += 1
        return ('L', self.logo_next)

    def evolve(self):
        """Iterated learning: lineages of scribes pass the lexicon through a bottleneck."""
        th = self.th
        ids, ps = self.marginal()
        ps = ps / ps.sum()
        nlin = 1 if th['shared'] > 0.5 else th['n_scribes']
        cap = int(10 ** th['lg_logo']) if th['lg_logo'] > 0.3 else 0
        B = th['bottleneck'] * 1500
        lexs = []
        for _ in range(nlin):
            lex = {}
            for g in range(th['n_gen']):
                cnt = self.rng.poisson(ps * B)
                seen = ids[cnt > 0]
                cs = cnt[cnt > 0]
                newlex = {}
                order = np.argsort(-cs)
                logo_ok = set(int(seen[i]) for i in order[:cap]) if cap else set()
                for c, k in zip(seen.tolist(), cs.tolist()):
                    f = lex.get(c)
                    if f is None:
                        f = self.spell()
                    if c in logo_ok:
                        if not (len(f) == 1 and isinstance(f[0], tuple)):
                            f = (self.new_logo(),)
                    elif len(f) == 1 and isinstance(f[0], tuple):
                        f = self.spell()
                    else:
                        if len(f) > 1 and self.r.random() < th['abbrev'] * min(1.0, k / 5.0):
                            f = f[:-1]
                        if self.r.random() < th['p_var']:
                            f = list(f); j = self.r.randrange(len(f)); f[j] = self.r.randrange(self.ns); f = tuple(f)
                    newlex[c] = f
                lex = newlex
            lexs.append(lex)
        self.lexs = lexs

    def form(self, lin, c):
        lex = self.lexs[lin]
        f = lex.get(c)
        if f is None:
            f = self.spell()
            lex[c] = f
        return f

    def write_word(self, lin, c):
        th = self.th
        f = list(self.form(lin, c))
        if self.r.random() < th['p_var'] * 0.3 and not isinstance(f[0], tuple):
            j = self.r.randrange(len(f)); f[j] = self.r.randrange(self.ns)
        out = []
        h = 1 + int(3 * th['secrecy'])
        pnull = 0.15 * th['secrecy'] if th['secrecy'] > 0.5 else 0.0
        for s in f:
            if pnull and self.r.random() < pnull:
                out.append('n%d' % self.r.randrange(3))
            if isinstance(s, tuple):
                out.append('L%d' % s[1])
            else:
                out.append('s%d.%d' % (s, self.r.randrange(h)) if h > 1 else 's%d' % s)
        return tuple(out)

    def draw(self, p, n, pool, topical):
        out = []
        for _ in range(n):
            if pool is not None and self.r.random() < topical:
                out.append(self.r.choice(pool))
            else:
                out.append(int(self.rng.choice(len(p), p=p)))
        return out

    def corpus(self, ntok=NTOK + 300):
        th = self.th
        R = self.rng; r = self.r
        docs = []; n = 0
        # cumulative tables for fast sampling
        cp = np.cumsum(self.pp); cg = np.cumsum(self.pg); cv = np.cumsum(self.pv); cf = np.cumsum(self.pf)
        samp = lambda c: int(min(np.searchsorted(c, r.random() * c[-1]), len(c) - 1))
        nlin = len(self.lexs)
        while n < ntok:
            lin = r.randrange(nlin)
            ppool = [samp(cp) for _ in range(5)]
            vpool = [samp(cv) for _ in range(8)]
            ne = 1 + R.poisson(max(0.0, 10 ** th['lg_epd'] - 1))
            doc = []
            if r.random() < th['p_header']:
                c = self.off_v + vpool[0] if r.random() > th['p_ledger'] else self.off_p + ppool[0]
                doc.append(([self.write_word(lin, c)], None))
            for _ in range(ne):
                if r.random() < th['p_ledger']:
                    cs = []
                    for _ in range(1 + R.poisson(th['lam_name'])):
                        cs.append(self.off_p + (r.choice(ppool) if r.random() < th['topical'] else samp(cp)))
                    cs.append(self.off_g + samp(cg))
                    q = None
                    if r.random() < th['p_num']:
                        q = max(1, int(round(10 ** R.normal(th['q_mu'], th['q_sd']))))
                        if q >= 10 and r.random() < th['p_round']:
                            b = 10 ** int(math.log10(q))
                            q = int(round(q / b)) * b
                else:
                    cs = []
                    for _ in range(1 + R.poisson(max(0.0, th['text_len'] - 1))):
                        if r.random() < th['p_func']:
                            cs.append(self.off_f + samp(cf))
                        else:
                            cs.append(self.off_v + (r.choice(vpool) if r.random() < th['topical'] else samp(cv)))
                    q = None
                ws = [self.write_word(lin, c) for c in cs]
                # word dividers: unwritten dividers merge neighbouring words
                merged = [ws[0]]
                for w in ws[1:]:
                    if r.random() < th['p_div']:
                        merged.append(w)
                    else:
                        merged[-1] = merged[-1] + w
                doc.append((merged, q))
                n += sum(len(w) for w in merged)
            docs.append(doc)
        return docs


def simulate(th, seed):
    W = World(th, seed)
    W.evolve()
    C = W.corpus()
    rng = random.Random(seed + 1)
    return panel(sample_docs(C, rng), rng)


def theta_vec(th):
    return [float(th[k]) for k in PNAMES]
