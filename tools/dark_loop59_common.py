"""Loop 59: HOW MUCH OF THE SCRIPT IS EXPLAINED? Shared code.
Every model is a predictive distribution P(next sign or END | text so far, object class, site) scored in bits
(cross-entropy) on held-out texts, so that models of very different kinds (slot grammar, Markov chains, the S366
generator, the loop 33 automaton, text caches) are compared on one scale. Numbers only; no interpretation.

Corpus: data/derived/merged-corpus-canonical.json (S-DARK-23 caution), complete, direction-recorded texts, one copy
per die (loop 41 regime; 'rows' regime optional), three merge levels. Fit = Mohenjo-daro + Harappa. Evaluation
(a) held-out sites (every other known site), (b) the 324 IM77-only texts of S-DARK-27 (M -> W through the completed
bridge, most frequent Wells form per M sign; unbridged M signs score as UNK).
"""
import json, csv, math, random, collections, sys, os, bisect, itertools
ROOT = '/home/user/Indus-/'
DARK = ROOT + 'data/derived/dark/'
sys.path.insert(0, ROOT + 'tools')
BIG = ('Mohenjo-daro', 'Harappa')
END = 'E'; UNK = 'U'

# sign classes (GRAMMAR.md; loop 33 / loop 54 constants)
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; MJAR = {741, 742, 745}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]; CLS = set(CL)
FISHQ = {235, 240, 233, 231}; FISH = {220}; TREE = {390, 405, 407}; LEAF = {803, 806, 838}
NUMS = {1, 3, 4, 5, 16, 17, 18}; NUMT = {31, 32, 33, 34, 55, 56}; NUM = NUMS | NUMT
TITLE = {255, 435, 690, 760, 100, 176, 923, 904, 636, 61, 752, 48, 590, 142, 575, 585}

def otype(t):
    t = t.split(':')[0]
    return 'SEAL' if t == 'SEAL' else 'TAB' if t == 'TAB' else 'OTHER'

def load_corpus(level, regime='die'):
    """-> list of dict(site, ot, seq, cisi). complete + direction-recorded texts, one copy per die."""
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    out = []; seen = set()
    for r in C:
        s = tuple(r[level])
        if not s or r['complete'] != 'Y' or r['dir.'].strip() == '-' or r['site'] == 'Unknown': continue
        ot = otype(r['type'])
        if regime == 'die':
            if ot == 'TAB' or r['type'].startswith('TAG') or r['type'].startswith('POT:T:s'): k = (r['site'], ot, s)
            else: k = (r['cisi'] if r['cisi'] not in ('-', '') else id(r), s)
            if k in seen: continue
            seen.add(k)
        out.append(dict(site=r['site'], ot=ot, seq=s, cisi=r['cisi'], type=r['type']))
    return out

def load_im77_new(level, fit):
    """the 324 IM77-only texts, M -> W (most frequent Wells form at this level); unbridged -> UNK."""
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    RAW2LV = collections.defaultdict(collections.Counter)
    for r in C:
        for a, b in zip(r['seq_raw'], r[level]): RAW2LV[a][b] += 1
    def canon(w):
        c = RAW2LV.get(w); return c.most_common(1)[0][0] if c else w
    BR = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
    L27 = json.load(open(DARK + 'loop27_sets.json'))
    bridge = dict(L27['bridge']); bridge.update({k: v for k, v in BR.items() if k not in bridge})
    freq = collections.Counter(x for o in fit for x in o['seq'])
    m2w = collections.defaultdict(list)
    for w, ms in bridge.items():
        for m in ms: m2w[m].append(canon(int(w)))
    M2W = {m: max(ws, key=lambda w: freq.get(w, 0)) for m, ws in m2w.items()}
    want = set((a, b) for a, b in L27['new'])
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')):
        k = (r['text_no'], r['side'])
        if k in want and r['signs_clean'].strip():
            by[k].append((int(r['line']), [int(x) for x in r['signs_clean'].split() if x != '0'], r['object_type'], r['site']))
    out = []; unb = tot = 0
    for k, lines in by.items():
        lines.sort(); s = [x for ln in lines for x in ln[1]]
        ws = []
        for m in s:
            tot += 1
            if m in M2W: ws.append(M2W[m])
            else: ws.append(UNK); unb += 1
        if not ws: continue
        ot = {'seal': 'SEAL', 'miniature tablet': 'TAB', 'copper tablet': 'TAB'}.get(lines[0][2], 'OTHER')
        site = {'Mohenjodaro': 'Mohenjo-daro', 'Harappa': 'Harappa'}.get(lines[0][3], lines[0][3])
        out.append(dict(site=site, ot=ot, seq=tuple(ws), cisi='IM' + k[0] + '/' + k[1], type='IM77:' + lines[0][2]))
    return out, unb, tot

# ---------------------------------------------------------------- slot labels (S310 parse, as in strat_adequacy.parse)
def learn_qual(texts):
    left = collections.defaultdict(collections.Counter)
    for s in texts:
        s = list(s)
        while len(s) > 1 and s[-1] in SUF: s.pop()
        if len(s) >= 2 and s[-1] in CLS: left[s[-1]][s[-2]] += 1
    Q = {}
    for c, cnt in left.items():
        tot = sum(cnt.values()); acc = 0; q = set()
        for a, n in cnt.most_common():
            if acc / tot >= 0.6: break
            q.add(a); acc += n
        Q[c] = q
    return Q

def parse(s, QUAL):
    """slot label per token: OPENER MARKER NAME COUNT TITLE CLOSER SUFFIX"""
    lab = ['NAME'] * len(s); i = 0; j = len(s)
    if s[0] in OPEN:
        lab[0] = 'OPENER'; i = 1
        if len(s) > 1 and s[1] in MARK: lab[1] = 'MARKER'; i = 2
    while j - 1 > i and s[j - 1] in SUF and j >= 2 and (s[j - 2] in CLS or s[j - 2] in SUF): lab[j - 1] = 'SUFFIX'; j -= 1
    if j - 1 >= i and s[j - 1] in CLS:
        c = s[j - 1]; lab[j - 1] = 'CLOSER'; j -= 1
        if c == 520:
            if j - 2 >= i and s[j - 1] == 33 and s[j - 2] in (705, 706): lab[j - 1] = lab[j - 2] = 'TITLE'; j -= 2
            while j - 1 >= i and s[j - 1] in FISH | FISHQ: lab[j - 1] = 'TITLE'; j -= 1
        elif c == 740:
            if j - 1 >= i and s[j - 1] == 100: lab[j - 1] = 'TITLE'; j -= 1
            if j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        elif j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        if j - 1 >= i and s[j - 1] in NUM and lab[j] == 'TITLE': lab[j - 1] = 'TITLE'; j -= 1
    for k in range(i, j):
        if s[k] in NUM: lab[k] = 'COUNT'
    return lab

# ---------------------------------------------------------------- sign class (for the frame component)
def sign_class(x):
    if x == UNK: return 'OTHER'
    if x in OPEN: return 'OPEN'
    if x in MARK: return 'CONN'
    if x in MJAR: return 'MJAR'
    if x in SUF: return 'SUF'
    if x in CLS: return 'C' + str(x)
    if x in FISHQ: return 'FISHQ'
    if x in FISH: return 'FISH'
    if x in TREE: return 'TREE'
    if x in LEAF: return 'LEAF'
    if x in NUMS: return 'NUMs'
    if x in NUMT: return 'NUMt'
    if x in TITLE: return 'TITLE'
    return 'OTHER'

# ---------------------------------------------------------------- base distributions
class Unigram:
    """P(x | ot) with UNK mass (Witten-Bell style: singletons' share), shrunk to pooled; includes END."""
    def __init__(self, texts, V):
        self.V = V
        self.c = collections.defaultdict(collections.Counter)
        for o in texts:
            for x in o['seq']: self.c[o['ot']][x] += 1
            self.c[o['ot']][END] += 1
        self.pool = collections.Counter()
        for cnt in self.c.values(): self.pool.update(cnt)
        self.P = {}
        for g, cnt in list(self.c.items()) + [('ALL', self.pool)]:
            N = sum(cnt.values()); n1 = max(1, sum(1 for v in cnt.values() if v == 1))
            lam = 50.0
            d = {}
            for x in V | {END, UNK}:
                pg = (cnt.get(x, 0) + (lam * (self.pool.get(x, 0) / sum(self.pool.values())) if g != 'ALL' else 0.0)) / (N + (lam if g != 'ALL' else 0))
                d[x] = pg
            # UNK mass = singletons' share, taken proportionally from the rest
            pu = n1 / (N + n1)
            tot = sum(d.values())
            for x in d: d[x] = d[x] / tot * (1 - pu)
            d[UNK] += pu
            self.P[g] = d
    def p(self, x, ot):
        d = self.P.get(ot, self.P['ALL'])
        return d.get(x, d[UNK])
    def dist(self, ot):
        return self.P.get(ot, self.P['ALL'])

class KN:
    """interpolated Kneser-Ney, context = last `order` tokens (START padding, with a type token first), backoff
    to the Unigram above (which carries UNK and END)."""
    def __init__(self, texts, order, uni, D=0.75, site_token=False):
        self.k = order; self.uni = uni; self.D = D; self.site_token = site_token
        self.c = [collections.defaultdict(collections.Counter) for _ in range(order + 1)]   # c[n][ctx][x], n = len(ctx)
        self.cont = [collections.defaultdict(collections.Counter) for _ in range(order + 1)]  # continuation counts
        for o in texts:
            toks = self.prefix(o) + list(o['seq']) + [END]
            for i in range(len(self.prefix(o)), len(toks)):
                for n in range(0, order + 1):
                    ctx = tuple(toks[max(0, i - n):i]) if n else ()
                    if n and len(ctx) < n: ctx = ('S',) * (n - len(ctx)) + ctx
                    self.c[n][ctx][toks[i]] += 1
        # continuation counts: N1+(• ctx x) for lower orders
        for n in range(1, order + 1):
            for ctx, cnt in self.c[n].items():
                for x in cnt: self.cont[n - 1][ctx[1:]][x] += 1
        self.N1p = [{ctx: len(cnt) for ctx, cnt in tab.items()} for tab in self.c]
        self.N1pc = [{ctx: len(cnt) for ctx, cnt in tab.items()} for tab in self.cont]
    def prefix(self, o):
        p = ['T:' + o['ot']]
        if self.site_token: p = ['S:' + o['site']] + p
        return p
    def hist(self, o, h):
        toks = self.prefix(o) + list(h)
        ctx = tuple(toks[-self.k:]) if self.k else ()
        if self.k and len(ctx) < self.k: ctx = ('S',) * (self.k - len(ctx)) + ctx
        return ctx
    def p(self, x, o, h):
        ctx = self.hist(o, h)
        return self._p(x, ctx, self.k, top=True, ot=o['ot'])
    def _p(self, x, ctx, n, top, ot):
        if n == 0:
            cnt = self.cont[0][()] if not top else self.c[0][()]
            N = sum(cnt.values()); n1 = len(cnt)
            if N == 0: return self.uni.p(x, ot)
            return max(cnt.get(x, 0) - self.D, 0) / N + self.D * n1 / N * self.uni.p(x, ot)
        tab = self.c[n] if top else self.cont[n]
        cnt = tab.get(ctx)
        lower = self._p(x, ctx[1:], n - 1, False, ot)
        if not cnt: return lower
        N = sum(cnt.values()); n1 = len(cnt)
        return max(cnt.get(x, 0) - self.D, 0) / N + self.D * n1 / N * lower

# ---------------------------------------------------------------- structural model components
class Frame:
    """slot-state automaton: P(class | state, ot) x P(sign | class, zone). OTHER-class signs (middle elements) come
    from one pan-Indus shared pool (S-DARK-45/53)."""
    @staticmethod
    def zone(h):
        """-> (zone, prevclass, opener_flag, midlen_bucket)"""
        if not h: return ('START', '-', 0, 0)
        op = 1 if h[0] in OPEN else 0
        last = h[-1]; pc = sign_class(last)
        if op and len(h) == 1: return ('OP', pc, 1, 0)
        if op and len(h) == 2 and h[1] in MARK: return ('MK', pc, 1, 0)
        if op and len(h) == 3 and h[1] == 60 and h[2] in MJAR: return ('MK', 'MJAR', 1, 0)
        if last in CLS: return ('CL', pc, op, 0)
        if last in SUF and any(x in CLS for x in h): return ('SUF', pc, op, 0)
        # middle: how many middle signs so far
        start = 1 if op else 0
        if op and len(h) > 1 and h[1] in MARK: start = 2
        midlen = len(h) - start
        return ('MID', pc, op, min(midlen, 4))
    def __init__(self, texts, V, pool_uni):
        self.V = V
        self.cls_tab = collections.defaultdict(collections.Counter)   # (state tuple, ot) -> class counts
        self.sign_tab = collections.defaultdict(collections.Counter)  # (class, zone) -> sign counts
        self.sign_all = collections.defaultdict(collections.Counter)  # class -> sign counts
        for o in texts:
            s = list(o['seq'])
            for i in range(len(s) + 1):
                st = self.zone(s[:i]); x = s[i] if i < len(s) else END
                c = 'END' if x == END else sign_class(x)
                for key in self.keys(st, o['ot']): self.cls_tab[key][c] += 1
                if x != END:
                    self.sign_tab[(c, st[0])][x] += 1; self.sign_all[c][x] += 1
        self.pool = pool_uni   # P(sign) for OTHER class (shared pool, incl. UNK)
        self.classes = sorted(set(c for cnt in self.cls_tab.values() for c in cnt))
    @staticmethod
    def keys(st, ot):
        z, pc, op, ml = st
        return [(z, pc, op, ml, ot), (z, pc, op, ml, '*'), (z, pc, op, '*', '*'), (z, pc, '*', '*', '*'), (z, '*', '*', '*', '*')]
    def pclass(self, st, ot):
        """Witten-Bell backoff along the key chain"""
        p = None
        for key in reversed(self.keys(st, ot)):
            cnt = self.cls_tab.get(key)
            if not cnt: continue
            N = sum(cnt.values()); T = len(cnt)
            if p is None:
                p = {c: cnt[c] / N for c in cnt}; base = p
            else:
                lam = N / (N + T)
                p = {c: lam * cnt.get(c, 0) / N + (1 - lam) * p.get(c, 0.0) for c in set(cnt) | set(p)}
        if p is None: p = {'END': 0.5, 'OTHER': 0.5}
        return p
    def psign(self, x, c, z, ot):
        if c == 'OTHER': return self.pool.p(x, ot) / self.pool_other_mass(ot)
        if c.startswith('C') and c[1:].isdigit(): return 1.0 if ('C' + str(x)) == c else 0.0
        cnt = self.sign_tab.get((c, z)); allc = self.sign_all.get(c)
        if not allc: return 0.0
        Na = sum(allc.values()); Ta = len(allc)
        pa = (allc.get(x, 0) + 0.5) / (Na + 0.5 * Ta)
        if not cnt: return pa
        N = sum(cnt.values()); T = len(cnt); lam = N / (N + T)
        return lam * cnt.get(x, 0) / N + (1 - lam) * pa
    def pool_other_mass(self, ot):
        if not hasattr(self, '_pom'): self._pom = {}
        if ot not in self._pom:
            d = self.pool.dist(ot); self._pom[ot] = sum(p for x, p in d.items() if x != END and sign_class(x) == 'OTHER')
        return self._pom[ot]
    def p(self, x, o, h):
        st = self.zone(h); pc = self.pclass(st, o['ot'])
        if x == END: return pc.get('END', 1e-6)
        c = sign_class(x)
        return pc.get(c, 1e-6) * self.psign(x, c, st[0], o['ot'])

class Rules:
    """frozen pairs / qualifier -> head / numeral -> item: bigram continuations that are rules in the fit set
    (count >= minc, P(b|a) >= minp, O/E >= mine). Inactive (returns None) when the previous sign has no rule."""
    def __init__(self, texts, minc=4, minp=0.12, mine=3.0):
        big = collections.defaultdict(collections.Counter); uni = collections.Counter(); N = 0
        for o in texts:
            s = list(o['seq']) + [END]
            for i in range(len(s)): uni[s[i]] += 1; N += 1
            for a, b in zip(s, s[1:]): big[a][b] += 1
        self.R = {}
        for a, cnt in big.items():
            na = sum(cnt.values()); keep = {}
            for b, c in cnt.items():
                oe = (c / na) / (uni[b] / N)
                if c >= minc and c / na >= minp and oe >= mine: keep[b] = c
            if keep: self.R[a] = keep
        self.npairs = sum(len(v) for v in self.R.values())
    def p(self, x, o, h):
        if not h or h[-1] not in self.R: return None
        keep = self.R[h[-1]]; tot = sum(keep.values())
        return keep.get(x, 0) / tot

class Cache:
    """whole-text reuse: prefix tree over texts of the same site (local) or of all sites (global); adaptive (texts
    scored so far are added, so held-out sites build their own cache). Returns None when the prefix is unseen."""
    def __init__(self, texts, local=True):
        self.local = local; self.tree = collections.defaultdict(collections.Counter)
        for o in texts: self.add(o)
    def key(self, o, h): return ((o['site'] if self.local else '*'), o['ot'], tuple(h))
    def add(self, o):
        s = list(o['seq']) + [END]
        for i in range(len(s)): self.tree[self.key(o, s[:i])][s[i]] += 1
    def p(self, x, o, h):
        cnt = self.tree.get(self.key(o, h))
        if not cnt: return None
        return cnt.get(x, 0) / sum(cnt.values())

ZONES = ['START', 'OP', 'MK', 'MID', 'CL', 'SUF']

class Structural:
    """interpolation of Frame, Rules, local Cache, global Cache and the type Unigram, with weights per zone and per
    activity pattern, fitted by EM on deleted-interpolation folds."""
    names = ['frame', 'rules', 'cache_site', 'cache_all', 'unigram']
    def __init__(self, texts, V, uni, weights=None, use=None):
        self.uni = uni
        self.frame = Frame(texts, V, uni)
        self.rules = Rules(texts)
        self.cs = Cache(texts, local=True); self.cg = Cache(texts, local=False)
        self.use = set(use) if use else set(self.names)
        self.w = weights or {}
    def comps(self, x, o, h):
        out = {}
        if 'frame' in self.use: out['frame'] = self.frame.p(x, o, h)
        if 'rules' in self.use:
            r = self.rules.p(x, o, h)
            if r is not None: out['rules'] = r
        if 'cache_site' in self.use:
            r = self.cs.p(x, o, h)
            if r is not None: out['cache_site'] = r
        if 'cache_all' in self.use:
            r = self.cg.p(x, o, h)
            if r is not None: out['cache_all'] = r
        if 'unigram' in self.use: out['unigram'] = self.uni.p(x, o['ot'])
        return out
    def wkey(self, h, active):
        return (Frame.zone(h)[0], tuple(sorted(active)))
    def p(self, x, o, h):
        c = self.comps(x, o, h); key = self.wkey(h, c)
        w = self.w.get(key) or self.w.get(('*', key[1])) or {k: 1.0 / len(c) for k in c}
        tot = sum(w.get(k, 0.0) for k in c)
        if tot <= 0: w = {k: 1.0 for k in c}; tot = len(c)
        return sum(w.get(k, 0.0) / tot * c[k] for k in c)
    def adapt(self, o):
        self.cs.add(o); self.cg.add(o)

def fit_weights(texts, V, uni_fn, folds=5, iters=25, seed=59, use=None):
    """deleted interpolation: for each fold, fit the components on the rest and collect component probabilities on the
    fold; then EM the mixture weights per (zone, active set). Returns weights dict."""
    rnd = random.Random(seed); idx = list(range(len(texts))); rnd.shuffle(idx)
    obs = collections.defaultdict(list)   # key -> list of dict(name -> p)
    for f in range(folds):
        test = [texts[i] for i in idx[f::folds]]; train = [texts[i] for i in idx if i % folds != f or True]
        train = [texts[i] for j, i in enumerate(idx) if j % folds != f]
        uni = uni_fn(train)
        M = Structural(train, V, uni, use=use)
        for o in test:
            s = list(o['seq']) + [END]
            for i in range(len(s)):
                c = M.comps(s[i], o, s[:i]); obs[M.wkey(s[:i], c)].append(c)
            M.adapt(o)
    W = {}
    # EM per key; keys with few observations share weights at zone '*'
    byact = collections.defaultdict(list)
    for key, L in obs.items(): byact[key[1]] += L
    def em(L, names):
        w = {k: 1.0 / len(names) for k in names}
        for _ in range(iters):
            acc = collections.Counter()
            for c in L:
                tot = sum(w[k] * c[k] for k in c)
                if tot <= 0: continue
                for k in c: acc[k] += w[k] * c[k] / tot
            Z = sum(acc.values())
            if Z: w = {k: (acc[k] + 0.01) / (Z + 0.01 * len(names)) for k in names}
        return w
    for act, L in byact.items(): W[('*', act)] = em(L, list(act))
    for key, L in obs.items():
        if len(L) >= 200: W[key] = em(L, list(key[1]))
    return W, obs

# ---------------------------------------------------------------- evaluation
def evaluate(models, texts, QUAL, adapt=None):
    """models: dict name -> callable p(x, o, h). Returns per-token records: (text_idx, pos, slot, prevclass, zone, sign, {model: bits})."""
    recs = []
    for ti, o in enumerate(texts):
        s = list(o['seq']); lab = parse(s, QUAL) + ['END']
        toks = s + [END]
        for i in range(len(toks)):
            h = s[:i]; z = Frame.zone(h)
            bits = {}
            for name, f in models.items():
                p = f(toks[i], o, h)
                bits[name] = -math.log2(max(p, 1e-12))
            recs.append(dict(t=ti, i=i, slot=lab[i], pc=z[1], zone=z[0], x=toks[i], prev=(h[-1] if h else 'S'), ot=o['ot'], site=o['site'], n=len(s), bits=bits))
        if adapt:
            for a in adapt: a(o)
    return recs

def ce(recs, name, sel=lambda r: True):
    L = [r['bits'][name] for r in recs if sel(r)]
    return (sum(L) / len(L) if L else float('nan')), len(L)

def bootstrap_ce(recs, name, ntexts, reps=1000, seed=1):
    """CI over texts for per-sign cross-entropy"""
    import numpy as np
    rnd = np.random.default_rng(seed)
    tb = np.zeros(ntexts); tn = np.zeros(ntexts)
    for r in recs: tb[r['t']] += r['bits'][name]; tn[r['t']] += 1
    out = []
    for _ in range(reps):
        ix = rnd.integers(0, ntexts, ntexts); out.append(tb[ix].sum() / tn[ix].sum())
    out.sort()
    return out[int(0.025 * reps)], out[int(0.975 * reps)]
