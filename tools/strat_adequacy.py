"""S366: a Turing test for the credential grammar.

Fit the best generative model we have of a seal text, a slot grammar
    P(opener) x P(connective | opener) x P(middle: Markov chain of order k over middle signs)
    x P(closer | ...) x P(qualifiers | closer) x P(suffix | object type),
with every table estimated from Mohenjo-daro + Harappa texts (the parse of tools/parse_all.py, S310).
Generate 100 synthetic corpora of the same size and object mix and compare the real corpus with them on a
battery of ~45 statistics that the model was NOT fitted on directly. Statistics where the real corpus falls
outside the synthetic range are the structure the model is missing. Then add one mechanism at a time
(object-type effect, site effect, middle lexicon with reuse, closer <- middle dependency, frame package,
order-2 middle chain, whole-text reuse, open sign inventory) and report which gaps close. Also: held-out sites (model fitted on
MD+H), and controls (a no-slot Markov chain as generator; a Markov no-slot corpus and a planted syllabic
corpus as the 'real' target).

Usage: python3 tools/strat_adequacy.py [--n 100] [--seqkey seq] [--quick]
Output: data/derived/strat_adequacy.txt and .json
"""
import json, math, random, collections, sys, time, os, statistics
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

ARGS = sys.argv[1:]
def arg(name, default):
    if name in ARGS:
        return ARGS[ARGS.index(name) + 1]
    return default
NSYN = int(arg('--n', 100)); SEQKEY = arg('--seqkey', 'seq'); QUICK = '--quick' in ARGS
if QUICK: NSYN = min(NSYN, 20)

C = json.load(open(os.path.join(HERE, 'data/derived/merged-corpus-canonical.json')))
FIT_SITES = {'Mohenjo-daro', 'Harappa'}

def cls_of(t):
    ty = t['type']
    return 'SEAL' if ty.startswith('SEAL') else 'TAB' if ty.startswith('TAB') else 'OTHER'

ALL = [(t['site'], cls_of(t), tuple(t[SEQKEY])) for t in C if t[SEQKEY]]
FIT = [x for x in ALL if x[0] in FIT_SITES]
HELD = [x for x in ALL if x[0] not in FIT_SITES and x[0] != 'Unknown']

# ------------------------------------------------------------------ parser (S310, tools/parse_all.py)
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]; CLS = set(CL)
FISH = {235, 240, 233, 231, 220}; FISHQ = [235, 240, 233, 231]
NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}

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

QUAL = learn_qual([s for _, _, s in FIT])

def parse(s):
    lab = ['NAME'] * len(s); i = 0; j = len(s)
    if s[0] in OPEN:
        lab[0] = 'OPENER'; i = 1
        if len(s) > 1 and s[1] in MARK: lab[1] = 'MARKER'; i = 2
    while j - 1 > i and s[j - 1] in SUF and j >= 2 and (s[j - 2] in CLS or s[j - 2] in SUF): lab[j - 1] = 'SUFFIX'; j -= 1
    if j - 1 >= i and s[j - 1] in CLS:
        c = s[j - 1]; lab[j - 1] = 'CLOSER'; j -= 1
        if c == 520:
            if j - 2 >= i and s[j - 1] == 33 and s[j - 2] in (705, 706): lab[j - 1] = lab[j - 2] = 'TITLE'; j -= 2
            while j - 1 >= i and s[j - 1] in FISH: lab[j - 1] = 'TITLE'; j -= 1
        elif c == 740:
            if j - 1 >= i and s[j - 1] == 100: lab[j - 1] = 'TITLE'; j -= 1
            if j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        elif j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        if j - 1 >= i and s[j - 1] in NUM and lab[j] == 'TITLE': lab[j - 1] = 'TITLE'; j -= 1
    for k in range(i, j - 1):
        if s[k] in NUM and lab[k] == 'NAME' and lab[k + 1] == 'NAME': lab[k] = lab[k + 1] = 'COUNT'
    for k in range(i, j):
        if s[k] in NUM and lab[k] == 'NAME': lab[k] = 'COUNT'
    return lab

def decompose(s):
    """-> dict(opener, marker, middle(tuple), title(tuple), closer, suffix(tuple))"""
    L = parse(s)
    d = {'opener': None, 'marker': None, 'middle': [], 'title': [], 'closer': None, 'suffix': []}
    for x, l in zip(s, L):
        if l == 'OPENER': d['opener'] = x
        elif l == 'MARKER': d['marker'] = x
        elif l in ('NAME', 'COUNT'): d['middle'].append(x)
        elif l == 'TITLE': d['title'].append(x)
        elif l == 'CLOSER': d['closer'] = x
        elif l == 'SUFFIX': d['suffix'].append(x)
    d['middle'] = tuple(d['middle']); d['title'] = tuple(d['title']); d['suffix'] = tuple(d['suffix'])
    return d

# ------------------------------------------------------------------ helpers
class Cat:
    """categorical with optional backoff to a parent Cat (shrinkage lam)"""
    def __init__(self, counts, parent=None, lam=0.0):
        self.c = collections.Counter(counts); self.n = sum(self.c.values()); self.parent = parent; self.lam = lam
        self.keys = None
    def sample(self, rng):
        if self.parent is not None and self.n + self.lam > 0 and rng.random() < self.lam / (self.n + self.lam):
            return self.parent.sample(rng)
        if self.n == 0: return self.parent.sample(rng)
        if self.keys is None:
            self.keys = list(self.c.keys()); self.w = [self.c[k] for k in self.keys]
        return rng.choices(self.keys, self.w)[0]

class Markov:
    """order-k chain over middle signs with START/END, absolute discounting, interpolated with lower orders.
    Optional parent chain (pooled) for shrinkage of sparse (per-class / per-site) tables."""
    START = 'S'; END = 'E'; NEW = 'NEW'
    def __init__(self, seqs, k=1, D=0.5, parent=None, lam=5.0, open_inv=False):
        self.k = k; self.D = D; self.parent = parent; self.lam = lam; self.open_inv = open_inv; self.fresh = 0
        self.ctx = [collections.defaultdict(collections.Counter) for _ in range(k + 1)]
        for s in seqs:
            s = [Markov.START] * k + list(s) + [Markov.END]
            for i in range(k, len(s)):
                for o in range(0, k + 1):
                    self.ctx[o][tuple(s[i - o:i])][s[i]] += 1
        self.uni = self.ctx[0][()]
        # Kneser-Ney style continuation counts for the unigram backoff (number of distinct left contexts),
        # so that rare signs are not starved by the discount mass flowing to frequent signs
        self.cont = collections.Counter()
        for h, cnt in self.ctx[1].items():
            for v in cnt: self.cont[v] += 1
        self.vocab = [v for v in self.uni if v != Markov.START]
        # open inventory: Good-Turing mass for signs never seen = number of hapax signs / tokens
        self.n1 = sum(1 for v, c in self.uni.items() if c == 1 and v not in (Markov.START, Markov.END))
        if open_inv: self.vocab.append(Markov.NEW)
        self.V = len(self.vocab)
        self._cache = {}
    def dist(self, hist):
        hist = tuple(hist[-self.k:]) if self.k else ()
        if hist in self._cache: return self._cache[hist]
        # base: smoothed unigram
        tot = sum(self.cont.values())
        p = {v: (self.cont[v] + 0.5) / (tot + 0.5 * self.V) for v in self.vocab}
        if self.open_inv:
            z = sum(p.values()); share = self.n1 / max(1, sum(self.uni.values()))
            p = {v: q / z * (1 - share) for v, q in p.items()}; p[Markov.NEW] = share
        for o in range(1, self.k + 1):
            h = hist[len(hist) - o:] if o <= len(hist) else None
            if h is None: break
            cnt = self.ctx[o].get(h)
            if not cnt: continue
            n = sum(cnt.values()); n1 = len(cnt)
            q = {}
            back = self.D * n1 / n
            for v in self.vocab:
                q[v] = max(cnt.get(v, 0) - self.D, 0) / n + back * p[v]
            p = q
        if self.parent is not None:
            pk, pw = self.parent.dist(hist); pp = dict(zip(pk, pw))
            n = sum(self.ctx[self.k].get(hist, {}).values()) if self.k else sum(self.uni.values())
            w = n / (n + self.lam)
            p = {v: w * p.get(v, 0) + (1 - w) * pp.get(v, 0) for v in set(p) | set(pp)}
        keys = list(p.keys()); wts = [p[v] for v in keys]
        self._cache[hist] = (keys, wts)
        return self._cache[hist]
    def generate(self, rng, maxlen=14):
        out = []; hist = [Markov.START] * self.k
        while len(out) < maxlen:
            keys, wts = self.dist(hist)
            v = rng.choices(keys, wts)[0]
            if v == Markov.END: break
            if v == Markov.NEW:
                self.fresh += 1; out.append(-self.fresh); hist = (hist + [v])[-max(self.k, 1):]; continue
            out.append(v); hist = (hist + [v])[-max(self.k, 1):]
        return tuple(out)
    def p_empty(self):
        keys, wts = self.dist([Markov.START] * self.k)
        return dict(zip(keys, wts)).get(Markov.END, 0.0)

class CRP:
    """reuse an already-drawn item with prob n/(n+theta), else draw a new one"""
    def __init__(self, theta): self.theta = theta; self.items = []; self.n = 0
    def draw(self, rng, new):
        if self.n > 0 and rng.random() < self.n / (self.n + self.theta):
            x = rng.choice(self.items)
        else:
            x = new()
        self.items.append(x); self.n += 1
        return x

def fit_theta(n, k):
    """theta with E[#distinct among n CRP draws] = k"""
    lo, hi = 1e-3, 1e6
    for _ in range(60):
        th = math.sqrt(lo * hi)
        ek = sum(th / (th + i) for i in range(n))
        if ek < k: lo = th
        else: hi = th
    return math.sqrt(lo * hi)

# ------------------------------------------------------------------ the generative model
class SlotModel:
    def __init__(self, data, mech=(), k=1):
        """data: list of (site, cls, seq). mech: subset of
        'type' (tables | object class), 'site' (tables | site), 'lexicon' (middle reuse CRP),
        'closerdep' (closer | last middle sign), 'package' (joint opener+marker+closer+suffix),
        'markov2' (order-2 middle chain), 'textreuse' (whole-text CRP)"""
        self.mech = set(mech); self.k = 2 if 'markov2' in self.mech else k
        self.D = [(site, cls, s, decompose(s)) for site, cls, s in data]
        groups = collections.defaultdict(list)
        for site, cls, s, d in self.D:
            groups[self.key(site, cls)].append(d)
        self.pooled = self.fit_tables([d for *_, d in self.D], None)
        self.tab = {g: self.fit_tables(ds, self.pooled) for g, ds in groups.items()} if len(groups) > 1 else {}
        # lexicon / text reuse
        n_mid = [d['middle'] for *_, d in self.D if d['middle']]
        self.theta_mid = fit_theta(len(n_mid), len(set(n_mid)))
        texts = [s for _, _, s, _ in self.D]
        self.theta_text = fit_theta(len(texts), len(set(texts)))
        if 'textreuse' in self.mech:
            # stock texts are reused within a city (S17/S76), so one restaurant per site; theta calibrated by
            # simulation so that the generated share of distinct texts matches the fit set (collisions of
            # independently generated short texts already produce repeats, so the closed-form fit overshoots)
            target = len(set(texts)) / len(texts); meta = [(site, cls) for site, cls, _, _ in self.D]
            lo, hi = 10.0, 1e5; rng = random.Random(99)
            for _ in range(12):
                self.theta_text = math.sqrt(lo * hi)
                g = self.generate_corpus(meta, rng); u = len(set(x[2] for x in g)) / len(g)
                if u < target: lo = self.theta_text
                else: hi = self.theta_text
            self.theta_text = math.sqrt(lo * hi)
    def key(self, site, cls):
        return ((site if 'site' in self.mech else None), (cls if 'type' in self.mech else None))
    def fit_tables(self, ds, parent):
        T = {}
        P = parent or {}
        lam = 5.0 if parent else 0.0
        T['opener'] = Cat([d['opener'] for d in ds], P.get('opener'), lam)
        mk = collections.defaultdict(list)
        for d in ds: mk[d['opener']].append(d['marker'])
        T['marker'] = {o: Cat(v, P.get('marker', {}).get(o) if parent else None, lam) for o, v in mk.items()}
        T['middle'] = Markov([d['middle'] for d in ds], k=self.k, parent=P.get('middle'), lam=lam, open_inv='open' in self.mech)
        cl = collections.defaultdict(list)
        for d in ds: cl[d['opener'] is not None].append(d['closer'])
        T['closer'] = {o: Cat(v, P.get('closer', {}).get(o) if parent else None, lam) for o, v in cl.items()}
        cd = collections.defaultdict(list)
        for d in ds: cd[d['middle'][-1] if d['middle'] else 'S'].append(d['closer'])
        T['closer_all'] = Cat([d['closer'] for d in ds], P.get('closer_all'), lam)
        T['closer_last'] = {a: Cat(v, P['closer_last'].get(a, P['closer_all']) if parent else T['closer_all'], 3.0) for a, v in cd.items()}
        ti = collections.defaultdict(list)
        for d in ds:
            if d['closer'] is not None: ti[d['closer']].append(d['title'])
        T['title'] = {c: Cat(v, P.get('title', {}).get(c) if parent else None, lam) for c, v in ti.items()}
        su = collections.defaultdict(list)
        for d in ds:
            if d['closer'] is not None: su[True].append(d['suffix'])
        T['suffix'] = Cat(su[True], P.get('suffix'), lam)
        T['package'] = Cat([(d['opener'], d['marker'], d['closer'], d['suffix']) for d in ds], P.get('package'), lam)
        return T
    def tables(self, site, cls):
        if not self.tab: return self.pooled
        return self.tab.get(self.key(site, cls), self.pooled)
    def generate_corpus(self, meta, rng):
        """meta: list of (site, cls) -> list of (site, cls, seq)"""
        out = []
        mid_crp = CRP(self.theta_mid); text_crps = collections.defaultdict(lambda: CRP(self.theta_text))
        for site, cls in meta:
            T = self.tables(site, cls); text_crp = text_crps[site]
            def one():
                for _ in range(50):
                    s = self.generate_text(T, rng, mid_crp)
                    if s: return s
                return (740,)
            s = text_crp.draw(rng, one) if 'textreuse' in self.mech else one()
            out.append((site, cls, s))
        return out
    def generate_text(self, T, rng, mid_crp):
        if 'package' in self.mech:
            op, mk, cl, suf = T['package'].sample(rng)
        else:
            op = T['opener'].sample(rng)
            mk = T['marker'][op].sample(rng) if op in T['marker'] else None
            cl = suf = None
        # middle
        M = T['middle']
        if 'lexicon' in self.mech:
            if rng.random() < M.p_empty(): mid = ()
            else:
                def newmid():
                    for _ in range(50):
                        m = M.generate(rng)
                        if m: return m
                    return M.generate(rng)
                mid = mid_crp.draw(rng, newmid)
        else:
            mid = M.generate(rng)
        if 'package' not in self.mech:
            if 'closerdep' in self.mech:
                a = mid[-1] if mid else 'S'
                cat = T['closer_last'].get(a) or T['closer'].get(op is not None) or T['closer'][False]
            else:
                cat = T['closer'].get(op is not None) or T['closer'][False]
            cl = cat.sample(rng)
        ti = ()
        if cl is not None:
            ti = T['title'][cl].sample(rng) if cl in T['title'] else ()
            if 'package' not in self.mech:
                suf = T['suffix'].sample(rng)
        else:
            suf = ()
        s = []
        if op is not None: s.append(op)
        if mk is not None: s.append(mk)
        s += list(mid) + list(ti)
        if cl is not None: s.append(cl)
        s += list(suf or ())
        return tuple(s)

class PlainMarkov:
    """control generator: a no-slot chain of order k over whole texts, lengths emerge from END"""
    def __init__(self, data, k=2):
        self.M = Markov([s for _, _, s in data], k=k)
    def generate_corpus(self, meta, rng):
        out = []
        for site, cls in meta:
            s = ()
            while not s: s = self.M.generate(rng, maxlen=17)
            out.append((site, cls, s))
        return out

# ------------------------------------------------------------------ the battery
def entropy(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c) if n else float('nan')

def mi_pairs(pairs):
    if not pairs: return float('nan')
    ja = collections.Counter(pairs); a = collections.Counter(x for x, _ in pairs); b = collections.Counter(y for _, y in pairs)
    n = len(pairs)
    return sum(c / n * math.log2(c * n / (a[x] * b[y])) for (x, y), c in ja.items())

def zipf_slope(counter, top=50):
    f = sorted(counter.values(), reverse=True)[:top]
    if len(f) < 5: return float('nan')
    xs = [math.log(r + 1) for r in range(len(f))]; ys = [math.log(v) for v in f]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)

def battery(data):
    """data: list of (site, cls, seq). Returns an ordered dict of statistics."""
    S = collections.OrderedDict()
    texts = [s for _, _, s in data]; n = len(texts)
    lens = [len(s) for s in texts]
    S['len_mean'] = statistics.mean(lens)
    S['len_sd'] = statistics.pstdev(lens)
    S['len_ge6_share'] = sum(l >= 6 for l in lens) / n
    S['len1_share'] = sum(l == 1 for l in lens) / n
    S['len_max'] = max(lens)
    distinct = collections.Counter(texts)
    S['unique_text_share'] = len(distinct) / n
    S['texts_rep3'] = sum(1 for t, c in distinct.items() if len(t) >= 3 and c >= 3)
    # nesting
    D = set(distinct)
    subs = set()
    for t in D:
        if len(t) >= 4:
            for m in (3, 4, 5):
                if m < len(t):
                    for i in range(len(t) - m + 1): subs.add(t[i:i + m])
    small = [t for t in D if 3 <= len(t) <= 5]
    S['nesting_rate'] = sum(t in subs for t in small) / max(1, len(small))
    # cross-site sharing of 3-runs and texts
    run_sites = collections.defaultdict(set); text_sites = collections.defaultdict(set)
    for site, _, s in data:
        if len(s) >= 3: text_sites[s].add(site)
        for i in range(len(s) - 2): run_sites[s[i:i + 3]].add(site)
    S['run3_multisite_share'] = sum(len(v) >= 2 for v in run_sites.values()) / max(1, len(run_sites))
    S['text3_multisite_share'] = sum(len(v) >= 2 for v in text_sites.values()) / max(1, len(text_sites))
    # MI at distance 1-4
    for d in (1, 2, 3, 4):
        S[f'mi_d{d}'] = mi_pairs([(s[i], s[i + d]) for s in texts for i in range(len(s) - d)])
    # repetition
    S['repeat_rate'] = sum(len(set(s)) < len(s) for s in texts if len(s) >= 3) / max(1, sum(l >= 3 for l in lens))
    S['adj_double_rate'] = sum(any(s[i] == s[i + 1] for i in range(len(s) - 1)) for s in texts) / n
    # fish qualifier order
    ok = tot = 0
    rank = {x: i for i, x in enumerate(FISHQ)}
    for s in texts:
        pos = [(i, rank[x]) for i, x in enumerate(s) if x in rank]
        for a in range(len(pos)):
            for b in range(a + 1, len(pos)):
                if pos[a][1] != pos[b][1]:
                    tot += 1; ok += pos[a][1] < pos[b][1]
    S['fish_order_consistency'] = ok / tot if tot >= 5 else float('nan')
    # long-range co-occurrence (distance >= 2) vs independence
    has = collections.Counter(); both = collections.Counter()
    for s in texts:
        st = set(s)
        for x in st: has[x] += 1
        seen = set()
        for i in range(len(s)):
            for j in range(i + 2, len(s)):
                if s[i] != s[j]:
                    p = (s[i], s[j]) if s[i] < s[j] else (s[j], s[i])
                    seen.add(p)
        for p in seen: both[p] += 1
    attract = avoid = 0
    for (x, y), c in both.items():
        e = has[x] * has[y] / n
        if c >= 5 and c / e >= 3: attract += 1
    # avoidance: pairs of common signs expected >= 5 times but seen <= 1/3
    common = [x for x, c in has.items() if c >= 40]
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            x, y = common[i], common[j]; p = (x, y) if x < y else (y, x)
            e = has[x] * has[y] / n
            if e >= 5 and both.get(p, 0) <= e / 3: avoid += 1
    S['longrange_attract_pairs'] = attract
    S['longrange_avoid_pairs'] = avoid
    # parse-based: middle-initial / middle-final sets, same middle under different closer
    mi = collections.Counter(); mf = collections.Counter(); midclos = collections.defaultdict(set)
    midlen_c = []; midlen_nc = []; op_cl = []; op_suf = [0, 0]; closer_nonfinal = 0
    for s in texts:
        d = decompose(s)
        if d['middle']:
            mi[d['middle'][0]] += 1; mf[d['middle'][-1]] += 1
            if len(d['middle']) >= 2: midclos[d['middle']].add(d['closer'])
        (midlen_c if d['closer'] is not None else midlen_nc).append(len(d['middle']))
        op_cl.append((d['opener'], d['closer']))
        if d['opener'] is not None:
            op_suf[0] += 1; op_suf[1] += bool(d['suffix'])
        body = list(s)
        while len(body) > 1 and body[-1] in SUF: body.pop()
        if any(x in CLS for x in body[:-1]): closer_nonfinal += 1
    A = {x for x, c in mi.items() if c >= 3}; B = {x for x, c in mf.items() if c >= 3}
    S['mid_init_final_jaccard'] = len(A & B) / max(1, len(A | B))
    S['mid_init_top10_share'] = sum(c for _, c in mi.most_common(10)) / max(1, sum(mi.values()))
    S['mid_final_top10_share'] = sum(c for _, c in mf.most_common(10)) / max(1, sum(mf.values()))
    rep = [v for m, v in midclos.items()]
    S['same_middle_diff_closer'] = float('nan')
    cnt_mid = collections.Counter()
    for s in texts:
        d = decompose(s)
        if len(d['middle']) >= 2: cnt_mid[d['middle']] += 1
    reps = [m for m, c in cnt_mid.items() if c >= 2]
    if len(reps) >= 5: S['same_middle_diff_closer'] = sum(len(midclos[m]) >= 2 for m in reps) / len(reps)
    S['mid_len_closer_diff'] = (statistics.mean(midlen_c) if midlen_c else 0) - (statistics.mean(midlen_nc) if midlen_nc else 0)
    S['opener_closer_mi'] = mi_pairs(op_cl)
    S['opener_suffix_share'] = op_suf[1] / op_suf[0] if op_suf[0] else float('nan')
    S['closer_sign_nonfinal_share'] = closer_nonfinal / n
    # numerals
    after = collections.defaultdict(collections.Counter); num_tok = num_init = 0; two_num = 0
    for s in texts:
        k = sum(x in NUM for x in s); two_num += k >= 2
        for i, x in enumerate(s):
            if x in NUM:
                num_tok += 1; num_init += (i == 0)
                if i + 1 < len(s): after[s[i + 1]][x] += 1
    S['numeral_attach_types'] = sum(1 for v in after.values() if sum(v.values()) >= 2)
    fixed = [v for v in after.values() if sum(v.values()) >= 5]
    S['fixed_numeral_share'] = sum(v.most_common(1)[0][1] / sum(v.values()) >= 0.8 for v in fixed) / len(fixed) if fixed else float('nan')
    S['numeral_initial_share'] = num_init / num_tok if num_tok else float('nan')
    S['two_numerals_share'] = two_num / n
    # positional entropies (texts of >= 3 signs)
    T3 = [s for s in texts if len(s) >= 3]
    for p in (1, 2, 3):
        S[f'H_start{p}'] = entropy(collections.Counter(s[p - 1] for s in T3))
        S[f'H_end{p}'] = entropy(collections.Counter(s[-p] for s in T3))
    # Zipf slopes
    S['zipf_all'] = zipf_slope(collections.Counter(x for s in texts for x in s))
    S['zipf_initial'] = zipf_slope(collections.Counter(s[0] for s in texts if len(s) >= 2))
    S['zipf_final'] = zipf_slope(collections.Counter(s[-1] for s in texts if len(s) >= 2))
    S['zipf_middle'] = zipf_slope(mi + mf)
    # inventory
    uni = collections.Counter(x for s in texts for x in s)
    S['n_sign_types'] = len(uni)
    S['hapax_sign_share'] = sum(c == 1 for c in uni.values()) / len(uni)
    bg = collections.Counter(s[i:i + 2] for s in texts for i in range(len(s) - 1))
    tg = collections.Counter(s[i:i + 3] for s in texts for i in range(len(s) - 2))
    S['bigram_hapax_share'] = sum(c == 1 for c in bg.values()) / max(1, len(bg))
    S['trigram_hapax_share'] = sum(c == 1 for c in tg.values()) / max(1, len(tg))
    r4 = collections.Counter(s[i:i + 4] for s in texts for i in range(len(s) - 3))
    S['run4_rep3'] = sum(c >= 3 for c in r4.values())
    # bigram conditional entropy H(next | prev) incl. END
    ctx = collections.defaultdict(collections.Counter)
    for s in texts:
        ss = list(s) + ['E']
        for i in range(len(ss) - 1): ctx[ss[i]][ss[i + 1]] += 1
    tot = sum(sum(v.values()) for v in ctx.values())
    S['H_cond_bigram'] = sum(sum(v.values()) / tot * entropy(v) for v in ctx.values())
    # object classes
    by = collections.defaultdict(list)
    for _, cls, s in data: by[cls].append(len(s))
    S['seal_minus_tab_len'] = (statistics.mean(by['SEAL']) if by['SEAL'] else 0) - (statistics.mean(by['TAB']) if by['TAB'] else 0)
    S['seal_jar_final_share'] = sum(s[-1] == 740 for _, c, s in data if c == 'SEAL') / max(1, len(by['SEAL']))
    S['tab_suffix_share'] = sum(s[-1] in SUF for _, c, s in data if c == 'TAB') / max(1, len(by['TAB']))
    return S

def compare(real_stats, syn_list):
    rows = []
    for k, rv in real_stats.items():
        vals = [s[k] for s in syn_list if not (isinstance(s[k], float) and math.isnan(s[k]))]
        if not vals or (isinstance(rv, float) and math.isnan(rv)):
            rows.append(dict(stat=k, real=rv, mean=float('nan'), sd=float('nan'), z=float('nan'), lo=float('nan'), hi=float('nan'), outside=False)); continue
        m = statistics.mean(vals); sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        z = (rv - m) / sd if sd > 1e-12 else (0.0 if abs(rv - m) < 1e-9 else math.copysign(99.0, rv - m))
        rows.append(dict(stat=k, real=rv, mean=m, sd=sd, z=z, lo=min(vals), hi=max(vals), outside=bool(rv < min(vals) or rv > max(vals))))
    return rows

def run_model(model, meta, real_stats, nsyn, seed=1):
    rng = random.Random(seed); syn = []
    for i in range(nsyn):
        syn.append(battery(model.generate_corpus(meta, rng)))
    return compare(real_stats, syn)

def summary(rows):
    out = sum(r['outside'] for r in rows); n = sum(not math.isnan(r['z']) for r in rows)
    big = sum(abs(r['z']) > 3 for r in rows if not math.isnan(r['z']))
    sumz = sum(min(abs(r['z']), 20) for r in rows if not math.isnan(r['z']))
    return dict(outside=out, n=n, z_gt3=big, sum_abs_z_capped=sumz)

def fmt_rows(rows, top=None):
    rs = sorted([r for r in rows if not math.isnan(r['z'])], key=lambda r: -abs(r['z']))
    if top: rs = rs[:top]
    lines = [f"{'statistic':28s} {'real':>9s} {'syn mean':>9s} {'sd':>8s} {'z':>7s}  {'range':>19s} out"]
    for r in rs:
        lines.append(f"{r['stat']:28s} {r['real']:9.3f} {r['mean']:9.3f} {r['sd']:8.3f} {r['z']:7.2f}  [{r['lo']:8.3f},{r['hi']:8.3f}] {'*' if r['outside'] else ''}")
    for r in rows:
        if math.isnan(r['z']): lines.append(f"{r['stat']:28s} {str(r['real']):>9s}  (not computable)")
    return '\n'.join(lines)

# ------------------------------------------------------------------ main
def main():
    t0 = time.time(); OUT = []; J = {'seqkey': SEQKEY, 'nsyn': NSYN}
    def log(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)
    log(f"# strat_adequacy: Turing test for the credential grammar ({time.strftime('%Y-%m-%d')}; seqkey={SEQKEY}, {NSYN} synthetic corpora per model)")
    log(f"fit set (Mohenjo-daro + Harappa): {len(FIT)} texts; held-out sites: {len(HELD)} texts from {len(set(s for s,_,_ in HELD))} sites")
    meta = [(s, c) for s, c, _ in FIT]
    real = battery(FIT)
    log(f"battery: {len(real)} statistics")

    # baseline
    log("\n## 1. Baseline slot grammar M0: opener | marker|opener | middle (order-1 chain) | closer|opener? | qualifiers|closer | suffix")
    M0 = SlotModel(FIT); rows0 = run_model(M0, meta, real, NSYN)
    s0 = summary(rows0); log(f"M0: {s0['outside']} of {s0['n']} statistics outside the synthetic range; {s0['z_gt3']} with |z|>3; sum|z| (capped 20) = {s0['sum_abs_z_capped']:.1f}")
    log(fmt_rows(rows0))
    J['M0'] = rows0
    missing = [r['stat'] for r in sorted(rows0, key=lambda r: -abs(r['z']) if not math.isnan(r['z']) else 0) if r['outside']]
    log("\nMISSING STRUCTURE (outside range), ranked by |z|: " + ', '.join(missing))

    # one mechanism at a time
    log("\n## 2. One mechanism at a time (added to M0)")
    MECHS = ['type', 'site', 'lexicon', 'closerdep', 'package', 'markov2', 'textreuse', 'open']
    DESC = {'type': 'all tables conditioned on object class (seal / tablet / other), shrunk to pooled',
            'site': 'all tables conditioned on site (Mohenjo-daro vs Harappa), shrunk to pooled',
            'lexicon': f'middle lexicon with reuse: Chinese-restaurant reuse of middles (theta fitted to the distinct-middle count = {M0.theta_mid:.1f})',
            'closerdep': 'closer chosen given the last middle sign (shrunk to the marginal)',
            'package': 'frame package: opener + marker + closer + suffix drawn jointly',
            'markov2': 'order-2 chain for the middle (interpolated)',
            'open': 'open sign inventory: the middle chain emits a never-seen sign with the Good-Turing mass (hapax signs / tokens, S309)',
            'textreuse': f'whole-text reuse (stock texts): one CRP per site, theta calibrated by simulation to the distinct-text share = {SlotModel(FIT, mech=["textreuse"]).theta_text:.0f} (closed-form fit would give {M0.theta_text:.0f}; fits unique_text_share directly)'}
    single = {}
    for m in MECHS:
        Mm = SlotModel(FIT, mech=[m]); rows = run_model(Mm, meta, real, NSYN)
        s = summary(rows); single[m] = rows; J['single_' + m] = rows
        fixed = [r['stat'] for r in rows if not r['outside'] and next(x for x in rows0 if x['stat'] == r['stat'])['outside']]
        broke = [r['stat'] for r in rows if r['outside'] and not next(x for x in rows0 if x['stat'] == r['stat'])['outside']]
        log(f"+{m:10s} ({DESC[m]}): outside {s['outside']}/{s['n']}, |z|>3: {s['z_gt3']}, sum|z| {s['sum_abs_z_capped']:.1f}")
        log(f"    brings inside range: {', '.join(fixed) or '-'}")
        log(f"    pushes outside range: {', '.join(broke) or '-'}")
        log(f"    biggest residual gaps: " + '; '.join(f"{r['stat']} z={r['z']:.1f}" for r in sorted([r for r in rows if not math.isnan(r['z'])], key=lambda r: -abs(r['z']))[:5]))

    # greedy forward selection on sum|z|
    log("\n## 3. Greedy combination (add the mechanism that lowers sum|z| most, until no gain)")
    cur = []; cur_rows = rows0; cur_score = s0['sum_abs_z_capped']; path = []
    while True:
        best = None
        for m in MECHS:
            if m in cur: continue
            rows = run_model(SlotModel(FIT, mech=cur + [m]), meta, real, max(10, NSYN // 2))
            sc = summary(rows)['sum_abs_z_capped']
            log(f"   try {'+'.join(cur + [m]):45s} sum|z| {sc:.1f}  outside {summary(rows)['outside']}")
            if best is None or sc < best[0]: best = (sc, m, rows)
        if best is None or best[0] >= cur_score - 1.0: break
        cur.append(best[1]); cur_rows = best[2]; cur_score = best[0]; path.append((best[1], best[0]))
    log(f"best combination: {'+'.join(cur) or 'M0'}; path: " + ' -> '.join(f"{m} ({sc:.1f})" for m, sc in path))
    cur_rows = run_model(SlotModel(FIT, mech=cur), meta, real, NSYN, seed=2)
    sb = summary(cur_rows); log(f"M* ({'+'.join(cur) or 'M0'}): outside {sb['outside']}/{sb['n']}, |z|>3: {sb['z_gt3']}, sum|z| {sb['sum_abs_z_capped']:.1f}")
    log(fmt_rows(cur_rows))
    J['best'] = {'mech': cur, 'rows': cur_rows}
    # mechanism per gap: for each M0-outside stat, which single mechanism closes it (brings inside range or |z| < 2)
    log("\n### What fixed each gap (single mechanisms that bring the statistic inside the synthetic range; in brackets: best |z|)")
    gapfix = {}
    for r in sorted(rows0, key=lambda r: -abs(r['z']) if not math.isnan(r['z']) else 0):
        if not r['outside']: continue
        fx = []
        for m in MECHS:
            rr = next(x for x in single[m] if x['stat'] == r['stat'])
            if not rr['outside']: fx.append(f"{m} (z={rr['z']:.1f})")
        rb = next(x for x in cur_rows if x['stat'] == r['stat'])
        gapfix[r['stat']] = {'M0_z': r['z'], 'fixed_by': fx, 'best_z': rb['z'], 'best_outside': rb['outside']}
        log(f"  {r['stat']:28s} M0 z={r['z']:6.1f}  fixed by: {', '.join(fx) or 'none'}  | M* z={rb['z']:.1f}{' still outside' if rb['outside'] else ''}")
    J['gapfix'] = gapfix
    unexplained = [k for k, v in gapfix.items() if v['best_outside']]
    log("STILL OUTSIDE RANGE under M*: " + (', '.join(unexplained) or 'none'))

    # held-out sites
    log("\n## 4. Held-out sites (model fitted on Mohenjo-daro + Harappa, synthetic corpora with the held-out site/object mix)")
    meta_h = [(s, c) for s, c, _ in HELD]; real_h = battery(HELD)
    for name, mech in (('M0', []), ('M*', cur)):
        rows = run_model(SlotModel(FIT, mech=mech), meta_h, real_h, NSYN, seed=7)
        s = summary(rows); J['heldout_' + name] = rows
        log(f"{name} on held-out: outside {s['outside']}/{s['n']}, |z|>3: {s['z_gt3']}, sum|z| {s['sum_abs_z_capped']:.1f}")
        log("  top gaps: " + '; '.join(f"{r['stat']} real={r['real']:.3f} syn={r['mean']:.3f} z={r['z']:.1f}" for r in sorted([r for r in rows if not math.isnan(r['z'])], key=lambda r: -abs(r['z']))[:8]))

    # controls
    log("\n## 5. Controls")
    log("### 5a. No-slot generator: an order-2 Markov chain over whole texts (no frame), same corpus size; does the battery tell it from the real corpus?")
    rowsP = run_model(PlainMarkov(FIT, k=2), meta, real, NSYN, seed=3); sP = summary(rowsP); J['control_plainmarkov2'] = rowsP
    log(f"plain order-2 chain: outside {sP['outside']}/{sP['n']}, |z|>3: {sP['z_gt3']}, sum|z| {sP['sum_abs_z_capped']:.1f} (M0: {s0['outside']}, {s0['z_gt3']}, {s0['sum_abs_z_capped']:.1f}; M*: {sb['outside']}, {sb['z_gt3']}, {sb['sum_abs_z_capped']:.1f})")
    log(fmt_rows(rowsP, top=12))
    rowsP1 = run_model(PlainMarkov(FIT, k=1), meta, real, NSYN, seed=3); sP1 = summary(rowsP1); J['control_plainmarkov1'] = rowsP1
    log(f"plain order-1 chain: outside {sP1['outside']}/{sP1['n']}, |z|>3: {sP1['z_gt3']}, sum|z| {sP1['sum_abs_z_capped']:.1f}")

    log("\n### 5b. Language-like planted target: a corpus drawn from the no-slot order-2 chain is treated as 'real'; the slot grammar M0 is fitted to it and tested with the same battery")
    rng = random.Random(11); planted = PlainMarkov(FIT, k=2).generate_corpus(meta, rng)
    realp = battery(planted); Mp = SlotModel(planted); rowsp = run_model(Mp, meta, realp, NSYN, seed=5); sp = summary(rowsp); J['control_planted_markov_target'] = rowsp
    log(f"slot grammar fitted to a no-slot Markov corpus: outside {sp['outside']}/{sp['n']}, |z|>3: {sp['z_gt3']}, sum|z| {sp['sum_abs_z_capped']:.1f}")
    log("  top gaps: " + '; '.join(f"{r['stat']} z={r['z']:.1f}" for r in sorted([r for r in rowsp if not math.isnan(r['z'])], key=lambda r: -abs(r['z']))[:8]))
    # plain chain fitted to its own kind of corpus (should fit well): calibration of the battery's false-alarm rate
    rowspp = run_model(PlainMarkov(planted, k=2), meta, realp, NSYN, seed=5); spp = summary(rowspp); J['control_planted_markov_selffit'] = rowspp
    log(f"calibration: order-2 chain refitted to that Markov corpus and tested on it: outside {spp['outside']}/{spp['n']}, |z|>3: {spp['z_gt3']}, sum|z| {spp['sum_abs_z_capped']:.1f} (false-alarm level of the battery)")

    log("\n### 5c. Planted syllabic language corpus (synth.planted_corpus, Tamil lexicon, same text lengths; signs are syllable codes, so the parser sees no frame and the slot grammar reduces to the middle chain)")
    try:
        import synth, indus_core as IC
        lex = IC.load_lexicon('tamil')
        ptexts, _ = synth.planted_corpus([[list(s)] for s in [x[2] for x in FIT]], lex, seed=1)
        plant = [(site, cls, tuple(t[0])) for (site, cls, _), t in zip(FIT, ptexts)]
        realq = battery(plant)
        for k_, nm in ((1, 'M0 (order-1 middle chain)'), (2, 'order-2 middle chain')):
            rowsq = run_model(SlotModel(plant, mech=['markov2'] if k_ == 2 else []), meta, realq, NSYN, seed=9); sq = summary(rowsq); J[f'control_syllabic_k{k_}'] = rowsq
            log(f"{nm} fitted to planted Tamil: outside {sq['outside']}/{sq['n']}, |z|>3: {sq['z_gt3']}, sum|z| {sq['sum_abs_z_capped']:.1f}")
            log("  top gaps: " + '; '.join(f"{r['stat']} real={r['real']:.3f} syn={r['mean']:.3f} z={r['z']:.1f}" for r in sorted([r for r in rowsq if not math.isnan(r['z'])], key=lambda r: -abs(r['z']))[:8]))
    except Exception as e:
        log(f"  (planted syllabic control skipped: {e!r})")

    # transcription robustness: M0 on the other seq variants
    if not QUICK:
        log("\n## 6. Merge robustness: M0 and M* on seq_raw and seq_strong (outside / |z|>3 / sum|z|)")
        for key in ('seq_raw', 'seq_strong'):
            data = [(t['site'], cls_of(t), tuple(t[key])) for t in C if t[key] and t['site'] in FIT_SITES]
            rb_ = battery(data); mt = [(s, c) for s, c, _ in data]
            for name, mech in (('M0', []), ('M*', cur)):
                rows = run_model(SlotModel(data, mech=mech), mt, rb_, max(30, NSYN // 2), seed=13); s = summary(rows); J[f'{key}_{name}'] = rows
                log(f"{key:11s} {name}: outside {s['outside']}/{s['n']}, |z|>3 {s['z_gt3']}, sum|z| {s['sum_abs_z_capped']:.1f}; top: " + '; '.join(f"{r['stat']} z={r['z']:.1f}" for r in sorted([r for r in rows if not math.isnan(r['z'])], key=lambda r: -abs(r['z']))[:5]))

    log(f"\n(elapsed {time.time() - t0:.0f} s)")
    with open(os.path.join(HERE, 'data/derived/strat_adequacy.txt'), 'w') as f: f.write('\n'.join(OUT) + '\n')
    with open(os.path.join(HERE, 'data/derived/strat_adequacy.json'), 'w') as f: json.dump(J, f, indent=0, default=str)

if __name__ == '__main__':
    main()
