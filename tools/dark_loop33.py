"""S-DARK-33: induce the whole code as a finite-state machine.

If the Indus texts are a designed business code, the complete grammar should be a small probabilistic finite
automaton (PFA), learnable from Mohenjo-daro + Harappa and predictive outside them.
Learners: ALERGIA (Hoeffding state merging on a prefix-tree acceptor), an MDL-greedy merger (MDI-like), a MAP-EM
HMM with explicit termination (K = 5..40), and baselines (unigram, Markov-1, Markov-2 as PFAs).
Alphabets: CLASS (frame slots, numerals, fish marks, trees, titles, each paradigm closer, the S-DARK-10 MDL
classes, frequent lexicon signs, rest OTHER) and RAW (120 commonest signs + OTHER).
Controls: within-text shuffle, within-slot shuffle (S310 parse), Markov-2 synthetic corpus, Ur III seal legends,
Linear B tablet lines, Proto-Elamite lines (loaders as in tools/dark_loop25.py).
Held-out: other sites; the 324 IM77-only certainly-new texts (loop27_sets.json; W<->M via bridge_extended +
loop27 completed bridge); versus shuffled copies and random unigram strings of the same lengths.
Usage: python3 tools/dark_loop33.py <cycle 1|2|3|4> <seq_raw|seq_strong|seq_all>
"""
import json, sys, random, collections, math, re, csv, os, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
SP = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
OUT = 'data/derived/dark/'
C = json.load(open('data/derived/merged-corpus-canonical.json'))
BR = json.load(open('data/derived/bridge_extended.json'))
L27 = json.load(open('data/derived/dark/loop27_sets.json'))
CY = int(sys.argv[1]); LV = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'
rnd = random.Random(33)
BIG = {'Mohenjo-daro', 'Harappa'}
END = '#'

def otype(t):
    t = t.split(':')[0]
    return 'SEAL' if t == 'SEAL' else 'TAB' if t == 'TAB' else 'OTHER'

# ---------------------------------------------------------------- Indus corpus
RAW2LV = collections.defaultdict(collections.Counter)
for r in C:
    for a, b in zip(r['seq_raw'], r[LV]): RAW2LV[a][b] += 1
def canon(w):
    c = RAW2LV.get(w)
    return c.most_common(1)[0][0] if c else w
OBJ = []
for r in C:
    s = r[LV]
    if not s or r['complete'] != 'Y' or r['dir.'].strip() == '-': continue
    OBJ.append(dict(id=r['cisi'], site=r['site'], ot=otype(r['type']), seq=list(s),
                    big=r['site'] in BIG, held=r['site'] not in BIG and r['site'] != 'Unknown'))
FIT = [o for o in OBJ if o['big']]
HELD = [o for o in OBJ if o['held']]

# ---------------------------------------------------------------- IM77 certainly-new texts (M numbers -> W numbers)
def load_im77_new():
    rows = list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
    want = set((a, b) for a, b in L27['new'])
    bridge = dict(L27['bridge']); bridge.update({k: v for k, v in BR.items() if k not in bridge})
    freq = collections.Counter(x for o in OBJ for x in o['seq'])
    m2w = collections.defaultdict(list)
    for w, ms in bridge.items():
        for m in ms: m2w[m].append(canon(int(w)))
    M2W = {m: max(ws, key=lambda w: freq.get(w, 0)) for m, ws in m2w.items()}
    by = collections.defaultdict(list)
    for r in rows:
        k = (r['text_no'], r['side'])
        if k in want and r['signs_clean'].strip():
            by[k].append((int(r['line']), [int(x) for x in r['signs_clean'].split()], r['object_type'], r['site']))
    out = []; unb = 0; tot = 0
    for k, lines in by.items():
        lines.sort(); s = [x for ln in lines for x in ln[1]]
        ws = []
        for m in s:
            tot += 1
            if m in M2W: ws.append(M2W[m])
            else: ws.append(-m); unb += 1      # negative = unbridged M sign -> OTHER in any alphabet
        ot = {'seal': 'SEAL', 'miniature tablet': 'TAB', 'copper tablet': 'TAB'}.get(lines[0][2], 'OTHER')
        out.append(dict(id='IM' + k[0] + '/' + k[1], site=lines[0][3], ot=ot, seq=ws, big=False, held=True))
    return out, unb, tot

# ---------------------------------------------------------------- frame parser (S310/S331) for slot shuffles
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; MJAR = {741, 742, 745}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]; CLS = set(CL)
FISHQ = {235, 240, 233, 231}; FISH = {220}; TREE = {390, 405, 407}; LEAF = {803, 806, 838}
NUMS = {1, 3, 4, 5, 16, 17, 18}; NUMT = {31, 32, 33, 34, 55, 56}
TITLE = {255, 435, 690, 760, 100, 176, 923}
def slots(s):
    lab = ['NAME'] * len(s); i = 0; j = len(s)
    if s and canon(s[0]) in OPEN or s and s[0] in OPEN:
        lab[0] = 'OPENER'; i = 1
        if len(s) > 1 and s[1] in MARK:
            lab[1] = 'MARKER'; i = 2
            if s[0] == 920 and len(s) > 2 and s[2] in MJAR: lab[2] = 'MARKER'; i = 3
    while j - 1 > i and s[j - 1] in SUF: lab[j - 1] = 'SUFFIX'; j -= 1
    if j - 1 >= i and s[j - 1] in CLS: lab[j - 1] = 'CLOSER'; j -= 1
    return lab
def shuffle_within(seqs, R):
    out = []
    for s in seqs:
        t = list(s); R.shuffle(t); out.append(t)
    return out
def shuffle_slots(seqs, R):
    out = []
    for s in seqs:
        lab = slots(s); t = list(s)
        idx = [k for k, l in enumerate(lab) if l == 'NAME']
        vals = [t[k] for k in idx]; R.shuffle(vals)
        for k, v in zip(idx, vals): t[k] = v
        out.append(t)
    return out

# ---------------------------------------------------------------- alphabets
def classify_fn(fit_seqs, mode):
    freq = collections.Counter(x for s in fit_seqs for x in s)
    if mode == 'RAW':
        top = set(w for w, _ in freq.most_common(120))
        return lambda w: str(w) if w in top else 'OTHER'
    # CLASS alphabet
    fixed = {}
    for g, name in [(OPEN, 'OPEN'), (MARK, 'CONN'), (MJAR, 'MJAR'), (SUF, 'SUF'), (FISHQ, 'FISHQ'), (FISH, 'FISH'),
                    (TREE, 'TREE'), (LEAF, 'LEAF'), (NUMS, 'NUMs'), (NUMT, 'NUMt'), (TITLE, 'TITLE'),
                    ({900}, 'BRACKET'), ({630, 904, 482}, 'TABTITLE'), ({368}, 'CONN')]:
        for w in g: fixed[w] = name
    for c in CL: fixed[c] = 'C' + str(c)                     # each paradigm closer is its own symbol
    rest = [w for w, n in freq.most_common() if w not in fixed and n >= 30][:40]
    for w in rest: fixed[w] = 'w' + str(w)
    return lambda w: fixed.get(w, 'OTHER')
def encode(objs, f, with_type=False):
    out = []
    for o in objs:
        s = [f(x) for x in o['seq']]
        if with_type: s = ['T:' + o['ot']] + s
        out.append(s)
    return out

# ---------------------------------------------------------------- PFA (deterministic) with smoothing
class PFA:
    """states: list of dict symbol->(count,next); final counts; n counts. Smoothed with beta * unigram backoff;
    an absent transition leads to BACK (unigram state)."""
    def __init__(self, trans, final, n, alphabet, uni, beta=0.5):
        self.trans = trans; self.final = final; self.n = n; self.A = alphabet; self.uni = uni; self.beta = beta
        self.S = len(trans)
    def logp_sym(self, q, a):
        if q == -1: return math.log(self.uni.get(a, 1e-9))
        c = self.trans[q].get(a, (0, None))[0]
        return math.log((c + self.beta * self.uni.get(a, 1e-9)) / (self.n[q] + self.beta))
    def next(self, q, a):
        if q == -1: return -1
        t = self.trans[q].get(a)
        return t[1] if t else -1
    def logp_end(self, q):
        if q == -1: return math.log(self.uni.get(END, 1e-9))
        return math.log((self.final[q] + self.beta * self.uni.get(END, 1e-9)) / (self.n[q] + self.beta))
    def ll(self, s, skip_first=False):
        q = 0; tot = 0.0; path = [0]
        for k, a in enumerate(s):
            lp = self.logp_sym(q, a)
            if not (skip_first and k == 0): tot += lp
            q = self.next(q, a); path.append(q)
        return tot + self.logp_end(q), path
    def ntrans(self):
        return sum(len(t) for t in self.trans) + sum(1 for f in self.final if f > 0)
    def model_bits(self, N):
        # each transition: target state + symbol + count precision 0.5 log2 N
        return self.ntrans() * (math.log2(max(self.S, 2)) + math.log2(len(self.A)) + 0.5 * math.log2(N))

def build_pta(seqs):
    trans = [dict()]; final = [0]; n = [0]
    for s in seqs:
        q = 0; n[0] += 1
        for a in s:
            if a not in trans[q]:
                trans.append(dict()); final.append(0); n.append(0); trans[q][a] = [0, len(trans) - 1]
            trans[q][a][0] += 1; q = trans[q][a][1]; n[q] += 1
        final[q] += 1
    return trans, final, n

def alergia(seqs, alpha=0.05, t0=2, criterion='hoeffding'):
    """Standard ALERGIA (Carrasco & Oncina 1994) with red/blue ordering; merge + fold.
    criterion 'mdl': merge if it does not increase the two-part description length (an MDI-like variant)."""
    trans, final, n = build_pta(seqs)
    N = len(trans)
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    bound = math.sqrt(0.5 * math.log(2.0 / alpha))
    def different(f1, n1, f2, n2):
        return abs(f1 / n1 - f2 / n2) > bound * (1 / math.sqrt(n1) + 1 / math.sqrt(n2))
    def compatible(q1, q2, depth=0):
        q1 = find(q1); q2 = find(q2)
        if q1 == q2: return True
        if n[q1] < t0 or n[q2] < t0: return True
        if different(final[q1], n[q1], final[q2], n[q2]): return False
        for a in set(trans[q1]) | set(trans[q2]):
            c1 = trans[q1].get(a, (0, None))[0]; c2 = trans[q2].get(a, (0, None))[0]
            if different(c1, n[q1], c2, n[q2]): return False
            if c1 and c2 and depth < 12 and not compatible(trans[q1][a][1], trans[q2][a][1], depth + 1): return False
        return True
    def merge(q1, q2):   # fold q2 into q1 (both roots)
        parent[q2] = q1; final[q1] += final[q2]; n[q1] += n[q2]
        for a, (c, t) in list(trans[q2].items()):
            if a in trans[q1]:
                trans[q1][a][0] += c
                t1 = find(trans[q1][a][1]); t2 = find(t)
                if t1 != t2: merge(t1, t2)
            else:
                trans[q1][a] = [c, t]
        trans[q2] = {}
    # MDL criterion helpers
    def dl_state(q):
        tot = n[q]; bits = 0.0
        for a, (c, t) in trans[q].items(): bits -= c * math.log2(c / tot)
        if final[q]: bits -= final[q] * math.log2(final[q] / tot)
        return bits
    def mdl_gain(q1, q2):
        # approximate: local change in data bits from pooling the two distributions, minus saved parameter bits
        tot1 = n[q1]; tot2 = n[q2]; tot = tot1 + tot2; before = dl_state(q1) + dl_state(q2); after = 0.0
        for a in set(trans[q1]) | set(trans[q2]):
            c = trans[q1].get(a, (0, None))[0] + trans[q2].get(a, (0, None))[0]
            after -= c * math.log2(c / tot)
        f = final[q1] + final[q2]
        if f: after -= f * math.log2(f / tot)
        saved = (len(trans[q2]) + (1 if final[q2] else 0)) * (0.5 * math.log2(len(seqs) + 1) + 6)
        return (before - after) + saved     # >= 0 means merging does not cost
    red = [0]
    blue = sorted(set(find(t) for a, (c, t) in trans[0].items()), key=lambda q: -n[q])
    while blue:
        blue = [find(b) for b in blue]; blue = [b for b in dict.fromkeys(blue) if b not in red]
        if not blue: break
        b = max(blue, key=lambda q: n[q]); blue.remove(b)
        merged = False
        def sim(r):   # cosine between outgoing distributions (symbols + end); root is never a merge target
            if r == 0: return -1
            ks = set(trans[r]) | set(trans[b]) | {END}
            v1 = [trans[r].get(a, (0,))[0] if a != END else final[r] for a in ks]
            v2 = [trans[b].get(a, (0,))[0] if a != END else final[b] for a in ks]
            d = math.sqrt(sum(x * x for x in v1) * sum(x * x for x in v2))
            return sum(x * y for x, y in zip(v1, v2)) / d if d else 0
        for r in sorted(red, key=lambda q: -sim(q)):
            if r == 0: continue
            if criterion == 'mdl':
                ok = mdl_gain(r, b) >= 0 and compatible(r, b) if alpha < 1 else mdl_gain(r, b) >= 0
            else:
                ok = compatible(r, b)
            if ok:
                merge(r, b); merged = True; break
        if not merged:
            red.append(b)
        # refresh blue: children of red states not red
        blue = []
        for r in red:
            for a, (c, t) in trans[r].items():
                t = find(t)
                if t not in red: blue.append(t)
    # compact
    idx = {q: i for i, q in enumerate(red)}
    T = [dict() for _ in red]; F = [0] * len(red); NN = [0] * len(red)
    for q in red:
        i = idx[q]; F[i] = final[q]; NN[i] = n[q]
        for a, (c, t) in trans[q].items(): T[i][a] = (c, idx[find(t)])
    return T, F, NN

def unigram_of(seqs):
    cnt = collections.Counter(x for s in seqs for x in s); cnt[END] = len(seqs)
    tot = sum(cnt.values()); return {a: c / tot for a, c in cnt.items()}

def make_pfa(seqs, alphabet, kind, **kw):
    uni = unigram_of(seqs)
    if kind == 'alergia':
        T, F, NN = alergia(seqs, alpha=kw.get('alpha', 0.05), t0=kw.get('t0', 2))
    elif kind == 'mdl':
        T, F, NN = alergia(seqs, alpha=kw.get('alpha', 1.0), t0=kw.get('t0', 2), criterion='mdl')
    elif kind == 'unigram':
        T = [dict()]; F = [len(seqs)]; NN = [0]
        for s in seqs:
            for a in s: T[0][a] = (T[0].get(a, (0, 0))[0] + 1, 0); NN[0] += 1
        NN[0] += len(seqs)
    elif kind in ('markov1', 'markov2'):
        k = 1 if kind == 'markov1' else 2
        states = {(): 0}; T = [dict()]; F = [0]; NN = [0]
        for s in seqs:
            h = (); NN[0] += 1
            for a in s:
                h2 = (h + (a,))[-k:]
                if h2 not in states:
                    states[h2] = len(T); T.append(dict()); F.append(0); NN.append(0)
                q = states[h]; c = T[q].get(a, (0, states[h2]))[0]; T[q][a] = (c + 1, states[h2])
                h = h2; NN[states[h]] += 1
            F[states[h]] += 1
    return PFA(T, F, NN, alphabet, uni, beta=kw.get('beta', 0.5))

# ---------------------------------------------------------------- HMM with termination (MAP-EM)
class HMM:
    def __init__(self, K, A, seed=0, prior=0.1):
        R = np.random.RandomState(seed); self.K = K; self.A = list(A); self.ai = {a: i for i, a in enumerate(self.A)}
        self.pi = R.dirichlet(np.ones(K)); self.T = R.dirichlet(np.ones(K + 1), size=K)  # last column = stop
        self.E = R.dirichlet(np.ones(len(A)), size=K); self.prior = prior
    def enc(self, s): return [self.ai.get(a, self.ai.get('OTHER', 0)) for a in s]
    def fwd(self, x):
        K = self.K; alpha = np.zeros((len(x), K)); c = np.zeros(len(x))
        alpha[0] = self.pi * self.E[:, x[0]]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
        for t in range(1, len(x)):
            alpha[t] = (alpha[t - 1] @ self.T[:, :K]) * self.E[:, x[t]]; c[t] = alpha[t].sum(); alpha[t] /= c[t]
        stop = (alpha[-1] * self.T[:, K]).sum()
        return alpha, c, stop
    def ll(self, s):
        x = self.enc(s)
        if not x: return math.log(max(1e-12, (self.pi * self.T[:, self.K]).sum()))
        alpha, c, stop = self.fwd(x)
        return float(np.log(c).sum() + np.log(max(stop, 1e-300)))
    def fit(self, seqs, iters=40):
        K = self.K; X = [self.enc(s) for s in seqs if s]
        for it in range(iters):
            Npi = np.full(K, self.prior); NT = np.full((K, K + 1), self.prior); NE = np.full((K, len(self.A)), self.prior)
            for x in X:
                alpha, c, stop = self.fwd(x); n = len(x)
                beta = np.zeros((n, K)); beta[-1] = self.T[:, K]
                for t in range(n - 2, -1, -1):
                    beta[t] = (self.T[:, :K] @ (self.E[:, x[t + 1]] * beta[t + 1])) / c[t + 1]
                g = alpha * beta; g /= g.sum(1, keepdims=True)
                Npi += g[0]
                for t in range(n): NE[:, x[t]] += g[t]
                for t in range(n - 1):
                    xi = np.outer(alpha[t], self.E[:, x[t + 1]] * beta[t + 1]) * self.T[:, :K] / c[t + 1]
                    NT[:, :K] += xi
                NT[:, K] += g[-1]
            self.pi = Npi / Npi.sum(); self.T = NT / NT.sum(1, keepdims=True); self.E = NE / NE.sum(1, keepdims=True)
        return self
    def nparams(self):
        return self.K - 1 + self.K * self.K + self.K * (len(self.A) - 1)
    def model_bits(self, N): return self.nparams() * 0.5 * math.log2(N)

# ---------------------------------------------------------------- control corpora (as tools/dark_loop25.py)
def load_ur3():
    legs = json.load(open(SP + 'ur3_legends.json')); seen = set(); out = []
    for lg in legs:
        flat = tuple(x for ln in lg['lines'] for x in ln)
        if len(flat) < 2 or flat in seen or 'x' in flat: continue
        seen.add(flat); out.append(dict(site='ur3', ot='OTHER', seq=list(flat), big=rnd.random() < 0.7))
    return out
def load_linb():
    out = []; seen = set()
    for l in open('other-scripts/linear-a/data/damos_items.jsonl'):
        d = json.loads(l); c = d.get('content') or ''
        for ln in c.split('\n'):
            m = re.match(r'\s*\.(\d+[a-z]?)\s+(.*)', ln)
            if not m: continue
            toks = []
            for t in re.split(r'\s+', m.group(2).strip()):
                t = t.strip('[],')
                if not t or t in ('/', 'vac.', 'vacat', 'vest.') or t.startswith('vac'): continue
                if re.fullmatch(r'\d+', t): t = 'NUM'
                elif re.fullmatch(r"'?[a-z0-9*]+(-[a-z0-9*]+)+'?", t): t = 'w:' + t.strip("'")
                elif re.fullmatch(r'[A-Z][A-Z0-9*+]*', t): t = 'I:' + t
                elif t in ('S', 'V', 'T', 'Z', 'M', 'N', 'P', 'Q', 'L'): t = 'U:' + t
                else: continue
                toks.append(t)
            if len(toks) >= 2 and tuple(toks) not in seen:
                seen.add(tuple(toks)); out.append(dict(site='linb', ot='OTHER', seq=toks, big=rnd.random() < 0.7))
    return out
def load_pe():
    pe = json.load(open('other-scripts/proto-elamite/data/pe_corpus.json')); out = []; seen = set()
    for t in pe:
        for ln in t['lines']:
            s = [x.split('~')[0] for x in ln['signs'] if x != 'x' and not x.startswith('x')]
            if ln.get('lacuna'): continue
            if ln['numerals']: s = s + ['N:' + ln['numerals'][0][1]]
            if len(s) >= 2 and tuple(s) not in seen:
                seen.add(tuple(s)); out.append(dict(site='pe', ot='OTHER', seq=s, big=rnd.random() < 0.7))
    return out
def markov2_synthetic(seqs, R, n=None):
    """synthetic corpus from an order-2 Markov chain fitted on seqs (same size), with termination"""
    tri = collections.defaultdict(collections.Counter)
    for s in seqs:
        h = ('^', '^')
        for a in list(s) + [END]: tri[h][a] += 1; h = (h[1], a)
    out = []
    for _ in range(n or len(seqs)):
        h = ('^', '^'); s = []
        while True:
            c = tri[h]; a = R.choices(list(c), list(c.values()))[0]
            if a == END or len(s) > 20: break
            s.append(a); h = (h[1], a)
        if s: out.append(s)
    return out
def random_unigram(seqs, R, uni):
    syms = [a for a in uni if a != END]; w = [uni[a] for a in syms]
    return [R.choices(syms, w, k=len(s)) for s in seqs]

# ---------------------------------------------------------------- evaluation
def auc(pos, neg):
    allv = sorted(set(pos) | set(neg)); rank = {}
    # average ranks with ties
    vals = sorted(pos + neg); i = 0
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j] == vals[i]: j += 1
        for k in range(i, j): rank[vals[i]] = (i + j + 1) / 2
        i = j
    rp = sum(rank[v] for v in pos); n1 = len(pos); n2 = len(neg)
    return (rp - n1 * (n1 + 1) / 2) / (n1 * n2) if n1 and n2 else float('nan')
def per_sign(model, seqs, skip_first=False):
    out = []
    for s in seqs:
        if isinstance(model, PFA): ll = model.ll(s, skip_first)[0]
        else: ll = model.ll(s[1:] if skip_first else s)
        L = len(s) - (1 if skip_first else 0) + 1
        out.append(ll / L)
    return out
def total_ll(model, seqs, skip_first=False):
    return sum(model.ll(s, skip_first)[0] if isinstance(model, PFA) else model.ll(s[1:] if skip_first else s) for s in seqs)

def split(seqs, R, frac=0.8):
    idx = list(range(len(seqs))); R.shuffle(idx); k = int(frac * len(seqs))
    return [seqs[i] for i in idx[:k]], [seqs[i] for i in idx[k:]]

def fit_all_models(train, dev, alphabet, tag, log, hmm_ks=(5, 10, 20, 40), hmm_iters=20):
    """fit baselines, ALERGIA at several alphas, MDL-merge, HMMs; choose each family's member by dev LL; report sizes + MDL"""
    N = sum(len(s) + 1 for s in train); res = {}
    def rec(name, m, S, ntr):
        llf = total_ll(m, train); lld = total_ll(m, dev)
        mb = m.model_bits(N); db = -llf / math.log(2)
        res[name] = dict(states=S, trans=ntr, model_bits=round(mb), data_bits=round(db), mdl_bits=round(mb + db),
                         fit_nats_per_sign=round(-llf / N, 4), dev_nats_per_sign=round(-lld / sum(len(s) + 1 for s in dev), 4))
        log(f'  {tag:34s} {name:14s} states {S:5d} trans {ntr:6d} MDL {mb + db:10.0f} bits (model {mb:8.0f})  '
            f'fit {res[name]["fit_nats_per_sign"]:.3f}  dev {res[name]["dev_nats_per_sign"]:.3f} nats/sign')
        return m
    models = {}
    for k in ('unigram', 'markov1', 'markov2'):
        m = make_pfa(train, alphabet, k); models[k] = rec(k, m, m.S, m.ntrans())
    best = None
    for al in (0.5, 0.1, 0.01, 0.001):
        m = make_pfa(train, alphabet, 'alergia', alpha=al, t0=3); rec(f'alergia@{al}', m, m.S, m.ntrans())
        d = total_ll(m, dev)
        if best is None or d > best[0]: best = (d, al, m)
    models['alergia'] = best[2]; res['alergia_best_alpha'] = best[1]
    m = make_pfa(train, alphabet, 'mdl', alpha=1.0, t0=3); models['mdl'] = rec('mdl-merge', m, m.S, m.ntrans())
    bestH = None
    for K in hmm_ks:
        h = HMM(K, alphabet, seed=K).fit(train, iters=hmm_iters); rec(f'hmm{K}', h, K, K * K + K * len(alphabet))
        d = total_ll(h, dev)
        if bestH is None or d > bestH[0]: bestH = (d, K, h)
    models['hmm'] = bestH[2]; res['hmm_best_K'] = bestH[1]
    return models, res

def acceptance(models, dev, genuine_sets, R, uni, alphabet, log, tag, skip_first=False):
    """threshold = 5th percentile of dev per-sign LL (so dev accept = 95%); accept rates + AUC vs shuffle and random"""
    out = {}
    for name, m in models.items():
        dv = per_sign(m, dev, skip_first); thr = sorted(dv)[int(0.05 * len(dv))]
        row = {'thr': round(thr, 3)}
        for gname, G in genuine_sets.items():
            if not G: continue
            if skip_first:
                body = [s[1:] for s in G]; heads = [s[0] for s in G]
                shuf = [[h] + t for h, t in zip(heads, shuffle_within(body, R))]
                slot = [[h] + t for h, t in zip(heads, shuffle_slots_sym(body, R))]
                rand = [[h] + t for h, t in zip(heads, random_unigram(body, R, uni))]
            else:
                shuf = shuffle_within(G, R); slot = shuffle_slots_sym(G, R); rand = random_unigram(G, R, uni)
            g = per_sign(m, G, skip_first); sh = per_sign(m, shuf, skip_first); sl = per_sign(m, slot, skip_first); rd = per_sign(m, rand, skip_first)
            L3 = [i for i, s in enumerate(G) if len(s) - (1 if skip_first else 0) >= 3]
            acc = lambda v: sum(x >= thr for x in v) / len(v)
            row[gname] = dict(n=len(G), accept=round(acc(g), 3), accept_shuf=round(acc(sh), 3), accept_slot=round(acc(sl), 3),
                              accept_rand=round(acc(rd), 3), auc_shuf=round(auc(g, sh), 3), auc_slot=round(auc(g, sl), 3),
                              auc_rand=round(auc(g, rd), 3),
                              auc_shuf_len3=round(auc([g[i] for i in L3], [sh[i] for i in L3]), 3) if L3 else None,
                              mean_nats=round(-sum(g) / len(g), 3))
            r = row[gname]
            log(f'  {tag:30s} {name:9s} {gname:10s} n {r["n"]:4d} accept {r["accept"]:.2f} | shuf {r["accept_shuf"]:.2f} '
                f'slot {r["accept_slot"]:.2f} rand {r["accept_rand"]:.2f} | AUC shuf {r["auc_shuf"]:.3f} (len>=3 {r["auc_shuf_len3"]}) '
                f'slot {r["auc_slot"]:.3f} rand {r["auc_rand"]:.3f}')
        out[name] = row
    return out

SYM_SLOT = None
def shuffle_slots_sym(seqs, R):
    """within-slot shuffle in symbol space: keep OPEN/CONN/MJAR at the start, closers/SUF at the end, shuffle the rest"""
    out = []
    for s in seqs:
        t = list(s); i = 0; j = len(t)
        if t and t[0] in ('OPEN',) or (t and t[0].isdigit() and int(t[0]) in OPEN): i = 1
        if i == 1 and len(t) > 1 and (t[1] == 'CONN' or (t[1].isdigit() and int(t[1]) in MARK)): i = 2
        while j - 1 > i and (t[j - 1] == 'SUF' or (t[j - 1].isdigit() and int(t[j - 1]) in SUF)): j -= 1
        if j - 1 >= i and (t[j - 1].startswith('C') and t[j - 1][1:].isdigit() or (t[j - 1].isdigit() and int(t[j - 1]) in CLS)): j -= 1
        mid = t[i:j]; R.shuffle(mid); t[i:j] = mid; out.append(t)
    return out

# ---------------------------------------------------------------- reading an automaton
def describe_pfa(m, train, log, maxstates=60):
    """name each state by its dominant incoming symbols and outgoing symbols; list loops and forbidden transitions"""
    inc = collections.defaultdict(collections.Counter); visits = collections.Counter(); pos = collections.defaultdict(list)
    for s in train:
        q = 0; visits[0] += 1
        for k, a in enumerate(s):
            q2 = m.next(q, a)
            if q2 >= 0: inc[q2][a] += 1; visits[q2] += 1; pos[q2].append(k + 1)
            q = q2
            if q < 0: break
    order = sorted(range(m.S), key=lambda q: -visits[q])
    lines = []
    for q in order[:maxstates]:
        outs = sorted(m.trans[q].items(), key=lambda kv: -kv[1][0])
        o = ', '.join(f'{a}->s{t}({c})' for a, (c, t) in outs[:6])
        i = ', '.join(f'{a}({c})' for a, c in inc[q].most_common(4))
        fin = m.final[q] / m.n[q] if m.n[q] else 0
        mp = sum(pos[q]) / len(pos[q]) if pos[q] else 0
        lines.append(f'  s{q:<3d} visits {visits[q]:5d} end {fin:.2f} meanpos {mp:.1f} | in: {i} | out: {o}')
    for l in lines: log(l)
    # cycles: find edges that return to an earlier (lower mean-position) state
    meanpos = {q: (sum(pos[q]) / len(pos[q]) if pos[q] else 0) for q in range(m.S)}
    loops = []
    for q in range(m.S):
        for a, (c, t) in m.trans[q].items():
            if q == 0 or t == 0: continue
            if t == q and c >= 5: loops.append((q, a, t, c, 'self'))
            elif meanpos[t] < meanpos[q] - 0.5 and c >= 5 and visits[t] >= 10: loops.append((q, a, t, c, 'back'))
    for q, a, t, c, kind in sorted(loops, key=lambda x: -x[3])[:20]:
        log(f'  LOOP {kind}: s{q} --{a}--> s{t}  ({c}x)')
    return visits, inc, loops

def to_dot(m, visits, inc, path, minc=5):
    lines = ['digraph PFA {', 'rankdir=LR; node [shape=circle,fontsize=9];']
    for q in range(m.S):
        if visits[q] < minc: continue
        lab = f's{q}\\n' + '/'.join(a for a, _ in inc[q].most_common(2)) if q else 'START'
        fin = m.final[q] / m.n[q] if m.n[q] else 0
        lines.append(f's{q} [label="{lab}\\nend {fin:.2f}"{",peripheries=2" if fin > 0.3 else ""}];')
    for q in range(m.S):
        for a, (c, t) in m.trans[q].items():
            if c >= minc and visits[q] >= minc and visits[t] >= minc:
                lines.append(f's{q} -> s{t} [label="{a} {c}",penwidth={0.5 + math.log(c) / 2:.1f}];')
    lines.append('}'); open(path, 'w').write('\n'.join(lines))

def enumerate_texts(m, pmin=1e-4, maxlen=12, cap=200000):
    """all strings with probability >= pmin (deterministic PFA, smoothed probs on known transitions only)"""
    import heapq
    out = []; heap = [(-0.0, 0, ())]; seen = 0
    while heap and seen < cap:
        nlp, q, pre = heapq.heappop(heap); lp = -nlp; seen += 1
        pe = lp + m.logp_end(q)
        if pe >= math.log(pmin): out.append((tuple(pre), math.exp(pe)))
        if len(pre) >= maxlen: continue
        for a, (c, t) in m.trans[q].items():
            lp2 = lp + m.logp_sym(q, a)
            if lp2 >= math.log(pmin): heapq.heappush(heap, (-lp2, t, pre + (a,)))
    return out

# =================================================================================================================
def main():
    t0 = time.time()
    logf = open(OUT + f'loop33_cycle{CY}_{LV}.txt', 'w')
    def log(s=''):
        print(s); logf.write(s + '\n'); logf.flush()
    IMNEW, unb, tot = load_im77_new()
    log(f'== S-DARK-33 cycle {CY} level {LV}: fit (MD+H complete, direction recorded) {len(FIT)} texts '
        f'[seals {sum(o["ot"]=="SEAL" for o in FIT)}, tablets {sum(o["ot"]=="TAB" for o in FIT)}]; held-out sites {len(HELD)}; '
        f'IM77 certainly-new {len(IMNEW)} texts ({unb}/{tot} tokens unbridged -> OTHER)')
    fit_seqs = [o['seq'] for o in FIT]
    results = {}
    if CY == 1:
        # -------- learning: Indus seals, Indus all types with type start symbol, controls
        for mode in ('CLASS', 'RAW'):
            f = classify_fn(fit_seqs, mode)
            for name, objs, with_type in (('seals', [o for o in FIT if o['ot'] == 'SEAL'], False), ('all+type', FIT, True)):
                enc = encode(objs, f, with_type); A = sorted(set(x for s in enc for x in s)) + [END]
                train, dev = split(enc, random.Random(1))
                log(f'\n## Indus {name} alphabet {mode} (|A| = {len(A) - 1}), train {len(train)} dev {len(dev)}')
                models, res = fit_all_models(train, dev, A, f'indus-{name}-{mode}', log)
                results[f'indus-{name}-{mode}'] = res
                # controls on the same alphabet: within-text shuffle, within-slot shuffle, Markov-2 synthetic
                R = random.Random(2)
                if with_type:
                    body = [s[1:] for s in train]; heads = [s[0] for s in train]
                    ctrl = {'shuffle': [[h] + t for h, t in zip(heads, shuffle_within(body, R))],
                            'slotshuf': [[h] + t for h, t in zip(heads, shuffle_slots_sym(body, R))],
                            'markov2': [[R.choice(heads)] + t for t in markov2_synthetic(body, R)]}
                else:
                    ctrl = {'shuffle': shuffle_within(train, R), 'slotshuf': shuffle_slots_sym(train, R), 'markov2': markov2_synthetic(train, R)}
                for cname, cs in ctrl.items():
                    ctr, cdv = split(cs, random.Random(3))
                    log(f'-- control {cname} ({len(ctr)} train)')
                    _, cres = fit_all_models(ctr, cdv, A, f'ctrl-{cname}-{name}-{mode}', log, hmm_ks=(10, 20), hmm_iters=20)
                    results[f'ctrl-{cname}-{name}-{mode}'] = cres
        # other scripts, RAW-style alphabet (120 commonest + OTHER)
        if LV == 'seq_raw':
            for cname, loader in (('ur3', load_ur3), ('linb', load_linb), ('pe', load_pe)):
                objs = loader(); seqs = [o['seq'] for o in objs]
                if len(seqs) > 3000: seqs = random.Random(11).sample(seqs, 3000)   # size-matched to the Indus fit set
                freq = collections.Counter(x for s in seqs for x in s); top = set(w for w, _ in freq.most_common(120))
                enc = [[str(x) if x in top else 'OTHER' for x in s] for s in seqs]
                A = sorted(set(x for s in enc for x in s)) + [END]
                train, dev = split(enc, random.Random(1))
                log(f'\n## control script {cname}: {len(enc)} texts, train {len(train)} (mean len {sum(map(len,enc))/len(enc):.1f})')
                _, cres = fit_all_models(train, dev, A, f'script-{cname}', log, hmm_ks=(10, 20, 40), hmm_iters=20)
                results[f'script-{cname}'] = cres
        json.dump(results, open(OUT + f'loop33_c1_{LV}.json', 'w'), indent=1)
    elif CY == 2:
        # -------- held-out acceptance and ROC
        for mode in ('CLASS', 'RAW'):
            f = classify_fn(fit_seqs, mode)
            for name, objs, with_type in (('seals', [o for o in FIT if o['ot'] == 'SEAL'], False), ('all+type', FIT, True)):
                enc = encode(objs, f, with_type); A = sorted(set(x for s in enc for x in s)) + [END]
                train, dev = split(enc, random.Random(1)); uni = unigram_of(train)
                log(f'\n## Indus {name} alphabet {mode}: models fitted on train {len(train)}, threshold from dev {len(dev)} (5th percentile)')
                models = {'markov1': make_pfa(train, A, 'markov1'), 'markov2': make_pfa(train, A, 'markov2'),
                          'alergia': make_pfa(train, A, 'alergia', alpha=0.05, t0=3), 'mdl': make_pfa(train, A, 'mdl', alpha=1.0, t0=3),
                          'hmm20': HMM(20, A, seed=20).fit(train, iters=30)}
                held = [o for o in HELD if (o['ot'] == 'SEAL' or with_type)]
                imn = [o for o in IMNEW if (o['ot'] == 'SEAL' or with_type)]
                G = {'dev': dev, 'heldsites': encode(held, f, with_type), 'im77new': encode(imn, f, with_type),
                     'im77new_s': encode([o for o in imn if o['ot'] == 'SEAL'], f, with_type)}
                # fresh unigram-random: use train unigram over symbols
                acc = acceptance(models, dev, G, random.Random(5), uni, A, log, f'{name}-{mode}', skip_first=with_type)
                results[f'{name}-{mode}'] = acc
                # how much of the IM77 new material is OTHER?
                oth = sum(x == 'OTHER' for s in G['im77new'] for x in s) / max(1, sum(len(s) for s in G['im77new']))
                log(f'  OTHER share: train {sum(x=="OTHER" for s in train for x in s)/sum(len(s) for s in train):.3f}  held sites '
                    f'{sum(x=="OTHER" for s in G["heldsites"] for x in s)/max(1,sum(len(s) for s in G["heldsites"])):.3f}  im77new {oth:.3f}')
        json.dump(results, open(OUT + f'loop33_c2_{LV}.json', 'w'), indent=1)
    elif CY == 5:
        # -------- cycle 2b: the same acceptance / ROC test on control corpora (what does a real code or a real language give?)
        def planted_frame(ntext=3200, seed=8):
            R = random.Random(seed); names = [f'n{i}' for i in range(150)]
            wn = [1 / (i + 1) for i in range(150)]; closers = [f'c{i}' for i in range(10)]; wc = [1 / (i + 1) ** 0.9 for i in range(10)]
            quals = [f'q{i}' for i in range(25)]; QS = {c: R.sample(quals, 3) for c in closers}; out = []
            for _ in range(ntext):
                s = []
                if R.random() < 0.45: s += [R.choice(['o0', 'o0', 'o1', 'o2']), 'mk']
                s += R.choices(names, wn, k=R.choice([1, 2, 2, 3, 3, 4]))
                if R.random() < 0.8:
                    c = R.choices(closers, wc)[0]
                    if R.random() < 0.6: s.append(R.choice(QS[c]))
                    s.append(c)
                    if R.random() < 0.15: s.append('sf')
                if len(s) >= 2: out.append(s)
            return out
        def strict_code(ntext=3200, seed=9):
            """a designed code: 6 ordered fields, each optional with p = 0.7, each with its own 4-12 symbols; no field repeats"""
            R = random.Random(seed); fields = [[f'f{k}_{i}' for i in range(n)] for k, n in enumerate([4, 6, 12, 8, 10, 5])]
            out = []
            for _ in range(ntext):
                s = [R.choice(F) for F in fields if R.random() < 0.7]
                if len(s) >= 2: out.append(s)
            return out
        fs = [o['seq'] for o in FIT if o['ot'] == 'SEAL']
        corp = {'indus-seals': fs, 'ur3': [o['seq'] for o in load_ur3()], 'linb': [o['seq'] for o in load_linb()],
                'pe': [o['seq'] for o in load_pe()], 'planted-frame': planted_frame(), 'strict-code': strict_code(),
                'markov2-synth': markov2_synthetic(fs, random.Random(4)), 'indus-shuffled': shuffle_within(fs, random.Random(6))}
        for cname, seqs in corp.items():
            if len(seqs) > 3000: seqs = random.Random(11).sample(seqs, 3000)
            freq = collections.Counter(x for s in seqs for x in s); top = set(w for w, _ in freq.most_common(120))
            enc = [[str(x) if x in top else 'OTHER' for x in s] for s in seqs]
            A = sorted(set(x for s in enc for x in s)) + [END]
            train, dev = split(enc, random.Random(1)); tr2, test = split(train, random.Random(2))
            uni = unigram_of(tr2)
            models = {'markov1': make_pfa(tr2, A, 'markov1'), 'alergia': make_pfa(tr2, A, 'alergia', alpha=0.05, t0=3),
                      'hmm20': HMM(20, A, seed=20).fit(tr2, iters=25)}
            log(f'\n## control {cname}: {len(enc)} texts, mean len {sum(map(len, enc))/len(enc):.1f}, |A| {len(A)-1}; '
                f'alergia states {models["alergia"].S}; train {len(tr2)} / threshold set {len(test)} / genuine held-out {len(dev)}')
            acc = acceptance(models, test, {'heldout': dev}, random.Random(5), uni, A, log, cname)
            results[cname] = acc
        json.dump(results, open(OUT + f'loop33_c5_{LV}.json', 'w'), indent=1)
    elif CY == 3:
        # -------- read the automaton (CLASS, all types with type symbol; and seals), compare with the frame, test predictions
        f = classify_fn(fit_seqs, 'CLASS')
        for name, objs, with_type in (('seals', [o for o in FIT if o['ot'] == 'SEAL'], False), ('all+type', FIT, True)):
            enc = encode(objs, f, with_type); A = sorted(set(x for s in enc for x in s)) + [END]
            m = make_pfa(enc, A, 'alergia', alpha=0.05, t0=3)
            log(f'\n## ALERGIA automaton, Indus {name}, CLASS alphabet, fitted on all {len(enc)} MD+H texts: {m.S} states, {m.ntrans()} transitions')
            visits, inc, loops = describe_pfa(m, enc, log)
            to_dot(m, visits, inc, OUT + f'loop33_automaton_{name}_{LV}.dot')
            json.dump(dict(states=m.S, trans=[{a: [c, t] for a, (c, t) in tr.items()} for tr in m.trans], final=m.final, n=m.n,
                           alphabet=A), open(OUT + f'loop33_automaton_{name}_{LV}.json', 'w'))
            # most probable complete paths (the 'form'): top strings
            top = sorted(enumerate_texts(m, pmin=2e-3, maxlen=10), key=lambda x: -x[1])[:25]
            log('  top strings by automaton probability:')
            for s, p in top: log(f'    {p:.4f}  {" ".join(s)}')
            # forbidden transitions: symbol bigrams with expected >= 5 under state-independent emission but 0 observed in fit
            heldE = encode([o for o in HELD if (o['ot'] == 'SEAL' or with_type)], f, with_type)
            imE = encode([o for o in IMNEW if (o['ot'] == 'SEAL' or with_type)], f, with_type)
            big = collections.Counter(); uni = collections.Counter(); nb = 0
            strip = lambda s: [x for x in s if not x.startswith('T:')]
            heldE = [strip(s) for s in heldE]; imE = [strip(s) for s in imE]
            for s in enc:
                s = strip(s)
                for a, b in zip(s, s[1:]): big[(a, b)] += 1; nb += 1
                for a in s: uni[a] += 1
            tot = sum(uni.values()); forb = []
            for a in uni:
                for b in uni:
                    E = nb * uni[a] / tot * uni[b] / tot
                    if E >= 8 and big[(a, b)] == 0: forb.append((a, b, E))
            forb.sort(key=lambda x: -x[2])
            log(f'  forbidden bigrams (E >= 8, observed 0 in fit): {len(forb)}')
            def count_in(seqs, pairs):
                c = 0; n = 0
                for s in seqs:
                    for a, b in zip(s, s[1:]):
                        n += 1
                        if (a, b) in pairs: c += 1
                return c, n
            P = set((a, b) for a, b, _ in forb)
            R = random.Random(7)
            for gname, G in (('heldsites', heldE), ('im77new', imE)):
                c, n = count_in(G, P)
                sh = [count_in(shuffle_within(G, R), P)[0] for _ in range(200)]
                log(f'    {gname}: forbidden bigrams observed {c} of {n} ({c/n:.4f}); within-text shuffle median {sorted(sh)[100]} '
                    f'[{sorted(sh)[5]}-{sorted(sh)[194]}]')
                for a, b, E in forb[:12]:
                    cc = sum(1 for s in G for x, y in zip(s, s[1:]) if (x, y) == (a, b))
                    log(f'      {a:>8s} -> {b:<8s} E_fit {E:5.1f}  {gname} {cc}')
            # the chain of S-DARK-19 as a prediction: order of class symbols along the automaton's mean position
            meanpos = collections.defaultdict(list)
            for s in enc:
                for k, a in enumerate(s): meanpos[a].append(k / max(1, len(s) - 1))
            order = sorted((sum(v) / len(v), a, len(v)) for a, v in meanpos.items() if len(v) >= 20)
            log('  class order by mean relative position (fit): ' + ' > '.join(f'{a}({p:.2f})' for p, a, n in order))
            heldpos = collections.defaultdict(list)
            for s in heldE + imE:
                for k, a in enumerate(s): heldpos[a].append(k / max(1, len(s) - 1))
            horder = sorted((sum(v) / len(v), a) for a, v in heldpos.items() if len(v) >= 10 and a in dict((x[1], 1) for x in order))
            fo = [a for p, a, n in order if a in dict((x[1], 1) for x in horder)]; ho = [a for p, a in horder]
            # Kendall tau between fit order and held-out order
            conc = disc = 0
            for i in range(len(fo)):
                for j in range(i + 1, len(fo)):
                    a, b = fo[i], fo[j]; d = ho.index(a) < ho.index(b)
                    conc += d; disc += (not d)
            log(f'  held-out (sites + IM77 new) class order Kendall tau vs fit: {(conc - disc) / max(1, conc + disc):.3f} over {len(fo)} classes')
    elif CY == 4:
        # -------- size of the code
        f = classify_fn(fit_seqs, 'RAW'); fc = classify_fn(fit_seqs, 'CLASS')
        for mode, fn in (('RAW', f), ('CLASS', fc)):
            for name, objs, with_type in (('seals', [o for o in FIT if o['ot'] == 'SEAL'], False),):
                enc = encode(objs, fn, with_type); A = sorted(set(x for s in enc for x in s)) + [END]
                seen = collections.Counter(tuple(s) for s in enc)
                for kind, kw in (('alergia', dict(alpha=0.05, t0=3)), ('markov1', {}), ('markov2', {})):
                    m = make_pfa(enc, A, kind, **kw)
                    for pmin in (1e-3, 1e-4, 1e-5):
                        gen = enumerate_texts(m, pmin=pmin, maxlen=12, cap=400000)
                        mass = sum(p for _, p in gen); nseen = sum(1 for s, _ in gen if s in seen)
                        pseen = sum(p for s, p in gen if s in seen)
                        # probability that the next text is one already seen (under the model): sum P(seen distinct texts)
                        p_seen_total = sum(math.exp(m.ll(list(s))[0]) for s in seen)
                        log(f'  {mode:5s} {name} {kind:8s} states {m.S:4d} pmin {pmin:g}: {len(gen):6d} texts generable (mass {mass:.3f}); '
                            f'{nseen} of them seen ({nseen/max(1,len(gen)):.3f}); seen texts hold mass {pseen:.3f} of that set; '
                            f'P(next text already seen) = {p_seen_total:.3f} -> P(new) {1-p_seen_total:.3f}; distinct seen {len(seen)}')
                # Good-Turing on the same texts for comparison
                n1 = sum(1 for s, c in seen.items() if c == 1); log(f'  {mode:5s} Good-Turing P(next text new) = {n1/len(enc):.3f} (singletons {n1} of {len(enc)} texts)')
    log(f'\n[{time.time() - t0:.0f}s]')

if __name__ == '__main__':
    main()
