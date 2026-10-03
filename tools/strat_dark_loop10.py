"""Arrow-in-the-dark loop 10: RANDOM ONTOLOGIES.

Each cycle draws >= 40 random small ontologies (K = 2..6 categories with random, decorative semantics) and assigns
the 60 commonest signs to categories by random start + hill-climb on ONE randomly chosen criterion:
  fact      : naive-Bayes prediction of an outside fact (object type / emblem / size class / material / period)
              optimised cross-city (Mohenjo-daro <-> Harappa), then scored once on held-out sites (all other sites)
  quantity  : the category sequence of 'quantity seals' (opener-2-N-item) and Harappa voucher text faces predicts the
              count N (2/3/4/other), fitted on vouchers, scored on quantity seals and vice versa
  reuse     : share of texts (3-8 signs) whose category sequence recurs at another site
  mdl       : description length of the corpus under the category alphabet (Brown-clustering objective)
Category sizes are kept between 60/(2K) and 2*60/K so that collapsing everything into one class is not a move.
Controls, same search: (P) facts/sites/counts permuted across texts; (S) sign order shuffled within texts.
Across ontologies the co-category rate of every sign pair is recorded (label-invariant); the pairs that stay together
regardless of the semantics form the stable co-category graph, compared with GRAMMAR.md slots and the dictionary.

This is deliberately different from tools/strat_anneal.py (fixed 16-label ontology, 80 signs, annealing on summed
held-out facts): here the ontology itself is random, K is small, and the object of study is the pair-stability graph.

Usage: python3 tools/strat_dark_loop10.py --cycle N [--n 48] [--seed S] [--level seq|seq_raw|seq_strong]
                                          [--subset all|md|ha|other]
Writes data/derived/dark/loop10_cycleN.txt and .json.
"""
import json, csv, re, os, sys, random, argparse, collections, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NTOP = 60
NUMERALS = set(range(1, 61))
OPENERS = {861, 817, 820}
SEMANTIC_POOL = [
    ['perishable', 'durable'], ['inbound', 'outbound', 'stored'], ['royal', 'temple', 'private'],
    ['male', 'female', 'child'], ['day', 'month', 'year'], ['gold', 'copper', 'grain', 'cloth', 'wood', 'fish'],
    ['north', 'south', 'east', 'west'], ['debt', 'credit'], ['sea', 'river', 'land'], ['owner', 'witness', 'scribe', 'carrier'],
    ['clan', 'office', 'commodity', 'place', 'count'], ['sacred', 'profane'], ['raw', 'worked', 'finished', 'waste'],
    ['morning', 'noon', 'evening', 'night'], ['father', 'son', 'brother', 'servant', 'guest'],
    ['wheat', 'barley', 'sesame', 'dates', 'beer', 'oil'], ['high', 'low'], ['tax', 'gift', 'wage', 'loan'],
    ['stone', 'clay', 'metal', 'bone'], ['open', 'sealed'], ['first', 'second', 'third', 'fourth', 'fifth', 'sixth'],
]
KNOWN = {  # Wells numbers; GRAMMAR.md slots and WORKING-DICTIONARY.md families
    'opener': {861, 817, 820, 920, 692}, 'connective': {2, 60, 1}, 'mid-initial': {235, 31},
    'closer': {740, 390, 405, 156, 151, 527, 520, 595, 226, 617}, 'suffix': {400, 90},
    'pre-jar title': {176, 904, 636, 923, 61, 752, 48, 100, 760, 690},
    'fish': {220, 240, 235, 233, 231, 226}, 'short numeral': {1, 2, 3, 4, 5, 16, 17, 18, 55},
    'tall numeral': {31, 32, 33, 34}, 'counted item': {390, 407, 405, 220, 900, 740, 904, 645, 384, 700},
    'name-initial': {692, 575, 125, 416, 413, 920, 495},
}


# ------------------------------------------------------------------------------------------------- data
def numval(w):
    if w in (1, 2, 3, 4, 5): return w
    if w in (16, 17, 18): return w - 10
    if 31 <= w <= 40: return w - 30
    return None


def load(level, subset):
    C = json.load(open(os.path.join(ROOT, 'data/derived/merged-corpus-canonical.json')))
    raw2m = {}
    for r in C:
        if len(r['seq_raw']) == len(r[level]):
            for a, b in zip(r['seq_raw'], r[level]): raw2m[a] = b
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'data/raw/inscriptions.csv'))))
    parse = lambda t: tuple(int(x) for x in re.findall(r'\d+', t))
    bykey = {}
    for r in rows:
        for k in ((r['site'], parse(r['text'])), (r['site'], parse(r['text'])[::-1])):
            bykey.setdefault(k, r)
    texts = []
    for r in C:
        s = [x for x in r[level] if x != 0]
        if len(s) < 1: continue
        if subset == 'md' and r['site'] != 'Mohenjo-daro': continue
        if subset == 'ha' and r['site'] != 'Harappa': continue
        if subset == 'other' and r['site'] in ('Mohenjo-daro', 'Harappa'): continue
        row = bykey.get((r['site'], tuple(r['seq_raw'])), {})
        typ = r['type'].split(':')[0]
        typ = typ if typ in ('SEAL', 'TAB', 'POT', 'TAG') else 'other'
        sym = r['symbol']
        emb = None
        if r['type'].startswith('SEAL') and sym not in ('', '-'):
            emb = 'unicorn' if sym.startswith('Bull1') else ('none' if sym == 'None' else 'other')
        mat = r['material'].lower()
        mat = None if mat in ('-', '') else ('steatite' if 'steatite' in mat else
                                             'clay' if mat in ('clay', 'terracotta', 'ceramic') else
                                             'faience' if mat == 'faience' else 'copper' if mat == 'copper' else 'other')
        per = r['period']
        per = None if per in ('-', '') else ('early' if per in ('Early', '1', '2', 'Layer 2') else
                                             'mid' if per in ('Intermediate', '3') else
                                             'late' if per in ('Late', '4', '5', '6') else None)
        size = None
        try:
            h = float(row.get('horizontal(mm)', '0') or 0)
            if h > 0 and r['type'].startswith('SEAL'): size = h
        except ValueError: pass
        texts.append(dict(seq=s, site=r['site'], area=r['area-section'], type=typ, emblem=emb, material=mat,
                          period=per, size=size, cisi=r['cisi'], two=r['type'].startswith('TAB')))
    # size classes: tertiles among seals
    hs = sorted(t['size'] for t in texts if t['size'])
    if hs:
        q1, q2 = hs[len(hs) // 3], hs[2 * len(hs) // 3]
        for t in texts:
            if t['size']: t['size'] = 'small' if t['size'] < q1 else 'large' if t['size'] >= q2 else 'mid'
    # vouchers: two-faced objects in the raw file with one count face (numerals + W700 only) and one text face
    faces = collections.defaultdict(list)
    for r in rows:
        faces[r['id'].split('.')[0]].append(r)
    vouchers = []
    for oid, fs in faces.items():
        if len(fs) != 2: continue
        s0, s1 = [tuple(raw2m.get(x, x) for x in parse(f['text']) if x != 0) for f in fs]
        for cnt, txt, fr in ((s0, s1, fs[1]), (s1, s0, fs[0])):
            if cnt and all(x in NUMERALS or x == 700 for x in cnt) and 700 in cnt and txt and not any(x == 700 for x in txt):
                vals = [numval(x) for x in cnt if numval(x)]
                if vals and (subset == 'all' or (subset == 'ha' and fr['site'] == 'Harappa') or
                             (subset == 'md' and fr['site'] == 'Mohenjo-daro') or
                             (subset == 'other' and fr['site'] not in ('Harappa', 'Mohenjo-daro'))):
                    vouchers.append((list(txt), vals[0], fr['site']))
    quantity = []
    for t in texts:
        s = t['seq']
        if len(s) == 4 and s[0] in OPENERS and s[1] == 2 and numval(s[2]) and s[3] not in NUMERALS:
            quantity.append(([s[0], s[1], s[3]], numval(s[2]), t['site']))
    return texts, vouchers, quantity


class Data:
    def __init__(self, texts, vouchers, quantity, rng, permute=False, shuffle=False):
        self.rng = rng
        cnt = collections.Counter(x for t in texts for x in t['seq'])
        self.top = [s for s, _ in cnt.most_common(NTOP)]
        self.idx = {s: i for i, s in enumerate(self.top)}
        O = NTOP  # OTHER column
        seqs = [[self.idx.get(x, O) for x in t['seq']] for t in texts]
        if shuffle:
            for s in seqs: rng.shuffle(s)
        self.seqs = seqs
        n = len(texts)
        self.X = np.zeros((n, NTOP + 1), dtype=np.float32)
        for i, s in enumerate(seqs):
            for x in s: self.X[i, x] += 1
        self.first = np.array([s[0] for s in seqs]); self.last = np.array([s[-1] for s in seqs])
        self.lenb = np.array([min(len(s), 7) // 2 for s in seqs])  # 0..3
        self.B = np.zeros((NTOP + 1, NTOP + 1))
        for s in seqs:
            for a, b in zip(s, s[1:]): self.B[a, b] += 1
        self.start = np.bincount(self.first, minlength=NTOP + 1).astype(float)
        self.uni = self.X.sum(0).astype(float)
        self.site = np.array([t['site'] for t in texts])
        self.facts = {}
        for f in ('type', 'emblem', 'material', 'period', 'size'):
            v = [t[f] for t in texts]
            if permute:
                v = list(v); rng.shuffle(v)
            levels = sorted(set(x for x in v if x))
            self.facts[f] = np.array([levels.index(x) if x else -1 for x in v])
        if permute:
            sites = list(self.site); rng.shuffle(sites); self.site = np.array(sites)
        self.md = self.site == 'Mohenjo-daro'; self.ha = self.site == 'Harappa'; self.other = ~(self.md | self.ha)
        # reuse: texts of 3-8 signs padded
        keep = [i for i, s in enumerate(seqs) if 3 <= len(s) <= 8]
        self.R = np.full((len(keep), 8), -1, dtype=np.int64)
        for j, i in enumerate(keep): self.R[j, :len(seqs[i])] = seqs[i]
        self.Rsite = self.site[keep]
        # quantity / vouchers
        def enc(lst):
            return [([self.idx.get(x, O) for x in s], min(v, 5), site) for s, v, site in lst]
        self.vou = enc(vouchers); self.qty = enc(quantity)
        if shuffle:
            for s, _, _ in self.vou + self.qty: rng.shuffle(s)
        if permute:
            for grp in (self.vou, self.qty):
                vals = [v for _, v, _ in grp]; rng.shuffle(vals)
                for k in range(len(grp)): grp[k] = (grp[k][0], vals[k], grp[k][2])


# -------------------------------------------------------------------------------------------- criteria
def nb_score(F, y, train, test):
    """multinomial NB on feature matrix F; y integer-coded (-1 missing); mean log2 gain over the prior on test rows"""
    tr = train & (y >= 0); te = test & (y >= 0)
    if te.sum() < 20 or tr.sum() < 20: return 0.0
    nc = int(y.max()) + 1
    Y = np.zeros((tr.sum(), nc), dtype=np.float32); Y[np.arange(tr.sum()), y[tr]] = 1
    cnt = Y.T @ F[tr] + 0.5                      # classes x features
    prior = Y.sum(0) + 0.5
    if (prior > 1).sum() < 2: return 0.0
    logth = np.log2(cnt / cnt.sum(1, keepdims=True)); lprior = np.log2(prior / prior.sum())
    lp = F[te] @ logth.T + lprior
    lp -= lp.max(1, keepdims=True)
    post = lp - np.log2(np.exp2(lp).sum(1, keepdims=True))
    yt = y[te]
    gain = post[np.arange(len(yt)), yt] - lprior[yt]
    return float(gain.mean())


def features(D, A, K):
    """A: assignment vector length NTOP+1 (OTHER = K). Features: category counts, first, last, length bucket."""
    M = np.zeros((NTOP + 1, K + 1), dtype=np.float32); M[np.arange(NTOP + 1), A] = 1
    cc = D.X @ M
    fi = np.zeros((len(A), K + 1), dtype=np.float32)
    F1 = M[D.first]; F2 = M[D.last]
    L = np.zeros((len(D.first), 4), dtype=np.float32); L[np.arange(len(D.first)), D.lenb] = 1
    return np.hstack([cc, F1, F2, L])


def crit_fact(D, A, K, fact, mode):
    F = features(D, A, K); y = D.facts[fact]
    if mode == 'opt':   # cross-city
        return 0.5 * (nb_score(F, y, D.md, D.ha) + nb_score(F, y, D.ha, D.md))
    return nb_score(F, y, D.md | D.ha, D.other)


def crit_mdl(D, A, K):
    M = np.zeros((NTOP + 1, K + 1)); M[np.arange(NTOP + 1), A] = 1
    Bc = M.T @ D.B @ M + 0.5
    rowtot = Bc.sum(1, keepdims=True)
    bits = -(Bc * np.log2(Bc / rowtot)).sum()
    st = M.T @ D.start + 0.5
    bits -= (st * np.log2(st / st.sum())).sum()
    catn = M.T @ D.uni
    bits -= (D.uni * np.log2((D.uni + 1e-9) / catn[A])).sum()
    return -bits / 1000.0  # maximise = minimise bits (kbits)


def crit_reuse(D, A, K):
    cat = np.where(D.R >= 0, A[np.clip(D.R, 0, None)], K + 1)
    code = (cat * ((K + 2) ** np.arange(8))).sum(1)
    u, inv = np.unique(code, return_inverse=True)
    sites = collections.defaultdict(set)
    for c, s in zip(inv, D.Rsite): sites[c].add(s)
    multi = np.array([len(sites[c]) > 1 for c in range(len(u))])
    return float(multi[inv].mean())


def crit_quantity(D, A, K, mode):
    def catseq(s): return tuple(A[x] for x in s)
    def fit_score(train, test):
        tab = collections.defaultdict(collections.Counter)
        prior = collections.Counter()
        for s, v, _ in train: tab[catseq(s)][v] += 1; prior[v] += 1
        vals = sorted(prior); tot = sum(prior.values())
        g = []
        for s, v, _ in test:
            c = tab.get(catseq(s), collections.Counter())
            p = (c[v] + 0.5) / (sum(c.values()) + 0.5 * len(vals))
            g.append(np.log2(p) - np.log2((prior[v] + 0.5) / (tot + 0.5 * len(vals))))
        return float(np.mean(g)) if g else 0.0
    if mode == 'opt':   # leave-one-site-out within vouchers+quantity pooled, by site
        pool = D.vou + D.qty
        sites = sorted(set(s for _, _, s in pool))
        sc = [fit_score([p for p in pool if p[2] != st], [p for p in pool if p[2] == st]) for st in sites]
        return float(np.mean(sc))
    return 0.5 * (fit_score(D.vou, D.qty) + fit_score(D.qty, D.vou))


def evaluate(D, A, K, crit, mode='opt'):
    kind, arg = crit
    if kind == 'fact': return crit_fact(D, A, K, arg, mode)
    if kind == 'mdl': return crit_mdl(D, A, K)
    if kind == 'reuse': return crit_reuse(D, A, K)
    if kind == 'quantity': return crit_quantity(D, A, K, mode)


# ------------------------------------------------------------------------------------------ hill-climb
def hillclimb(D, K, crit, rng, max_pass=8):
    lo, hi = max(1, NTOP // (2 * K)), 2 * NTOP // K
    A = np.array([i % K for i in range(NTOP)] + [K]); rng.shuffle(A[:NTOP])
    A = np.array(list(A[:NTOP]) + [K])
    best = evaluate(D, A, K, crit)
    sizes = np.bincount(A[:NTOP], minlength=K)
    evals = 1
    for _ in range(max_pass):
        improved = False
        order = list(range(NTOP)); rng.shuffle(order)
        for s in order:
            cur = A[s]
            if sizes[cur] - 1 < lo: continue
            cand = [k for k in range(K) if k != cur and sizes[k] + 1 <= hi]
            bestk, bestv = None, best
            for k in cand:
                A[s] = k
                v = evaluate(D, A, K, crit); evals += 1
                if v > bestv + 1e-9: bestk, bestv = k, v
            if bestk is None: A[s] = cur
            else:
                A[s] = bestk; sizes[cur] -= 1; sizes[bestk] += 1; best = bestv; improved = True
        if not improved: break
    return A, best, evals


def run_ontologies(D, specs, rng):
    out = []
    for (K, sem, crit, seed) in specs:
        r = random.Random(seed)
        A, best, ev = hillclimb(D, K, crit, r)
        held = evaluate(D, A, K, crit, mode='held')
        out.append(dict(K=K, semantics=sem, crit=crit, opt=best, held=held, A=A[:NTOP].tolist(), evals=ev))
    return out


def cograph(results, ntop):
    S = np.zeros((ntop, ntop)); n = 0
    for r in results:
        A = np.array(r['A'])
        S += (A[:, None] == A[None, :]); n += 1
    return S / max(n, 1)


def expected_same(results):
    """chance co-category rate for balanced random assignment, averaged over the ontologies"""
    e = []
    for r in results:
        sizes = np.bincount(r['A'], minlength=r['K']) / NTOP
        e.append((sizes ** 2).sum())
    return float(np.mean(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cycle', type=int, required=True); ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--seed', type=int, default=None); ap.add_argument('--level', default='seq')
    ap.add_argument('--subset', default='all'); ap.add_argument('--tag', default='')
    ap.add_argument('--crits', default='fact,quantity,reuse,mdl')
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else 1000 + a.cycle
    rng = random.Random(seed)
    texts, vouchers, quantity = load(a.level, a.subset)
    t0 = time.time()
    D = Data(texts, vouchers, quantity, random.Random(seed))
    DP = Data(texts, vouchers, quantity, random.Random(seed + 1), permute=True)
    DS = Data(texts, vouchers, quantity, random.Random(seed + 2), shuffle=True)
    crit_kinds = a.crits.split(',')
    facts = [f for f in ('type', 'emblem', 'material', 'period', 'size')
             if (D.facts[f] >= 0).sum() > 200 and len(set(D.facts[f].tolist()) - {-1}) > 1]
    specs = []
    for i in range(a.n):
        K = rng.randint(2, 6)
        sem = rng.choice([p for p in SEMANTIC_POOL if len(p) >= K]); sem = rng.sample(sem, K)
        kind = rng.choice(crit_kinds)
        if kind == 'quantity' and len(D.vou) + len(D.qty) < 30: kind = 'mdl'
        if kind == 'reuse' and len(set(D.site)) < 2: kind = 'mdl'
        crit = (kind, rng.choice(facts) if kind == 'fact' else None)
        specs.append((K, sem, crit, rng.randrange(10 ** 9)))
    real = run_ontologies(D, specs, rng)
    ctlP = run_ontologies(DP, specs, rng)
    ctlS = run_ontologies(DS, specs, rng)
    G, GP, GS = cograph(real, NTOP), cograph(ctlP, NTOP), cograph(ctlS, NTOP)
    top = D.top
    exp = expected_same(real)
    # stable pairs: real co-rate high and above both controls
    pairs = []
    for i in range(NTOP):
        for j in range(i + 1, NTOP):
            pairs.append((G[i, j], GP[i, j], GS[i, j], top[i], top[j]))
    pairs.sort(reverse=True)
    thr = 0.70
    stable = [(g, gp, gs, x, y) for g, gp, gs, x, y in pairs if g >= thr and g - max(gp, gs) >= 0.20]
    stable_any = [(g, gp, gs, x, y) for g, gp, gs, x, y in pairs if g >= thr]
    # components of the stable graph (real, above controls)
    adj = collections.defaultdict(set)
    for g, gp, gs, x, y in stable: adj[x].add(y); adj[y].add(x)
    seen = set(); comps = []
    for v in adj:
        if v in seen: continue
        stack = [v]; comp = set()
        while stack:
            u = stack.pop()
            if u in comp: continue
            comp.add(u); stack.extend(adj[u] - comp)
        seen |= comp; comps.append(sorted(comp))
    comps.sort(key=len, reverse=True)
    # compare to known classes
    def label(x):
        return ','.join(k for k, v in KNOWN.items() if x in v) or 'new'
    # criterion-level: does any real ontology beat the controls on held-out?
    bykind = collections.defaultdict(lambda: dict(real=[], P=[], S=[]))
    for r, p, s in zip(real, ctlP, ctlS):
        k = r['crit'][0] + ('/' + r['crit'][1] if r['crit'][1] else '')
        bykind[k]['real'].append(r['held']); bykind[k]['P'].append(p['held']); bykind[k]['S'].append(s['held'])
    lines = []
    tag = a.tag or ('cycle%d' % a.cycle)
    lines.append('LOOP 10 (random ontologies) %s  level=%s subset=%s n=%d seed=%d  texts=%d vouchers=%d quantity=%d  %.0fs'
                 % (tag, a.level, a.subset, a.n, seed, len(texts), len(D.vou), len(D.qty), time.time() - t0))
    lines.append('top-60 signs: %s' % top)
    lines.append('chance co-category rate (balanced random, mean over ontologies): %.3f' % exp)
    lines.append('criteria drawn: %s' % dict(collections.Counter(r['crit'][0] for r in real)))
    lines.append('')
    lines.append('HELD-OUT SCORES by criterion (bits/text gain for fact & quantity; share for reuse; -kbits for mdl). '
                 'real mean [max] vs P=facts-permuted vs S=order-shuffled, mean [max]:')
    for k, v in sorted(bykind.items()):
        f = lambda l: '%.3f [%.3f]' % (np.mean(l), np.max(l))
        beats = sum(1 for r in v['real'] if r > max(v['P'] + v['S']))
        lines.append('  %-16s n=%2d  real %s  P %s  S %s  real-ontologies-beating-max-control: %d/%d'
                     % (k, len(v['real']), f(v['real']), f(v['P']), f(v['S']), beats, len(v['real'])))
    lines.append('')
    lines.append('STABLE CO-CATEGORY PAIRS (real rate >= %.2f and >= 0.20 above both controls): %d of %d pairs'
                 % (thr, len(stable), len(pairs)))
    for g, gp, gs, x, y in stable[:60]:
        lines.append('  W%-4d W%-4d  real %.2f  P %.2f  S %.2f   [%s | %s]' % (x, y, g, gp, gs, label(x), label(y)))
    lines.append('pairs >= %.2f in real regardless of controls: %d; of these also >= %.2f in P: %d, in S: %d'
                 % (thr, len(stable_any), thr, sum(1 for p in stable_any if p[1] >= thr), sum(1 for p in stable_any if p[2] >= thr)))
    lines.append('')
    lines.append('STABLE CLASSES (connected components of the stable graph):')
    for c in comps:
        lines.append('  {%s}  -> known: %s' % (', '.join('W%d' % x for x in c), '; '.join('W%d=%s' % (x, label(x)) for x in c)))
    # known-class recovery: mean co-rate within each known class vs chance
    lines.append('')
    lines.append('KNOWN CLASSES: mean within-class co-category rate (real / P / S), pairs among top-60:')
    for k, v in KNOWN.items():
        ids = [top.index(x) for x in v if x in top]
        if len(ids) < 2: continue
        pr = [(G[i, j], GP[i, j], GS[i, j]) for ii, i in enumerate(ids) for j in ids[ii + 1:]]
        m = np.array(pr).mean(0)
        lines.append('  %-14s %2d signs %3d pairs  real %.2f  P %.2f  S %.2f  (chance %.2f)' % (k, len(ids), len(pr), m[0], m[1], m[2], exp))
    # by criterion kind: which pairs each criterion kind drives
    lines.append('')
    lines.append('PER-CRITERION co-graphs: top 8 pairs for each criterion kind (real):')
    for kind in crit_kinds:
        sub = [r for r in real if r['crit'][0] == kind]
        if len(sub) < 3: continue
        Gk = cograph(sub, NTOP)
        pk = sorted(((Gk[i, j], top[i], top[j]) for i in range(NTOP) for j in range(i + 1, NTOP)), reverse=True)[:8]
        lines.append('  %-9s n=%2d: %s' % (kind, len(sub), '  '.join('%d-%d(%.2f)' % (x, y, g) for g, x, y in pk)))
    txt = '\n'.join(lines)
    os.makedirs(os.path.join(ROOT, 'data/derived/dark'), exist_ok=True)
    open(os.path.join(ROOT, 'data/derived/dark/loop10_%s.txt' % tag), 'w').write(txt + '\n')
    json.dump(dict(top=top, G=G.tolist(), GP=GP.tolist(), GS=GS.tolist(), real=[dict(r, crit=list(r['crit'])) for r in real],
                   ctlP=[dict(r, crit=list(r['crit'])) for r in ctlP], ctlS=[dict(r, crit=list(r['crit'])) for r in ctlS],
                   stable=[list(map(float, p[:3])) + list(p[3:]) for p in stable], comps=comps, exp=exp),
              open(os.path.join(ROOT, 'data/derived/dark/loop10_%s.json' % tag), 'w'))
    print(txt)


if __name__ == '__main__':
    main()
