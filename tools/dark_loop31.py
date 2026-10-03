"""S-DARK-31: THE PARALLELOGRAM TEST. Does the Indus sign system have analogical (vector-offset) structure?

Embeddings: PPMI + truncated SVD (the closed-form skip-gram of Levy & Goldberg 2014) over left/right neighbours
(distance 1 and 2), position class, S310 frame-slot label of the sign itself, and object class. Vocabulary = the 150
commonest signs in the corpus under test. Analogies are mined with the 4-corner parallelogram rule: an ordered pair
(a,b) and a disjoint ordered pair (c,d) are LINKED iff all four 3CosAdd predictions hold
    NN(a - b + c) = d, NN(b - a + d) = c, NN(c - d + a) = b, NN(d - c + b) = a  (nearest neighbour excludes a, b, c).
Linked pairs form a pair graph; an OFFSET CLASS is a connected component holding >= 3 mutually disjoint pairs.
A weaker 2-corner rule (NN(a-b+c)=d and NN(c-d+a)=b) is reported too.
Nulls: (i) tokens shuffled within frame-slot label across the corpus (frame kept, pairing destroyed);
       (ii) a slot-grammar synthetic corpus (skeleton of slot labels kept, each slot filled from its own distribution,
            qualifiers conditional on the closer, middle an order-1 chain);
       (iii) for the calibration corpora (Ur III seal legends, Linear B lines, Proto-Elamite lines) and the Indus for
            fairness: tokens shuffled within position class (initial / medial / final).
Usage: python3 tools/dark_loop31.py embed  <seq_raw|seq_strong|seq_all> [nrep]   (cycle 1, Indus)
       python3 tools/dark_loop31.py calib  [nrep]                                  (cycle 1, controls)
Output: data/derived/dark/loop31_c1_<level>.txt and loop31_c1_<level>.json
"""
import json, sys, random, collections, math, re, csv, os, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'data/derived/dark/')
C = json.load(open(os.path.join(HERE, 'data/derived/merged-corpus-canonical.json')))
BR = json.load(open(os.path.join(HERE, 'data/derived/bridge_extended.json')))

# ---------------- S310 frame parser (copied from tools/parse_all.py via strat_adequacy.py) ----------------
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]; CLS = set(CL)
FISH = {235, 240, 233, 231, 220}
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

QUAL = {}
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

def posclass(i, n):
    if n == 1: return 'ALONE'
    return 'INIT' if i == 0 else 'FIN' if i == n - 1 else 'MED'

# ---------------- corpora ----------------
def otype(t):
    t = t.split(':')[0]
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing'}.get(t, 'other')

def load_indus(level, sites=('Mohenjo-daro', 'Harappa'), complete_only=True, use_slots=True):
    """-> list of dict(seq, ot, site, lab)"""
    global QUAL
    objs = []
    for r in C:
        s = r[level]
        if not s or len(s) < 2: continue
        if complete_only and r['complete'] != 'Y': continue
        if sites is not None and r['site'] not in sites: continue
        objs.append(dict(seq=list(s), ot=otype(r['type']), site=r['site']))
    QUAL = learn_qual([o['seq'] for o in objs])
    for o in objs:
        o['lab'] = parse(o['seq']) if use_slots else [posclass(i, len(o['seq'])) for i in range(len(o['seq']))]
    return objs

def load_im77(sites=None):
    rows = list(csv.DictReader(open(os.path.join(HERE, 'data/im77/im77_corpus_lines.csv'))))
    by = collections.defaultdict(list)
    for r in rows:
        if r['direction'] == 'single sign' and False: pass
        by[(r['text_no'], r['side'])].append(r)
    objs = []
    for k, rs in by.items():
        rs.sort(key=lambda r: int(r['line']))
        seq = []
        for r in rs:
            seq += [int(x) for x in r['signs_clean'].split() if x.isdigit()]
        if len(seq) < 2: continue
        if sites is not None and rs[0]['site'] not in sites: continue
        objs.append(dict(seq=seq, ot=rs[0]['object_type'], site=rs[0]['site'],
                         lab=[posclass(i, len(seq)) for i in range(len(seq))]))
    return objs

def load_ur3():
    legs = json.load(open(os.path.join(HERE, 'data/derived/dark/loop4_ur3_legends.json'))); seen = set(); out = []
    for lg in legs:
        flat = tuple(x for ln in lg for x in ln)
        if len(flat) < 2 or flat in seen or 'x' in flat: continue
        seen.add(flat); out.append(dict(seq=list(flat), ot='legend', site='ur3'))
    for o in out: o['lab'] = [posclass(i, len(o['seq'])) for i in range(len(o['seq']))]
    return out

def load_linb():
    out = []; seen = set()
    for l in open(os.path.join(HERE, 'other-scripts/linear-a/data/damos_items.jsonl')):
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
                seen.add(tuple(toks)); out.append(dict(seq=toks, ot='line', site='linb'))
    for o in out: o['lab'] = [posclass(i, len(o['seq'])) for i in range(len(o['seq']))]
    return out

def load_linb_syll():
    """Linear B words split into syllabic signs (sign-level control, like Ur III syllables)."""
    out = []; seen = set()
    for o in load_linb():
        s = []
        for t in o['seq']:
            if t.startswith('w:'): s += t[2:].split('-')
            else: s.append(t)
        if len(s) >= 2 and tuple(s) not in seen:
            seen.add(tuple(s)); out.append(dict(seq=s, ot='line', site='linb'))
    for o in out: o['lab'] = [posclass(i, len(o['seq'])) for i in range(len(o['seq']))]
    return out

def load_pe():
    pe = json.load(open(os.path.join(HERE, 'other-scripts/proto-elamite/data/pe_corpus.json'))); out = []; seen = set()
    for t in pe:
        for ln in t['lines']:
            s = [x.split('~')[0] for x in ln['signs'] if x != 'x' and not x.startswith('x')]
            if ln.get('lacuna'): continue
            if ln['numerals']: s = s + ['N:' + ln['numerals'][0][1]]
            if len(s) >= 2 and tuple(s) not in seen:
                seen.add(tuple(s)); out.append(dict(seq=s, ot=t.get('object_type', 'tablet'), site='pe'))
    for o in out: o['lab'] = [posclass(i, len(o['seq'])) for i in range(len(o['seq']))]
    return out

# ---------------- embeddings ----------------
def embed(objs, top=150, dim=30, ctx_min=3, use_lab=True, use_ot=True, vocab=None, alpha=0.75, return_ctx=False):
    """PPMI + SVD embeddings for the `top` commonest signs (or a given vocab). Returns (vocab list, matrix rows L2-normalised)."""
    freq = collections.Counter(x for o in objs for x in o['seq'])
    if vocab is None: vocab = [s for s, _ in freq.most_common(top)]
    V = {s: i for i, s in enumerate(vocab)}
    ctxsigns = {s for s, n in freq.items() if n >= ctx_min}
    feats = collections.Counter(); rows = collections.defaultdict(collections.Counter)
    for o in objs:
        s = o['seq']; n = len(s)
        for i, x in enumerate(s):
            if x not in V: continue
            f = []
            if i > 0 and s[i - 1] in ctxsigns: f.append(('L1', s[i - 1]))
            if i < n - 1 and s[i + 1] in ctxsigns: f.append(('R1', s[i + 1]))
            if i > 1 and s[i - 2] in ctxsigns: f.append(('L2', s[i - 2]))
            if i < n - 2 and s[i + 2] in ctxsigns: f.append(('R2', s[i + 2]))
            if i == 0: f.append(('P', 'INIT'))
            if i == n - 1: f.append(('P', 'FIN'))
            if 0 < i < n - 1: f.append(('P', 'MED'))
            if use_lab: f.append(('S', o['lab'][i]))
            if use_ot: f.append(('O', o['ot']))
            for ff in f:
                rows[V[x]][ff] += 1; feats[ff] += 1
    F = [f for f, n in feats.items() if n >= 2]
    FI = {f: i for i, f in enumerate(F)}
    M = np.zeros((len(vocab), len(F)), dtype=np.float64)
    for r, cnt in rows.items():
        for f, n in cnt.items():
            if f in FI: M[r, FI[f]] += n
    tot = M.sum(); pr = M.sum(1, keepdims=True) / tot
    pc = M.sum(0, keepdims=True) ** alpha; pc = pc / pc.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log((M / tot) / (pr * pc))
    pmi[~np.isfinite(pmi)] = 0; pmi[pmi < 0] = 0
    k = min(dim, min(pmi.shape) - 1)
    U, S, Vt = np.linalg.svd(pmi, full_matrices=False)
    E = U[:, :k] * np.sqrt(S[:k])
    nrm = np.linalg.norm(E, axis=1, keepdims=True); nrm[nrm == 0] = 1
    E = (E / nrm).astype(np.float32)
    if return_ctx: return vocab, E, pmi, F
    return vocab, E

# ---------------- analogy mining ----------------
TAU_SIM = 0.5     # a and b (and c and d) must not be near-synonyms: cos < TAU_SIM, so the offset is not ~0
TAU_OFF = 0.5     # the two offsets must point the same way: cos(a-b, c-d) >= TAU_OFF

def nn_tensor(E):
    """NN[a,b,c] = argmax_x cos(E_a - E_b + E_c, E_x) over x not in {a,b,c}; -1 where a,b,c not distinct."""
    n = E.shape[0]
    NN = np.full((n, n, n), -1, dtype=np.int16)
    idx = np.arange(n)
    for a in range(n):
        Q = E[a][None, None, :] - E[:, None, :] + E[None, :, :]          # b, c, k
        Q = Q.reshape(n * n, -1)
        qn = np.linalg.norm(Q, axis=1, keepdims=True); qn[qn == 0] = 1
        S = ((Q / qn) @ E.T).reshape(n, n, n)
        S[:, :, a] = -2; S[idx, :, idx] = -2; S[:, idx, idx] = -2
        NN[a] = S.argmax(2)
        NN[a, a, :] = -1; NN[a, :, a] = -1; NN[a, idx, idx] = -1
    return NN

def mine(E, rule=4, tau_sim=TAU_SIM, tau_off=TAU_OFF, nontrivial=True, NN=None):
    """Linked pairs ((a,b),(d,c)) with a - b = d - c: a,b,c,d distinct, a < c; cos(a,b) < tau_sim and cos(c,d) < tau_sim;
    NN(a-b+c) = d and NN(c-d+a) = b (rule 2), plus NN(b-a+d) = c and NN(d-c+b) = a (rule 4);
    d != NN1(c) and b != NN1(a) (the offset must beat plain similarity); cos(a-b, c-d) >= tau_off."""
    n = E.shape[0]
    if NN is None: NN = nn_tensor(E)
    cs = E @ E.T
    nn1 = (cs - 2 * np.eye(n, dtype=np.float32)).argmax(1)
    A, B, Cc = np.indices((n, n, n), dtype=np.int32)
    D = NN.astype(np.int32)
    ok = (D >= 0) & (A < Cc) & (D != A) & (D != B) & (D != Cc)
    ok &= cs[A, B] < tau_sim
    Dc = np.where(ok, D, 0)
    ok &= cs[Cc, Dc] < tau_sim
    ok &= NN[Cc, Dc, A] == B
    if rule == 4:
        ok &= (NN[B, A, Dc] == Cc) & (NN[Dc, Cc, B] == A)
    if nontrivial:
        ok &= (nn1[Cc] != Dc) & (nn1[A] != B)
    a, b, c = np.nonzero(ok); d = D[a, b, c]
    O1 = E[a] - E[b]; O2 = E[d] - E[c]          # a - b + c = d  <=>  a - b = d - c : the parallel pair is (d, c)
    agree = (O1 * O2).sum(1) / (np.linalg.norm(O1, axis=1) * np.linalg.norm(O2, axis=1) + 1e-9)
    keep = agree >= tau_off
    links = [((int(x), int(y)), (int(w), int(z))) for x, y, z, w in zip(a[keep], b[keep], c[keep], d[keep])]
    return links, NN, None

def components(links):
    adj = collections.defaultdict(set)
    for p, q in links:
        adj[p].add(q); adj[q].add(p)
    seen = set(); comps = []
    for v in adj:
        if v in seen: continue
        st = [v]; comp = []
        seen.add(v)
        while st:
            u = st.pop(); comp.append(u)
            for w in adj[u]:
                if w not in seen: seen.add(w); st.append(w)
        comps.append(comp)
    return comps

def max_disjoint(pairs):
    """greedy size of a set of mutually disjoint pairs (exact for small sets)."""
    pairs = list(pairs)
    if len(pairs) <= 12:
        best = 0
        def rec(i, used, k):
            nonlocal best
            if k + (len(pairs) - i) <= best: return
            if i == len(pairs): best = max(best, k); return
            a, b = pairs[i]
            if a not in used and b not in used:
                rec(i + 1, used | {a, b}, k + 1)
            rec(i + 1, used, k)
        rec(0, frozenset(), 0); return best
    used = set(); k = 0
    for a, b in sorted(pairs):
        if a not in used and b not in used: used |= {a, b}; k += 1
    return k

def offset_stats(E, links, min_disjoint=3, min_cos=0.5):
    """Offset classes by greedy stars: the pair with most links seeds a class = seed + every pair linked to it (all of
    them share the seed's offset direction, cos >= TAU_OFF); the class counts if it holds >= min_disjoint mutually disjoint
    pairs and the mean pairwise offset cosine over all its members is >= min_cos. Members are removed and the next seed taken.
    Also reports sharper counts: classes with mean cos >= 0.7, and with >= 5 disjoint pairs."""
    import heapq
    adj = collections.defaultdict(set)
    for p, q in links:
        adj[p].add(q); adj[q].add(p)
    deg = {p: len(v) for p, v in adj.items()}
    heap = [(-d, p) for p, d in deg.items()]; heapq.heapify(heap)
    alive = set(adj); classes = []; stars = 0; sharp = 0; big = 0; sharp_big = 0
    while heap:
        d, seed = heapq.heappop(heap)
        if seed not in alive or -d != deg[seed]: continue
        mem = [seed] + sorted(adj[seed] & alive)
        if len(mem) < 2: alive.discard(seed); continue
        O = np.array([E[a] - E[b] for a, b in mem]); On = O / np.linalg.norm(O, axis=1, keepdims=True)
        cs = On @ On.T; m = len(mem)
        meancos = float((cs.sum() - m) / (m * (m - 1)))
        md = max_disjoint(mem)
        stars += 1
        if md >= min_disjoint and meancos >= min_cos:
            classes.append(dict(pairs=mem, n_pairs=m, disjoint=md, meancos=meancos))
            if meancos >= 0.7: sharp += 1
            if md >= 5: big += 1
            if meancos >= 0.7 and md >= 5: sharp_big += 1
        for x in mem: alive.discard(x)
        for x in mem:
            for y in adj[x]:
                if y in alive:
                    deg[y] -= 1; heapq.heappush(heap, (-deg[y], y))
    comps = components(links)
    return dict(n_links=len(links), n_comps=len(comps), n_stars=stars, n_classes=len(classes), n_sharp=sharp, n_big=big,
                n_sharp_big=sharp_big, n_pairs_linked=len(adj), n_pairs_in_classes=sum(c['n_pairs'] for c in classes),
                classes=sorted(classes, key=lambda c: (-c['disjoint'], -c['meancos'])))

# ---------------- nulls ----------------
def shuffle_within(objs, rnd, key='lab'):
    pools = collections.defaultdict(list)
    for o in objs:
        for x, l in zip(o['seq'], o[key]): pools[l].append(x)
    for l in pools: rnd.shuffle(pools[l])
    idx = collections.Counter(); out = []
    for o in objs:
        s = []
        for l in o[key]:
            s.append(pools[l][idx[l]]); idx[l] += 1
        out.append(dict(seq=s, ot=o['ot'], site=o['site'], lab=list(o[key])))
    return out

class SlotGrammar:
    """skeleton of slot labels kept per text; OPENER/MARKER/SUFFIX/COUNT from slot unigrams; CLOSER from its unigram;
    TITLE conditional on the closer (right-to-left chain); middle (NAME) an order-1 chain with the slot's own start."""
    def __init__(self, objs, rnd):
        self.rnd = rnd
        self.uni = collections.defaultdict(collections.Counter)
        self.title_given = collections.defaultdict(collections.Counter)   # (closer or previous title) -> title sign
        self.name_start = collections.Counter(); self.name_tr = collections.defaultdict(collections.Counter)
        self.skel = []
        for o in objs:
            s, L = o['seq'], o['lab']; self.skel.append((L, o['ot'], o['site']))
            for x, l in zip(s, L): self.uni[l][x] += 1
            closer = None
            for i in range(len(s) - 1, -1, -1):
                if L[i] == 'CLOSER': closer = s[i]
                elif L[i] == 'TITLE':
                    self.title_given[closer][s[i]] += 1; closer = s[i]
            prev = None
            for x, l in zip(s, L):
                if l in ('NAME', 'COUNT'):
                    if prev is None: self.name_start[x] += 1
                    else: self.name_tr[prev][x] += 1
                    prev = x
                else: prev = None if l != 'MARKER' else None
    def draw(self, cnt):
        ks = list(cnt.keys()); ws = list(cnt.values()); return self.rnd.choices(ks, ws)[0]
    def generate(self):
        out = []
        for L, ot, site in self.skel:
            s = [None] * len(L)
            closer = None
            for i in range(len(L) - 1, -1, -1):
                if L[i] == 'CLOSER': s[i] = self.draw(self.uni['CLOSER']); closer = s[i]
                elif L[i] == 'TITLE':
                    cnt = self.title_given.get(closer) or self.uni['TITLE']; s[i] = self.draw(cnt); closer = s[i]
            prev = None
            for i, l in enumerate(L):
                if s[i] is not None: prev = None; continue
                if l in ('NAME', 'COUNT'):
                    cnt = self.name_tr.get(prev) if prev is not None else None
                    if not cnt or sum(cnt.values()) < 3: cnt = self.name_start if prev is None else self.uni[l]
                    s[i] = self.draw(cnt); prev = s[i]
                else:
                    s[i] = self.draw(self.uni[l]); prev = None
            out.append(dict(seq=s, ot=ot, site=site, lab=list(L)))
        return out

def summarize(E, vocab, links, label=''):
    st = offset_stats(E, links)
    st['vocab'] = vocab
    return st

def run_real_and_nulls(objs, tag, nrep, rnd, nulls=('slot', 'grammar', 'pos'), top=150, dim=30, log=print, nrep_by=None):
    t0 = time.time()
    vocab, E = embed(objs, top=top, dim=dim)
    res = {}
    NN = nn_tensor(E)
    for rule in (4, 2):
        links, _, _ = mine(E, rule, NN=NN)
        st = offset_stats(E, links); st['vocab'] = vocab
        res[f'real_rule{rule}'] = st
        log(f'[{tag}] REAL rule{rule}: links {st["n_links"]}, pairs linked {st["n_pairs_linked"]}, stars {st["n_stars"]}, offset classes (>=3 disjoint pairs, mean offset cos >= 0.5) {st["n_classes"]} holding {st["n_pairs_in_classes"]} pairs; sharp (cos >= 0.7) {st["n_sharp"]}, big (>= 5 disjoint) {st["n_big"]}, sharp+big {st["n_sharp_big"]}  ({time.time()-t0:.1f}s)')
    nullres = {}
    for nt in nulls:
        stats = collections.defaultdict(list)
        G = SlotGrammar(objs, rnd) if nt == 'grammar' else None
        nrep_t = (nrep_by or {}).get(nt, nrep)
        for r in range(nrep_t):
            if nt == 'slot': syn = shuffle_within(objs, rnd, 'lab')
            elif nt == 'pos':
                for o in objs: o['pos'] = [posclass(i, len(o['seq'])) for i in range(len(o['seq']))]
                syn = shuffle_within(objs, rnd, 'pos')
                for o in syn: o['lab'] = o['seq'] and [posclass(i, len(o['seq'])) for i in range(len(o['seq']))]
            else: syn = G.generate()
            v2, E2 = embed(syn, top=top, dim=dim)
            NN2 = nn_tensor(E2)
            for rule in (4, 2):
                links2, _, _ = mine(E2, rule, NN=NN2); st2 = offset_stats(E2, links2)
                for k in ('n_links', 'n_comps', 'n_classes', 'n_sharp', 'n_big', 'n_sharp_big', 'n_pairs_linked', 'n_pairs_in_classes'): stats[(rule, k)].append(st2[k])
        nullres[nt] = {f'rule{r}_{k}': v for (r, k), v in stats.items()}
        for rule in (4, 2):
            parts = []
            for k in ('n_links', 'n_classes', 'n_sharp', 'n_big', 'n_sharp_big'):
                a = stats[(rule, k)]; realv = res[f'real_rule{rule}'][k]
                p_hi = (sum(1 for x in a if x >= realv) + 1) / (len(a) + 1); p_lo = (sum(1 for x in a if x <= realv) + 1) / (len(a) + 1)
                parts.append(f'{k} real {realv} vs null median {np.median(a):.0f} [5-95% {np.percentile(a,5):.0f}-{np.percentile(a,95):.0f}] P(>=) {p_hi:.3f} P(<=) {p_lo:.3f}')
            log(f'[{tag}] NULL {nt} rule{rule} ({nrep_t}x): ' + '; '.join(parts) + f'  ({time.time()-t0:.0f}s)')
    res['nulls'] = nullres
    return res, vocab, E

def fmt_sign(x, corpus='indus'):
    if corpus != 'indus': return str(x)
    m = BR.get(str(x)); return f'W{x}(M{"/".join(map(str,m))})' if m else f'W{x}'

def describe_classes(st, vocab, corpus='indus', log=print, maxc=40):
    for ci, c in enumerate(st['classes'][:maxc]):
        prs = ', '.join(f'{fmt_sign(vocab[a],corpus)}:{fmt_sign(vocab[b],corpus)}' for a, b in c['pairs'][:12])
        log(f'  class {ci+1}: {c["n_pairs"]} pairs, {c["disjoint"]} disjoint, offset cos {c["meancos"]:.2f}: {prs}{" ..." if c["n_pairs"]>12 else ""}')

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'embed':
        LV = sys.argv[2]; NREP = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        rnd = random.Random(31)
        objs = load_indus(LV)
        fo = open(OUT + f'loop31_c1_{LV}.txt', 'w')
        def log(*a):
            s = ' '.join(str(x) for x in a); print(s, flush=True); fo.write(s + '\n'); fo.flush()
        log(f'== S-DARK-31 cycle 1, level {LV}: Indus Mohenjo-daro + Harappa complete texts >= 2 signs: {len(objs)} objects, '
            f'{sum(len(o["seq"]) for o in objs)} tokens; embeddings PPMI+SVD dim 30 over L1/R1/L2/R2/position/slot/object; vocab 150; nulls {NREP}x')
        res, vocab, E = run_real_and_nulls(objs, LV, NREP, rnd, log=log, nrep_by={'pos': max(50, NREP // 2)})
        for rule in (4, 2):
            log(f'-- real offset classes, rule {rule}:')
            describe_classes(res[f'real_rule{rule}'], vocab, log=log)
        # also dump the full link list for cycle 2
        links4, NN, SC = mine(E, 4)
        res['links_rule4'] = [[vocab[a], vocab[b], vocab[c], vocab[d]] for (a, b), (c, d) in links4]
        links2, _, _ = mine(E, 2, NN=NN)
        res['links_rule2'] = [[vocab[a], vocab[b], vocab[c], vocab[d]] for (a, b), (c, d) in links2]
        for k in ('real_rule4', 'real_rule2'):
            res[k]['classes'] = [dict(c, pairs=[[vocab[a], vocab[b]] for a, b in c['pairs']]) for c in res[k]['classes']]
        np.save(OUT + f'loop31_c1_{LV}_E.npy', E)
        json.dump(res, open(OUT + f'loop31_c1_{LV}.json', 'w'))
        log('done')
    elif mode == 'calib':
        NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        rnd = random.Random(32)
        fo = open(OUT + 'loop31_c1_calib.txt', 'w')
        def log(*a):
            s = ' '.join(str(x) for x in a); print(s, flush=True); fo.write(s + '\n'); fo.flush()
        allres = {}
        corpora = [('Ur III legends (syllables)', load_ur3), ('Linear B lines (words+ideograms)', load_linb),
                   ('Linear B lines (syllabic signs)', load_linb_syll), ('Proto-Elamite lines', load_pe)]
        for name, loader in corpora:
            objs = loader()
            if name.startswith('Ur III'):
                # size-match to the Indus (about 4,000 texts) by random subsample
                rnd.shuffle(objs); objs = objs[:4400]
            log(f'== calibration corpus {name}: {len(objs)} texts, {sum(len(o["seq"]) for o in objs)} tokens, '
                f'{len({x for o in objs for x in o["seq"]})} distinct signs; nulls {NREP}x (shuffle within position class)')
            res, vocab, E = run_real_and_nulls(objs, name, NREP, rnd, nulls=('pos',), log=log)
            for rule in (4, 2):
                log(f'-- {name} offset classes rule {rule}:')
                describe_classes(res[f'real_rule{rule}'], vocab, corpus='x', log=log, maxc=15)
            for k in ('real_rule4', 'real_rule2'):
                res[k]['classes'] = [dict(c, pairs=[[str(vocab[a]), str(vocab[b])] for a, b in c['pairs']]) for c in res[k]['classes']]
                res[k]['vocab'] = [str(v) for v in vocab]
            allres[name] = res
        # Indus with position-only context (no slot labels, same null) for a fair comparison
        objs = load_indus('seq_raw', use_slots=False)
        log(f'== Indus seq_raw MD+H with position-only labels (no S310 slots): {len(objs)} texts')
        res, vocab, E = run_real_and_nulls(objs, 'indus-pos', NREP, rnd, nulls=('pos',), log=log)
        for k in ('real_rule4', 'real_rule2'):
            res[k]['classes'] = [dict(c, pairs=[[vocab[a], vocab[b]] for a, b in c['pairs']]) for c in res[k]['classes']]
        allres['indus_pos_only'] = res
        json.dump(allres, open(OUT + 'loop31_c1_calib.json', 'w'))
        log('done')
