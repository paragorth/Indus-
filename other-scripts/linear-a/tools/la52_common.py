#!/usr/bin/env python3
"""LA-52 shared code: COPY IT A THOUSAND TIMES AND SEE WHAT REFUSES TO DIE (iterated learning).

A chain of simulated learners each learns the corpus from the previous generation's output and
writes a new corpus of the same size for the next generation. Learner types (drawn at random per
generation): back-off n-gram (order 1-4, optional site conditioning), slot-template grammar,
topic-clustered bigram, small recurrent neural net (GRU). Every generation has a random
bottleneck (share of documents seen), copy noise and a memory limit (rare words forgotten and
re-spelled from a sign model).

For every relation instance f (la46 vocabulary + logogram order + long-range word->logogram) we
track its EXCESS: the number of documents containing it minus the number in a kind-preserving
shuffle of the same corpus. Retention d_f = (e_real(G) - e_shufstart(G)) / (e_real(0) - e_shufstart(0)):
the share of the real corpus's excess above the learners' own attractor that survives G
generations. Low d = structure the writing system alone cannot keep = candidates for structure
held in place by the recorded world.

doc = {'id', 'site', 'toks': [('W', str) | ('L', str) | ('N', float) | ('NL',)]}
"""
import json, os, re, sys, math, random, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import la46_common as C46  # corpora loaders and the relation vocabulary (read only)

D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la52_ckpt')
os.makedirs(CK, exist_ok=True)
seed = C46.seed
nbin = C46.nbin


# ------------------------------------------------------------------ corpora
def merge_sides(docs):
    """Join the a/b sides of one tablet into one document (la46 cycle-3 lesson)."""
    out, idx = [], {}
    for d in docs:
        m = re.match(r'^(.*\d)([ab])$', d['id'])
        key = m.group(1) if m else d['id']
        if key in idx:
            o = out[idx[key]]
            o['toks'] = o['toks'] + [('NL',)] + list(d['toks'])
        else:
            idx[key] = len(out)
            out.append({'id': key, 'site': d['site'], 'toks': list(d['toks'])})
    return out


def la_docs():
    return merge_sides(C46.la_docs())


def lb_docs(target, rng):
    return C46.sample_like(C46.lb_docs_all(), target, rng)


def ur_docs(target, rng):
    return C46.sample_like(C46.ur3_docs_all(), target, rng)


def ntok(docs):
    return C46.ntok(docs)


# ------------------------------------------------------------------ features
def relations(doc):
    """la46 relations + LO (ordered logogram pair) + WR (word -> logogram 3+ tokens later)."""
    R = C46.relations(doc)
    T = [t for t in doc['toks'] if t[0] != 'NL']
    logos = [(i, t[1]) for i, t in enumerate(T) if t[0] == 'L']
    seen = set()
    for a in range(len(logos)):
        for b in range(a + 1, len(logos)):
            if logos[a][1] != logos[b][1]:
                k = (logos[a][1], logos[b][1])
                if k not in seen:
                    seen.add(k); R.add(('LO',) + k)
    for i, t in enumerate(T):
        if t[0] == 'W':
            for j, l in logos:
                if j >= i + 3:
                    R.add(('WR', t[1], l))
    return R


def counts(docs):
    c = collections.Counter()
    for d in docs:
        c.update(relations(d))
    return c


def kind_shuffle(docs, rng):
    """Keep every document's skeleton (kinds, line breaks, site); permute tokens of each kind
    across the whole corpus."""
    pools = collections.defaultdict(list)
    for d in docs:
        for t in d['toks']:
            if t[0] != 'NL':
                pools[t[0]].append(t)
    for k in pools:
        rng.shuffle(pools[k])
    ptr = collections.Counter()
    out = []
    for d in docs:
        nt = []
        for t in d['toks']:
            if t[0] == 'NL':
                nt.append(t)
            else:
                nt.append(pools[t[0]][ptr[t[0]]]); ptr[t[0]] += 1
        out.append({'id': d['id'], 'site': d['site'], 'toks': nt})
    return out


def excess(docs, rng, nshuf=1):
    c = counts(docs)
    b = collections.Counter()
    for _ in range(nshuf):
        b.update(counts(kind_shuffle(docs, rng)))
    return c, {k: v / nshuf for k, v in b.items()}


# ------------------------------------------------------------------ symbols
def sym(t):
    return C46.sym(t)


class Spell:
    """Sign-bigram model used to re-spell forgotten words (memory limit)."""
    def __init__(self, words, rng):
        self.rng = rng
        self.big = collections.defaultdict(collections.Counter)
        self.lens = []
        for w in words:
            s = w.split('-')
            self.lens.append(len(s))
            prev = '^'
            for x in s:
                self.big[prev][x] += 1; prev = x
        self.cache = {k: (list(v.keys()), np.array(list(v.values()), float)) for k, v in self.big.items()}

    def word(self):
        if not self.lens:
            return 'X'
        n = self.rng.choice(self.lens)
        prev, out = '^', []
        for _ in range(n):
            if prev not in self.cache:
                prev = '^'
            ks, ps = self.cache[prev]
            x = ks[int(self.rng.choices(range(len(ks)), weights=ps)[0])]
            out.append(x); prev = x
        return '-'.join(out)


def _choice(rng, counter):
    ks = list(counter.keys()); ws = list(counter.values())
    return rng.choices(ks, weights=ws)[0]


class Base:
    """Shared bottleneck / memory / noise / decoding machinery."""
    def __init__(self, rng, p):
        self.rng = rng; self.p = p

    def prepare(self, docs):
        rng, p = self.rng, self.p
        n = max(5, int(round(p['bottleneck'] * len(docs))))
        sample = rng.sample(docs, min(n, len(docs)))
        uni = collections.Counter()
        binvals = collections.defaultdict(list)
        sites = []
        words = []
        for d in sample:
            sites.append(d['site'])
            for t in d['toks']:
                if t[0] == 'NL':
                    continue
                uni[sym(t)] += 1
                if t[0] == 'N':
                    binvals[nbin(t[1])].append(t[1])
                if t[0] == 'W':
                    words.append(t[1])
        keep = set(s for s, _ in uni.most_common(p['memory']))
        self.keep = keep
        self.binvals = binvals
        self.sites = sites
        self.spell = Spell(words, rng)
        self.uni_kind = collections.defaultdict(collections.Counter)
        for s, c in uni.items():
            self.uni_kind[s[0]][s] += c
        seqs = []
        for d in sample:
            seq = []
            for t in d['toks']:
                s = sym(t)
                if s != 'NL' and s not in keep:
                    s = 'U|' + s[0]
                seq.append(s)
            seqs.append((d['site'], seq))
        return seqs

    def decode(self, site, seq):
        rng, p = self.rng, self.p
        toks = []
        for s in seq:
            if s == 'NL':
                if toks and toks[-1][0] != 'NL':
                    toks.append(('NL',))
                continue
            if rng.random() < p['noise'] and s[0] in self.uni_kind:
                s = _choice(rng, self.uni_kind[s[0]])
            if s.startswith('U|'):
                k = s[2]
                if k == 'W':
                    toks.append(('W', self.spell.word()))
                    continue
                s = _choice(rng, self.uni_kind[k]) if self.uni_kind[k] else None
                if s is None:
                    continue
            k, v = s.split('|', 1)
            if k == 'W':
                toks.append(('W', v))
            elif k == 'L':
                toks.append(('L', v))
            elif k == 'N':
                vals = self.binvals.get(v)
                toks.append(('N', rng.choice(vals) if vals else float(2 ** int(v)) if v != 'f' else 0.5))
        while toks and toks[-1][0] == 'NL':
            toks.pop()
        return {'id': 'g', 'site': site, 'toks': toks}


class NGram(Base):
    name = 'ngram'

    def fit(self, docs):
        seqs = self.prepare(docs)
        k = self.p['order']
        self.tab = [collections.defaultdict(collections.Counter) for _ in range(k)]
        self.maxlen = max(len(s) for _, s in seqs) + 5
        self.site_seqs = seqs
        for site, seq in seqs:
            ctx0 = ['<S:' + site + '>'] if self.p['site'] else ['<S>']
            full = ctx0 * k + seq + ['</S>']
            for i in range(k, len(full)):
                for o in range(k):
                    ctx = tuple(full[i - o:i]) if o else ()
                    self.tab[o][ctx][full[i]] += 1
        return self

    def sample(self, n):
        rng, k = self.rng, self.p['order']
        out = []
        for _ in range(n):
            site = rng.choice(self.sites)
            hist = (['<S:' + site + '>'] if self.p['site'] else ['<S>']) * k
            if self.p['site'] and not self.tab[min(1, k - 1)].get(tuple(hist[-1:])):
                hist = ['<S>'] * k
            seq = []
            for _ in range(self.maxlen):
                o = k - 1
                while o > 0:
                    ctx = tuple(hist[len(hist) - o:])
                    if ctx in self.tab[o] and rng.random() > self.p['backoff']:
                        break
                    o -= 1
                ctx = tuple(hist[len(hist) - o:]) if o else ()
                s = _choice(rng, self.tab[o][ctx])
                if s == '</S>':
                    break
                if s.startswith('<S'):
                    continue
                seq.append(s); hist.append(s)
            out.append(self.decode(site, seq))
        return out


class Template(Base):
    """Copies whole document skeletons; fills each slot from P(symbol | kind, slot bucket)."""
    name = 'template'

    def fit(self, docs):
        seqs = self.prepare(docs)
        self.skel = []
        self.slot = collections.defaultdict(collections.Counter)
        for site, seq in seqs:
            kinds = [s if s == 'NL' else s[0] for s in seq]
            self.skel.append((site, kinds))
            m = max(1, sum(k != 'NL' for k in kinds))
            j = 0
            for s, kk in zip(seq, kinds):
                if kk == 'NL':
                    continue
                self.slot[(kk, self._b(j, m))][s] += 1; j += 1
        return self

    def _b(self, j, m):
        if j == 0:
            return 'first'
        if j == m - 1:
            return 'last'
        return int(self.p['buckets'] * j / m)

    def sample(self, n):
        out = []
        for _ in range(n):
            site, kinds = self.rng.choice(self.skel)
            m = max(1, sum(k != 'NL' for k in kinds))
            seq, j = [], 0
            for kk in kinds:
                if kk == 'NL':
                    seq.append('NL'); continue
                c = self.slot.get((kk, self._b(j, m)))
                if not c:
                    c = collections.Counter({s: 1 for s in self.keep if s[0] == kk}) or collections.Counter({'U|' + kk: 1})
                seq.append(_choice(self.rng, c)); j += 1
            out.append(self.decode(site, seq))
        return out


class Topic(Base):
    """Documents clustered by their word/logogram bag (K-means on binary bags); one bigram per
    cluster, backed off to the global bigram."""
    name = 'topic'

    def fit(self, docs):
        seqs = self.prepare(docs)
        vocab = {}
        rows = []
        for _, seq in seqs:
            r = set()
            for s in seq:
                if s[0] in 'WL' and not s.startswith('U|'):
                    r.add(vocab.setdefault(s, len(vocab)))
            rows.append(r)
        K = min(self.p['K'], len(seqs))
        X = np.zeros((len(seqs), max(1, len(vocab))))
        for i, r in enumerate(rows):
            for j in r:
                X[i, j] = 1
        X /= np.maximum(1, np.linalg.norm(X, axis=1, keepdims=True))
        nrng = np.random.default_rng(self.rng.randrange(2 ** 31))
        cent = X[nrng.choice(len(X), K, replace=False)]
        for _ in range(8):
            lab = np.argmax(X @ cent.T, axis=1)
            for k in range(K):
                if (lab == k).any():
                    cent[k] = X[lab == k].mean(0)
        self.lab = lab
        self.big = [collections.defaultdict(collections.Counter) for _ in range(K)]
        self.gbig = collections.defaultdict(collections.Counter)
        self.maxlen = max(len(s) for _, s in seqs) + 5
        self.members = collections.defaultdict(list)
        for i, (site, seq) in enumerate(seqs):
            k = int(lab[i]); self.members[k].append(site)
            full = ['<S>'] + seq + ['</S>']
            for a, b in zip(full, full[1:]):
                self.big[k][a][b] += 1; self.gbig[a][b] += 1
        self.ks = list(self.members.keys())
        self.kw = [len(self.members[k]) for k in self.ks]
        return self

    def sample(self, n):
        rng = self.rng
        out = []
        for _ in range(n):
            k = rng.choices(self.ks, weights=self.kw)[0]
            site = rng.choice(self.members[k])
            prev, seq = '<S>', []
            for _ in range(self.maxlen):
                tab = self.big[k] if (prev in self.big[k] and rng.random() > self.p['backoff']) else self.gbig
                s = _choice(rng, tab[prev])
                if s == '</S>':
                    break
                seq.append(s); prev = s
            out.append(self.decode(site, seq))
        return out


class Neural(Base):
    """Small GRU language model over symbols (site token first), trained a few epochs."""
    name = 'neural'

    def fit(self, docs):
        import torch
        torch.set_num_threads(1)
        seqs = self.prepare(docs)
        voc = {'<S>': 0, '</S>': 1}
        for site, seq in seqs:
            voc.setdefault('<S:' + site + '>', len(voc))
            for s in seq:
                voc.setdefault(s, len(voc))
        self.voc = voc; self.inv = {v: k for k, v in voc.items()}
        V = len(voc); H = self.p['hidden']
        torch.manual_seed(self.rng.randrange(2 ** 31))
        self.emb = torch.nn.Embedding(V, H)
        self.gru = torch.nn.GRU(H, H, batch_first=True)
        self.out = torch.nn.Linear(H, V)
        params = list(self.emb.parameters()) + list(self.gru.parameters()) + list(self.out.parameters())
        opt = torch.optim.Adam(params, lr=self.p['lr'])
        data = []
        for site, seq in seqs:
            ids = [voc['<S:' + site + '>']] + [voc[s] for s in seq] + [1]
            data.append(ids[:160])
        self.maxlen = max(len(x) for x in data) + 3
        L = max(len(x) for x in data)
        X = torch.zeros(len(data), L, dtype=torch.long)
        M = torch.zeros(len(data), L - 1)
        for i, ids in enumerate(data):
            X[i, :len(ids)] = torch.tensor(ids)
            M[i, :len(ids) - 1] = 1
        for ep in range(self.p['epochs']):
            perm = torch.randperm(len(data))
            for b0 in range(0, len(data), 32):
                idx = perm[b0:b0 + 32]
                xb = X[idx]; mb = M[idx]
                h, _ = self.gru(self.emb(xb[:, :-1]))
                lo = self.out(h)
                ll = torch.nn.functional.cross_entropy(lo.reshape(-1, V), xb[:, 1:].reshape(-1), reduction='none')
                loss = (ll * mb.reshape(-1)).sum() / mb.sum()
                opt.zero_grad(); loss.backward(); opt.step()
        self.site_ids = [voc['<S:' + s + '>'] for s in self.sites]
        return self

    def sample(self, n):
        import torch
        out = []
        with torch.no_grad():
            starts = [self.rng.choice(self.site_ids) for _ in range(n)]
            x = torch.tensor(starts).unsqueeze(1)
            h = None
            seqs = [[] for _ in range(n)]
            alive = np.ones(n, bool)
            g = torch.Generator().manual_seed(self.rng.randrange(2 ** 31))
            for _ in range(self.maxlen):
                o, h = self.gru(self.emb(x), h)
                pr = torch.softmax(self.out(o[:, -1]) / self.p['temp'], -1)
                nx = torch.multinomial(pr, 1, generator=g)
                for i in range(n):
                    if alive[i]:
                        t = int(nx[i])
                        if t == 1:
                            alive[i] = False
                        elif not self.inv[t].startswith('<S'):
                            seqs[i].append(self.inv[t])
                if not alive.any():
                    break
                x = nx
        for i in range(n):
            site = self.inv[starts[i]][3:-1]
            out.append(self.decode(site, seqs[i]))
        return out


TYPES = ['ngram', 'template', 'topic', 'neural']
REGIME = {'mode': os.environ.get('LA52_REGIME', 'harsh')}


def random_learner(rng, kind=None, weights=(0.4, 0.15, 0.25, 0.2)):
    kind = kind or rng.choices(TYPES, weights=weights)[0]
    if REGIME['mode'] == 'gentle':
        p = {'bottleneck': rng.uniform(0.75, 1.0), 'noise': rng.uniform(0.0, 0.02),
             'memory': rng.choice([600, 100000, 100000]), 'backoff': rng.uniform(0.0, 0.1)}
    else:
        p = {'bottleneck': rng.uniform(0.3, 1.0), 'noise': rng.uniform(0.0, 0.08),
             'memory': rng.choice([150, 300, 600, 100000]), 'backoff': rng.uniform(0.0, 0.3)}
    if kind == 'ngram':
        p.update(order=rng.choice([1, 2, 2, 3, 3, 4]), site=rng.random() < 0.5)
        return NGram(rng, p)
    if kind == 'template':
        p.update(buckets=rng.choice([2, 3, 4, 6]))
        return Template(rng, p)
    if kind == 'topic':
        p.update(K=rng.choice([2, 4, 8, 16, 32]))
        return Topic(rng, p)
    p.update(hidden=rng.choice([32, 48]), epochs=rng.choice([4, 8, 12]), lr=0.01, temp=rng.uniform(0.8, 1.1))
    return Neural(rng, p)


def generation(docs, rng, kind=None, objective=None, weights=(0.4, 0.15, 0.25, 0.2)):
    lr = random_learner(rng, kind, weights)
    lr.fit(docs)
    if objective is None:
        new = lr.sample(len(docs))
    else:
        cand = lr.sample(3 * len(docs))
        new = select_objective(cand, len(docs), objective, rng)
    new = [d for d in new if sum(t[0] != 'NL' for t in d['toks']) >= 1] or new
    return new, lr.name


def select_objective(cand, n, target, rng):
    """Extra objective that preserves ARBITRARY structure: greedily pick n of 3n candidate
    documents so that the counts of the target relation set approach their generation-0 counts."""
    rels = [relations(d) & target.keys() for d in cand]
    have = collections.Counter()
    chosen, used = [], set()
    order = list(range(len(cand)))
    rng.shuffle(order)
    for _ in range(n):
        best, bs = None, -1e9
        for i in order[:]:
            if i in used:
                continue
            s = sum(1.0 for f in rels[i] if have[f] < target[f]) - 0.5 * sum(1.0 for f in rels[i] if have[f] >= target[f])
            if s > bs:
                best, bs = i, s
                if s >= 1:
                    break
        used.add(best); chosen.append(cand[best]); have.update(rels[best])
        if len(used) == len(cand):
            break
    return chosen


# ------------------------------------------------------------------ planted controls (LA copy)
def plant(docs, rng):
    """World-anchored plants: PLW-A opens and PLW-B closes 25 tablets (one latent official: long
    range); PLW-T + exact running sum closes 20 tablets (arithmetic). Learnable-but-meaningless
    plants: PLM-X written right before every GRA on 25 tablets (adjacent W->L), PLM-N always
    followed by a number of the same scale 16-31 (adjacent W->N) on 25 tablets."""
    docs = [dict(d, toks=list(d['toks'])) for d in docs]
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    big = [i for i in idx if sum(t[0] == 'W' for t in docs[i]['toks']) >= 2]
    A = big[:25]
    for i in A:
        t = docs[i]['toks']
        docs[i]['toks'] = [('W', 'PLW-A')] + t + [('NL',), ('W', 'PLW-B'), ('N', float(rng.randint(1, 9)))]
    rest = [i for i in idx if i not in set(A)]
    T = [i for i in rest if sum(t[0] == 'N' for t in docs[i]['toks']) >= 2][:20]
    for i in T:
        s = sum(math.floor(t[1]) for t in docs[i]['toks'] if t[0] == 'N')
        if s > 0:
            docs[i]['toks'] = docs[i]['toks'] + [('NL',), ('W', 'PLW-T'), ('N', float(s))]
    rest2 = [i for i in rest if i not in set(T)]
    G = [i for i in rest2 if any(t == ('L', 'GRA') for t in docs[i]['toks'])]
    X = (G + [i for i in rest2 if i not in set(G)])[:25]
    for i in X:
        t = docs[i]['toks']
        nt = []
        hasg = False
        for x in t:
            if x == ('L', 'GRA'):
                nt.append(('W', 'PLM-X')); hasg = True
            nt.append(x)
        if not hasg:
            nt = nt + [('NL',), ('W', 'PLM-X'), ('L', 'GRA'), ('N', float(rng.randint(1, 20)))]
        docs[i]['toks'] = nt
    rest3 = [i for i in rest2 if i not in set(X)]
    for i in rest3[:25]:
        docs[i]['toks'] = docs[i]['toks'] + [('NL',), ('W', 'PLM-N'), ('N', float(rng.randint(16, 31)))]
    return docs


PLANT_FEATS = {
    ('CO', 'PLW-A', 'PLW-B'): 'world: one latent official opens and closes the tablet (long range)',
    ('TOT', 'PLW-T', 'SUM'): 'world: arithmetic total',
    ('WL', 'PLM-X', 'GRA'): 'meaningless: word written before GRA (adjacent)',
    ('WN', 'PLM-N', '4'): 'meaningless: word always followed by 16-31 (adjacent)',
}


# ------------------------------------------------------------------ chains
def chain(docs0, rng, G, feats, measure_at, kind=None, objective=None, weights=(0.4, 0.15, 0.25, 0.2)):
    """Run one chain; return {g: (count vector, baseline vector)} for the tracked features and
    the learner names."""
    res = {}
    names = []
    cur = docs0
    for g in range(G + 1):
        if g > 0:
            cur, nm = generation(cur, rng, kind, objective, weights)
            names.append(nm)
        if g in measure_at:
            c, b = excess(cur, rng, 1)
            res[g] = ([c.get(f, 0) for f in feats], [b.get(f, 0.0) for f in feats])
    return res, names


def fkey(f):
    return '\t'.join(f)


def unkey(s):
    return tuple(s.split('\t'))
