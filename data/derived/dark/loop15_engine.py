"""Loop 15 (NEGATIVE SPACE): gaps = patterns the grammar's own generative model expects but the corpus never shows.

Model: order-2 Markov chain over whole texts (START,START ... END), absolute discounting with KN continuation
backoff, one table per object class (SEAL / TAB / OTHER) shrunk to a pooled table.  Fitted on Mohenjo-daro +
Harappa (home) and separately on the held-out sites.  Expected count of a pattern = mean presence count over
NSYN synthetic corpora of the same size and object mix.

Patterns (presence per text, signs with >= MINTOK tokens in the fitting set):
  co   : {X,Y} co-occur anywhere in a text (either order)           -> paradigm candidates
  ord  : X ... Y with at least one sign between (X before Y, gap>=1) -> ordering candidates (beyond bigrams)
  skip : X _ Y (exactly one sign between)
  tri  : X ... Y ... Z in order, any gaps

Gap: expected >= EMIN and observed 0.  Controls: (1) each synthetic corpus scored against the expectation of the
others -> number of chance gaps; (2) real corpus shuffled within text, refit, same search; (3) held-out sites:
own model, expected >= EHELD and observed 0 -> persistent gap.

Usage: python3 loop15_engine.py --seqkey seq_raw --nsyn 200 --out loop15_c1_seq_raw
"""
import json, sys, os, random, collections, math, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
ARGS = sys.argv[1:]
def arg(n, d):
    return ARGS[ARGS.index(n) + 1] if n in ARGS else d
SEQKEY = arg('--seqkey', 'seq_raw'); NSYN = int(arg('--nsyn', 200)); OUT = arg('--out', 'loop15_c1_' + SEQKEY)
EMIN = float(arg('--emin', 5)); EHELD = float(arg('--eheld', 2)); MINTOK = int(arg('--mintok', 15))
NCTRL = int(arg('--nctrl', 100)); SEED = int(arg('--seed', 15))
MAXLEN = 14

C = json.load(open(os.path.join(ROOT, 'data/derived/merged-corpus-canonical.json')))
def cls_of(ty):
    return 'SEAL' if ty.startswith('SEAL') else 'TAB' if ty.startswith('TAB') else 'OTHER'
ALL = [(t['site'], cls_of(t['type']), tuple(t[SEQKEY])) for t in C if t[SEQKEY]]
HOME = [x for x in ALL if x[0] in ('Mohenjo-daro', 'Harappa')]
HELD = [x for x in ALL if x[0] not in ('Mohenjo-daro', 'Harappa', 'Unknown')]

S, E = -1, -2

class KN2:
    """order-2 chain over whole texts, absolute discounting, KN continuation unigram, optional parent shrinkage."""
    def __init__(self, seqs, D=0.75, parent=None, lam=5.0):
        self.D = D; self.parent = parent; self.lam = lam
        self.c2 = collections.defaultdict(collections.Counter); self.c1 = collections.defaultdict(collections.Counter)
        self.cont = collections.Counter()
        for s in seqs:
            s = [S, S] + list(s) + [E]
            for i in range(2, len(s)):
                self.c2[(s[i-2], s[i-1])][s[i]] += 1; self.c1[(s[i-1],)][s[i]] += 1
        for h, cnt in self.c1.items():
            for v in cnt: self.cont[v] += 1
        self.vocab = sorted(self.cont); self.V = len(self.vocab); self.cache = {}; self.ccache = {}
        if parent is not None:
            self.vocab = sorted(set(self.vocab) | set(parent.vocab)); self.V = len(self.vocab)
    def ckey(self, h):
        # the distribution depends on the full bigram history only if some table has seen it; else on h[1] alone
        if h in self.c2 or (self.parent is not None and h in self.parent.c2): return h
        return ('u', h[1])
    def dist(self, h):
        ck = self.ckey(h)
        if ck in self.cache: return self.cache[ck]
        tot = sum(self.cont.values())
        p = {v: (self.cont[v] + 0.5) / (tot + 0.5 * self.V) for v in self.vocab}
        for ctx in ((h[1],), h):
            cnt = self.c1.get(ctx) if len(ctx) == 1 else self.c2.get(ctx)
            if not cnt: continue
            n = sum(cnt.values()); back = self.D * len(cnt) / n
            p = {v: max(cnt.get(v, 0) - self.D, 0) / n + back * p[v] for v in self.vocab}
        if self.parent is not None:
            pk, pw = self.parent.dist(h); pp = dict(zip(pk, pw))
            n = sum(self.c2.get(h, {}).values()); w = n / (n + self.lam)
            p = {v: w * p.get(v, 0) + (1 - w) * pp.get(v, 0) for v in self.vocab}
        keys = list(p); wts = [p[v] for v in keys]
        self.cache[ck] = (keys, wts); return self.cache[ck]
    def cum(self, h):
        ck = self.ckey(h)
        if ck in self.ccache: return self.ccache[ck]
        k, w = self.dist(h); acc = 0.0; cw = []
        for x in w: acc += x; cw.append(acc)
        self.ccache[ck] = (k, cw, acc); return self.ccache[ck]
    def gen(self, rng):
        import bisect
        out = []; h = (S, S)
        while len(out) < MAXLEN:
            k, cw, tot = self.cum(h); v = k[min(bisect.bisect_left(cw, rng.random() * tot), len(k) - 1)]
            if v == E: break
            out.append(v); h = (h[1], v)
        return tuple(out) if out else self.gen(rng)

class Model:
    def __init__(self, data):
        self.pool = KN2([s for _, _, s in data])
        g = collections.defaultdict(list)
        for _, c, s in data: g[c].append(s)
        self.tab = {c: KN2(v, parent=self.pool) for c, v in g.items()}
    def corpus(self, meta, rng):
        return [(site, c, self.tab.get(c, self.pool).gen(rng)) for site, c in meta]

TRIVOC = set()
def patterns(seq, voc):
    """set of pattern keys present in one text (triples only over TRIVOC, the commoner signs)"""
    out = set(); L = len(seq)
    idx = [(i, x) for i, x in enumerate(seq) if x in voc]
    for a in range(len(idx)):
        i, x = idx[a]
        for b in range(a + 1, len(idx)):
            j, y = idx[b]
            out.add(('co', min(x, y), max(x, y)))
            if j - i >= 2: out.add(('ord', x, y))
            if j - i == 2: out.add(('skip', x, y))
            if x in TRIVOC and y in TRIVOC:
                for c in range(b + 1, len(idx)):
                    k, z = idx[c]
                    if z in TRIVOC: out.add(('tri', x, y, z))
    return out

def count(corpus, voc):
    cnt = collections.Counter()
    for _, _, s in corpus:
        for p in patterns(s, voc): cnt[p] += 1
    return cnt

def expected(model, meta, voc, n, seed, keys=None):
    """mean presence count over n synthetic corpora (corpus i generated with seed+i, so it can be regenerated).
    With keys: also return per-corpus counts restricted to those keys (memory-safe)."""
    tot = collections.Counter(); per = []
    for i in range(n):
        c = count(model.corpus(meta, random.Random(seed + i)), voc); tot.update(c)
        if keys is not None: per.append({k: c[k] for k in keys if c[k]})
        del c
    return {k: v / n for k, v in tot.items() if v / n >= 0.3}, per

def gaps(exp, obs, emin):
    return sorted([(k, e) for k, e in exp.items() if e >= emin and obs.get(k, 0) == 0], key=lambda t: -t[1])

def vocab_of(data, mintok):
    tok = collections.Counter(x for _, _, s in data for x in s)
    return {x for x, n in tok.items() if n >= mintok}

def shuffle_within(data, rng):
    out = []
    for site, c, s in data:
        s = list(s); rng.shuffle(s); out.append((site, c, tuple(s)))
    return out

def run():
    t0 = time.time(); rng = random.Random(SEED)
    log = []
    def P(*a):
        line = ' '.join(str(x) for x in a); print(line, flush=True); log.append(line)
    P(f'LOOP15 cycle1 seqkey={SEQKEY} nsyn={NSYN} emin={EMIN} eheld={EHELD} mintok={MINTOK} home={len(HOME)} held={len(HELD)}')
    voc = vocab_of(HOME, MINTOK); P('vocab size', len(voc))
    global TRIVOC; TRIVOC = vocab_of(HOME, 2 * MINTOK); P('triple vocab size', len(TRIVOC))
    meta = [(a, b) for a, b, _ in HOME]
    M = Model(HOME)
    obs = count(HOME, voc)
    exp, _ = expected(M, meta, voc, NSYN, SEED * 1000)
    G = gaps(exp, obs, EMIN)
    bykind = collections.Counter(k[0] for k, _ in G)
    P('REAL gaps (E>=%.0f, O=0):' % EMIN, dict(bykind), 'total', len(G), 'patterns with E>=emin:', sum(1 for e in exp.values() if e >= EMIN))
    # control 1: synthetic corpora scored against the leave-one-out expectation (regenerate, keep only candidate keys)
    keys = [k for k, e in exp.items() if e * NSYN / (NSYN - 1) >= EMIN]
    _, per = expected(M, meta, voc, min(NCTRL, NSYN), SEED * 1000, keys)
    ctrl = []
    for c in per:
        exp_i = {k: (exp[k] * NSYN - c.get(k, 0)) / (NSYN - 1) for k in keys}
        g = gaps(exp_i, c, EMIN); ctrl.append(collections.Counter(k[0] for k, _ in g))
    for kind in ('co', 'ord', 'skip', 'tri'):
        xs = sorted(x[kind] for x in ctrl)
        P(f'  synthetic null {kind}: mean {sum(xs)/len(xs):.1f} p95 {xs[int(0.95*len(xs))-1]} max {xs[-1]}  | real {bykind[kind]}')
    # control 2: shuffled-within-text corpus, refit
    SH = shuffle_within(HOME, rng); MS = Model(SH); obs_s = count(SH, voc)
    exp_s, _ = expected(MS, meta, voc, max(40, NSYN // 4), SEED * 2000)
    gs = collections.Counter(k[0] for k, _ in gaps(exp_s, obs_s, EMIN))
    P('  shuffled-within-text corpus (refit) gaps:', dict(gs))
    # held-out
    vocH = voc  # same sign set
    metaH = [(a, b) for a, b, _ in HELD]
    MH = Model(HELD); obsH = count(HELD, vocH)
    expH, _ = expected(MH, metaH, vocH, NSYN, SEED * 3000)
    # held-out chance: how many of the real gaps would persist in a synthetic held-out corpus
    persist = [(k, e, expH.get(k, 0)) for k, e in G if expH.get(k, 0) >= EHELD and obsH.get(k, 0) == 0]
    tested = [(k, e) for k, e in G if expH.get(k, 0) >= EHELD]
    P(f'held-out: {len(tested)} real gaps have E_held>={EHELD}; {len(persist)} persist (O_held=0)')
    # chance persistence: synthetic held-out corpora
    pers_null = []
    for i in range(20):
        synH = count(MH.corpus(metaH, rng), vocH)
        pers_null.append(sum(1 for k, e in tested if synH.get(k, 0) == 0))
    P('  chance persistence (20 synthetic held-out corpora): mean %.1f max %d' % (sum(pers_null) / 20, max(pers_null)))
    P('PERSISTENT GAPS (pattern, E_home, E_held, obs in all sites incl. Unknown):')
    allobs = count(ALL, voc)
    for k, e, eh in persist:
        P('  ', k, 'E_home=%.1f' % e, 'E_held=%.1f' % eh, 'O_all=%d' % allobs.get(k, 0))
    P('TOP 40 home gaps by expected count:')
    for k, e in G[:40]:
        P('  ', k, 'E=%.1f' % e, 'E_held=%.1f' % expH.get(k, 0), 'O_held=%d' % obsH.get(k, 0))
    json.dump({'seqkey': SEQKEY, 'gaps': [(list(k), e, expH.get(k, 0), obsH.get(k, 0)) for k, e in G],
               'persist': [(list(k), e, eh) for k, e, eh in persist], 'null': [dict(x) for x in ctrl],
               'shuffled': dict(gs), 'pers_null': pers_null},
              open(os.path.join(HERE, OUT + '.json'), 'w'))
    P('time %.0fs' % (time.time() - t0))
    open(os.path.join(HERE, OUT + '.txt'), 'w').write('\n'.join(log) + '\n')

if __name__ == '__main__':
    run()
