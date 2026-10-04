"""v26 EVOLVE THE SCRIBE: shared library.

A modular word generator (the 'scribe') whose genome switches and tunes every mechanism found so far:
  junction table (none / last unit / rich key), section and position tables, paragraph-first-line tables,
  line-initial chain, reduplication, local copy-and-vary from the lines above (window, distance weights,
  edit rate), page-wide urn (page seeds: words already on the page, with edits), slot template (novel words
  from an in-word unit trigram), and log-linear experts: ch/sh state, first-unit alliteration, last-unit
  echo of the word above, paragraph drift of q-/a-/l- words (driven by the previous line's make-up),
  one-m/g-per-line quota, word-length autocorrelation; line-width filling (sampling only).
Every mechanism is conditioned on OBSERVABLE history only, so the same step function both samples text and
gives an exact per-word probability (used for held-out perplexity).

Discriminators are the v21 page-feature battery (fast near-repeat implementation, identical output).
"""
import os, sys, re, json, math, random, inspect, textwrap, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
import v21_lib
from v21_lib import voynich_pages, tokens, near1, Featurizer, cv_auc, _auc, folds, paired_z

LOOPS = os.path.join(vlib.ROOT, 'loops')
CK = os.path.join(vlib.DATA, 'v26_ckpt'); os.makedirs(CK, exist_ok=True)


def save(name, obj):
    with open(os.path.join(CK, name), 'w') as f: json.dump(obj, f, indent=1, default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


# ------------------------------------------------------------------ corpora and split
def herbals():
    return json.load(open(os.path.join(vlib.DATA, 'derived', 'v21_herbals.json')))


def corpus(name):
    if name == 'V': return voynich_pages('ZL3b')
    if name == 'VI': return voynich_pages('IT2a')
    if name in ('LA', 'IT'): return herbals()[name]
    raise KeyError(name)


def split_half(C, seed=26):
    """Half the folios (pages) per section for fitting, the other half held out."""
    rng = random.Random(seed)
    bysec = defaultdict(list)
    for i, p in enumerate(C): bysec[p['sec']].append(i)
    tr = set()
    for s, ix in sorted(bysec.items()):
        rng.shuffle(ix); tr |= set(ix[:(len(ix) + 1) // 2])
    return [C[i] for i in range(len(C)) if i in tr], [C[i] for i in range(len(C)) if i not in tr]


# ------------------------------------------------------------------ fast featurizer (same output as v21)
def _near_classes(toks, tl):
    """v21 near-repeat loop: for each token, the EARLIEST token in the 40 before it that is an edit-1 variant
    (v21 scans j ascending and breaks). Uses deletion keys instead of 40 pairwise checks."""
    n = len(toks)
    uw = {}
    ids = [uw.setdefault(w, len(uw)) for w in toks]
    U = list(uw)
    keys = defaultdict(list)
    for u, w in enumerate(U):
        if len(w) < 2: continue
        for i in range(len(w)):
            d = w[:i] + w[i + 1:]
            keys[('s', len(w), i, d)].append(u)    # substitution partner: same length, same deletion position
            keys[('d', d)].append(u)               # w minus one unit
        keys[('w', w)].append(u)
    nb = [set() for _ in U]
    for k, lst in keys.items():
        if k[0] == 'w': continue
        if k[0] == 's':
            for a in lst:
                for b in lst:
                    if a != b: nb[a].add(b)
        else:
            d = k[1]
            if ('w', d) in keys:
                for a in lst:
                    for b in keys[('w', d)]: nb[a].add(b); nb[b].add(a)
    nbr = [set(b for b in s if near1(U[a], U[b])) for a, s in enumerate(nb)]
    out = Counter()
    for i in range(n):
        s = nbr[ids[i]]
        if not s: continue
        for j in range(max(0, i - 40), i):
            if ids[j] in s:
                d = tl[i] - tl[j]
                out['same' if d == 0 else ('prev' if d == 1 else 'far')] += 1
                break
    return out


def _make_fast_page():
    src = textwrap.dedent(inspect.getsource(Featurizer.page))
    old = ("    nr = Counter()\n"
           "    for i in range(n):\n"
           "        for j in range(max(0, i - 40), i):\n"
           "            if near1(toks[i], toks[j]):\n"
           "                d = tl[i] - tl[j]\n"
           "                nr['same' if d == 0 else ('prev' if d == 1 else 'far')] += 1\n"
           "                break\n")
    assert old in src, 'v21 featurizer changed'
    src = src.replace(old, '    nr = _near_classes(toks, tl)\n')
    g = dict(v21_lib.__dict__); g['_near_classes'] = _near_classes
    exec(src, g)
    return g['page']


class FastFZ(Featurizer):
    page = _make_fast_page()


# ------------------------------------------------------------------ genome
# name: (kind, choices or (lo, hi), default=off)
GENES = {
    'junc':     ('cat', [0, 1, 2], 1),      # 0 unigram by position, 1 last-unit junction, 2 rich junction key
    'sec':      ('cat', [0, 1], 0),         # section-specific tables
    'pos':      ('cat', [0, 1], 0),         # 2nd / 3rd word tables
    'pfl':      ('cat', [0, 1], 0),         # paragraph-first-line junction tables
    'width':    ('cat', [0, 1], 0),         # fill lines to the real line width (sampling only)
    'cs_line':  ('cat', [0, 1], 0),         # ch/sh state resets at each line start
    'cwin':     ('cat', [1, 2, 3, 5, 8, 99], 1),
    'pi_chain': ('num', (0.0, 1.0), 0.0),
    'pi_redup': ('num', (0.0, 0.05), 0.0),
    'pi_copy':  ('num', (0.0, 0.5), 0.0),
    'w_prev':   ('num', (0.0, 3.0), 1.0),   # copy weight of a token one line up (same line = 1)
    'w_far':    ('num', (0.0, 3.0), 1.0),   # copy weight of a token 2+ lines up
    'edit':     ('num', (0.0, 1.0), 0.0),   # copied word replaced by an edit-1 neighbour
    'pi_urn':   ('num', (0.0, 0.3), 0.0),   # page urn (page seeds)
    'urn_edit': ('num', (0.0, 1.0), 0.0),
    'pi_slot':  ('num', (0.0, 0.2), 0.0),   # novel word from the slot template
    'b_cs':     ('num', (-3.0, 3.0), 0.0),
    'b_al':     ('num', (-2.0, 2.0), 0.0),
    'b_vl':     ('num', (-2.0, 2.0), 0.0),
    'b_dr':     ('num', (-2.0, 8.0), 0.0),
    'b_mg':     ('num', (-3.0, 1.5), 0.0),
    'b_len':    ('num', (-2.0, 2.0), 0.0),
}
# genes that only matter if a parent gene is on
DEPENDS = {'w_prev': 'pi_copy', 'w_far': 'pi_copy', 'edit': 'pi_copy', 'cwin': 'pi_copy', 'urn_edit': 'pi_urn',
           'cs_line': 'b_cs'}
OFF = {k: v[2] for k, v in GENES.items()}
OFF['junc'] = 1   # the plain v17 junction rule is the base generator (= v21 F1-F3 family)


def active(g):
    """Mechanisms switched on (relative to the plain junction base)."""
    on = []
    for k, (kind, rng_, d) in GENES.items():
        if k in DEPENDS and not _on(g, DEPENDS[k]): continue
        if _on(g, k): on.append(k)
    return on


def _on(g, k):
    kind, r, d = GENES[k]
    if kind == 'cat': return g[k] != OFF[k]
    if k in ('w_prev', 'w_far'): return abs(g[k] - 1.0) > 0.05
    return abs(g[k]) > 1e-9


def bits(g):
    """Description length of the genome: 1 switch bit per gene + log2(choices) or 5 bits per active value."""
    b = 0.0
    for k in active(g):
        kind, r, d = GENES[k]
        b += 1 + (math.log2(len(r)) if kind == 'cat' else 5)
    return b


def random_genome(rng, p_on=0.35):
    g = dict(OFF)
    for k, (kind, r, d) in GENES.items():
        if rng.random() < p_on:
            g[k] = rng.choice(r) if kind == 'cat' else rng.uniform(*r)
    return g


def mutate(g, rng, rate=0.15):
    h = dict(g)
    for k, (kind, r, d) in GENES.items():
        if rng.random() < rate:
            if kind == 'cat': h[k] = rng.choice(r)
            elif rng.random() < 0.25: h[k] = OFF[k]          # switch off
            else:
                lo, hi = r
                x = h[k] + rng.gauss(0, 0.2 * (hi - lo)) if _on(h, k) else rng.uniform(lo, hi)
                h[k] = min(hi, max(lo, x))
    return h


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in GENES}


def gstr(g):
    s = []
    for k in active(g):
        v = g[k]
        s.append(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}')
    return ' '.join(s) or 'base'


# ------------------------------------------------------------------ scribe model
PCS = ('p1', 'p2', 'mid', 'end')


class Scribe:
    """Tables fitted on a corpus; generate(page skeleton, genome) and logprob(page, genome)."""

    def __init__(self, C, cs_units=('C', 'S')):
        self.cs = cs_units
        self.w2i = {}; self.words = []
        cap = 4 * len(set(w for p in C for w in tokens(p))) + 5000
        self.fu = np.zeros(cap, np.int16); self.lu = np.zeros(cap, np.int16); self.ln = np.zeros(cap, np.float64)
        self.csm = np.zeros(cap, np.float64); self.mg = np.zeros(cap, np.float64)
        self.Q = np.zeros((cap, 3), np.float64)
        self.u2i = {}
        for p in C:
            for w in tokens(p): self.wid(w)
        self.nvocab = len(self.words)
        allw = [w for p in C for w in tokens(p)]
        self.base_q = np.array([np.mean([self._q(w)[j] for w in allw]) for j in range(3)])
        L = np.array([len(w) for w in allw], float); self.mu_len = L.mean(); self.var_len = L.var()
        self._neighbours()
        self._tables(C)
        self._slot(C)

    # vocabulary with features
    def uid(self, u):
        return self.u2i.setdefault(u, len(self.u2i))

    @staticmethod
    def _q(w):
        return (float(w[0] == 'q'), float('a' in w), float(w[0] == 'l'))

    def wid(self, w):
        i = self.w2i.get(w)
        if i is not None: return i
        i = len(self.words); self.w2i[w] = i; self.words.append(w)
        if i >= len(self.fu):
            for a in ('fu', 'lu', 'ln', 'csm', 'mg', 'Q'):
                arr = getattr(self, a); setattr(self, a, np.concatenate([arr, np.zeros_like(arr)]))
        self.fu[i] = self.uid(w[0]); self.lu[i] = self.uid(w[-1]); self.ln[i] = len(w)
        c, s = self.cs[0] in w, self.cs[1] in w
        self.csm[i] = 1.0 if (c and not s) else (-1.0 if (s and not c) else 0.0)
        self.mg[i] = float('m' in w or 'g' in w)
        self.Q[i] = self._q(w)
        return i

    def _neighbours(self):
        keys = defaultdict(list)
        W = self.words[:self.nvocab]
        for u, w in enumerate(W):
            for i in range(len(w)):
                d = w[:i] + w[i + 1:]
                keys[('s', len(w), i, d)].append(u); keys[('d', d)].append(u)
            keys[('w', w)].append(u)
        nb = [set() for _ in W]
        for k, lst in keys.items():
            if k[0] == 's':
                if len(lst) > 1:
                    for a in lst:
                        nb[a].update(lst)
            elif k[0] == 'd' and ('w', k[1]) in keys:
                for a in lst:
                    for b in keys[('w', k[1])]: nb[a].add(b); nb[b].add(a)
        self.nbr = [np.array(sorted(b for b in s if near1(W[a], W[b])), np.int32) for a, s in enumerate(nb)]

    def _arr(self, c):
        ks = list(c.keys()); v = np.array([c[k] for k in ks], float)
        p = v / v.sum()
        return (np.array(ks, np.int32), p, v.sum(), np.cumsum(p))

    def _tables(self, C):
        T = defaultdict(Counter)
        for p in C:
            for s in (p['sec'], '*'):
                for qi, pa in enumerate(p['paras']):
                    prevfirst = None
                    for li, ws in enumerate(pa):
                        I = [self.w2i[w] for w in ws]
                        lt = 'pf' if li == 0 else 'li'
                        T[('L', s, lt)][I[0]] += 1
                        if li > 0: T[('CH', s, prevfirst)][I[0]] += 1
                        prevfirst = ws[0][0]
                        n = len(ws)
                        for k in range(1, n):
                            pc = 'end' if k == n - 1 else ('p1' if k == 1 else ('p2' if k == 2 else 'mid'))
                            ne = 'end' if pc == 'end' else 'ne'
                            lu = ws[k - 1][-1]
                            for pk in (pc, ne):
                                T[('U', s, pk)][I[k]] += 1
                                T[('J', s, pk, lu)][I[k]] += 1
                            T[('U', s, 'any')][I[k]] += 1
                            T[('J', s, 'any', lu)][I[k]] += 1
                            if pc != 'end':
                                T[('R', s, lu, ws[k - 1][0], ws[k - 2][0] if k >= 2 else '^')][I[k]] += 1
                            if li == 0: T[('F', s, lu)][I[k]] += 1
        self.T = {k: self._arr(v) for k, v in T.items()}

    def get(self, *keys, minn=1):
        for k in keys:
            t = self.T.get(k)
            if t is not None and t[2] >= minn: return t
        return None

    # slot template: in-word unit trigram per section x class, add-0.05 smoothing, backoff to '*'
    def _slot(self, C):
        T = defaultdict(Counter)
        units = set()
        for p in C:
            for pa in p['paras']:
                for li, ws in enumerate(pa):
                    for k, w in enumerate(ws):
                        cls = 'lf' if k == 0 else ('end' if k == len(ws) - 1 else 'mid')
                        x = '^^' + w + '$'; units.update(w)
                        for i in range(2, len(x)):
                            for s in (p['sec'], '*'):
                                T[(s, cls, x[i - 2:i])][x[i]] += 1
        self.units = sorted(units) + ['$']
        self.ST = T

    def slot_dist(self, s, cls, ctx):
        c = self.ST.get((s, cls, ctx)) or Counter()
        c2 = self.ST.get(('*', cls, ctx)) or Counter()
        n, n2 = sum(c.values()), sum(c2.values())
        a = 0.05
        tot = n + 0.5 * n2 + a * len(self.units)
        return {u: (c[u] + 0.5 * c2[u] + a) / tot for u in self.units}

    def slot_lp(self, w, s, cls):
        x = '^^' + w + '$'; lp = 0.0
        for i in range(2, len(x)):
            d = self.slot_dist(s, cls, x[i - 2:i]); lp += math.log(d.get(x[i], 1e-9))
        return lp

    def slot_sample(self, rng, s, cls):
        x = '^^'
        while len(x) < 18:
            d = self.slot_dist(s, cls, x[-2:])
            ks = list(d); r = rng.random(); acc = 0.0
            for u in ks:
                acc += d[u]
                if acc >= r: break
            if u == '$': break
            x += u
        return x[2:] or 'o'

    # ---------------------------------------------------------- one word
    def pool(self, e):
        return CopyPool(self, e)

    def comps(self, g, st):
        """Mixture components [(table (idx, p, n[, cum]), weight)] for the next word (finite part)."""
        s = st['s']; k = st['k']; pc = st['pc']
        comps = []
        if k == 0:
            lt = 'pf' if st['li'] == 0 else 'li'
            base = self.get(('L', s, lt), ('L', '*', lt))
            pch = g['pi_chain'] if (st['li'] > 0 and st['prevfirst'] is not None) else 0.0
            ch = self.get(('CH', s, st['prevfirst']), minn=5) if pch > 0 else None
            if ch is None: pch = 0.0
            pj = 1.0 - g['pi_copy'] - g['pi_urn']
            comps.append((base, pj * (1 - pch)))
            if pch: comps.append((ch, pj * pch))
        else:
            lu = st['prev'][-1]
            pk = pc if g['pos'] else ('end' if pc == 'end' else 'ne')
            tab = None
            if g['junc'] == 0:
                tab = self.get(('U', s, pk), ('U', s, 'any'))
            else:
                if g['pfl'] and st['li'] == 0 and pc != 'end': tab = self.get(('F', s, lu), minn=5)
                if tab is None and g['junc'] == 2 and pc != 'end':
                    tab = self.get(('R', s, lu, st['prev'][0], st['prev2f']), minn=8)
                if tab is None: tab = self.get(('J', s, pk, lu), ('J', s, 'any', lu), ('U', s, pk), ('U', s, 'any'))
            pr = g['pi_redup']
            pj = 1.0 - g['pi_copy'] - g['pi_urn'] - pr
            comps.append((tab, pj))
            if pr: comps.append(((np.array([st['previ']], np.int32), np.ones(1), 1), pr))
        if g['pi_copy'] > 0:
            cp = st['cpool'].part(g['cwin'], g['w_prev'], g['w_far']) if st.get('nrng') is None else st['cpool'].nonempty(g['cwin'], g['w_prev'], g['w_far'])
            if cp is not None: comps.append(((cp[0], cp[1], 1, None, ('pool', st['cpool'], g['cwin'], g['w_prev'], g['w_far'])), g['pi_copy']))
            else: comps[0] = (comps[0][0], comps[0][1] + g['pi_copy'])
        if g['pi_urn'] > 0:
            cp = st['upool'].part(10 ** 6, 1.0, 1.0) if st.get('nrng') is None else st['upool'].nonempty(10 ** 6, 1.0, 1.0)
            if cp is not None: comps.append(((cp[0], cp[1], 1, None, ('pool', st['upool'], 10 ** 6, 1.0, 1.0)), g['pi_urn']))
            else: comps[0] = (comps[0][0], comps[0][1] + g['pi_urn'])
        return comps

    def energy(self, g, st, idx):
        k = st['k']; E = None
        if g['b_cs'] and st['cs']:
            E = g['b_cs'] * st['cs'] * self.csm[idx]
        if g['b_al'] and k > 0:
            e = g['b_al'] * (self.fu[idx] == st['prevfu']); E = e if E is None else E + e
        if g['b_vl'] and st['above_lu'] is not None:
            e = g['b_vl'] * (self.lu[idx] == st['above_lu']); E = e if E is None else E + e
        if g['b_dr'] and st['drift'] is not None:
            e = g['b_dr'] * (self.Q[idx] @ st['drift']); E = e if E is None else E + e
        if g['b_mg'] and st['hasmg']:
            e = g['b_mg'] * self.mg[idx]; E = e if E is None else E + e
        if g['b_len'] and k > 0:
            e = g['b_len'] * np.clip((self.ln[idx] - self.mu_len) * (st['prevlen'] - self.mu_len) / self.var_len, -2, 2)
            E = e if E is None else E + e
        if st.get('room') is not None:
            e = -1.0 * np.abs(self.ln[idx] - st['room']); E = e if E is None else E + e
        return E

    def dist(self, g, st):
        """Exact candidate distribution (scoring)."""
        comps = self.comps(g, st)
        idx = np.concatenate([c[0][0] for c in comps])
        q = np.concatenate([c[0][1] * c[1] for c in comps])
        E = self.energy(g, st, idx)
        if E is not None: q = q * np.exp(E - E.max())
        return idx, q / q.sum()

    def draw(self, g, st, rng, K=24):
        """Sampling: K proposals from the mixture, then one chosen with weight exp(E) (sampling-importance
        resampling; exact when no expert is on)."""
        comps = self.comps(g, st)
        nr = st['nrng']
        if not any(g[x] for x in ('b_cs', 'b_al', 'b_vl', 'b_dr', 'b_mg', 'b_len')) and st.get('room') is None:
            K = 1
        if len(comps) == 1:
            cnt = [K]
        else:
            ws = np.array([c[1] for c in comps]); cnt = nr.multinomial(K, ws / ws.sum())
        cand = []
        for (tab, w), c in zip(comps, cnt):
            if not c: continue
            if len(tab) > 4:
                _, pool, cw, wp, wf = tab[4]; cand.append(pool.sample(cw, wp, wf, nr, c)); continue
            cum = tab[3] if len(tab) > 3 else np.cumsum(tab[1])
            j = np.minimum(np.searchsorted(cum, nr.random(c) * cum[-1]), len(cum) - 1)
            cand.append(tab[0][j])
        idx = np.concatenate(cand)
        if K == 1: return int(idx[0])
        E = self.energy(g, st, idx)
        if E is None: return int(idx[0])
        q = np.exp(E - E.max()); cq = np.cumsum(q)
        return int(idx[min(int(np.searchsorted(cq, nr.random() * cq[-1])), K - 1)])

    # ---------------------------------------------------------- page walker (sampling or scoring)
    def walk(self, p, g, rng=None, eps=0.0):
        """rng given: sample a forged page on p's skeleton. rng None: return (log2 prob, n_glyph_events) of p.
        eps: slot escape weight used in scoring (P = (1-eps) P_model + eps P_slot); in sampling pi_slot is used."""
        s = p['sec']
        if ('L', s, 'pf') not in self.T: s = '*'
        sec_for_slot = p['sec']
        sample = rng is not None
        hist = []; lineno = 0; paras = []; lp = 0.0; nev = 0
        cs_state = 0.0
        nrng = np.random.default_rng(rng.getrandbits(63)) if sample else None
        cpool = CopyPool(self, g['edit']) if g['pi_copy'] > 0 else None
        upool = CopyPool(self, g['urn_edit']) if g['pi_urn'] > 0 else None
        for qi, pa in enumerate(p['paras']):
            out = []; prevfirst = None; prevline_words = None
            if True: cs_state = 0.0
            for li, ws in enumerate(pa):
                if g['cs_line']: cs_state = 0.0
                n = len(ws)
                target = sum(len(x) for x in ws) + n - 1
                line = []; hasmg = False
                if prevline_words is not None and g['b_dr']:
                    qv = np.mean([self._q(w) for w in prevline_words], 0)
                    drift = qv - self.base_q
                else:
                    drift = None
                k = 0
                while True:
                    room = None
                    if sample and g['width']:
                        if k > 0:
                            room = target - (sum(len(x) for x in line) + len(line) - 1) - 1
                            if room < 2 or k > 3 * n + 5: break
                            pc = 'end' if room <= 6 else ('p1' if k == 1 else ('p2' if k == 2 else 'mid'))
                            if pc != 'end': room = None
                        else:
                            pc = 'lf'
                    else:
                        if k >= n: break
                        pc = 'lf' if k == 0 else ('end' if k == n - 1 else ('p1' if k == 1 else ('p2' if k == 2 else 'mid')))
                    above = prevline_words[k][-1] if (prevline_words is not None and k < len(prevline_words)) else None
                    st = dict(s=s, k=k, pc=pc, li=li, prevfirst=prevfirst, hist=hist, lineno=lineno, cs=cs_state,
                              above_lu=(self.u2i.get(above, -1) if above is not None else None), drift=drift,
                              hasmg=hasmg, room=room, cpool=cpool, upool=upool, nrng=nrng)
                    if k > 0:
                        pw = line[-1]
                        st.update(prev=pw, previ=self.w2i[pw], prevfu=self.u2i[pw[0]], prevlen=len(pw),
                                  prev2f=line[-2][0] if k >= 2 else '^')
                    cls = 'lf' if k == 0 else ('end' if pc == 'end' else 'mid')
                    if sample:
                        if g['pi_slot'] and rng.random() < g['pi_slot']:
                            w = self.slot_sample(rng, sec_for_slot, cls)
                        else:
                            w = self.words[self.draw(g, st, rng)]
                    else:
                        w = ws[k]
                        idx, pr = self.dist(g, st)
                        wi = self.w2i.get(w, -1)
                        pm = float(pr[idx == wi].sum()) if wi >= 0 else 0.0
                        ps = math.exp(self.slot_lp(w, sec_for_slot, cls))
                        lp += math.log2((1 - eps) * pm + eps * ps + 1e-300); nev += len(w) + 1
                    wi = self.wid(w)
                    line.append(w)
                    if cpool: cpool.add(wi)
                    if upool: upool.add(wi)
                    c = self.csm[wi]
                    if c: cs_state = c
                    if self.mg[wi]: hasmg = True
                    k += 1
                    if sample and g['width'] and pc == 'end': break
                out.append(line); prevfirst = line[0][0]; prevline_words = line; lineno += 1
                if cpool: cpool.newline()
                if upool: upool.newline()
            paras.append(out)
        if sample:
            q = dict(p); q['paras'] = paras; return q
        return lp, nev

    def forge(self, C, g, rng):
        return [self.walk(p, g, rng) for p in C]


class CopyPool:
    """Tokens already written on the page, each expanded to itself (1 - e) and its edit-1 vocabulary
    neighbours (e, shared equally). part() weights: same line 1, one line up w_prev, 2..cwin-1 lines up w_far."""

    def __init__(self, sc, e):
        self.sc = sc; self.e = e; self.cache = {}
        self.lines = []      # completed lines: (idx, w, ntok)
        self.cur = []        # current line expansions
        self.curtok = []; self.toklines = []
        self.farc = {}

    def exp(self, ix):
        r = self.cache.get(ix)
        if r is None:
            nb = self.sc.nbr[ix] if (self.e > 0 and ix < self.sc.nvocab) else ()
            if len(nb):
                r = (np.concatenate([np.array([ix], np.int32), nb]), np.concatenate([[1 - self.e], np.full(len(nb), self.e / len(nb))]))
            else:
                r = (np.array([ix], np.int32), np.ones(1))
            self.cache[ix] = r
        return r

    def add(self, ix):
        self.cur.append(self.exp(ix)); self.curtok.append(ix)

    def nonempty(self, cwin, w_prev, w_far):
        ok = bool(self.curtok) or (cwin >= 2 and self.toklines and w_prev > 0 and self.toklines[-1]) or \
            (cwin >= 3 and w_far > 0 and any(self.toklines[max(0, len(self.toklines) - cwin + 1):-1]))
        return (None, None) if ok else None

    def sample(self, cwin, w_prev, w_far, nr, c):
        """c draws from part() without building it: choose a token by distance weight, then itself or a neighbour."""
        T = list(self.curtok); W = [1.0] * len(T)
        if cwin >= 2 and self.toklines and w_prev > 0:
            T += self.toklines[-1]; W += [w_prev] * len(self.toklines[-1])
        if cwin >= 3 and len(self.toklines) >= 2 and w_far > 0:
            for L in self.toklines[max(0, len(self.toklines) - cwin + 1):-1]:
                T += L; W += [w_far] * len(L)
        cw = np.cumsum(W)
        js = np.minimum(np.searchsorted(cw, nr.random(c) * cw[-1]), len(T) - 1)
        out = [T[j] for j in js]
        if self.e > 0:
            us = nr.random(c)
            for t in range(c):
                if us[t] < self.e:
                    ix = out[t]
                    if ix < self.sc.nvocab:
                        nb = self.sc.nbr[ix]
                        if len(nb): out[t] = int(nb[int(us[t] / self.e * len(nb)) % len(nb)])
        return np.array(out, np.int32)

    def newline(self):
        if self.cur:
            self.lines.append((np.concatenate([a for a, _ in self.cur]), np.concatenate([b for _, b in self.cur]), len(self.cur)))
        else:
            self.lines.append((np.zeros(0, np.int32), np.zeros(0), 0))
        self.toklines.append(self.curtok)
        self.cur = []; self.curtok = []; self.farc = {}

    def part(self, cwin, w_prev, w_far):
        I = []; W = []; tot = 0.0
        if self.cur:
            I += [a for a, _ in self.cur]; W += [b for _, b in self.cur]; tot += len(self.cur)
        if cwin >= 2 and self.lines and w_prev > 0:
            a, b, n = self.lines[-1]
            if n: I.append(a); W.append(b * w_prev); tot += n * w_prev
        if cwin >= 3 and len(self.lines) >= 2 and w_far > 0:
            key = (cwin, w_far)
            fc = self.farc.get(key)
            if fc is None:
                L = self.lines[max(0, len(self.lines) - cwin + 1):-1]
                L = [x for x in L if x[2]]
                fc = (np.concatenate([x[0] for x in L]), np.concatenate([x[1] for x in L]) * w_far,
                      sum(x[2] for x in L) * w_far) if L else None
                self.farc[key] = fc
            if fc is not None: I.append(fc[0]); W.append(fc[1]); tot += fc[2]
        if not I or tot <= 0: return None
        return np.concatenate(I), np.concatenate(W) / tot


# ------------------------------------------------------------------ discriminators
def ridge_fit(X, y, C=0.05):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    A = Z.T @ Z + C * len(Z) * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ y)
    return mu, sd, w


def ridge_score(m, X):
    mu, sd, w = m; return ((X - mu) / sd) @ w


def cv_ridge(Xr, Xf, fold, cols=None, C=0.05, keep=False):
    n = len(Xr)
    if cols is not None: Xr = Xr[:, cols]; Xf = Xf[:, cols]
    sr = np.zeros(n); sf = np.zeros(n); models = []
    for f in sorted(set(fold)):
        tr = fold != f; te = fold == f
        X = np.vstack([Xr[tr], Xf[tr]]); y = np.r_[np.ones(tr.sum()), -np.ones(tr.sum())]
        m = ridge_fit(X, y, C)
        sr[te] = ridge_score(m, Xr[te]); sf[te] = ridge_score(m, Xf[te])
        if keep: models.append((f, cols, m))
    return _auc(sr, sf), models


def archive_auc(models, Xr, Xf, fold):
    """Apply frozen per-fold discriminators (trained on earlier generations' elite) to new forgeries, each page
    scored only by the model that did not see it."""
    n = len(Xr); sr = np.zeros(n); sf = np.zeros(n)
    for f, cols, m in models:
        te = fold == f
        A = Xr[te] if cols is None else Xr[te][:, cols]; B = Xf[te] if cols is None else Xf[te][:, cols]
        sr[te] = ridge_score(m, A); sf[te] = ridge_score(m, B)
    return _auc(sr, sf)


def gbm_auc(Xr, Xf, seed=0):
    import warnings; warnings.filterwarnings('ignore')
    return cv_auc(Xr, Xf, 'gbm', seed=seed)


# ------------------------------------------------------------------ baselines for held-out perplexity
class WordBigram:
    """Witten-Bell bigram over words, with unigram backoff and a page cache (lam_cache); slot escape eps."""

    def __init__(self, C, lam_cache=0.0):
        self.u = Counter(); self.b = defaultdict(Counter)
        for p in C:
            for pa in p['paras']:
                for ws in pa:
                    prev = '<s>'
                    for w in ws: self.u[w] += 1; self.b[prev][w] += 1; prev = w
        self.N = sum(self.u.values()); self.lam = lam_cache

    def pw(self, w, prev, cache, ncache):
        pu = self.u[w] / self.N
        c = self.b.get(prev)
        if c:
            n = sum(c.values()); t = len(c)
            pb = (c[w] + t * pu) / (n + t)
        else:
            pb = pu
        if self.lam and ncache: pb = (1 - self.lam) * pb + self.lam * cache[w] / ncache
        return pb

    def logprob(self, p, sc, eps):
        lp = 0.0; nev = 0; cache = Counter(); nc = 0
        for pa in p['paras']:
            for li, ws in enumerate(pa):
                prev = '<s>'
                for k, w in enumerate(ws):
                    cls = 'lf' if k == 0 else ('end' if k == len(ws) - 1 else 'mid')
                    pm = self.pw(w, prev, cache, nc)
                    ps = math.exp(sc.slot_lp(w, p['sec'], cls))
                    lp += math.log2((1 - eps) * pm + eps * ps + 1e-300); nev += len(w) + 1
                    cache[w] += 1; nc += 1; prev = w
        return lp, nev

    def generate(self, p, rng):
        q = dict(p); paras = []
        ks = list(self.u); cu = np.cumsum([self.u[k] for k in ks])
        for pa in p['paras']:
            o = []
            for ws in pa:
                prev = '<s>'; line = []
                for k in range(len(ws)):
                    c = self.b.get(prev)
                    if c and rng.random() < sum(c.values()) / (sum(c.values()) + len(c)):
                        kk = list(c); cc = np.cumsum([c[x] for x in kk]); w = kk[int(np.searchsorted(cc, rng.random() * cc[-1]))]
                    else:
                        w = ks[int(np.searchsorted(cu, rng.random() * cu[-1]))]
                    line.append(w); prev = w
                o.append(line)
            paras.append(o)
        q['paras'] = paras; return q


class GlyphWB:
    """Witten-Bell interpolated unit n-gram over each line ('_' = end of word; line start context '^')."""

    def __init__(self, C, order=5):
        self.o = order; self.c = defaultdict(Counter)
        self.V = set('_')
        for p in C:
            for pa in p['paras']:
                for ws in pa:
                    x = '^' * (order - 1) + '_'.join(ws) + '_'
                    self.V.update(x)
                    for i in range(order - 1, len(x)):
                        for m in range(order):
                            self.c[x[i - m:i]][x[i]] += 1
        self.V.discard('^'); self.nv = len(self.V)
        self.cache = {}

    def p(self, ctx, u):
        key = (ctx, u)
        r = self.cache.get(key)
        if r is not None: return r
        pr = 1.0 / (self.nv + 1)
        for m in range(0, len(ctx) + 1):
            h = ctx[len(ctx) - m:]
            c = self.c.get(h)
            if not c: break
            n = sum(c.values()); t = len(c)
            pr = (c[u] + t * pr) / (n + t)
        self.cache[key] = pr
        return pr

    def logprob(self, p):
        lp = 0.0; nev = 0
        for pa in p['paras']:
            for ws in pa:
                x = '^' * (self.o - 1) + '_'.join(ws) + '_'
                for i in range(self.o - 1, len(x)):
                    lp += math.log2(self.p(x[i - self.o + 1:i], x[i])); nev += 1
        return lp, nev
