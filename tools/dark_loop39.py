"""S-DARK-39: THE CHECKLIST GENERATOR. A non-chain generative model of an Indus text as a filled FORM:
an ordered list of FIELDS (cliques of mutually exclusive signs, ordered by position), each text selecting a
subset of fields through a pairwise log-linear (Ising) co-selection model, one element per field, written in
field order; optional second unit; site-local whole-text reuse, object type and open inventory kept from S366.
Battery = the 50 statistics of tools/strat_adequacy.py, imported unchanged.

Usage: python3 tools/dark_loop39.py CYCLE LEVEL [--n 100]
  CYCLE 1: fit the form, generate, battery, controls (independent fields, shuffled field order, refit floor), held-out sites
  CYCLE 2: cycle 1 + element-level dependence across fields (frozen pairs) and unit model variants
  CYCLE 3: read the form (specification), distinct forms covering 90%, validity of the 324 IM77-new texts vs shuffles
Output: data/derived/dark/loop39_c<CYCLE>_<LEVEL>.txt (+ .json)
"""
import sys, os, json, math, random, collections, statistics, time, csv
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE); sys.path.insert(0, os.path.join(HERE, 'tools'))
CY = int(sys.argv[1]); LV = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'
NSYN = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 100
sys.argv = [sys.argv[0], '--seqkey', LV]          # strat_adequacy reads its seqkey from argv
import numpy as np
import strat_adequacy as SA                        # identical battery, parser, Cat, CRP, fit_theta, compare, summary
OUTD = 'data/derived/dark/'
LOG = []
CHAIN_LAM = 1.0   # shrinkage of P(element | field, previous written element) to the field's pooled menu
import bisect, itertools
def _fast_sample(self, rng):
    """same distribution as strat_adequacy.Cat.sample (shrinkage to parent, then categorical), cumulative weights cached"""
    if self.parent is not None and self.n + self.lam > 0 and rng.random() < self.lam / (self.n + self.lam):
        return self.parent.sample(rng)
    if self.n == 0: return self.parent.sample(rng)
    if self.keys is None:
        self.keys = list(self.c.keys()); self.w = [self.c[k] for k in self.keys]; self.cw = list(itertools.accumulate(self.w))
    return self.keys[bisect.bisect(self.cw, rng.random() * self.cw[-1])]
SA.Cat.sample = _fast_sample
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

FIT = SA.FIT; HELD = SA.HELD; battery = SA.battery

# ------------------------------------------------------------------ IM77 certainly-new texts (loop 27), M -> W at this level
def load_im77_new():
    C = SA.C
    RAW2LV = collections.defaultdict(collections.Counter)
    for r in C:
        for a, b in zip(r['seq_raw'], r[LV]): RAW2LV[a][b] += 1
    def canon(w):
        c = RAW2LV.get(w); return c.most_common(1)[0][0] if c else w
    BR = json.load(open('data/derived/bridge_extended.json'))
    L27 = json.load(open('data/derived/dark/loop27_sets.json'))
    rows = list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
    want = set((a, b) for a, b in L27['new'])
    bridge = dict(L27['bridge']); bridge.update({k: v for k, v in BR.items() if k not in bridge})
    freq = collections.Counter(x for _, _, s in FIT for x in s)
    m2w = collections.defaultdict(list)
    for w, ms in bridge.items():
        for m in ms: m2w[m].append(canon(int(w)))
    M2W = {m: max(ws, key=lambda w: freq.get(w, 0)) for m, ws in m2w.items()}
    by = collections.defaultdict(list)
    for r in rows:
        k = (r['text_no'], r['side'])
        if k in want and r['signs_clean'].strip():
            by[k].append((int(r['line']), [int(x) for x in r['signs_clean'].split()], r['object_type'], r['site']))
    out = []; unb = tot = 0
    for k, lines in by.items():
        lines.sort(); s = [x for ln in lines for x in ln[1]]
        ws = []
        for m in s:
            tot += 1
            if m in M2W: ws.append(M2W[m])
            else: ws.append(-m); unb += 1
        ot = {'seal': 'SEAL', 'miniature tablet': 'TAB', 'copper tablet': 'TAB'}.get(lines[0][2], 'OTHER')
        out.append((lines[0][3], ot, tuple(ws)))
    return out, unb, tot

# ------------------------------------------------------------------ the form
class Form:
    """fields = ordered list of sign sets; every sign of the fit set belongs to exactly one field"""
    def __init__(self, texts, kmax=25, kmin=10, excl=0.33, win=0.12, minc=10):
        n = len(texts)
        has = collections.Counter(); both = collections.Counter(); pos = collections.defaultdict(list)
        before = collections.Counter()
        for s in texts:
            st = set(s)
            for x in st: has[x] += 1
            L = len(s)
            for i, x in enumerate(s):
                pos[x].append(i / (L - 1) if L > 1 else 0.5)
            lst = sorted(st)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)): both[(lst[i], lst[j])] += 1
            for i in range(L):
                for j in range(i + 1, L):
                    if s[i] != s[j]: before[(s[i], s[j])] += 1
        self.n = n; self.has = has; self.both = both; self.before = before
        rank = {x: statistics.mean(v) for x, v in pos.items()}
        self.rank = rank
        def compatible(x, y):
            p = (x, y) if x < y else (y, x)
            e = has[x] * has[y] / n
            return both.get(p, 0) <= max(1, excl * e)
        fields = []   # list of dict(members=set, rank=float, w=tokens)
        for x in sorted(has, key=lambda v: (-has[v], v)):
            cands = []
            for fi, f in enumerate(fields):
                if abs(f['rank'] - rank[x]) > win: continue
                if all(compatible(x, y) for y in f['members'] if has[y] >= minc or has[x] >= minc):
                    cands.append((abs(f['rank'] - rank[x]), fi))
            if cands:
                fi = min(cands)[1]; f = fields[fi]
                f['rank'] = (f['rank'] * f['w'] + rank[x] * has[x]) / (f['w'] + has[x]); f['w'] += has[x]; f['members'].add(x)
            else:
                fields.append(dict(members={x}, rank=rank[x], w=has[x]))
        # merge the smallest fields into the nearest-rank field until <= kmax (cost = co-occurrence it creates)
        def viol(fa, fb):
            v = 0
            for x in fa['members']:
                for y in fb['members']:
                    p = (x, y) if x < y else (y, x); v += both.get(p, 0)
            return v
        while len(fields) > kmax:
            small = min(range(len(fields)), key=lambda i: fields[i]['w'])
            f = fields[small]
            best = min((viol(f, g) + 1e-6 * abs(f['rank'] - g['rank']) * 1000, gi) for gi, g in enumerate(fields) if gi != small)
            g = fields[best[1]]
            g['rank'] = (g['rank'] * g['w'] + f['rank'] * f['w']) / (g['w'] + f['w']); g['w'] += f['w']; g['members'] |= f['members']
            fields.pop(small)
        fields.sort(key=lambda f: f['rank'])
        fo = {x: i for i, f in enumerate(fields) for x in f['members']}
        K = len(fields)
        # ---- refinement: coordinate descent on the number of violated pair instances
        #      (x before y but field(x) > field(y); x, y co-occurring in one field), alternating with a
        #      re-ordering of the fields (insertion local search on the field-level before/after matrix)
        nb_bef = collections.defaultdict(dict); nb_aft = collections.defaultdict(dict); co = collections.defaultdict(dict)
        for (x, y), c in before.items(): nb_bef[x][y] = c; nb_aft[y][x] = c
        for (x, y), c in both.items(): co[x][y] = c; co[y][x] = c
        signs = sorted(has, key=lambda v: (-has[v], v))
        def cost(x, g):
            c = 0
            for y, v in nb_bef[x].items():
                if g > fo[y]: c += v
            for y, v in nb_aft[x].items():
                if g < fo[y]: c += v
            for y, v in co[x].items():
                if fo[y] == g: c += v
            return c
        def reorder():
            B = [[0] * K for _ in range(K)]
            for (x, y), c in before.items():
                if fo[x] != fo[y]: B[fo[x]][fo[y]] += c
            perm = list(range(K))
            def back(perm):
                return sum(B[perm[i]][perm[j]] for i in range(K) for j in range(i))
            cur = back(perm); improved = True
            while improved:
                improved = False
                for i in range(K):
                    for j in range(K):
                        if i == j: continue
                        p = perm[:]; f = p.pop(i); p.insert(j, f); c = back(p)
                        if c < cur - 1e-9: perm, cur, improved = p, c, True
            newpos = {f: i for i, f in enumerate(perm)}
            for x in fo: fo[x] = newpos[fo[x]]
        self.moves = []
        for rnd in range(4):
            reorder(); moved = 0
            for x in signs:
                g0 = fo[x]; c0 = cost(x, g0); best = (c0, g0)
                for g in range(K):
                    if g != g0:
                        c = cost(x, g)
                        if c < best[0] - 1e-9: best = (c, g)
                if best[1] != g0: fo[x] = best[1]; moved += 1
            self.moves.append(moved)
            if moved == 0: break
        reorder()
        used = sorted(set(fo.values())); remap = {g: i for i, g in enumerate(used)}
        self.fields = [[] for _ in used]
        for x in signs: self.fields[remap[fo[x]]].append(x)
        self.K = len(self.fields)
        self.field_of = {x: i for i, f in enumerate(self.fields) for x in f}
        self.rank = {x: statistics.mean(v) for x, v in pos.items()}
        # order of fields re-estimated from the actual before/after counts (field i before field j)
        self.order_ok = self.order_stats()
    def order_stats(self):
        ok = tot = 0
        for (x, y), c in self.before.items():
            fx, fy = self.field_of.get(x), self.field_of.get(y)
            if fx is None or fy is None or fx == fy: continue
            tot += c; ok += c * (fx < fy)
        return ok / max(1, tot)
    def parse(self, s, perm=None):
        """-> list of units; each unit = list of (field, sign); 'viol' counts same-field repeats inside a unit.
        perm: optional field permutation (control). Unknown signs (not in the form) get field None and are skipped."""
        units = [[]]; viol = 0; unknown = 0; last = -1
        for x in s:
            f = self.field_of.get(x)
            if f is None: unknown += 1; continue
            if perm is not None: f = perm[f]
            if f < last: units.append([]); last = -1
            if f == last: viol += 1
            units[-1].append((f, x)); last = f
        return units, viol, unknown
    def valid(self, s, forbid, perm=None, max_units=1):
        units, viol, unknown = self.parse(s, perm)
        if viol or len(units) > max_units: return False
        for u in units:
            fs = [f for f, _ in u]
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)):
                    if (fs[i], fs[j]) in forbid: return False
        return True

# ------------------------------------------------------------------ the generative model
def sigmoid(z): return 1 / (1 + np.exp(-z))

class FormModel:
    def __init__(self, data, mech=(), form=None, kmax=25, seed=0):
        """data: list of (site, cls, seq); mech subset of
        'ising' (field co-selection; else independent fields), 'type', 'site', 'reuse' (site-local whole-text CRP),
        'open' (Good-Turing new element per field), 'shuffleorder' (control: fields written in a random order),
        'elemchain' (element given the element of the previous selected field), 'units' (second units)"""
        self.mech = set(mech); self.rng = random.Random(seed)
        texts = [s for _, _, s in data]
        self.form = form or Form(texts, kmax=kmax)
        F = self.form; K = F.K; self.K = K
        self.perm = None
        if 'shuffleorder' in self.mech:
            p = list(range(K)); random.Random(seed + 1).shuffle(p); self.perm = p
        # parse into units ('partial': the whole text is one form instance = one field subset, written in a partial order)
        self.units = []   # (site, cls, unit_index, [(field, sign)])
        nun = collections.defaultdict(collections.Counter)
        self.mult = collections.defaultdict(collections.Counter)   # multiplicity of a field given present (partial model)
        for site, cls, s in data:
            if 'partial' in self.mech:
                items = [(F.field_of[x], x) for x in s if x in F.field_of]
                units = [items] if items else []
                for f, c in collections.Counter(f for f, _ in items).items(): self.mult[f][min(c, 3)] += 1
            else:
                units, viol, unk = F.parse(s)
            nun[self.tkey(cls)][min(len(units), 3)] += 1
            for ui, u in enumerate(units): self.units.append((site, cls, ui, u))
        self.n_units = {k: SA.Cat(dict(v)) for k, v in nun.items()}
        self.multcat = {f: SA.Cat(dict(c)) for f, c in self.mult.items()}
        # pairwise field order model (Babington Smith weights): p_ij = P(field i written before field j | both present)
        A = np.zeros((K, K))
        for (x, y), c in F.before.items():
            fx, fy = F.field_of.get(x), F.field_of.get(y)
            if fx is not None and fy is not None and fx != fy: A[fx, fy] += c
        if 'strictorder' in self.mech:      # ablation: linear field order only
            self.logp = np.where(np.arange(K)[:, None] < np.arange(K)[None, :], 0.0, -20.0)
        else:
            self.logp = np.log((A + 1) / (A + A.T + 2))
        self.order_agree = float(np.triu(A, 1).sum() / max(1, A.sum()))
        # design matrix
        self.sites = sorted(set(s for s, _, _ in data)); self.clss = ['SEAL', 'TAB', 'OTHER']
        N = len(self.units); X = np.zeros((N, K)); Z = np.zeros((N, self.ncov()))
        for i, (site, cls, ui, u) in enumerate(self.units):
            for f, _ in u: X[i, f] = 1
            Z[i] = self.cov(site, cls, ui)
        self.X = X; self.Z = Z
        self.fit_ising(X, Z)
        # element tables per field (shrunk to pooled over site/type)
        self.elem = {}; self.elem_key = collections.defaultdict(dict); self.newmass = {}
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for site, cls, ui, u in self.units:
            for f, x in u:
                per[f][None].append(x); per[f][self.key(site, cls)].append(x)
        for f in range(K):
            pooled = SA.Cat(per[f][None]); self.elem[f] = pooled
            cnt = collections.Counter(per[f][None])
            self.newmass[f] = sum(1 for v in cnt.values() if v == 1) / max(1, sum(cnt.values()))
            for k, v in per[f].items():
                if k is not None: self.elem_key[f][k] = SA.Cat(v, pooled, 5.0)
        # element chain across fields: P(elem | field, previous element in the unit)
        if 'elemchain' in self.mech:
            ch = collections.defaultdict(lambda: collections.defaultdict(list))
            for site, cls, ui, u in self.units:
                prev = 'S'
                for f, x in u: ch[f][prev].append(x); prev = x
            self.chain = {f: {p: SA.Cat(v, self.elem[f], CHAIN_LAM) for p, v in d.items()} for f, d in ch.items()}
        self.fresh = 0
        # whole-text reuse
        if 'reuse' in self.mech:
            target = len(set(texts)) / len(texts); meta = [(site, cls) for site, cls, _ in data]
            lo, hi = 10.0, 1e5; rng = random.Random(99)
            for _ in range(10):
                self.theta = math.sqrt(lo * hi)
                g = self.generate_corpus(meta, rng); u = len(set(x[2] for x in g)) / len(g)
                if u < target: lo = self.theta
                else: hi = self.theta
            self.theta = math.sqrt(lo * hi)
    def tkey(self, cls): return cls if 'type' in self.mech else None
    def key(self, site, cls): return ((site if 'site' in self.mech else None), (cls if 'type' in self.mech else None))
    def ncov(self): return 1 + 2 + len(self.sites) + 1
    def cov(self, site, cls, ui):
        z = np.zeros(self.ncov()); z[0] = 1
        if 'type' in self.mech:
            if cls == 'SEAL': z[1] = 1
            elif cls == 'TAB': z[2] = 1
        if 'site' in self.mech:
            if site in self.sites: z[3 + self.sites.index(site)] = 1
            else:
                # held-out site: the intercept and the two site dummies are collinear, so an all-zero site block
                # is an arbitrary extrapolation (cycle 1 bug: held-out len_mean 5.8 vs 3.1). Use the fit-site mixture.
                if not hasattr(self, 'site_mix'):
                    c = collections.Counter(s for s, _, _, _ in self.units); t = sum(c.values())
                    self.site_mix = [c[s] / t for s in self.sites]
                for k, p in enumerate(self.site_mix): z[3 + k] = p
        z[-1] = 1 if ui > 0 else 0
        return z
    def fit_ising(self, X, Z, lam=1.0, iters=25):
        """pseudo-likelihood: field i present ~ logistic(b_i . z + sum_j w_ij x_j); W symmetrised"""
        N, K = X.shape; P = Z.shape[1]
        W = np.zeros((K, K)); B = np.zeros((K, P))
        for i in range(K):
            if 'ising' in self.mech:
                A = np.hstack([np.delete(X, i, axis=1), Z]); idx = [j for j in range(K) if j != i]
            else:
                A = Z; idx = []
            y = X[:, i]; w = np.zeros(A.shape[1])
            reg = np.full(A.shape[1], lam); reg[len(idx):] = 0.01   # covariate weights barely regularised
            for _ in range(iters):
                p = sigmoid(A @ w); g = A.T @ (p - y) + reg * w
                H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg) + 1e-6 * np.eye(A.shape[1])
                step = np.linalg.solve(H, g); w -= step
                if np.max(np.abs(step)) < 1e-6: break
            for jj, j in enumerate(idx): W[i, j] = w[jj]
            B[i] = w[len(idx):]
        self.W = (W + W.T) / 2; self.B = B
        self.marg = X.mean(axis=0)
    def sample_fields(self, Zrow, rng, sweeps=40):
        """Gibbs sample one field-presence vector"""
        K = self.K; x = (np.array([rng.random() for _ in range(K)]) < self.marg).astype(float)
        bz = self.B @ Zrow
        for _ in range(sweeps):
            for i in range(K):
                p = sigmoid(bz[i] + self.W[i] @ x)
                x[i] = 1.0 if rng.random() < p else 0.0
        return x
    def sample_fields_batch(self, Zmat, rng_np, sweeps=40):
        N, K = Zmat.shape[0], self.K
        x = (rng_np.random((N, K)) < self.marg).astype(float)
        bz = Zmat @ self.B.T
        for _ in range(sweeps):
            for i in range(K):
                p = sigmoid(bz[:, i] + x @ self.W[i])
                x[:, i] = (rng_np.random(N) < p).astype(float)
        return x
    def pick(self, f, site, cls, prev, rng):
        if 'open' in self.mech and rng.random() < self.newmass[f]:
            self.fresh += 1; return -self.fresh
        if 'elemchain' in self.mech and f in self.chain and prev in self.chain[f]:
            return self.chain[f][prev].sample(rng)
        cat = self.elem_key[f].get(self.key(site, cls)) or self.elem[f]
        return cat.sample(rng)
    def unit_from_fields(self, x, site, cls, rng):
        order = range(self.K) if self.perm is None else sorted(range(self.K), key=lambda f: self.perm[f])
        out = []; prev = 'S'
        for f in order:
            if x[f] > 0:
                e = self.pick(f, site, cls, prev, rng); out.append(e); prev = e
        return out
    def order_items(self, items, rng):
        """items: list of field indices (a field may occur twice). Sequential draw: P(i first) ∝ Π_j p_ij over the rest."""
        R = list(items); out = []
        while len(R) > 1:
            lw = []
            for a, i in enumerate(R):
                l = 0.0
                for b, j in enumerate(R):
                    if a != b and i != j: l += self.logp[i, j]
                lw.append(l)
            m = max(lw); w = [math.exp(l - m) for l in lw]; r = rng.random() * sum(w); acc = 0.0
            for a, v in enumerate(w):
                acc += v
                if r <= acc: break
            out.append(R.pop(a))
        return out + R
    def text_from_fields(self, x, site, cls, rng):
        items = []
        for f in range(self.K):
            if x[f] > 0:
                m = self.multcat[f].sample(rng) if f in self.multcat else 1
                items += [f] * m
        if not items: return []
        out = []; prev = 'S'
        for f in self.order_items(items, rng):
            e = self.pick(f, site, cls, prev, rng); out.append(e); prev = e
        return out
    def generate_corpus(self, meta, rng):
        rng_np = np.random.default_rng(rng.randrange(1 << 30))
        N = len(meta)
        if 'partial' in self.mech:
            Z1 = np.array([self.cov(site, cls, 0) for site, cls in meta]); X1 = self.sample_fields_batch(Z1, rng_np)
            crps = collections.defaultdict(lambda: SA.CRP(getattr(self, 'theta', 1.0))); out = []
            for i, (site, cls) in enumerate(meta):
                def one():
                    s = self.text_from_fields(X1[i], site, cls, rng)
                    for _ in range(20):
                        if s: break
                        s = self.text_from_fields(self.sample_fields(self.cov(site, cls, 0), rng), site, cls, rng)
                    return tuple(s) if s else (740,)
                s = crps[site].draw(rng, one) if 'reuse' in self.mech else one()
                out.append((site, cls, s))
            return out
        nu = [self.n_units[self.tkey(cls)].sample(rng) if 'units' in self.mech else 1 for _, cls in meta]
        # pre-sample field vectors for unit 1 and unit 2 in batch
        Z1 = np.array([self.cov(site, cls, 0) for site, cls in meta]); X1 = self.sample_fields_batch(Z1, rng_np)
        need2 = [i for i, k in enumerate(nu) if k >= 2]
        X2 = {}
        if need2:
            Z2 = np.array([self.cov(meta[i][0], meta[i][1], 1) for i in need2]); xx = self.sample_fields_batch(Z2, rng_np)
            X2 = {i: xx[j] for j, i in enumerate(need2)}
        spare = collections.deque(); crps = collections.defaultdict(lambda: SA.CRP(getattr(self, 'theta', 1.0)))
        out = []
        for i, (site, cls) in enumerate(meta):
            def one():
                s = self.unit_from_fields(X1[i], site, cls, rng)
                for _ in range(20):
                    if s: break
                    s = self.unit_from_fields(self.sample_fields(self.cov(site, cls, 0), rng), site, cls, rng)
                if not s: s = [740]
                for k in range(1, nu[i]):
                    x2 = X2.get(i) if k == 1 else self.sample_fields(self.cov(site, cls, 1), rng)
                    s += self.unit_from_fields(x2, site, cls, rng)
                return tuple(s)
            s = crps[site].draw(rng, one) if 'reuse' in self.mech else one()
            out.append((site, cls, s))
        return out

def run(model, meta, real, nsyn, seed=1):
    rng = random.Random(seed); syn = [battery(model.generate_corpus(meta, rng)) for _ in range(nsyn)]
    return SA.compare(real, syn)

def show(name, rows, top=18):
    s = SA.summary(rows)
    log(f"{name}: outside {s['outside']}/{s['n']}, |z|>3: {s['z_gt3']}, sum|z| {s['sum_abs_z_capped']:.1f}")
    log(SA.fmt_rows(rows, top=top))
    return s

KEYSTATS = ['longrange_attract_pairs', 'longrange_avoid_pairs', 'repeat_rate', 'len_sd', 'two_numerals_share', 'H_start1', 'H_start2', 'H_start3', 'H_cond_bigram', 'mi_d2', 'hapax_sign_share', 'bigram_hapax_share', 'len_mean', 'unique_text_share', 'same_middle_diff_closer']
def keyline(rows):
    d = {r['stat']: r for r in rows}
    return '; '.join(f"{k} {d[k]['real']:.3f} vs {d[k]['mean']:.3f} z {d[k]['z']:.1f}{'*' if d[k]['outside'] else ''}" for k in KEYSTATS if k in d)

def field_diag(F, data):
    """share of texts with a repeated field; share of tokens whose field already occurred earlier in the text; repeat-sign share"""
    rt = rf = rs = tok = 0
    for _, _, s in data:
        fs = [F.field_of.get(x, ('?', x)) for x in s]
        seenf = set(); seens = set(); rep = False
        for f, x in zip(fs, s):
            tok += 1
            if f in seenf: rf += 1; rep = True
            if x in seens: rs += 1
            seenf.add(f); seens.add(x)
        rt += rep
    n = len(data)
    return f"texts with a repeated field {rt/n:.3f}, tokens in an already-used field {rf/tok:.3f}, repeated-sign tokens {rs/tok:.3f}"

def describe_form(F, texts, model=None, topn=8):
    has = F.has
    log(f"form: {F.K} fields; fixed-order consistency of sign pairs with the field order: {F.order_ok:.3f}; refinement moves per round {F.moves}")
    for i, f in enumerate(F.fields):
        toks = sum(has[x] for x in f)
        menu = ', '.join(f"{x}({has[x]})" for x in f[:topn]) + (f" ... +{len(f) - topn}" if len(f) > topn else '')
        extra = ''
        if model is not None:
            extra = f"  P(field)={model.marg[i]:.3f}"
        log(f"  F{i:02d} rank {F.rank[f[0]]:.2f} | {len(f):3d} signs, {toks:5d} tokens{extra} | {menu}")

# ------------------------------------------------------------------ cycles
def main():
    t0 = time.time()
    meta = [(s, c) for s, c, _ in FIT]; texts = [s for _, _, s in FIT]
    real = battery(FIT); J = {}
    log(f"# loop39 cycle {CY} level {LV}: checklist/form generator ({time.strftime('%Y-%m-%d %H:%M')}; {NSYN} synthetic corpora; fit {len(FIT)} MD+H texts, held-out {len(HELD)})")
    F = Form(texts)
    # parse diagnostics
    nunits = collections.Counter(); viol = 0; vtexts = 0; valid1 = 0
    for s in texts:
        u, v, _ = F.parse(s); nunits[min(len(u), 3)] += 1; viol += v; vtexts += v > 0; valid1 += (v == 0 and len(u) == 1)
    log(f"form parse of the fit set: units/text {dict(nunits)}; texts with a same-field repeat inside a unit {vtexts} ({vtexts/len(texts):.3f}); single-unit valid instances {valid1/len(texts):.3f}")
    describe_form(F, texts)
    J['fields'] = F.fields

    if CY in (1, 2):
        base = ['ising', 'type', 'site', 'reuse', 'open', 'units']
        configs = [('FORM (ising+type+site+reuse+open+units)', base)]
        if CY == 1:
            configs += [('control: independent fields (no co-selection)', [m for m in base if m != 'ising']),
                        ('control: shuffled field order', base + ['shuffleorder']),
                        ('ablation: no second units', [m for m in base if m != 'units']),
                        ('ablation: no whole-text reuse', [m for m in base if m != 'reuse'])]
        else:
            part = ['ising', 'type', 'site', 'reuse', 'open', 'partial', 'elemchain']
            configs += [('FORM + elemchain (element given previous written element)', base + ['elemchain']),
                        ('PARTIAL (one field set per text, pairwise field order, elemchain, reuse, open)', part),
                        ('PARTIAL control: independent fields (no co-selection)', [m for m in part if m != 'ising']),
                        ('PARTIAL ablation: strict linear field order', part + ['strictorder']),
                        ('PARTIAL ablation: no element chain', [m for m in part if m != 'elemchain']),
                        ('PARTIAL ablation: no whole-text reuse', [m for m in part if m != 'reuse'])]
        res = {}
        for name, mech in configs:
            t1 = time.time()
            M = FormModel(FIT, mech=mech, form=F if 'shuffleorder' not in mech else None)
            rows = run(M, meta, real, NSYN)
            log(f"\n## {name}  (fit {time.time()-t1:.0f} s)")
            res[name] = show(name, rows); J[name] = rows
            log("key stats: " + keyline(rows))
            if 'partial' in M.mech:
                g = M.generate_corpus(meta, random.Random(21))
                log(f"field-level check (real vs one synthetic corpus): " + field_diag(F, FIT) + " | " + field_diag(F, g) + f"; pair-order agreement with the linear field order {M.order_agree:.3f}")
            if name.startswith('FORM (') or name.startswith('PARTIAL ('):
                # held-out sites
                meta_h = [(s, c) for s, c, _ in HELD]; real_h = battery(HELD)
                rows_h = run(M, meta_h, real_h, NSYN, seed=7); J[name + ' heldout'] = rows_h
                log(f"held-out sites: " + f"outside {SA.summary(rows_h)['outside']}/{SA.summary(rows_h)['n']}, sum|z| {SA.summary(rows_h)['sum_abs_z_capped']:.1f}; top: " + '; '.join(f"{r['stat']} {r['real']:.3f} vs {r['mean']:.3f} z={r['z']:.1f}" for r in sorted([r for r in rows_h if not math.isnan(r['z'])], key=lambda r: -abs(r['z']))[:8]))
                # false-alarm floor: one synthetic corpus treated as real, the form refitted to it
                rng = random.Random(11); planted = M.generate_corpus(meta, rng)
                Mp = FormModel(planted, mech=M.mech); realp = battery(planted)
                rows_p = run(Mp, meta, realp, NSYN, seed=5); J[name + ' floor'] = rows_p
                sp = SA.summary(rows_p)
                log(f"false-alarm floor (form refitted to its own output): outside {sp['outside']}/{sp['n']}, |z|>3 {sp['z_gt3']}, sum|z| {sp['sum_abs_z_capped']:.1f}")
                # Ising couplings
                W = M.W; K = M.K
                pairs = sorted(((W[i, j], i, j) for i in range(K) for j in range(i + 1, K)), key=lambda t: -abs(t[0]))
                log("strongest field couplings (w>0 attract, w<0 repel): " + '; '.join(f"F{i:02d}-F{j:02d} {w:+.2f}" for w, i, j in pairs[:16]))
                J[name + ' W'] = W.tolist(); J[name + ' marg'] = M.marg.tolist()
        log("\n## Summary (outside / |z|>3 / sum|z|) vs S366 best chain model M* 23/50, 16, 134 and floor 19/50, 141")
        for name, s in res.items(): log(f"  {name:60s} {s['outside']:2d}/{s['n']}  {s['z_gt3']:2d}  {s['sum_abs_z_capped']:.1f}")

    if CY == 3:
        base = ['ising', 'type', 'site', 'reuse', 'open', 'units', 'elemchain']
        M = FormModel(FIT, mech=base, form=F)
        describe_form(F, texts, model=M)
        W = M.W; K = M.K
        # forbidden / required co-selections from the fit data on fields
        fx = collections.Counter(); fboth = collections.Counter(); nU = len(M.units)
        for _, _, _, u in M.units:
            fs = sorted(set(f for f, _ in u))
            for f in fs: fx[f] += 1
            for i in range(len(fs)):
                for j in range(i + 1, len(fs)): fboth[(fs[i], fs[j])] += 1
        forbid = set(); attract = []
        for i in range(K):
            for j in range(i + 1, K):
                e = fx[i] * fx[j] / nU; o = fboth.get((i, j), 0)
                if e >= 5 and o <= 0.2 * e: forbid.add((i, j))
                if o >= 5 and o >= 3 * e: attract.append((o / e, i, j, o))
        log(f"\nforbidden field pairs (O <= 0.2E, E >= 5): {len(forbid)}: " + ', '.join(f"F{i:02d}-F{j:02d}" for i, j in sorted(forbid)))
        log(f"attracting field pairs (O >= 3E, O >= 5): {len(attract)}: " + ', '.join(f"F{i:02d}-F{j:02d} {r:.1f}x({o})" for r, i, j, o in sorted(attract, reverse=True)[:20]))
        log("Ising couplings |w| >= 1: " + '; '.join(f"F{i:02d}-F{j:02d} {W[i,j]:+.2f}" for i in range(K) for j in range(i + 1, K) if abs(W[i, j]) >= 1))
        obl = [i for i in range(K) if M.marg[i] >= 0.5]
        log(f"fields present in >= 50% of units: {[f'F{i:02d}' for i in obl]}; most frequent field F{int(np.argmax(M.marg)):02d} at {M.marg.max():.3f} -> no field is obligatory" if M.marg.max() < 0.9 else f"obligatory fields: {obl}")
        # distinct forms (field subsets) covering 90% of units / texts
        subsets = collections.Counter(tuple(sorted(set(f for f, _ in u))) for _, _, _, u in M.units)
        tot = sum(subsets.values()); acc = 0; k90 = 0
        for sub, c in subsets.most_common():
            acc += c; k90 += 1
            if acc / tot >= 0.9: break
        log(f"distinct field subsets: {len(subsets)} over {tot} units; {k90} cover 90%; top 12: " + '; '.join(f"{'+'.join(f'F{f:02d}' for f in sub)} x{c}" for sub, c in subsets.most_common(12)))
        seals = collections.Counter(tuple(sorted(set(f for f, _ in u))) for _, cls, ui, u in M.units if cls == 'SEAL' and ui == 0)
        tot = sum(seals.values()); acc = 0; k90s = 0
        for sub, c in seals.most_common():
            acc += c; k90s += 1
            if acc / tot >= 0.9: break
        log(f"seal first units: {len(seals)} subsets over {tot}; {k90s} cover 90%; singletons {sum(1 for c in seals.values() if c == 1)}")
        J['forbid'] = sorted(forbid); J['subsets90'] = k90; J['subsets'] = len(subsets)
        # ---- validity outside: IM77-new, held-out sites, fit set; vs within-text shuffle and cross-text shuffle
        IMN, unb, tot_tok = load_im77_new()
        log(f"\nIM77 certainly-new texts: {len(IMN)} ({unb}/{tot_tok} tokens unbridged -> skipped in the parse)")
        rng = random.Random(3)
        def validity(data, label, nrep=200):
            res = {}
            for mu, tag in ((1, 'single unit'), (2, '<= 2 units')):
                tx = [s for _, _, s in data if len([x for x in s if x in F.field_of]) >= 2]
                obs = sum(F.valid(s, forbid, max_units=mu) for s in tx)
                # within-text shuffle (order only)
                sh = []
                for _ in range(nrep):
                    c = 0
                    for s in tx:
                        l = list(s); rng.shuffle(l); c += F.valid(tuple(l), forbid, max_units=mu)
                    sh.append(c)
                # cross-text permutation of signs (keeps lengths; tests field membership + co-selection + order)
                cr = []
                pool = [x for s in tx for x in s]
                for _ in range(nrep):
                    rng.shuffle(pool); c = 0; k = 0
                    for s in tx:
                        l = pool[k:k + len(s)]; k += len(s); c += F.valid(tuple(l), forbid, max_units=mu)
                    cr.append(c)
                n = len(tx)
                log(f"  {label:22s} {tag:12s}: valid {obs}/{n} = {obs/n:.3f}; within-text shuffle {statistics.mean(sh)/n:.3f} [{min(sh)/n:.3f},{max(sh)/n:.3f}]; cross-text permutation {statistics.mean(cr)/n:.3f} [{min(cr)/n:.3f},{max(cr)/n:.3f}]")
                res[tag] = dict(n=n, valid=obs, shuffle=statistics.mean(sh) / n, cross=statistics.mean(cr) / n)
            return res
        J['valid_fit'] = validity(FIT, 'fit set (MD+H)', 50)
        J['valid_held'] = validity(HELD, 'held-out sites', 200)
        J['valid_im77'] = validity(IMN, 'IM77-new (324)', 200)
        J['valid_im77_seals'] = validity([d for d in IMN if d[1] == 'SEAL'], 'IM77-new seals', 200)
        # Ur III-style control: a Markov-2 clone of the fit set parsed with the same form (how much validity does a chain give?)
        clone = SA.PlainMarkov(FIT, k=2).generate_corpus(meta, random.Random(5))
        J['valid_markov2'] = validity(clone, 'Markov-2 clone', 20)
        # invalid IM77 texts: why
        why = collections.Counter()
        for _, _, s in IMN:
            if len([x for x in s if x in F.field_of]) < 2: continue
            units, viol, unk = F.parse(s)
            if viol: why['same-field repeat'] += 1
            elif len(units) > 1: why['order violated (second unit)'] += 1
            elif not F.valid(s, forbid): why['forbidden co-selection'] += 1
            else: why['valid'] += 1
        log(f"  IM77-new breakdown: {dict(why)}")
        J['im77_why'] = dict(why)
        # one-page specification
        log("\n## FORM SPECIFICATION (candidate 'whole code', fitted on MD+H, level " + LV + ")")
        log("A text = 1 unit (" + f"{M.n_units[None].c[1]/sum(M.n_units[None].c.values()) if None in M.n_units else 0:.2f}" + " of texts) or 2-3 units written one after the other; a unit = a subset of the fields below, one element each, in field order.")
        for i, f in enumerate(F.fields):
            top = f[:10]; share = sum(F.has[x] for x in top) / sum(F.has[x] for x in f)
            req = [j for j in range(K) if j != i and W[i, j] >= 1.0]; rep = [j for j in range(K) if j != i and W[i, j] <= -1.0]
            log(f"  F{i:02d} P={M.marg[i]:.2f} menu({len(f)}; top10 cover {share:.2f}): {' '.join(str(x) for x in top)}" + (f" | attracts {['F%02d' % j for j in req]}" if req else '') + (f" | repels {['F%02d' % j for j in rep]}" if rep else ''))

    log(f"\n(elapsed {time.time()-t0:.0f} s)")
    open(f"{OUTD}loop39_c{CY}_{LV}.txt", 'w').write('\n'.join(LOG) + '\n')
    json.dump(J, open(f"{OUTD}loop39_c{CY}_{LV}.json", 'w'), default=str)

if __name__ == '__main__':
    main()
