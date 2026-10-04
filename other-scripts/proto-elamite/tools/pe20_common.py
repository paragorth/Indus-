"""pe20 shared code: THE FLOCK DOES THE ARITHMETIC.

Herd-entry signs are assigned to animal categories (two "species" s0/s1, each with
F adult breeding female, M adult breeding male, YF young female, YM young male,
Y unsexed young, T species total) or X (not constrained). Each assignment is scored
by how well the counts on herd records fit a herd-biology model, as a log-likelihood
ratio against a generic count distribution g (all-X scores 0):
  M | F  ~ NegBin(mean rho*F + eps), rho (males per female) prior 0.02-0.30
  Y | F  ~ NegBin(mean f*F + eps),   f (surviving young per female) prior 0.2-1.3
  YF+YM | F as Y, and YF | YF+YM ~ BetaBinomial(10, 10) (sex ratio ~1:1)
  only one of YF / YM observed: NegBin(mean f*F/2 + eps)
  T: P(n_T) = 0.5*[n_T == sum of the species' other categories] + 0.5*g(n_T)
Parameters are shared across records and marginalised over their prior grids
(uniform), separately for each species. Priors come from pastoral data ranges
(lambing / kidding 0.6-1.0 per ewe per year, with juvenile losses; one breeding male
per 4-50 females); they are data ranges, not readings of any sign.
"""
import json, math, os, re, sys
from collections import Counter, defaultdict
import numpy as np
from scipy.special import gammaln, logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe20_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
sys.path.insert(0, HERE)

CATS = ['F', 'M', 'YF', 'YM', 'Y', 'T']
NC = len(CATS)
NSTATE = 1 + 2 * NC          # 0 = X ; 1 + s*NC + c
F_GRID = np.linspace(0.2, 1.3, 12)
R_GRID = np.geomspace(0.02, 0.30, 10)
EPS = 0.3
K_NB = 4.0
K_GRID = np.array([0.7, 2.0, 6.0])      # NB dispersion, marginalised with the rates
PI_T = 0.5


def set_cats(cats):
    """Switch the category set (e.g. add 'W'); module-level, affects all functions."""
    global CATS, NC, NSTATE
    CATS = list(cats)
    NC = len(CATS)
    NSTATE = 1 + 2 * NC


def state_name(a):
    if a == 0:
        return 'X'
    s, c = divmod(a - 1, NC)
    return '%s%d' % (CATS[c], s)


def lnb(n, m, k=K_NB):
    n = np.asarray(n, float)
    m = np.asarray(m, float)
    return (gammaln(n + k) - gammaln(k) - gammaln(n + 1) + k * np.log(k / (k + m))
            + n * np.log(m / (k + m)))


def lbetabin(x, n, a=10.0, b=10.0):
    return (gammaln(n + 1) - gammaln(x + 1) - gammaln(n - x + 1)
            + gammaln(x + a) + gammaln(n - x + b) - gammaln(n + a + b)
            + gammaln(a + b) - gammaln(a) - gammaln(b))


def fit_generic(values):
    """Generic count distribution: NegBin fitted by ML on a grid (smooth baseline)."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], float)
    best = None
    for k in np.geomspace(0.2, 20, 40):
        ll = lnb(v, max(v.mean(), 0.1), k).sum()
        if best is None or ll > best[0]:
            best = (ll, k)
    m, k = max(v.mean(), 0.1), best[1]
    return lambda n: lnb(n, m, k)


# ---------------------------------------------------------------- scorer
class Scorer:
    def __init__(self, V, lg=None, fg=None, rg=None):
        """V: (R, K) float array of counts, nan = missing. fg / rg: prior grids for
        young per female and males per female (default: pastoral ranges)."""
        fg = F_GRID if fg is None else np.asarray(fg, float)
        rg = R_GRID if rg is None else np.asarray(rg, float)
        # flattened (rate x dispersion) grids
        self.fg = np.repeat(fg, len(K_GRID))
        self.rg = np.repeat(rg, len(K_GRID))
        self.fk = np.tile(K_GRID, len(fg))
        self.rk = np.tile(K_GRID, len(rg))
        self.V = np.asarray(V, float)
        R, K = self.V.shape
        self.R, self.K = R, K
        self.lg = lg or fit_generic(self.V[~np.isnan(self.V)])
        obs = ~np.isnan(self.V)
        Vz = np.where(obs, self.V, 0.0)
        G = np.where(obs, self.lg(Vz), 0.0)
        self.obs, self.Vz, self.G = obs, Vz, G
        f = self.fg[None, :]
        r = self.rg[None, :]
        fk, rk = self.fk[None, :], self.rk[None, :]
        self.Mp = np.zeros((K, K, len(self.rg)))
        self.Yp = np.zeros((K, K, len(self.fg)))
        self.Yh = np.zeros((K, K, len(self.fg)))
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                w = obs[:, i] & obs[:, j]
                if not w.any():
                    continue
                Fi, Nj, Gj = Vz[w, i][:, None], Vz[w, j][:, None], G[w, j][:, None]
                self.Mp[i, j] = (lnb(Nj, r * Fi + EPS, rk) - Gj).sum(0)
                self.Yp[i, j] = (lnb(Nj, f * Fi + EPS, fk) - Gj).sum(0)
                self.Yh[i, j] = (lnb(Nj, f * Fi / 2 + EPS, fk) - Gj).sum(0)
        self._tri = {}
        self._tot = {}
        # W (non-breeding adult, e.g. castrates): mean omega*F, omega 0.05-3, same dispersions
        wg = np.repeat(np.geomspace(0.05, 3.0, 10), len(K_GRID))
        wk = np.tile(K_GRID, 10)
        self.Wp = np.zeros((K, K, len(wg)))
        for i in range(K):
            for j in range(K):
                w = obs[:, i] & obs[:, j]
                if i != j and w.any():
                    self.Wp[i, j] = (lnb(Vz[w, j][:, None], wg[None, :] * Vz[w, i][:, None] + EPS,
                                         wk[None, :]) - G[w, j][:, None]).sum(0)
        self.LW = logsumexp(self.Wp, -1) - math.log(len(wg))
        self.LM = logsumexp(self.Mp, -1) - math.log(len(self.rg))
        self.LY = logsumexp(self.Yp, -1) - math.log(len(self.fg))
        self.LH = logsumexp(self.Yh, -1) - math.log(len(self.fg))
        self._ltri = {}

    def ltri(self, i, j, l):
        key = (i, j, l)
        v = self._ltri.get(key)
        if v is None:
            v = float(logsumexp(self.tri(i, j, l)) - math.log(len(self.fg)))
            self._ltri[key] = v
        return v

    def tri(self, i, j, l):
        key = (i, j, l)
        if key in self._tri:
            return self._tri[key]
        o = self.obs
        f = self.fg[None, :]
        fk = self.fk[None, :]
        out = np.zeros(len(self.fg))
        both = o[:, i] & o[:, j] & o[:, l]
        if both.any():
            Fi = self.Vz[both, i][:, None]
            a, b = self.Vz[both, j], self.Vz[both, l]
            out += (lnb((a + b)[:, None], f * Fi + EPS, fk)
                    - (self.G[both, j] + self.G[both, l])[:, None]
                    + lbetabin(a, a + b)[:, None]).sum(0)
        for x, y in ((j, l), (l, j)):
            only = o[:, i] & o[:, x] & ~o[:, y]
            if only.any():
                Fi = self.Vz[only, i][:, None]
                out += (lnb(self.Vz[only, x][:, None], f * Fi / 2 + EPS, fk)
                        - self.G[only, x][:, None]).sum(0)
        self._tri[key] = out
        return out

    def tot(self, t, members):
        key = (t, members)
        if key in self._tot:
            return self._tot[key]
        m = list(members)
        w = self.obs[:, t] & self.obs[:, m].all(1)
        s = 0.0
        if w.any():
            S = self.Vz[w][:, m].sum(1)
            nt = self.Vz[w, t]
            hit = (np.abs(S - nt) < 0.5)
            s = float(np.sum(np.log(PI_T * hit * np.exp(-self.G[w, t]) + (1 - PI_T))))
        self._tot[key] = s
        return s

    def score(self, a):
        """a: sequence of K states. Returns log-LR vs all-X, or None if invalid."""
        total = 0.0
        for s in (0, 1):
            slot = {}
            for j, st in enumerate(a):
                if st and (st - 1) // NC == s:
                    c = CATS[(st - 1) % NC]
                    if c in slot:
                        return None
                    slot[c] = j
            if 'Y' in slot and ('YF' in slot or 'YM' in slot):
                return None
            if 'F' in slot:
                i = slot['F']
                if 'M' in slot:
                    total += self.LM[i, slot['M']]
                if 'W' in slot:
                    total += self.LW[i, slot['W']]
                if 'Y' in slot:
                    total += self.LY[i, slot['Y']]
                elif 'YF' in slot and 'YM' in slot:
                    total += self.ltri(i, slot['YF'], slot['YM'])
                elif 'YF' in slot:
                    total += self.LH[i, slot['YF']]
                elif 'YM' in slot:
                    total += self.LH[i, slot['YM']]
            if 'T' in slot:
                mem = tuple(sorted(v for c, v in slot.items() if c != 'T'))
                if mem:
                    total += self.tot(slot['T'], mem)
        return total


# ---------------------------------------------------------------- search
def random_assignment(K, rng):
    while True:
        slots = np.array([0] * K + list(range(1, NSTATE)))
        rng.shuffle(slots)
        a = slots[:K].tolist()
        # X copies are all 0: fine. Reject Y/YF/YM conflicts.
        ok = True
        for s in (0, 1):
            cs = {CATS[(x - 1) % NC] for x in a if x and (x - 1) // NC == s}
            if 'Y' in cs and ('YF' in cs or 'YM' in cs):
                ok = False
        if ok:
            return a


def random_search(sc, n, rng, keep=200):
    scores = np.empty(n)
    best = []
    for t in range(n):
        a = random_assignment(sc.K, rng)
        v = sc.score(a)
        scores[t] = v
        best.append((v, a))
        if len(best) > 4 * keep:
            best.sort(key=lambda x: -x[0])
            best = best[:keep]
    best.sort(key=lambda x: -x[0])
    return scores, best[:keep]


def gibbs(sc, sweeps, rng, beta=1.0, init=None, burn=None):
    """Gibbs sampler over valid assignments (uniform prior). Returns marginal counts
    (K, NSTATE), best assignment, best score."""
    K = sc.K
    a = list(init) if init is not None else [0] * K
    cur = sc.score(a)
    marg = np.zeros((K, NSTATE))
    burn = sweeps // 4 if burn is None else burn
    best = (cur, list(a))
    for sw in range(sweeps):
        for j in rng.permutation(K):
            lp = np.full(NSTATE, -np.inf)
            for st in range(NSTATE):
                b = a[:]
                b[j] = st
                v = sc.score(b)
                if v is not None:
                    lp[st] = beta * v
            p = np.exp(lp - lp.max())
            p /= p.sum()
            a[j] = int(rng.choice(NSTATE, p=p))
        cur = sc.score(a)
        if cur > best[0]:
            best = (cur, list(a))
        if sw >= burn:
            for j in range(K):
                marg[j, a[j]] += 1
    return marg / max(1, sweeps - burn), best[1], best[0]


def maximise(sc, rng, restarts=6, sweeps=12):
    """Max score by annealed Gibbs from random starts (for null distributions)."""
    best = (-1e9, None)
    for _ in range(restarts):
        a = random_assignment(sc.K, rng)
        for beta in np.linspace(0.5, 8, sweeps):
            _, a, v = gibbs(sc, 1, rng, beta=beta, init=a, burn=0)
        v = sc.score(a)
        if v > best[0]:
            best = (v, a)
    return best


def collapse_marg(marg):
    """Species-symmetric summary: per sign, P(X), P(F), P(M), P(YF), P(YM), P(Y), P(T)."""
    out = np.zeros((marg.shape[0], 1 + NC))
    out[:, 0] = marg[:, 0]
    for s in (0, 1):
        out[:, 1:] += marg[:, 1 + s * NC: 1 + (s + 1) * NC]
    return out


def same_species(marg, i, j):
    """P(sign i and sign j are in the same species), from per-sign marginals is not
    enough; caller should use sampled assignments. Kept for API symmetry."""
    raise NotImplementedError


# ---------------------------------------------------------------- PE data
PE_SIGNS = ['M362', 'M367', 'M346', 'M006', 'M362~a', 'M367~a', 'M346~a', 'M006@g']
LETTER = {'a': 'M367', 'b': 'M346', 'c': 'M006', 'd': 'M362~a', 'e': 'M367~a',
          'f': 'M346~a', 'g': 'M006@g'}
LIST_TABS = {'P008283', 'P008295'}       # MDP 17,085 / 17,097: herd ration lists (pe5)
CHAIN_COPY = 'P008389'                    # MDP 17,191 = herd 3 of MDP 17,096+
MAIN = 'P008294'                          # MDP 17,096+325+380


def clean(s):
    return re.sub(r'[#?!\[\]]', '', s)


def canon(signs):
    sg = [clean(s) for s in signs]
    if not sg:
        return None
    last = sg[-1]
    m = {'M362~a': 'M362~a', 'M367': 'M367', 'M367~a': 'M367~a', 'M367~a1': 'M367~a',
         'M346': 'M346', 'M346~a': 'M346~a', 'M346~a2': 'M346~a', 'M346~a3': 'M346~a',
         'M346~a4': 'M346~a', 'M006': 'M006', 'M006@g': 'M006@g', 'M006@g~1': 'M006@g',
         'M006@g~2': 'M006@g', 'M362': 'M362'}
    if last in m:
        return m[last]
    if any(s.startswith('|M362+') or s.startswith('M362+') or s == 'M362' for s in sg):
        return 'M362'
    return None


def numval(l):
    raw = l['raw']
    if '[...]' in raw.split(',')[-1] or 'n(' in raw:
        return None
    if not l['numerals']:
        return None
    v = 0
    for n, u in l['numerals']:
        if n is None or u not in ('N01', 'N14'):
            return None
        v += n * (10 if u == 'N14' else 1)
    return float(v)


def pe_records(include=None):
    """Herd records: list of (tablet id, block label, {sign: value or None})."""
    T = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
    out = []
    for t in T:
        if include is not None and t['id'] not in include:
            continue
        if t['id'] in LIST_TABS:
            continue
        lines = t['lines']
        if t['id'] == MAIN:
            blocks = defaultdict(list)
            for l in lines:
                m = re.match(r"^(\d+)(?:\.([a-z]))?", l['label'].replace('?', ''))
                if not m:
                    continue
                blocks[(l['surface'], int(m.group(1)))].append((m.group(2), l))
            for key, bl in blocks.items():
                hdr = [l for x, l in bl if x is None]
                if not hdr:
                    continue
                rec = {s: None for s in PE_SIGNS}
                rec['M362'] = numval(hdr[0])
                present = {}
                for x, l in bl:
                    if x:
                        present[x] = l
                last_letter = max(present) if present else 'a'
                frag = any(l['lacuna'] or '[' in l['raw'] for _, l in bl)
                for x, sgn in LETTER.items():
                    if x in present:
                        rec[sgn] = numval(present[x])
                    elif x < last_letter or not frag:
                        rec[sgn] = 0.0
                out.append((t['id'], '%s%d' % (key[0][0], key[1]), rec))
            continue
        cur, cur_lab = {}, None
        for l in lines:
            c = canon(l['signs'])
            if c is None:
                continue
            v = numval(l)
            if c in cur or (c == 'M362' and cur):
                out.append((t['id'], cur_lab, cur))
                cur, cur_lab = {}, None
            cur[c] = v
            cur_lab = cur_lab or (l['surface'][0] + l['label'])
        if cur:
            out.append((t['id'], cur_lab, cur))
    res = []
    for tid, lab, rec in out:
        full = {s: rec.get(s) for s in PE_SIGNS}
        if sum(v is not None for v in full.values()) >= 2:
            res.append((tid, lab, full))
    return res


def to_matrix(recs, signs):
    return np.array([[np.nan if r[s] is None else r[s] for s in signs] for _, _, r in recs])


# ---------------------------------------------------------------- Ur III data
UR_SIGNS = ['u8', 'udu-nita2', 'sila4', 'kir11', 'ud5', 'masz2-nita2', 'masz2', 'asz2-gar3']
UR_TRUTH = {'u8': ('F', 'sheep'), 'udu-nita2': ('M', 'sheep'), 'sila4': ('YM|Y', 'sheep'),
            'kir11': ('YF', 'sheep'), 'ud5': ('F', 'goat'), 'masz2-nita2': ('M', 'goat'),
            'masz2': ('YM|Y', 'goat'), 'asz2-gar3': ('YF', 'goat')}


def ur_term(rest):
    w = rest.replace('-', ' ').split()
    if not w:
        return None
    cats = [x for x in w if x in ('u8', 'kir11', 'ud5', 'sila4')]
    if len(cats) > 1:
        return None
    if w[0] == 'u8':
        return 'u8'
    if w[0] == 'udu' and len(w) > 1 and w[1] == 'nita2':
        return 'udu-nita2'
    if w[0] == 'sila4':
        return 'sila4'
    if w[0] == 'kir11':
        return 'kir11'
    if w[0] == 'ud5':
        return 'ud5'
    if w[0] == 'masz2' and len(w) > 1 and w[1] in ('nita2', 'gal'):
        return 'masz2-nita2'
    if w[0] == 'masz2':
        return 'masz2'
    if (w[0] == 'munus' and len(w) > 1 and w[1] == 'asz2') or w[0] == 'asz2':
        return 'asz2-gar3'
    return None


HERD_PROV = ('Girsu', 'Umma')


def ur_herd_records():
    """Ur III herd records from Girsu and Umma (breeding-flock offices), with lines
    for fattened (niga) and dead (ba-usz2, ba-ug7) animals left out."""
    recs = ur_records(cache=os.path.join(CK, 'ur3_records_clean.json'), skipdead=True)
    prov = json.load(open(os.path.join(CK, 'ur3_prov.json')))
    return [r for r in recs if r[0] in prov and prov[r[0]][0].startswith(HERD_PROV)
            and prov[r[0]][1].startswith('Ur III')]


def ur_records(cache=os.path.join(CK, 'ur3_records.json'), skipdead=False):
    if os.path.exists(cache):
        return [tuple(x) for x in json.load(open(cache))]
    from pe5_common import parse_ur_line
    out = []
    cur, recs = None, []

    def flush():
        if cur is None:
            return
        rec = {}
        lab = 0
        for term, v in recs:
            if term in rec:
                out.append((cur, lab, rec))
                rec, lab = {}, lab + 1
            rec[term] = v
        if rec:
            out.append((cur, lab, rec))
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            flush()
            cur, recs = raw[1:8], []
            continue
        m = re.match(r"^\d+'?\.\s+(.*)$", raw.strip())
        if not m:
            continue
        p = parse_ur_line(m.group(1))
        if not p or p[1]:
            continue
        if skipdead and re.search(r'\b(niga|ba-usz2|ba-ug7|ri-ri-ga)\b', p[2]):
            continue
        term = ur_term(p[2])
        if term:
            recs.append((term, float(p[0])))
    flush()
    keep = []
    for pid, lab, rec in out:
        if len(rec) >= 3 and ('u8' in rec or 'ud5' in rec):
            keep.append((pid, lab, {s: rec.get(s) for s in UR_SIGNS}))
    json.dump(keep, open(cache, 'w'))
    return keep


def correct_role(state, truth):
    """Does a collapsed state (category name, species index) match the truth role?"""
    c = state
    return c in truth.split('|')


# ---------------------------------------------------------------- planted corpus
def planted(mask, rng, f=0.75, rho=0.08, Fmean=25.0):
    """Herd records with known roles in PE_SIGNS order:
    M362=F0, M367=M0, M346=YF0, M006=YM0, M362~a=F1, M367~a=M1, M346~a=X, M006@g=X.
    mask: (R, K) bool of observed cells (copied from the real search set)."""
    R, K = mask.shape
    V = np.full((R, K), np.nan)
    for r in range(R):
        row = np.zeros(K)
        for s, (iF, iM, iYF, iYM) in enumerate(((0, 1, 2, 3), (4, 5, None, None))):
            Fv = rng.negative_binomial(2, 2 / (2 + Fmean * (1.0 if s == 0 else 0.6)))
            row[iF] = Fv
            row[iM] = rng.poisson(rho * Fv + 0.2)
            if iYF is not None:
                y = rng.poisson(f * Fv)
                yf = rng.binomial(y, 0.5)
                row[iYF], row[iYM] = yf, y - yf
        row[6] = rng.negative_binomial(1, 1 / (1 + 4))
        row[7] = rng.negative_binomial(1, 1 / (1 + 2))
        V[r] = np.where(mask[r], row, np.nan)
    return V


PLANT_TRUTH = ['F0', 'M0', 'YF0', 'YM0', 'F1', 'M1', 'X', 'X']


def shuffle_all(V, rng):
    """Numbers shuffled across all observed entries."""
    W = V.copy()
    o = ~np.isnan(W)
    vals = W[o]
    rng.shuffle(vals)
    W[o] = vals
    return W


def shuffle_within_sign(V, rng):
    """Each sign's numbers permuted across records (keeps each sign's magnitudes,
    destroys within-herd coupling)."""
    W = V.copy()
    for j in range(W.shape[1]):
        o = ~np.isnan(W[:, j])
        vals = W[o, j]
        rng.shuffle(vals)
        W[o, j] = vals
    return W


def species_match(a, truth):
    """Best match of an assignment's names to a truth list under species relabelling;
    returns number of signs correct."""
    best = 0
    for swap in (False, True):
        n = 0
        for x, t in zip(a, truth):
            nm = state_name(x)
            if swap and nm != 'X':
                nm = nm[:-1] + ('1' if nm[-1] == '0' else '0')
            n += nm == t
        best = max(best, n)
    return best


def dump(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(obj, open(path, 'w'), indent=1, default=conv)
