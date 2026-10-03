"""pe4: are Proto-Elamite entry middles item codes (factorial attribute
descriptions) rather than personal names?  Shared code.

Models (type level, a corpus = list of distinct sign tuples):
  name model    : free strings spelled by a Markov chain (best of order 0 and 1),
                  Dirichlet-multinomial (alpha 0.5) codes, end symbol for length.
  factorial     : k ordered slots; every sign type belongs to exactly one slot;
                  a valid string has at most one sign per slot, in slot order;
                  slot presence = independent Bernoulli per slot; value = per-slot
                  multinomial; invalid strings are escaped and spelled order-0.
                  Assignment costs |V| log2 k bits.
All costs in bits.  No sound values or readings are used.
"""
import json, math, os, random, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, entries, base, is_sign  # noqa: E402

PEDATA = os.path.join(HERE, '..', 'data')
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DARK = os.path.join(ROOT, 'data', 'derived', 'dark', 'loop56_corpora')
LG2 = math.log(2)
A = 0.5

_r = json.load(open(os.path.join(PEDATA, 'res_a_slots.json')))['rows']
FINAL = {x['sign'] for x in _r if x['z_final'] >= 3}
INIT = {x['sign'] for x in _r if x['z_init'] >= 3}


# ------------------------------------------------------------- code lengths
def dm_bits(counts, K, a=A):
    """Dirichlet-multinomial marginal code length (bits) of a count vector over
    an alphabet of size K (only non-zero counts need be passed)."""
    n = sum(counts)
    if n == 0:
        return 0.0
    s = math.lgamma(K * a) - math.lgamma(K * a + n)
    for c in counts:
        if c:
            s += math.lgamma(a + c) - math.lgamma(a)
    return -s / LG2


def name_bits(corpus):
    """Best free-string model: min over Markov order 0 and 1."""
    V = {s for w in corpus for s in w}
    K = len(V) + 1
    c0 = Counter()
    c1 = defaultdict(Counter)
    for w in corpus:
        prev = '<s>'
        for s in list(w) + ['</s>']:
            c0[s] += 1
            c1[prev][s] += 1
            prev = s
    b0 = dm_bits(list(c0.values()), K)
    b1 = sum(dm_bits(list(c.values()), K) for c in c1.values())
    return min(b0, b1) + 1.0, {'order0': b0, 'order1': b1}


class Factorial:
    def __init__(self, corpus, k, assign=None, rng=None):
        self.C = corpus
        self.k = k
        self.V = sorted({s for w in corpus for s in w})
        self.rng = rng or random.Random(0)
        self.idx = defaultdict(list)
        for i, w in enumerate(corpus):
            for s in set(w):
                self.idx[s].append(i)
        self.g = assign if assign is not None else self.init_assign()
        self.rebuild()

    def init_assign(self):
        pos = defaultdict(list)
        for w in self.C:
            L = len(w)
            for j, s in enumerate(w):
                pos[s].append((j + 0.5) / L)
        mp = {s: sum(v) / len(v) for s, v in pos.items()}
        return {s: min(self.k - 1, int(mp[s] * self.k)) for s in self.V}

    def valid(self, w):
        last = -1
        for s in w:
            sl = self.g[s]
            if sl <= last:
                return False
            last = sl
        return True

    def rebuild(self):
        self.ok = [self.valid(w) for w in self.C]
        self.pres = [0] * self.k
        self.val = [Counter() for _ in range(self.k)]
        self.esc = Counter()
        self.nok = 0
        for w, ok in zip(self.C, self.ok):
            self._add(w, ok, +1)

    def _add(self, w, ok, d):
        if ok:
            self.nok += d
            for s in w:
                sl = self.g[s]
                self.pres[sl] += d
                self.val[sl][s] += d
        else:
            for s in list(w) + ['</s>']:
                self.esc[s] += d

    def bits(self):
        n = len(self.C)
        nok = self.nok
        b = dm_bits([nok, n - nok], 2)
        size = Counter(self.g.values())
        for sl in range(self.k):
            b += dm_bits([self.pres[sl], nok - self.pres[sl]], 2)
            b += dm_bits(list(self.val[sl].values()), max(1, size[sl]))
        b += dm_bits(list(self.esc.values()), len(self.V) + 1)
        b += len(self.V) * math.log2(self.k) if self.k > 1 else 0.0
        return b

    def move(self, s, new):
        old = self.g[s]
        if old == new:
            return
        aff = self.idx[s]
        for i in aff:
            self._add(self.C[i], self.ok[i], -1)
        self.g[s] = new
        for i in aff:
            self.ok[i] = self.valid(self.C[i])
            self._add(self.C[i], self.ok[i], +1)

    def fit(self, sweeps=6):
        cur = self.bits()
        for _ in range(sweeps):
            changed = 0
            order = sorted(self.V, key=lambda s: -len(self.idx[s]))
            for s in order:
                best, bsl = cur, self.g[s]
                old = self.g[s]
                for sl in range(self.k):
                    if sl == old:
                        continue
                    self.move(s, sl)
                    b = self.bits()
                    if b < best - 1e-9:
                        best, bsl = b, sl
                    self.move(s, old)
                if bsl != old:
                    self.move(s, bsl)
                    cur = best
                    changed += 1
            if not changed:
                break
        return cur

    def valid_share(self):
        return self.nok / len(self.C)


def best_factorial(corpus, kmax=6, seed=0):
    best = None
    for k in range(1, kmax + 1):
        F = Factorial(corpus, k, rng=random.Random(seed + k))
        b = F.fit()
        if best is None or b < best[0]:
            best = (b, k, F)
    return best


def mdl_compare(corpus, kmax=6, seed=0):
    """Return per-string bits saved by the factorial model vs the name model
    (positive = factorial better), plus details."""
    nb, nd = name_bits(corpus)
    fb, k, F = best_factorial(corpus, kmax, seed)
    n = len(corpus)
    return {'n': n, 'name_bits': nb, 'fact_bits': fb, 'k': k,
            'gain_per_str': (nb - fb) / n, 'valid_share': F.valid_share(),
            'mean_len': sum(map(len, corpus)) / n}, F


# ------------------------------------------------------------- corpora
def pe_records(clean=True):
    """PE entries with class = final class sign (or '-' if none), middle =
    signs before it, the numeral system and a value in base units."""
    T = load()
    E = entries(T, require_clean=clean)
    out = []
    for e in E:
        s = e['signs']
        if s[-1] in FINAL and len(s) >= 2:
            cls, mid = s[-1], tuple(s[:-1])
        else:
            cls, mid = '-', tuple(s)
        out.append({'t': e['tablet'], 'head': cls, 'attr': mid, 'sys': e['system'],
                    'n': value(e['numerals'], e['system']), 'prov': e['prov']})
    return out


CNT = {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 300, 'N48': 3000}   # FINDINGS attack 2, grade B
CAP = {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 720}


def value(nums, sys_):
    tab = CAP if sys_ == 'C' else CNT if sys_ == 'SDB' else None
    if tab is None:
        return None
    v = 0
    for n, c in nums:
        if c not in tab or not isinstance(n, int):
            return None
        v += n * tab[c]
    return v or None


def controls():
    return json.load(open(os.path.join(PEDATA, 'pe4_controls.json')))


def load_list(name):
    out = set()
    for line in open(os.path.join(DARK, name + '.jsonl'), encoding='utf-8'):
        seq = json.loads(line)['seq']
        if any((not t) or '$' in t or '(' in t or t in ('x', '...') for t in seq):
            continue
        out.add(tuple(seq))
    return sorted(out)


def synth_herd(n_records, rng, k=5, sizes=(2, 4, 5, 3, 20), p_present=(0.9, 0.8, 0.5, 0.4, 0.6)):
    """Planted factorial ledger: sex x age x colour x breed x owner-group, each
    value its own sign, Zipf value frequencies, independent attributes."""
    alph = []
    for a, m in enumerate(sizes):
        alph.append(['A%d_%d' % (a, j) for j in range(m)])
    recs = []
    for _ in range(n_records):
        w = []
        for a in range(k):
            if rng.random() < p_present[a]:
                m = sizes[a]
                wts = [1 / (j + 1) for j in range(m)]
                w.append(rng.choices(alph[a], wts)[0])
        if w:
            recs.append(tuple(w))
    return recs


def sample_types(types, n, rng, minlen=1):
    pool = [t for t in types if len(t) >= minlen]
    return rng.sample(pool, min(n, len(pool)))
