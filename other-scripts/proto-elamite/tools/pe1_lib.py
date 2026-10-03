"""PE-1 prosopography helpers: designations (entry middles), quantities, nulls.

Designation = entry sign string with the final class sign removed (when the last
sign is a final-slot class sign from test a/b). Only designations of >= 2 signs
with no 'x' are used (1-sign middles are too generic to be person-specific).
Data: data/pe_corpus.json (CDLI ATF, see FINDINGS.md)."""
import collections, math, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, entries, header, base, is_sign, DATA  # noqa

# final-slot class signs: z_final >= 3 in test a, plus capacity-tied M002, M243 (test b)
CLASS = {'M288', 'M297', 'M263', 'M346', 'M264', 'M072', 'M003', 'M354', 'M371',
         'M096', 'M376', 'M036', 'M317', 'M373', 'M002', 'M243'}
PREFIX = {'M387', 'M157', 'M370', 'M124', 'M305', 'M038', 'M111', 'M217', 'M304',
          'M059', 'M146'}

COUNT_V = {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 300}          # grade B (attack 2)
CAP_V = {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24,   # attack 2 a-priori set
         'N01': 120, 'N14': 720}
CAP_CODES = {'N39C', 'N30D', 'N30C', 'N24', 'N39B'}


def quantity(nums):
    """Return (system, value) or None. system 'C' (capacity) or 'K' (plain count)."""
    codes = {c for _, c in nums}
    if any(not isinstance(n, int) for n, _ in nums):
        return None
    if not codes or any('@' in c for c in codes):
        return None
    if codes & CAP_CODES:
        if codes <= set(CAP_V):
            return ('C', sum(n * CAP_V[c] for n, c in nums))
        return None
    if codes <= set(COUNT_V):
        return ('K', sum(n * COUNT_V[c] for n, c in nums))
    return None


def tablet_meta(T):
    meta = {}
    for t in T:
        h = header(t)
        hs = h[0] if h and h[0] != 'x' else None
        vol = t['designation'].split(',')[0].strip()
        meta[t['id']] = {'header': hs, 'header_full': tuple(h) if h else None, 'vol': vol,
                         'prov': t['provenience']}
    return meta


def designation_entries(T, strip_prefix=False, minlen=2, clean_only=False):
    """List of dicts: tablet, des (tuple), final (class sign or None), qty, system."""
    E = entries(T, require_clean=clean_only)
    out = []
    for e in E:
        s = list(e['signs'])
        fin = None
        if s and s[-1] in CLASS:
            fin = s[-1]
            s = s[:-1]
        if strip_prefix:
            while s and s[0] in PREFIX:
                s = s[1:]
        if len(s) < minlen or 'x' in s:
            continue
        q = quantity(e['numerals'])
        out.append({'tablet': e['tablet'], 'des': tuple(s), 'final': fin,
                    'sys': q[0] if q else None, 'val': q[1] if q else None,
                    'system': e['system']})
    return out


def one_variant(a, b):
    """True if a != b and they differ by one substitution, insertion or deletion."""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if abs(la - lb) != 1:
        return False
    if la > lb:
        a, b = b, a
    for i in range(len(b)):
        if b[:i] + b[i + 1:] == a:
            return True
    return False


def recurrence_stats(D):
    """D: list of (tablet, des). Stats of exact cross-tablet recurrence."""
    by = collections.defaultdict(list)
    for t, d in D:
        by[d].append(t)
    types_multi_tab = 0
    occ_recurring = 0
    within_dup = 0
    cross_pairs = 0
    for d, ts in by.items():
        c = collections.Counter(ts)
        within_dup += sum(v - 1 for v in c.values())
        if len(c) >= 2:
            types_multi_tab += 1
            occ_recurring += len(ts)
            k = len(c)
            cross_pairs += k * (k - 1) // 2
    return {'types': len(by), 'tokens': len(D), 'types_on_2plus_tablets': types_multi_tab,
            'tokens_in_recurring': occ_recurring, 'within_tablet_dups': within_dup,
            'cross_tablet_type_pairs': cross_pairs}


def variant_pairs(D, minlen=3):
    """Count pairs of distinct designation types (len>=minlen) that are one-sign
    variants of each other and occur on different tablets."""
    tabs = collections.defaultdict(set)
    for t, d in D:
        if len(d) >= minlen:
            tabs[d].add(t)
    types = list(tabs)
    # bucket by length for speed
    bylen = collections.defaultdict(list)
    for d in types:
        bylen[len(d)].append(d)
    n = 0
    for L, ds in bylen.items():
        cand = ds + bylen.get(L + 1, [])
        for i, a in enumerate(ds):
            for b in cand[i + 1:]:
                if one_variant(a, b) and (tabs[a] - tabs[b] or tabs[b] - tabs[a]):
                    n += 1
    return n


def shuffle_tokens(D, rng):
    """Null 1: keep every designation slot's length and tablet; refill with a
    global permutation of all sign tokens (keeps global sign frequencies)."""
    pool = [s for _, d in D for s in d]
    rng.shuffle(pool)
    out, i = [], 0
    for t, d in D:
        out.append((t, tuple(pool[i:i + len(d)])))
        i += len(d)
    return out


class Markov2:
    def __init__(self, strs):
        self.m2 = collections.defaultdict(collections.Counter)
        self.m1 = collections.defaultdict(collections.Counter)
        self.u = collections.Counter()
        for s in strs:
            seq = ['<s>', '<s>'] + list(s)
            for i in range(2, len(seq)):
                self.m2[(seq[i - 2], seq[i - 1])][seq[i]] += 1
                self.m1[seq[i - 1]][seq[i]] += 1
                self.u[seq[i]] += 1
        self.cache = {}

    def _draw(self, ctr, rng):
        ks, ws = zip(*ctr.items())
        return rng.choices(ks, ws)[0]

    def gen(self, L, rng):
        out = ['<s>', '<s>']
        for _ in range(L):
            c2 = self.m2.get((out[-2], out[-1]))
            if c2:
                out.append(self._draw(c2, rng))
                continue
            c1 = self.m1.get(out[-1])
            out.append(self._draw(c1 if c1 else self.u, rng))
        return tuple(out[2:])


def markov_null(D, model, rng):
    """Null 2: same slots (tablet, length); middle drawn from a Markov-2 model."""
    return [(t, model.gen(len(d), rng)) for t, d in D]


def zp(obs, null, greater=True):
    m = sum(null) / len(null)
    sd = (sum((x - m) ** 2 for x in null) / len(null)) ** .5
    if greater:
        p = (1 + sum(x >= obs for x in null)) / (1 + len(null))
    else:
        p = (1 + sum(x <= obs for x in null)) / (1 + len(null))
    z = (obs - m) / sd if sd > 0 else float('inf') if obs != m else 0.0
    return {'obs': obs, 'null_mean': round(m, 3), 'null_sd': round(sd, 3), 'z': round(z, 2), 'p': round(p, 4)}
