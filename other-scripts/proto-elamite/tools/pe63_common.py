#!/usr/bin/env python3
"""pe63 A DOSSIER IS A CONTROLLED EXPERIMENT: shared library.

Tablet record (all corpora, signs opaque to the method):
  {'id', 'ord' (museum/publication order key), 'lines': [{'s': [signs], 'sys': None|str, 'v': float|None}]}
Corpora: PE (pe_corpus.json, base signs, pe47 values), Ur III Drehem and Umma (pe38 docs, words as opaque
tokens; numerals are outside the token list). Truth roles for Ur III tokens are computed only for scoring.
"""
import os, sys, re, json, math, random, hashlib
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe63_ckpt')
os.makedirs(CK, exist_ok=True)


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ corpora
def pe_tabs():
    fn = os.path.join(CK, 'pe_tabs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    from common import load, base, is_sign, system_of
    from pe47_common import pe_value
    out = []
    for t in load():
        L = []
        for l in t['lines']:
            sg = [base(s) for s in l['signs'] if s and s != 'x' and is_sign(s)]
            if l['numerals']:
                codes = [[n, c] for n, c in l['numerals']]
                sy = system_of(codes) if all(c for _, c in codes) else None
                v = None
                tail = l['raw'].split(',')[-1] if ',' in l['raw'] else l['raw']
                if not ('...' in tail or '[' in tail):
                    try:
                        pv = pe_value(codes)
                        v = float(pv[0]) if pv else None
                    except Exception:
                        v = None
                L.append({'s': sg[:12], 'sys': sy or '?', 'v': v,
                          'raw': ' '.join('%s(%s)' % (n, c) for n, c in codes)})
            elif sg:
                L.append({'s': sg[:12], 'sys': None, 'v': None, 'raw': ''})
        if len(L) >= 2 and len(L) <= 60:
            out.append({'id': t['id'], 'ord': int(t['id'][1:]), 'des': t.get('designation', ''),
                        'lines': L})
    json.dump(out, open(fn, 'w'))
    return out


def ur3_tabs(site='Puzr', n=1500, tag='DR'):
    fn = os.path.join(CK, 'ur3_%s.json' % tag)
    if os.path.exists(fn):
        return json.load(open(fn))
    from fractions import Fraction as Fr
    src = [d for d in json.load(open(os.path.join(DATA, 'pe38_ckpt', 'ur3_docs.json')))
           if d['site'].startswith(site)]
    rng = random.Random(seed('pe63-' + tag))
    rng.shuffle(src)
    out = []
    for d in src:
        L = []
        for l in d['lines']:
            toks = [t for t in l['toks'] if t and '...' not in t][:12]
            v = None
            if l['val'] is not None and l['sys'] in (1, 2):
                try:
                    v = float(Fr(l['val']))
                except Exception:
                    v = None
                L.append({'s': toks, 'sys': 'u%d' % l['sys'], 'v': v, 'raw': str(l['val'])})
            elif toks:
                L.append({'s': toks, 'sys': None, 'v': None, 'raw': ''})
        if 3 <= len(L) <= 60:
            out.append({'id': d['id'], 'ord': int(re.sub(r'\D', '', d['id']) or 0), 'des': d['site'],
                        'lines': L})
        if len(out) >= n:
            break
    json.dump(out, open(fn, 'w'))
    return out


# Ur III truth roles (scoring only)
FUNC = {'ki', 'giri3', 'maszkim', 'i3-dab5', 'szu', 'ba-ti', 'kiszib3', 'mu-kux(DU)', 'ba-zi', 'sukkal',
        'dub-sar', 'u3', 'lu2', 'sza3', 'zi-ga', 'ba-ug7', 'e2-muhaldim', 'mu-DU', 'szu-ba-ti', 'ugula',
        'nu-banda3', 'dumu', 'sag-nig2-gur11-ra-kam', 'ba-an-zi', 'ba-zi-ge'}
PERSON_CUES = {'ki', 'giri3', 'maszkim', 'i3-dab5', 'kiszib3', 'mu-kux(DU)', 'ba-zi', 'ba-ti', 'ugula',
               'dub-sar', 'sukkal', 'nu-banda3'}


def ur3_role(line):
    """truth role of the variable content of a line: TIME / PERSON / COMMODITY / OTHER"""
    s = line['s']
    if not s:
        return 'OTHER'
    if s[0] in ('iti', 'mu') or (s[0] == 'u4' and any(t.endswith('-kam') for t in s)):
        return 'TIME'
    if set(s) & PERSON_CUES or any(t.endswith('-ta') or t.endswith('-sze3') for t in s):
        return 'PERSON'
    if line['sys'] is not None:
        return 'COMMODITY'
    return 'OTHER'


# ------------------------------------------------------------------ planting
def plant_dossier(T, rng, k=6, tag='PL'):
    """Copy a real tablet k times. Slot A (identifier): one sign in one numeral line swapped, number kept.
    Slot B (commodity/unit): sign swapped AND number system + scale changed. Slot C (step marker): sign
    swapped with a regular x2 step in quantity. Everything else fixed. Returns new corpus and truth."""
    cand = [t for t in T if sum(l['sys'] is not None and len(l['s']) >= 2 for l in t['lines']) >= 3
            and 5 <= len(t['lines']) <= 12]
    base_t = rng.choice(cand)
    nl = [i for i, l in enumerate(base_t['lines']) if l['sys'] is not None and len(l['s']) >= 2]
    a, b, c = rng.sample(nl, 3)
    allsig = [s for t in T for l in t['lines'] for s in l['s']]
    systems = sorted({l['sys'] for t in T for l in t['lines'] if l['sys']})
    stepsig = rng.sample(allsig, k)
    out = []
    for j in range(k):
        t = json.loads(json.dumps(base_t))
        t['id'] = '%s%02d' % (tag, j)
        t['ord'] = -1000 - j * 977
        la, lb, lc = t['lines'][a], t['lines'][b], t['lines'][c]
        la['s'][-1] = rng.choice(allsig)                       # identifier: swap, number fixed
        lb['s'][-1] = rng.choice(allsig)                       # commodity: swap + system + scale
        lb['sys'] = rng.choice(systems)
        lb['v'] = float(rng.choice([1, 3, 10, 30, 60, 120, 300]))
        lc['s'][-1] = stepsig[j]                               # step marker: swap + x2 step
        lc['v'] = float(2 ** j)
        out.append(t)
    truth = {'base': base_t['id'], 'ids': [t['id'] for t in out], 'A': a, 'B': b, 'C': c}
    return T + out, truth


# ------------------------------------------------------------------ features
def line_key(l):
    return '|'.join(sorted(set(l['s']))) + '#' + (l['sys'] or 'H')


def feats(t):
    """template feature families for a tablet"""
    F = set()
    L = t['lines']
    for l in L:
        if l['s']:
            F.add('L:' + line_key(l))
    for s in {s for l in L for s in l['s']}:
        F.add('S:' + s)
    if L[0]['s']:
        F.add('H:' + L[0]['s'][0])
    F.add('C:' + line_key(L[-1]))
    F.add('Q:' + '.'.join((l['sys'] or 'H') for l in L)[:60])
    F.add('N:%d' % min(len(L), 20))
    F.add('P:%d' % (t['ord'] // 6))
    F.add('P2:%d' % ((t['ord'] + 3) // 6))
    return F


class Index:
    """IDF-weighted similarity (line keys + signs) and feature bitsets."""

    def __init__(self, T):
        self.T = T
        n = len(T)
        self.F = [feats(t) for t in T]
        df = Counter(f for F in self.F for f in F)
        self.df = df
        self.idf = {f: math.log(n / c) for f, c in df.items()}
        # similarity vectors: line keys and signs only (the content)
        vocab = sorted(f for f in df if f[:2] in ('L:', 'S:'))
        self.vid = {f: i for i, f in enumerate(vocab)}
        import scipy.sparse as sp
        rows, cols, vals = [], [], []
        for i, F in enumerate(self.F):
            for f in F:
                if f in self.vid:
                    rows.append(i); cols.append(self.vid[f]); vals.append(self.idf[f])
        X = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)))
        self.norm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-9
        self.S = (X @ X.T).toarray() / np.outer(self.norm, self.norm)
        np.fill_diagonal(self.S, 0)
        # tablet-calibrated baseline: each tablet's typical similarity to the corpus
        self.mu = self.S.sum(1) / (n - 1)
        self.sd = np.sqrt(np.maximum((self.S ** 2).sum(1) / (n - 1) - self.mu ** 2, 1e-6))
        self.bits = defaultdict(int)
        for i, F in enumerate(self.F):
            for f in F:
                self.bits[f] |= (1 << i)

    def members(self, tmpl):
        b = -1
        for f in tmpl:
            b &= self.bits[f]
        return b

    @staticmethod
    def unpack(b):
        return [i for i, c in enumerate(bin(b)[:1:-1]) if c == '1']

    def heldout(self, G, tmpl):
        """mean pairwise similarity after removing the template's own content features"""
        G = np.asarray(G)
        sub = self.S[np.ix_(G, G)].copy()
        rm = 0.0
        exp = set()
        for f in tmpl:
            if f.startswith('L:') or f.startswith('C:'):
                key = f[2:]
                exp.add('L:' + key)
                for s in key.split('#')[0].split('|'):
                    if s:
                        exp.add('S:' + s)
            elif f.startswith('S:') or f.startswith('H:'):
                exp.add('S:' + f[2:])
        for f in exp:
            if f in self.vid:
                rm += self.idf[f] ** 2
        nn = self.norm[G]
        sub -= rm / np.outer(nn, nn)
        mu, sd = self.mu[G], self.sd[G]
        z = (sub - (mu[:, None] + mu[None, :]) / 2) / np.sqrt((sd[:, None] ** 2 + sd[None, :] ** 2) / 2)
        m = len(G)
        return float((z.sum() - np.trace(z)) / (m * (m - 1)))


def shuffle_lines(T, rng):
    """null corpus: every line moved to a random tablet slot of the same kind (header/numeral) and same
    position class (first / last / middle); tablet lengths and line frequencies kept, real copies destroyed"""
    pools = defaultdict(list)
    for t in T:
        for i, l in enumerate(t['lines']):
            pc = 'f' if i == 0 else ('l' if i == len(t['lines']) - 1 else 'm')
            pools[(l['sys'] is None, pc)].append(l)
    for p in pools.values():
        rng.shuffle(p)
    out = []
    for t in T:
        L = []
        for i, l in enumerate(t['lines']):
            pc = 'f' if i == 0 else ('l' if i == len(t['lines']) - 1 else 'm')
            L.append(pools[(l['sys'] is None, pc)].pop())
        out.append({'id': t['id'], 'ord': t['ord'], 'des': t.get('des', ''), 'lines': L})
    return out


FAMS = ['L', 'S', 'H', 'C', 'Q', 'N', 'P', 'P2']


def random_template(ix, rng):
    i = rng.randrange(len(ix.T))
    F = list(ix.F[i])
    k = rng.choice([1, 2, 2, 3, 3, 4])
    # bias towards rarer features
    w = [1.0 / math.sqrt(ix.df[f]) for f in F]
    tm = set()
    for _ in range(k):
        tm.add(rng.choices(F, weights=w)[0])
    return tuple(sorted(tm))


def search(ix, n_tmpl, rng, smin=3, smax=40):
    """random template search: unique (group, best held-out score, template)"""
    best = {}
    for _ in range(n_tmpl):
        tm = random_template(ix, rng)
        b = ix.members(tm)
        m = b.bit_count() if b >= 0 else 0
        if not (smin <= m <= smax):
            continue
        G = ix.unpack(b)
        key = tuple(G)
        h = ix.heldout(G, tm)
        if key not in best or h > best[key][0]:
            best[key] = (h, tm)
    return best


def size_class(m):
    return 0 if m <= 3 else (1 if m <= 5 else (2 if m <= 9 else 3))
