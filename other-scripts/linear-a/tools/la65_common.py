#!/usr/bin/env python3
"""la65 LINEAR A DOSSIERS AS CONTROLLED EXPERIMENTS (port of pe63): shared library.

Tablet record (both sides of a tablet merged; signs opaque to the method):
  {'id', 'site', 'meta': {'scribe', 'find', 'ord'}, 'ents': [{'w': [words], 'l': [logograms],
   'n': int|None, 'f': str}]}
An entry is a run of words/logograms closed by one number (or a run with no number = context).
No Linear B sound value or Linear B role is used for Linear A. Linear B series codes and the la63
answer key (place names, commodity-bound qualifiers) are used ONLY to score the Linear B control.
"""
import os, sys, re, json, math, random, hashlib
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(ROOT, 'data')
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(DATA, 'la65_ckpt')
os.makedirs(CK, exist_ok=True)


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def lbase(x):
    b = re.sub(r"[\[\]'?]", '', x.split('+')[0]).strip()
    return b or x


def _ents(toks):
    E, w, l = [], [], []
    for k, v, f in toks:
        if k == 'W':
            w.append(v)
        elif k == 'L':
            l.append(v)
        elif k == 'N':
            E.append({'w': w[:8], 'l': l[:4], 'n': int(v), 'f': f or ''})
            w, l = [], []
        elif k == 'NL':
            if w or l:
                E.append({'w': w[:8], 'l': l[:4], 'n': None, 'f': ''})
            w, l = [], []
    if w or l:
        E.append({'w': w[:8], 'l': l[:4], 'n': None, 'f': ''})
    return E


def _findspots():
    s = open(os.path.join(DATA, 'LinearAInscriptions.js')).read()
    out = {}
    for m in re.finditer(r'\["([^"]+)",\{(.*?)\n\}\]', s, re.S):
        f = re.search(r'"findspot": "([^"]*)"', m.group(2))
        out[m.group(1)] = f.group(1) if f else ''
    return out


def la_tabs():
    """Linear A administrative tablets (Tablet, lames, bars), sides merged, >= 2 entries, >= 1 number."""
    fn = os.path.join(CK, 'la_tabs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    import la60_common as C
    docs = C.admin_docs(C.load_la())
    fs = _findspots()
    by = defaultdict(list)
    for d in docs:
        if d['support'] in ('Tablet', 'Lames (short thin tablet)', '3-sided bar', '4-sided bar'):
            by[re.sub(r'(\d)[a-z]{1,2}$', r'\1', d['id'])].append(d)
    out = []
    for tab, ds in by.items():
        ds.sort(key=lambda d: d['id'])
        E = []
        for d in ds:
            E += _ents(d['toks'])
        if len(E) < 2 or not any(e['n'] is not None for e in E):
            continue
        m = re.match(r'([A-Z]+)(\d+)', tab)
        ordk = int(m.group(2)) if m else -1
        out.append({'id': tab, 'site': ds[0]['site'], 'sides': [d['id'] for d in ds],
                    'meta': {'scribe': ds[0]['scribe'] if 'nknown' not in ds[0]['scribe'] else '',
                             'find': fs.get(ds[0]['id'], ''), 'ord': ordk}, 'ents': E[:40]})
    json.dump(out, open(fn, 'w'))
    return out


def lb_tabs_all():
    fn = os.path.join(CK, 'lb_tabs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    import la60_common as C
    out = []
    for d in C.load_lb():
        E = _ents(d['toks'])
        if len(E) >= 2 and any(e['n'] is not None for e in E):
            ser = re.match(r'(\w+ \w+)', d['id']).group(1)
            out.append({'id': d['id'], 'site': d['site'], 'series': ser,
                        'meta': {'scribe': '', 'find': '', 'ord': -1}, 'ents': E[:40]})
    json.dump(out, open(fn, 'w'))
    return out


def lb_draw(n, s):
    T = lb_tabs_all()
    rng = random.Random(seed('la65-lb-%d' % s))
    return rng.sample(T, n)


# ------------------------------------------------------------------ planting
def plant_dossier(T, rng, k=6, tag='PL'):
    """copy a real tablet k times: slot A word swapped alone (number fixed), slot B word and logogram swapped
    together (3 fixed pairs) with a new quantity scale, slot C word fixed with a regular x(j+1) quantity step"""
    src = [t for t in T if sum(e['n'] is not None for e in t['ents']) >= 4 and
           sum(1 for e in t['ents'] if e['n'] is not None and e['w']) >= 3]
    t = rng.choice(src)
    words = sorted({w for x in T for e in x['ents'] for w in e['w']})
    logos = sorted({l for x in T for e in x['ents'] for l in e['l']})
    num = [i for i, e in enumerate(t['ents']) if e['n'] is not None and e['w']]
    a, b, c = rng.sample(num, 3)
    cw, cl = rng.sample(words, 3), rng.sample(logos, 3)          # commodity-bound word/logogram pairs
    out = []
    for j in range(k):
        E = [dict(e, w=list(e['w']), l=list(e['l'])) for e in t['ents']]
        E[a]['w'] = [rng.choice(words)]
        E[b]['w'], E[b]['l'] = [cw[j % 3]], [cl[j % 3]]
        E[b]['n'] = max(1, int(E[b]['n'] * rng.choice([3, 5, 7, 10, 20])))
        E[c]['n'] = max(1, E[c]['n']) * (j + 1)
        out.append({'id': '%s%d_%s' % (tag, j, t['id']), 'site': t['site'], 'planted': tag,
                    'meta': {'scribe': '', 'find': '', 'ord': -1}, 'ents': E,
                    'truth': {'A': a, 'B': b, 'C': c}})
    return t['id'], out


# ------------------------------------------------------------------ features
def ekey(e):
    return ' '.join(e['w'] + ['L:' + x for x in e['l']])


def units(t):
    """content units (similarity) of a tablet"""
    U = set()
    for e in t['ents']:
        for w in e['w']:
            U.add('w:' + w)
        for l in e['l']:
            U.add('l:' + lbase(l))
            U.add('lf:' + l)
        k = ekey(e)
        if k:
            U.add('e:' + k)
            if e['n'] is not None:
                U.add('en:%s|%d%s' % (k, e['n'], e['f']))
        for l in e['l']:
            if e['f']:
                U.add('lfr:%s|%s' % (lbase(l), e['f']))
        for i in range(len(e['w']) - 1):
            U.add('wb:%s %s' % (e['w'][i], e['w'][i + 1]))
    return U


def skel(t):
    s = []
    for e in t['ents']:
        c = ('W' if e['w'] else '') + ('L' if e['l'] else '') + ('N' if e['n'] is not None else 'c')
        if not s or s[-1] != c:
            s.append(c)
    return '.'.join(s)[:60]


def feats(t, meta=True):
    F = {'U:' + u for u in units(t) if u[:2] in ('w:', 'l:', 'e:', 'en') or u.startswith('lfr')}
    E = t['ents']
    if ekey(E[0]):
        F.add('H:' + ekey(E[0]))
    if ekey(E[-1]):
        F.add('C:' + ekey(E[-1]))
    F.add('Q:' + skel(t))
    F.add('N:%d' % min(len(E), 20))
    F.add('LS:' + '|'.join(sorted({lbase(l) for e in E for l in e['l']})))
    F.add('S:' + t['site'])
    if meta:
        if t['meta']['scribe']:
            F.add('M:' + t['meta']['scribe'])
        if t['meta']['find']:
            F.add('F:' + t['meta']['find'])
        o = t['meta']['ord']
        if o >= 0:
            F.add('P:%s%d' % (t['site'], o // 6))
            F.add('P2:%s%d' % (t['site'], (o + 3) // 6))
    return F


def implied(f):
    """content units a template feature implies (removed before held-out scoring)"""
    out = set()
    if f.startswith('U:'):
        u = f[2:]
        out.add(u)
        if u.startswith('e:') or u.startswith('en:'):
            k = u.split(':', 1)[1].split('|')[0]
            out.add('e:' + k)
            for x in k.split():
                if x.startswith('L:'):
                    out.add('l:' + lbase(x[2:])); out.add('lf:' + x[2:])
                else:
                    out.add('w:' + x)
        elif u.startswith('lfr:'):
            out.add('l:' + u[4:].split('|')[0])
    elif f[:2] in ('H:', 'C:'):
        k = f[2:]
        out.add('e:' + k)
        for x in k.split():
            if x.startswith('L:'):
                out.add('l:' + lbase(x[2:])); out.add('lf:' + x[2:])
            else:
                out.add('w:' + x)
    elif f.startswith('LS:'):
        for b in f[3:].split('|'):
            if b:
                out.add('l:' + b)
    return out


class Index:
    def __init__(self, T, meta=True):
        import scipy.sparse as sp
        self.T = T
        n = len(T)
        self.F = [feats(t, meta) for t in T]
        self.U = [units(t) for t in T]
        df = Counter(f for F in self.F for f in F)
        self.df = df
        udf = Counter(u for U in self.U for u in U)
        self.uidf = {u: math.log(n / c) for u, c in udf.items()}
        vocab = sorted(u for u, c in udf.items() if c >= 2)
        self.vid = {u: i for i, u in enumerate(vocab)}
        rows, cols, vals = [], [], []
        for i, U in enumerate(self.U):
            for u in U:
                if u in self.vid:
                    rows.append(i); cols.append(self.vid[u]); vals.append(self.uidf[u])
        X = sp.csr_matrix((vals, (rows, cols)), shape=(n, max(1, len(vocab))))
        self.norm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-9
        self.S = (X @ X.T).toarray() / np.outer(self.norm, self.norm)
        np.fill_diagonal(self.S, 0)
        self.mu = self.S.sum(1) / (n - 1)
        self.sd = np.sqrt(np.maximum((self.S ** 2).sum(1) / (n - 1) - self.mu ** 2, 1e-6))
        self.bits = defaultdict(int)
        for i, F in enumerate(self.F):
            for f in F:
                self.bits[f] |= (1 << i)
        self.Fl = [sorted(F) for F in self.F]
        self.Fw = [[1.0 / math.sqrt(df[f]) for f in F] for F in self.Fl]

    def members(self, tm):
        b = -1
        for f in tm:
            b &= self.bits[f]
        return b

    @staticmethod
    def unpack(b):
        return [i for i, c in enumerate(bin(b)[:1:-1]) if c == '1']

    def zmat(self, G):
        G = np.asarray(G)
        sub = self.S[np.ix_(G, G)]
        mu, sd = self.mu[G], self.sd[G]
        return (sub - (mu[:, None] + mu[None, :]) / 2) / np.sqrt((sd[:, None] ** 2 + sd[None, :] ** 2) / 2)

    def heldout(self, G, tm):
        G = np.asarray(G)
        sub = self.S[np.ix_(G, G)].copy()
        rm = 0.0
        exp = set()
        for f in tm:
            exp |= implied(f)
        for u in exp:
            if u in self.vid:
                rm += self.uidf[u] ** 2
        nn = self.norm[G]
        sub -= rm / np.outer(nn, nn)
        mu, sd = self.mu[G], self.sd[G]
        z = (sub - (mu[:, None] + mu[None, :]) / 2) / np.sqrt((sd[:, None] ** 2 + sd[None, :] ** 2) / 2)
        m = len(G)
        return float((z.sum() - np.trace(z)) / (m * (m - 1)))


def shuffle_ents(T, rng):
    """null: every entry moved to a random slot of the same site, same kind (number/context) and same
    position class (first / last / middle); tablet lengths, metadata and entry frequencies kept"""
    pools = defaultdict(list)

    def pc(i, n):
        return 'f' if i == 0 else ('l' if i == n - 1 else 'm')
    for t in T:
        for i, e in enumerate(t['ents']):
            pools[(t['site'], e['n'] is None, pc(i, len(t['ents'])))].append(e)
    for p in pools.values():
        rng.shuffle(p)
    out = []
    for t in T:
        E = [pools[(t['site'], e['n'] is None, pc(i, len(t['ents'])))].pop() for i, e in enumerate(t['ents'])]
        out.append(dict(t, ents=E))
    return out


def random_template(ix, rng):
    i = rng.randrange(len(ix.T))
    F, w = ix.Fl[i], ix.Fw[i]
    k = rng.choice([1, 2, 2, 3, 3, 4])
    return tuple(sorted(set(rng.choices(F, weights=w, k=k))))


def search(ix, n_tmpl, rng, smin=2, smax=40):
    best = {}
    seen = set()
    for _ in range(n_tmpl):
        tm = random_template(ix, rng)
        if tm in seen:
            continue
        seen.add(tm)
        b = ix.members(tm)
        if b < 0:
            continue
        m = b.bit_count()
        if not (smin <= m <= smax):
            continue
        G = ix.unpack(b)
        key = tuple(G)
        h = ix.heldout(G, tm)
        if key not in best or h > best[key][0]:
            best[key] = (h, tm)
    return best


def size_class(m):
    return 0 if m == 2 else (1 if m == 3 else (2 if m <= 5 else 3))


SC_NAMES = ['2', '3', '4-5', '6+']


def merge_dossiers(ix, sig, zmin=2.0):
    """greedy merge of significant groups (sorted by score): a group joins a dossier if >= 2 and >= 1/3
    of its members are in it; a new member must have mean pairwise z >= zmin to the dossier"""
    sig = sorted(sig, key=lambda x: -x[1])
    D = []
    for G, h, tm in sig:
        Gs = set(G)
        host = None
        for d in D:
            ov = len(Gs & d['m'])
            if ov >= min(2, len(Gs) - 1) and ov >= len(Gs) / 3:
                host = d
                break
        if host is None:
            if not any(Gs <= d['m'] for d in D):
                D.append({'m': set(G), 'score': h, 'tmpl': [tm]})
            continue
        for g in G:
            if g in host['m']:
                continue
            others = list(host['m'])
            zz = ix.zmat([g] + others)[0, 1:]
            if zz.mean() >= zmin:
                host['m'].add(g)
        host['tmpl'].append(tm)
    return D
