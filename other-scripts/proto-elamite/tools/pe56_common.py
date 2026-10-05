#!/usr/bin/env python3
"""pe56 shared code: HIRE FORGERS AND WATCH WHERE THEY FAIL (item-level inversion, Proto-Elamite).

Adapted from linear-a/tools/la46_common.py to the PE line structure.
doc = {'id', 'site', 'lines': [{'s': [base signs], 'sys': None | system class, 'v': float | None}]}
  sys None = a line without numeral (header / context); otherwise the numeral's system class
  (common.system_of: SDB, C, B, S-frac, N23, C*, mod*) and value (pe47 pe_value; None if unreadable).
Forgers see only surface streams: S|sign ... N|sys|bin NL ... END. Numbers are re-drawn from the
pool of real values with the same (sys, bin). Relations are presence sets per tablet.
Controls (sign identities only): proto-cuneiform (CDLI, pe2 corpus) and Ur III (CDLI, pe38 docs).
"""
import json, os, re, sys, math, random, hashlib, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'pe56_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ corpora
def _from_pe_lines(T, sign_ok, min_num=2):
    from common import base, system_of
    from pe47_common import pe_value
    out = []
    for t in T:
        L = []
        for l in t['lines']:
            sg = [base(s) for s in l['signs'] if s and s != 'x' and sign_ok(s)]
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
                L.append({'s': sg[:10], 'sys': sy or '?', 'v': v})
            elif sg:
                L.append({'s': sg[:10], 'sys': None, 'v': None})
        if sum(l['sys'] is not None for l in L) >= min_num and len(L) <= 60:
            site = t.get('provenience', '') or ''
            site = re.sub(r'^uncertain \(mod\. |\)$|\(mod\. .*$', '', site).strip()[:12] or '-'
            out.append({'id': t['id'], 'site': site, 'lines': L})
    return out


def pe_docs():
    fn = os.path.join(CK, 'pe_docs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    from common import load, is_sign
    out = _from_pe_lines(load(), is_sign)
    json.dump(out, open(fn, 'w'))
    return out


def pc_docs():
    fn = os.path.join(CK, 'pc_docs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    T = json.load(open(os.path.join(D, 'pe2_pc_corpus.json')))
    for t in T:
        t['provenience'] = (t.get('period') or '-')[:10]
        for l in t['lines']:
            l['numerals'] = [[n, ('N01' if c == 'N1' else c)] for n, c in l['numerals']]
            l['signs'] = ['P_' + re.sub(r'~[a-z0-9]+$', '', s) for s in l['signs']
                          if s != 'x' and not re.match(r'^N\d', s)]
    out = _from_pe_lines(T, lambda s: True)
    json.dump(out, open(fn, 'w'))
    return out


def ur3_docs(n=6000):
    fn = os.path.join(CK, 'ur3_docs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    from fractions import Fraction as Fr
    src = json.load(open(os.path.join(D, 'pe38_ckpt', 'ur3_docs.json')))
    rng = random.Random(seed('pe56-ur3'))
    rng.shuffle(src)
    out = []
    for d in src:
        L = []
        for l in d['lines']:
            toks = [t for t in l['toks'] if t][:10]
            if l['val'] is not None and l['sys'] in (1, 2):
                try:
                    v = float(Fr(l['val']))
                except Exception:
                    v = None
                L.append({'s': toks, 'sys': 'u%d' % l['sys'], 'v': v})
            elif toks:
                L.append({'s': toks, 'sys': None, 'v': None})
        if sum(l['sys'] is not None for l in L) >= 2 and len(L) <= 60:
            out.append({'id': d['id'], 'site': d['site'], 'lines': L})
        if len(out) >= n:
            break
    json.dump(out, open(fn, 'w'))
    return out


def ntok(docs):
    return sum(len(l['s']) + (l['sys'] is not None) for d in docs for l in d['lines'])


def sample_like(docs, target, rng):
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    out, c = [], 0
    for i in idx:
        if c >= target:
            break
        out.append(docs[i]); c += ntok([docs[i]])
    return out


# ------------------------------------------------------------------ symbols
def nbin(v):
    if v is None:
        return '?'
    if v < 1:
        return 'f'
    return str(int(math.floor(math.log2(v + 1e-9))))


def kind(s):
    return s[0] if s not in ('NL', 'END', 'BOS') else s


def stream(doc):
    out = []
    for l in doc['lines']:
        out += ['S|' + x for x in l['s']]
        if l['sys'] is not None:
            out.append('N|%s|%s' % (l['sys'], nbin(l['v'])))
        out.append('NL')
    return out


# ------------------------------------------------------------------ relations
FAMS = ['ES', 'EF', 'LR', 'HE', 'HS', 'SP', 'SO', 'SN', 'SM', 'MM', 'TOT', 'CO', 'LAST', 'WP']
FAM_DESC = {
    'ES': 'last sign of entry i -> last sign of entry i+1',
    'EF': 'first sign of entry i -> first sign of entry i+1',
    'LR': 'sign in an entry -> sign in an entry >= 2 entries later (ordered, distance)',
    'HE': 'header sign (line without numeral before the first entry) -> sign anywhere in the entries',
    'HS': 'header sign -> number system present on the tablet',
    'SP': 'two number systems on one tablet',
    'SO': 'number system of entry i -> different system of entry i+1',
    'SN': 'last sign of an entry -> its number system',
    'SM': 'last sign of an entry -> magnitude (log2 bin) of its numeral',
    'MM': 'magnitude of entry i -> magnitude of entry i+1',
    'TOT': 'line whose value = running sum of >= 2 earlier entries of the same system (key: its first sign)',
    'CO': 'two signs on non-adjacent lines of one tablet (unordered)',
    'LAST': 'sign on the last line',
    'WP': 'adjacent sign pair inside a line',
}

_PVSYS = {'C': 'cap', 'C*': 'cap'}


def relations(doc, freq):
    """Presence set of (fam, a, b). freq = set of signs allowed in the long-range families."""
    L = doc['lines']
    R = set()
    ents = [l for l in L if l['sys'] is not None]
    fi = next((i for i, l in enumerate(L) if l['sys'] is not None), len(L))
    head = set(x for l in L[:fi] for x in l['s'])
    syss = sorted(set(l['sys'] for l in ents))
    for i, a in enumerate(syss):
        for b in syss[i + 1:]:
            R.add(('SP', a, b))
    for h in head:
        if h in freq:
            for s in syss:
                R.add(('HS', h, s))
    ent_signs = set(x for l in ents for x in l['s'] if x in freq)
    for h in head:
        if h in freq:
            for x in ent_signs:
                R.add(('HE', h, x))
    for l in L:
        for a, b in zip(l['s'], l['s'][1:]):
            R.add(('WP', a, b))
    for k, e in enumerate(ents):
        if e['s']:
            R.add(('SN', e['s'][-1], e['sys']))
            R.add(('SM', e['s'][-1], nbin(e['v'])))
        if k + 1 < len(ents):
            f = ents[k + 1]
            if e['s'] and f['s']:
                R.add(('ES', e['s'][-1], f['s'][-1]))
                R.add(('EF', e['s'][0], f['s'][0]))
            if f['sys'] != e['sys']:
                R.add(('SO', e['sys'], f['sys']))
            R.add(('MM', nbin(e['v']), nbin(f['v'])))
    fs = [[x for x in set(e['s']) if x in freq] for e in ents]
    for i in range(len(ents)):
        for j in range(i + 2, len(ents)):
            for a in fs[i]:
                for b in fs[j]:
                    R.add(('LR', a, b))
    # co-occurrence on non-adjacent lines
    pos = collections.defaultdict(list)
    for i, l in enumerate(L):
        for x in set(l['s']):
            if x in freq:
                pos[x].append(i)
    ks = sorted(pos)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            if min(abs(p - q) for p in pos[ks[i]] for q in pos[ks[j]]) > 1:
                R.add(('CO', ks[i], ks[j]))
    if L:
        for x in L[-1]['s']:
            R.add(('LAST', x, 'END'))
    # running totals per system family
    run = collections.defaultdict(float)
    nt = collections.Counter()
    for l in L:
        if l['sys'] is None or l['v'] is None:
            continue
        g = _PVSYS.get(l['sys'], l['sys'] if l['sys'].startswith('u') else 'cnt')
        v = l['v']
        if nt[g] >= 2 and v > 0 and abs(v - run[g]) < 1e-6 * max(1.0, v):
            R.add(('TOT', l['s'][0] if l['s'] else '-', 'SUM'))
            run[g], nt[g] = 0.0, 0
        else:
            run[g] += v; nt[g] += 1
    return R


def freq_set(docs, min_docs=8):
    c = collections.Counter()
    for d in docs:
        c.update(set(x for l in d['lines'] for x in l['s']))
    return set(k for k, v in c.items() if v >= min_docs)


def rel_counts(docs, freq):
    c = collections.Counter()
    for d in docs:
        c.update(relations(d, freq))
    return c


# ------------------------------------------------------------------ forgers
def random_arch(rng):
    fam = rng.choice(['ngram', 'ngram', 'ngram', 'lines', 'lines', 'bag'])
    a = {
        'fam': fam,
        'k': rng.choice([1, 2, 2, 3, 3, 4, 5]),
        'fmin': rng.choice([1, 2, 3, 5, 10, 10 ** 9]),
        'nmode': rng.choice(['full', 'full', 'sys', 'kind', 'coarse']),
        'alpha': math.exp(rng.uniform(math.log(0.05), math.log(5))),
        'lineidx': rng.random() < 0.4,
        'site': rng.random() < 0.3,
        'K': rng.choice([0, 0, 2, 4, 8, 16, 32]),
        'template': rng.random() < 0.5,
        'nl_reset': rng.random() < 0.3,
        'order1': rng.random() < 0.6,      # lines forger: Markov on previous line key
        'renum': rng.random() < 0.5,       # lines forger: numeral re-drawn given last sign
        'hdr': rng.random() < 0.5,         # lines / bag: headers drawn from header pool
    }
    return a


def arch_group(a):
    return a['fam'][0] + ('C' if a['K'] else 'c') + ('T' if a['template'] else 't') + str(min(a['k'], 3))


def _cluster(docs, K, rng):
    voc = {}
    rows = []
    for d in docs:
        rows.append(set(voc.setdefault(x, len(voc)) for l in d['lines'] for x in l['s']))
    X = np.zeros((len(docs), max(1, len(voc))), dtype=np.float32)
    for i, r in enumerate(rows):
        for j in r:
            X[i, j] = 1
    df = X.sum(0) + 1
    X = X * np.log(len(docs) / df)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    npr = np.random.default_rng(rng.randrange(1 << 30))
    K = min(K, len(docs))
    C = X[npr.choice(len(docs), K, replace=False)]
    lab = np.zeros(len(docs), dtype=int)
    for _ in range(10):
        lab = (X @ C.T).argmax(1)
        for k in range(K):
            m = lab == k
            if m.any():
                C[k] = X[m].mean(0)
    return lab.tolist()


class Forger:
    def __init__(self, arch, docs, rng):
        self.a = arch
        self.rng = rng
        self.vals = collections.defaultdict(list)
        for d in docs:
            for l in d['lines']:
                if l['sys'] is not None:
                    self.vals[(l['sys'], nbin(l['v']))].append(l['v'])
        self.cl = _cluster(docs, arch['K'], rng) if arch['K'] else [0] * len(docs)
        self.meta = [(d['site'], self.cl[i]) for i, d in enumerate(docs)]
        self.nlines = [len(d['lines']) for d in docs]
        self.skel = [[('H' if l['sys'] is None else 'E') for l in d['lines']] for d in docs]
        if arch['fam'] == 'ngram':
            self._train_ngram(docs)
        else:
            self._train_lines(docs)

    # ---------- n-gram family
    def _extras(self, site, cl):
        ex = ()
        if self.a['site']:
            ex += ('s:' + site,)
        if self.a['K']:
            ex += ('c%d' % cl,)
        return ex

    def _abs(self, s):
        a = self.a
        if s.startswith('S|'):
            return s if self.sc[s] >= a['fmin'] else 'S'
        if s.startswith('N|'):
            m = a['nmode']
            if m == 'full':
                return s
            if m == 'kind':
                return 'N'
            _, sy, b = s.split('|')
            if m == 'sys':
                return 'N|' + sy
            if b in ('f', '?'):
                return 'N|' + sy + '|' + b
            b = int(b)
            return 'N|%s|c%d' % (sy, 0 if b == 0 else 1 if b <= 3 else 2 if b <= 6 else 3)
        return s

    def _keys(self, ex, ctx):
        out = []
        for j in range(min(self.a['k'], len(ctx)), 0, -1):
            out.append(ex + tuple(ctx[-j:]))
        if ex:
            out.append(ex)
        out.append(('',))
        return out

    def _train_ngram(self, docs):
        self.sc = collections.Counter(s for d in docs for s in stream(d) if s.startswith('S|'))
        self.tab = collections.defaultdict(lambda: collections.defaultdict(list))
        self.tskel = [[kind(s) for s in stream(d)] for d in docs]
        for di, d in enumerate(docs):
            seq = stream(d) + ['END']
            ex = self._extras(*self.meta[di])
            ctx = ['BOS']
            li = 0
            for s in seq:
                exl = ex + (('li%d' % min(li, 3),) if self.a['lineidx'] else ())
                for key in self._keys(exl, ctx):
                    e = self.tab[key]
                    e['*'].append(s)
                    e[kind(s)].append(s)
                if s == 'NL':
                    li += 1
                    if self.a['nl_reset']:
                        ctx = ['BOS']
                        continue
                ctx.append(self._abs(s))

    def _draw(self, ex, ctx, want):
        al = self.a['alpha']
        keys = self._keys(ex, ctx)
        for ki, key in enumerate(keys):
            e = self.tab.get(key)
            if e is None:
                continue
            Lst = e.get(want)
            if not Lst:
                continue
            if ki == len(keys) - 1 or self.rng.random() < len(Lst) / (len(Lst) + al):
                return self.rng.choice(Lst)
        return None

    def _num(self, s):
        _, sy, b = s.split('|')
        return sy, self.rng.choice(self.vals[(sy, b)])

    def _forge_ngram(self):
        rng = self.rng
        di = rng.randrange(len(self.meta))
        ex = self._extras(*self.meta[di])
        skel = self.tskel[di] if self.a['template'] else None
        ctx = ['BOS']
        lines, cur = [], {'s': [], 'sys': None, 'v': None}
        li, i = 0, 0
        while True:
            exl = ex + (('li%d' % min(li, 3),) if self.a['lineidx'] else ())
            if skel is not None:
                if i >= len(skel):
                    break
                want = skel[i]
                s = 'NL' if want == 'NL' else (self._draw(exl, ctx, want) or self._draw(('',), [], want) or 'NL')
            else:
                s = self._draw(exl, ctx, '*')
                if s is None or s == 'END' or i > 300:
                    break
            i += 1
            if s == 'NL':
                if cur['s'] or cur['sys'] is not None:
                    lines.append(cur)
                cur = {'s': [], 'sys': None, 'v': None}
                li += 1
                if self.a['nl_reset']:
                    ctx = ['BOS']
                    continue
                ctx.append('NL')
                continue
            if s.startswith('S|'):
                if cur['sys'] is None:
                    cur['s'].append(s[2:])
            elif s.startswith('N|'):
                if cur['sys'] is None:
                    cur['sys'], cur['v'] = self._num(s)
            ctx.append(self._abs(s))
        if cur['s'] or cur['sys'] is not None:
            lines.append(cur)
        return {'id': 'forg', 'site': self.meta[di][0], 'lines': lines[:60]}

    # ---------- line-bank family (entry-level grammar) and bag family
    def _train_lines(self, docs):
        self.hpool = collections.defaultdict(list)      # cluster -> header lines
        self.epool = collections.defaultdict(list)      # cluster -> entry lines
        self.succ = collections.defaultdict(list)       # (cluster, prev key) -> entry lines
        self.numby = collections.defaultdict(list)      # last sign -> (sys, v)
        for di, d in enumerate(docs):
            c = self.cl[di]
            prev = 'BOS'
            for l in d['lines']:
                if l['sys'] is None:
                    self.hpool[c].append(l)
                    continue
                self.epool[c].append(l)
                self.succ[(c, prev)].append(l)
                prev = l['s'][-1] if l['s'] else '-'
                if l['s']:
                    self.numby[l['s'][-1]].append((l['sys'], l['v']))
        self.allh = [l for v in self.hpool.values() for l in v]
        self.alle = [l for v in self.epool.values() for l in v]

    def _forge_lines(self):
        rng = self.rng
        di = rng.randrange(len(self.meta))
        c = self.meta[di][1]
        skel = self.skel[di] if self.a['template'] or self.a['fam'] == 'bag' else \
            self.skel[rng.randrange(len(self.skel))]
        out = []
        prev = 'BOS'
        al = self.a['alpha']
        for kd in skel:
            if kd == 'H':
                P = self.hpool.get(c) or self.allh
                if self.a['hdr'] or self.a['fam'] == 'bag':
                    l = rng.choice(P) if P else {'s': [], 'sys': None, 'v': None}
                else:
                    e = rng.choice(self.alle)
                    l = {'s': e['s'], 'sys': None, 'v': None}
                out.append(dict(l))
                continue
            if self.a['fam'] == 'lines' and self.a['order1']:
                S = self.succ.get((c, prev))
                if S and rng.random() < len(S) / (len(S) + al):
                    l = rng.choice(S)
                else:
                    l = rng.choice(self.epool.get(c) or self.alle)
            else:
                l = rng.choice(self.epool.get(c) or self.alle)
            l = dict(l)
            if self.a['renum'] and l['s'] and self.numby.get(l['s'][-1]):
                l['sys'], l['v'] = rng.choice(self.numby[l['s'][-1]])
            out.append(l)
            prev = l['s'][-1] if l['s'] else '-'
        return {'id': 'forg', 'site': self.meta[di][0], 'lines': out}

    def forge(self):
        if self.a['fam'] == 'ngram':
            return self._forge_ngram()
        return self._forge_lines()


# ------------------------------------------------------------------ the inside-out contest
def contest(docs, archs, rng, freq, folds=5, reps=1, log=None):
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    fold_of = {i: k % folds for k, i in enumerate(idx)}
    real = rel_counts(docs, freq)
    grp_E = collections.defaultdict(collections.Counter)
    grp_n = collections.Counter()
    for ai, a in enumerate(archs):
        g = arch_group(a)
        E = collections.Counter()
        for f in range(folds):
            tr = [docs[i] for i in range(len(docs)) if fold_of[i] != f]
            nte = sum(1 for i in range(len(docs)) if fold_of[i] == f)
            F = Forger(a, tr, random.Random(rng.randrange(1 << 30)))
            for _ in range(reps * nte):
                E.update(relations(F.forge(), freq))
        for key, v in E.items():
            grp_E[g][key] += v / reps
        grp_n[g] += 1
        if log and (ai + 1) % 10 == 0:
            log('  arch %d/%d' % (ai + 1, len(archs)))
    for g in grp_E:
        for key in grp_E[g]:
            grp_E[g][key] /= grp_n[g]
    return real, grp_E, dict(grp_n)


def pois_sf(k, lam):
    if k <= 0:
        return 1.0
    lam = max(lam, 1e-3)
    p = math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))
    s, j = 0.0, k
    while p > 1e-300 and j < k + 5000:
        s += p
        j += 1
        p *= lam / j
        if p < s * 1e-15:
            break
    return min(1.0, s)


def residues(real, grp_E, gn, min_real=3):
    """Per relation: R, ensemble-mean E, max-group E, p_mean, p_max."""
    tot = sum(gn.values())
    out = []
    for key, R in real.items():
        if R < min_real:
            continue
        Es = [grp_E[g].get(key, 0.0) for g in grp_E]
        E = sum(grp_E[g].get(key, 0.0) * gn[g] for g in grp_E) / tot
        M = max(Es)
        out.append((key, R, E, M, pois_sf(R, E), pois_sf(R, M)))
    return out


# ------------------------------------------------------------------ controls
def shuffle_corpus(docs, rng):
    """Signs shuffled globally (line lengths, numerals and layout kept)."""
    pool = [x for d in docs for l in d['lines'] for x in l['s']]
    rng.shuffle(pool)
    k = 0
    out = []
    for d in docs:
        L = []
        for l in d['lines']:
            L.append({'s': pool[k:k + len(l['s'])], 'sys': l['sys'], 'v': l['v']})
            k += len(l['s'])
        out.append({'id': d['id'], 'site': d['site'], 'lines': L})
    return out


def world(docs, rng, arch):
    F = Forger(arch, docs, rng)
    out = []
    while len(out) < len(docs):
        d = F.forge()
        if sum(l['sys'] is not None for l in d['lines']) >= 2:
            d['id'] = 'w%d' % len(out)
            out.append(d)
    return out


W_ARCHS = {
    'W1': {'fam': 'ngram', 'k': 3, 'fmin': 1, 'nmode': 'full', 'alpha': 1.0, 'lineidx': False, 'site': False,
           'K': 16, 'template': True, 'nl_reset': False, 'order1': False, 'renum': False, 'hdr': True},
    'W2': {'fam': 'lines', 'k': 1, 'fmin': 1, 'nmode': 'full', 'alpha': 0.5, 'lineidx': False, 'site': False,
           'K': 8, 'template': False, 'nl_reset': False, 'order1': True, 'renum': True, 'hdr': True},
}


def plant(docs, rng, n_keys=8, rate=0.8):
    """Planted LONG-RANGE constraints in a forger world (truth known):
    (a) 4 ordered distance pairs: key sign a in an entry -> partner sign b appended to an entry >= 2 entries later;
    (b) 2 header -> entry pairs: if header sign h present, entry sign e inserted into one entry;
    (c) 2 header -> system pairs: if header sign h present, one entry's numeral becomes system X.
    Signs are mid-frequency ones (12-60 docs)."""
    c = collections.Counter()
    for d in docs:
        c.update(set(x for l in d['lines'] for x in l['s']))
    mid = sorted(w for w, k in c.items() if 12 <= k <= 60)
    rng.shuffle(mid)
    A, B = mid[:4], mid[4:8]
    H, E = mid[8:10], mid[10:12]
    H2 = mid[12:14]
    sysl = sorted(set(l['sys'] for d in docs for l in d['lines'] if l['sys'] is not None),
                  key=lambda s: -sum(l['sys'] == s for d in docs for l in d['lines']))
    X = [sysl[1], sysl[2]] if len(sysl) > 2 else [sysl[-1], sysl[-1]]
    vals = collections.defaultdict(list)
    for d in docs:
        for l in d['lines']:
            if l['sys'] is not None:
                vals[l['sys']].append(l['v'])
    truth = set(('LR', a, b) for a, b in zip(A, B)) | set(('HE', h, e) for h, e in zip(H, E)) | \
        set(('HS', h, x) for h, x in zip(H2, X))
    out = []
    for d in docs:
        L = [dict(l, s=list(l['s'])) for l in d['lines']]
        ei = [i for i, l in enumerate(L) if l['sys'] is not None]
        fi = ei[0] if ei else len(L)
        head = set(x for l in L[:fi] for x in l['s'])
        for a, b in zip(A, B):
            for k, i in enumerate(ei):
                if a in L[i]['s'] and k + 2 < len(ei) and rng.random() < rate:
                    j = ei[rng.randrange(k + 2, len(ei))]
                    L[j]['s'].append(b)
                    break
        for h, e in zip(H, E):
            if h in head and ei and rng.random() < rate:
                L[rng.choice(ei)]['s'].append(e)
        for h, x in zip(H2, X):
            if h in head and ei and rng.random() < rate:
                j = rng.choice(ei)
                L[j]['sys'], L[j]['v'] = x, rng.choice(vals[x])
        out.append({'id': d['id'], 'site': d['site'], 'lines': L})
    return out, sorted(truth)
