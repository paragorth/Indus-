#!/usr/bin/env python3
"""LA-46 shared code: HIRE A FORGER AND WATCH WHERE HE FAILS (item-level inversion).

la14 already ran a document-level forger contest. LA-46 turns it inside out at ITEM level:
an ensemble of randomly drawn forger architectures (back-off n-grams over randomly abstracted
contexts, with or without tablet clusters, site, line index, copied layout templates) is trained
on 4/5 of the documents and forges the other 1/5 many times. For every specific relation instance
(sign pair across word edges, word->number scale, word->logogram, logogram->number, entry->next
entry, word co-occurring on a tablet, site x word, number->number, word = running total,
word in first / last slot) we count in how many held-out real documents it occurs and compare
with the LARGEST expectation over forger groups. A pair is a residue only if real documents beat
every forger group. Calibration: the same pipeline on forger-made worlds, shuffled corpora,
a planted constraint, Linear B and Ur III (sign identities only).

doc = {'id', 'site', 'toks': [('W', str) | ('L', str) | ('N', float) | ('NL',)]}
"""
import json, os, re, sys, math, random, hashlib, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la46_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
PE = os.path.join(HERE, '..', '..', 'proto-elamite', 'data')


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ corpora
def la_docs(min_content=4):
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        toks = []
        for t in ins['tokens']:
            if t['t'] == 'word':
                toks.append(('W', '-'.join(t['s'])))
            elif t['t'] == 'logo':
                toks.append(('L', t['v']))
            elif t['t'] == 'num':
                toks.append(('N', float(t['v'])))
            elif t['t'] == 'nl':
                if toks and toks[-1][0] != 'NL':
                    toks.append(('NL',))
        while toks and toks[-1][0] == 'NL':
            toks.pop()
        if sum(x[0] != 'NL' for x in toks) >= min_content:
            out.append({'id': ins['id'], 'site': ins['site'], 'toks': toks})
    return out


def lb_docs_all():
    fn = os.path.join(CK, 'lb_docs.json')
    if os.path.exists(fn):
        return [dict(d, toks=[tuple(x) for x in d['toks']]) for d in json.load(open(fn))]
    from la41_common import lb_docs
    out = []
    for k, d in lb_docs().items():
        toks = []
        for it in d['items']:
            if toks:
                toks.append(('NL',))
            if it['w']:
                toks.append(('W', '-'.join(it['w'])))
            for c in it.get('ctx', []):
                toks.append(('W', '-'.join(c)))
            if it['logo']:
                toks.append(('L', it['logo']))
            if it['num'] is not None:
                toks.append(('N', float(it['val'])))
        if sum(x[0] != 'NL' for x in toks) >= 4:
            out.append({'id': k, 'site': d['site'], 'series': d['support'], 'toks': toks})
    json.dump(out, open(fn, 'w'))
    return out


def ur3_docs_all(n=4000):
    fn = os.path.join(CK, 'ur3_docs.json')
    if os.path.exists(fn):
        return [dict(d, toks=[tuple(x) for x in d['toks']]) for d in json.load(open(fn))]
    from fractions import Fraction as Fr
    src = json.load(open(os.path.join(PE, 'pe38_ckpt', 'ur3_docs.json')))
    rng = random.Random(seed('la46-ur3'))
    rng.shuffle(src)
    out = []
    for d in src:
        toks = []
        for l in d['lines']:
            if toks:
                toks.append(('NL',))
            if l['val'] is not None and l['sys'] in (1, 2):
                try:
                    toks.append(('N', float(Fr(l['val']))))
                except Exception:
                    pass
            for t in l['toks']:
                if t:
                    toks.append(('W', t))
        while toks and toks[-1][0] == 'NL':
            toks.pop()
        if sum(x[0] != 'NL' for x in toks) >= 4 and len(toks) <= 120:
            out.append({'id': d['id'], 'site': d['site'], 'toks': toks})
        if len(out) >= n:
            break
    json.dump(out, open(fn, 'w'))
    return out


def ntok(docs):
    return sum(sum(x[0] != 'NL' for x in d['toks']) for d in docs)


def sample_like(docs, target_tokens, rng):
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    out, c = [], 0
    for i in idx:
        if c >= target_tokens:
            break
        out.append(docs[i]); c += sum(x[0] != 'NL' for x in docs[i]['toks'])
    return out


# ------------------------------------------------------------------ symbols
def nbin(v):
    if v < 1:
        return 'f'
    return str(int(math.floor(math.log2(v + 1e-9))))


def sym(t):
    if t[0] == 'W':
        return 'W|' + t[1]
    if t[0] == 'L':
        return 'L|' + t[1]
    if t[0] == 'N':
        return 'N|' + nbin(t[1])
    return 'NL'


def kind(s):
    return s[0] if s != 'NL' and s != 'END' else s


# ------------------------------------------------------------------ relations (the detector's vocabulary)
FAMS = ['XS', 'WN', 'WL', 'LN', 'ES', 'CO', 'SW', 'NN', 'TOT', 'PF']
FAM_DESC = {
    'XS': 'last sign of a word -> first sign of the next word',
    'WN': 'word -> scale (log2 bin) of the number right after it',
    'WL': 'word -> logogram within the next two content tokens',
    'LN': 'logogram -> scale of the number right after it',
    'ES': 'entry word -> next entry word (a number/logogram/line break between)',
    'CO': 'two word types on one tablet, not adjacent',
    'SW': 'site x word',
    'NN': 'number scale -> next number scale',
    'TOT': 'word followed by a number equal to the running sum (>= 2 terms)',
    'PF': 'word in the first content slot / as the last word',
}


def relations(doc):
    """Set of (fam, a, b) present in the doc (presence, not counts)."""
    T = [t for t in doc['toks'] if t[0] != 'NL']
    R = set()
    n = len(T)
    words = [(i, t[1]) for i, t in enumerate(T) if t[0] == 'W']
    for i, t in enumerate(T):
        nx = T[i + 1] if i + 1 < n else None
        if t[0] == 'W':
            if nx is not None and nx[0] == 'W':
                a = t[1].split('-')[-1]; b = nx[1].split('-')[0]
                R.add(('XS', a, b))
            if nx is not None and nx[0] == 'N':
                R.add(('WN', t[1], nbin(nx[1])))
            for j in (i + 1, i + 2):
                if j < n and T[j][0] == 'L':
                    R.add(('WL', t[1], T[j][1])); break
        elif t[0] == 'L':
            if nx is not None and nx[0] == 'N':
                R.add(('LN', t[1], nbin(nx[1])))
        elif t[0] == 'N':
            for j in range(i + 1, n):
                if T[j][0] == 'N':
                    R.add(('NN', nbin(t[1]), nbin(T[j][1]))); break
    # entry succession
    for k in range(len(words) - 1):
        i, w = words[k]; j, w2 = words[k + 1]
        if j > i + 1:
            R.add(('ES', w, w2))
    # co-occurrence (non adjacent)
    ws = {}
    for i, w in words:
        ws.setdefault(w, []).append(i)
    keys = sorted(ws)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            pa, pb = ws[keys[a]], ws[keys[b]]
            if min(abs(x - y) for x in pa for y in pb) > 1:
                R.add(('CO', keys[a], keys[b]))
    for w in ws:
        R.add(('SW', doc['site'], w))
    # running totals
    run, nterm = 0.0, 0
    for i, t in enumerate(T):
        if t[0] == 'N':
            v = math.floor(t[1])
            prev = T[i - 1] if i > 0 else None
            if nterm >= 2 and v > 0 and abs(v - run) < 1e-6:
                w = prev[1] if prev is not None and prev[0] == 'W' else (
                    T[i - 2][1] if i > 1 and T[i - 2][0] == 'W' else '-')
                R.add(('TOT', w, 'SUM'))
                run, nterm = 0.0, 0
            else:
                run += v; nterm += 1
    if T and T[0][0] == 'W':
        R.add(('PF', T[0][1], 'FIRST'))
    if words:
        R.add(('PF', words[-1][1], 'LAST'))
    return R


def rel_counts(docs):
    c = collections.Counter()
    for d in docs:
        c.update(relations(d))
    return c


# ------------------------------------------------------------------ forger ensemble
def random_arch(rng):
    a = {
        'k': rng.choice([1, 1, 2, 2, 2, 3, 3, 4]),
        'fmin': rng.choice([1, 2, 3, 5, 10, 10 ** 9]),
        'nmode': rng.choice(['bin', 'bin', 'coarse', 'kind']),
        'lmode': rng.choice(['exact', 'exact', 'base', 'kind']),
        'alpha': math.exp(rng.uniform(math.log(0.05), math.log(5))),
        'lineidx': rng.random() < 0.4,
        'site': rng.random() < 0.4,
        'K': rng.choice([0, 0, 0, 2, 3, 4, 6, 8]),
        'template': rng.random() < 0.5,
        'nl_reset': rng.random() < 0.3,
    }
    return a


def arch_group(a):
    """Forger groups used for the 'beat every group' rule."""
    g = 'T' if a['template'] else 'F'
    g += 'C' if a['K'] else 'c'
    g += 'S' if a['site'] else 's'
    g += str(min(a['k'], 3))
    return g


class Forger:
    def __init__(self, arch, docs, rng):
        self.a = arch
        self.rng = rng
        wc = collections.Counter(t[1] for d in docs for t in d['toks'] if t[0] == 'W')
        self.wc = wc
        self.vals = collections.defaultdict(list)
        for d in docs:
            for t in d['toks']:
                if t[0] == 'N':
                    self.vals[nbin(t[1])].append(t[1])
        self.sites = [d['site'] for d in docs]
        self.templates = [[kind(sym(t)) for t in d['toks']] for d in docs]
        # clusters
        self.cl = [0] * len(docs)
        if arch['K']:
            self.cl = self._cluster(docs, arch['K'])
        self.tab = collections.defaultdict(lambda: collections.defaultdict(list))
        self.docsite = {}
        for di, d in enumerate(docs):
            seq = [sym(t) for t in d['toks']] + ['END']
            ex = self._extras(d['site'], self.cl[di])
            ctx = ['BOS']
            li = 0
            for s in seq:
                exl = ex + (('li%d' % min(li, 3),) if arch['lineidx'] else ())
                for key in self._keys(exl, ctx):
                    e = self.tab[key]
                    e['*'].append(s)
                    e[kind(s)].append(s)
                if s == 'NL':
                    li += 1
                    if arch['nl_reset']:
                        ctx = ['BOS']
                        continue
                ctx.append(self._abs(s))
        self.doc_cl = list(zip(self.sites, self.cl))

    def _cluster(self, docs, K):
        voc = {}
        rows = []
        for d in docs:
            r = set()
            for t in d['toks']:
                if t[0] in 'WL':
                    r.add(voc.setdefault(t[0] + t[1], len(voc)))
            rows.append(r)
        X = np.zeros((len(docs), len(voc)), dtype=np.float32)
        for i, r in enumerate(rows):
            for j in r:
                X[i, j] = 1
        df = X.sum(0) + 1
        X = X * np.log(len(docs) / df)
        nr = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        X = X / nr
        npr = np.random.default_rng(self.rng.randrange(1 << 30))
        C = X[npr.choice(len(docs), K, replace=False)]
        lab = np.zeros(len(docs), dtype=int)
        for _ in range(12):
            lab = (X @ C.T).argmax(1)
            for k in range(K):
                m = lab == k
                if m.any():
                    C[k] = X[m].mean(0)
        return lab.tolist()

    def _extras(self, site, cl):
        ex = ()
        if self.a['site']:
            ex += ('s:' + site,)
        if self.a['K']:
            ex += ('c%d' % cl,)
        return ex

    def _abs(self, s):
        a = self.a
        if s.startswith('W|'):
            return s if self.wc[s[2:]] >= a['fmin'] else 'W'
        if s.startswith('N|'):
            if a['nmode'] == 'bin':
                return s
            if a['nmode'] == 'kind':
                return 'N'
            b = s[2:]
            if b == 'f':
                return 'N|f'
            b = int(b)
            return 'N|c%d' % (0 if b == 0 else 1 if b <= 3 else 2 if b <= 6 else 3)
        if s.startswith('L|'):
            if a['lmode'] == 'exact':
                return s
            if a['lmode'] == 'base':
                return 'L|' + re.split(r'[+ ]', s[2:])[0]
            return 'L'
        return s

    def _keys(self, ex, ctx):
        k = self.a['k']
        out = []
        for j in range(min(k, len(ctx)), 0, -1):
            out.append(ex + tuple(ctx[-j:]))
        if ex:
            out.append(ex)
        out.append(('',))
        return out

    def _draw(self, ex, ctx, want):
        al = self.a['alpha']
        keys = self._keys(ex, ctx)
        for ki, key in enumerate(keys):
            e = self.tab.get(key)
            if e is None:
                continue
            L = e.get(want)
            if not L:
                continue
            if ki == len(keys) - 1 or self.rng.random() < len(L) / (len(L) + al):
                return self.rng.choice(L)
        return None

    def forge(self):
        rng = self.rng
        site, cl = self.doc_cl[rng.randrange(len(self.doc_cl))]
        ex = self._extras(site, cl)
        ctx = ['BOS']
        out = []
        li = 0
        if self.a['template']:
            skel = self.templates[rng.randrange(len(self.templates))]
        else:
            skel = None
        i = 0
        while True:
            exl = ex + (('li%d' % min(li, 3),) if self.a['lineidx'] else ())
            if skel is not None:
                if i >= len(skel):
                    break
                want = skel[i]
                s = 'NL' if want == 'NL' else self._draw(exl, ctx, want)
                if s is None:
                    s = self._draw(('',), [], want) or 'NL'
            else:
                s = self._draw(exl, ctx, '*')
                if s is None or s == 'END' or i > 120:
                    break
            i += 1
            if s == 'NL':
                if out and out[-1][0] != 'NL':
                    out.append(('NL',))
                li += 1
                if self.a['nl_reset']:
                    ctx = ['BOS']
                    continue
                ctx.append('NL')
                continue
            if s.startswith('W|'):
                out.append(('W', s[2:]))
            elif s.startswith('L|'):
                out.append(('L', s[2:]))
            elif s.startswith('N|'):
                out.append(('N', rng.choice(self.vals[s[2:]])))
            ctx.append(self._abs(s))
        while out and out[-1][0] == 'NL':
            out.pop()
        return {'id': 'forg', 'site': site, 'toks': out}


# ------------------------------------------------------------------ the inside-out contest
def contest(docs, n_arch, rng, folds=5, reps=3, arch_list=None, log=None):
    """Returns real presence counts (held-out, every doc once), per-group expected counts, n_docs."""
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    fold_of = {i: k % folds for k, i in enumerate(idx)}
    real = rel_counts(docs)
    archs = arch_list or [random_arch(rng) for _ in range(n_arch)]
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
                E.update(relations(F.forge()))
        for key, v in E.items():
            grp_E[g][key] += v / reps
        grp_n[g] += 1
        if log and (ai + 1) % 25 == 0:
            log('  arch %d/%d' % (ai + 1, len(archs)))
    for g in grp_E:
        for key in grp_E[g]:
            grp_E[g][key] /= grp_n[g]
    contest.last_gn = dict(grp_n)
    return real, grp_E, len(docs), archs


def pois_sf(k, lam):
    """P(X >= k), X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    lam = max(lam, 1e-3)
    # sum_{j<k} pmf
    s, p = 0.0, math.exp(-lam)
    for j in range(k):
        s += p
        p *= lam / (j + 1)
    return max(0.0, 1.0 - s) if s < 0.999999 else _tail(k, lam)


def _tail(k, lam):
    p = math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))
    s, j = 0.0, k
    while p > 1e-300 and j < k + 2000:
        s += p
        j += 1
        p *= lam / j
    return s


def residues(real, grp_E, min_real=3):
    """Per relation: real count R, max expectation over groups M (and min), p = P(Pois(M) >= R)."""
    out = []
    groups = list(grp_E)
    for key, R in real.items():
        if R < min_real:
            continue
        Es = [grp_E[g].get(key, 0.0) for g in groups]
        M = max(Es)
        out.append((key, R, M, min(Es), pois_sf(R, M)))
    return out


def residues_mean(real, grp_E, gn, min_real=3):
    """Per relation: real count R vs the ensemble mean expectation over all architectures."""
    tot = sum(gn.values())
    out = []
    for key, R in real.items():
        if R < min_real:
            continue
        E = sum(grp_E[g].get(key, 0.0) * gn[g] for g in grp_E) / tot
        out.append((key, R, E, E, pois_sf(R, E)))
    return out


def summarize(res, thresholds=(1e-3, 1e-4)):
    by = collections.defaultdict(lambda: [0] * (len(thresholds) + 1))
    for key, R, M, m, p in res:
        f = key[0]
        by[f][0] += 1
        for i, t in enumerate(thresholds):
            if p < t:
                by[f][i + 1] += 1
    return {f: by[f] for f in FAMS if f in by}


# ------------------------------------------------------------------ controls
def shuffle_corpus(docs, rng, mode='kind'):
    """Global token shuffle keeping every doc's layout (kind sequence) and site."""
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
        toks = []
        for t in d['toks']:
            if t[0] == 'NL':
                toks.append(t)
            else:
                toks.append(pools[t[0]][ptr[t[0]]]); ptr[t[0]] += 1
        out.append({'id': d['id'], 'site': d['site'], 'toks': toks})
    return out


def world(docs, rng, arch=None):
    a = arch or {'k': 2, 'fmin': 2, 'nmode': 'bin', 'lmode': 'exact', 'alpha': 0.5, 'lineidx': False,
                 'site': True, 'K': 4, 'template': True, 'nl_reset': False}
    F = Forger(a, docs, rng)
    out = []
    while len(out) < len(docs):
        d = F.forge()
        if sum(x[0] != 'NL' for x in d['toks']) >= 4:
            out.append(d)
    return out


def plant(docs, rng, n_keys=6):
    """Planted hidden constraints in a forger world: key word w -> (a) the number after w is redrawn
    in a fixed scale bin b_w; (b) a partner word p_w is inserted on a later line of the same doc
    (prob 0.8). Returns docs and the planted relation set."""
    wc = collections.Counter(t[1] for d in docs for t in d['toks'] if t[0] == 'W')
    mid = [w for w, c in wc.items() if 6 <= c <= 20 and '-' in w]
    rng.shuffle(mid)
    keys = mid[:n_keys]
    partners = mid[n_keys:2 * n_keys]
    allvals = [t[1] for d in docs for t in d['toks'] if t[0] == 'N' and t[1] >= 1]
    bins = {}
    truth = set()
    for i, w in enumerate(keys):
        b = rng.choice(['2', '3', '4', '5'])
        bins[w] = b
        truth.add(('WN', w, b))
        a, c = sorted([w, partners[i]])
        truth.add(('CO', a, c))
    out = []
    for d in docs:
        toks = list(d['toks'])
        for i, t in enumerate(toks):
            if t[0] == 'W' and t[1] in bins and i + 1 < len(toks) and toks[i + 1][0] == 'N':
                b = int(bins[t[1]])
                toks[i + 1] = ('N', float(rng.randint(2 ** b, 2 ** (b + 1) - 1)))
        present = [t[1] for t in toks if t[0] == 'W' and t[1] in bins]
        for w in present[:1]:
            if rng.random() < 0.8:
                p = partners[keys.index(w)]
                toks = toks + [('NL',), ('W', p), ('N', float(rng.choice(allvals)))]
        out.append({'id': d['id'], 'site': d['site'], 'toks': toks})
    return out, truth


def plant_basket(docs, rng, n_keys=6, rate=0.8):
    """Planted ORDERED long-range constraint (the shape of the SA-RA2 basket): after key word w, with prob
    `rate`, the writer puts a logogram, a number and then a fixed partner word p_w (w L n p_w n).
    Truth: ('ES', w, p_w)."""
    wc = collections.Counter(t[1] for d in docs for t in d['toks'] if t[0] == 'W')
    lc = collections.Counter(t[1] for d in docs for t in d['toks'] if t[0] == 'L')
    mid = [w for w, c in wc.items() if 6 <= c <= 20 and '-' in w]
    rng.shuffle(mid)
    keys, partners = mid[:n_keys], mid[n_keys:2 * n_keys]
    logos = [l for l, c in lc.most_common(8)]
    vals = [t[1] for d in docs for t in d['toks'] if t[0] == 'N' and t[1] >= 1]
    P = dict(zip(keys, partners))
    truth = set(('ES', w, P[w]) for w in keys)
    out = []
    for d in docs:
        toks = []
        for t in d['toks']:
            toks.append(t)
            if t[0] == 'W' and t[1] in P and rng.random() < rate:
                toks += [('L', rng.choice(logos)), ('N', float(rng.choice(vals))), ('W', P[t[1]]),
                         ('N', float(rng.choice(vals)))]
        out.append({'id': d['id'], 'site': d['site'], 'toks': toks})
    return out, truth
