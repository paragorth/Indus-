"""la75: forensic audit of Linear A numbers (counted vs made-up), random fingerprint search.
Entries = numbers on tablets (one row per distinct (value, fraction) per document, totals excluded)."""
import json, os, re, collections, hashlib
import numpy as np

D = os.path.join(os.path.dirname(__file__), '..', 'data')
CK = os.path.join(D, 'la75_ckpt')
TOTAL_WORDS = {('KU', 'RO'), ('PO', 'TO', 'KU', 'RO'), ('KI', 'RO')}
FRACS = ['J', 'E', 'F', 'K', 'B', 'D', 'H', 'L2', 'A', 'JE', 'L', 'X', 'Y', 'W', 'Q']


def magbin(v):
    if v <= 0: return 0
    return min(int(np.log2(v)) + 1, 8)


def la_entries(read_only=True, supports=('Tablet', 'Lames (short thin tablet)')):
    d = json.load(open(os.path.join(D, 'corpus_ra.json')))
    rows = []; docs = {}
    for x in d:
        if x['support'] not in supports: continue
        docs[x['id']] = x
        tag = 'W'; words = []; is_tot = False; seen = set()
        for t in x['tokens']:
            if t['t'] == 'nl':
                tag = 'W'; words = []; is_tot = False; continue
            if t['t'] == 'logo':
                tag = t['v'].split('+')[0]; continue
            if t['t'] == 'word':
                w = tuple(t['s']); words.append('-'.join(w))
                if w in TOTAL_WORDS: is_tot = True
                continue
            if t['t'] == 'num':
                if read_only and t['st'] != 'read': continue
                if is_tot: continue
                fr = tuple(sorted(t['frac']))
                key = (t['v'], fr)
                if key in seen: continue          # collapse within-document repeats
                seen.add(key)
                rows.append(dict(doc=x['id'], site=x['site'], v=t['v'], frac=fr, tag=tag,
                                 word=words[-1] if words else '', scribe=x.get('scribe', ''),
                                 findspot=x.get('findspot', '')))
    return rows, docs


def lb_entries(prefix='KN'):
    """Linear B control: ideogram + number pairs from DAMOS; hand = parenthesised heading tail."""
    rows = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        if not h.startswith(prefix): continue
        m = re.search(r'\(([^()]*)\)\s*$', h)
        hand = m.group(1).strip() if m else ''
        doc = re.sub(r'\s*\([^()]*\)\s*$', '', h).strip()
        seen = set()
        for l in (d.get('content') or '').split('\n'):
            toks = [re.sub(r'[\[\]⟦⟧?]', '', t) for t in l.split()]
            if any(re.match(r'^to-s[oa]', t) for t in toks): continue
            tag = 'W'; word = ''
            for i, t in enumerate(toks):
                if re.match(r'^[a-z][a-z0-9\-]+$', t) and '-' in t: word = t
                elif re.match(r'^[A-Z*][A-Z0-9±+*]+$', t) and t not in ('S', 'V', 'Z', 'T', 'M', 'N', 'P', 'Q'):
                    tag = t.split('+')[0]
                elif t.isdigit():
                    prev = toks[i - 1] if i else ''
                    if prev in ('S', 'V', 'Z', 'T', 'M', 'N', 'P', 'Q'): continue   # sub-unit counts
                    key = (int(t), ())
                    if key in seen: continue
                    seen.add(key)
                    rows.append(dict(doc=doc, site=prefix, v=int(t), frac=(), tag=tag, word=word,
                                     scribe=hand, findspot=''))
    return rows


def prepare(rows, min_n=3, rare_tag=10):
    cnt = collections.Counter(r['doc'] for r in rows)
    rows = [r for r in rows if cnt[r['doc']] >= min_n]
    tc = collections.Counter(r['tag'] for r in rows)
    for r in rows:
        r['tg'] = r['tag'] if tc[r['tag']] >= rare_tag else ('W' if r['tag'] == 'W' else 'LOGO')
        r['mb'] = magbin(r['v'])
        r['stratum'] = (r['site'], r['tg'], r['mb'])
    return rows


# ---------------------------------------------------------------- random fingerprint features
def gen_features(rng, n):
    feats = []
    for _ in range(n):
        fam = rng.integers(0, 8)
        if fam == 0:
            m = int(rng.integers(2, 13)); k = int(rng.integers(1, m)); S = tuple(sorted(rng.choice(m, k, replace=False).tolist()))
            feats.append(('mod', m, S))
        elif fam == 1:
            k = int(rng.integers(1, 6)); feats.append(('last', tuple(sorted(rng.choice(10, k, replace=False).tolist()))))
        elif fam == 2:
            k = int(rng.integers(1, 5)); feats.append(('lead', tuple(sorted((1 + rng.choice(9, k, replace=False)).tolist()))))
        elif fam == 3:
            feats.append(('div', int(rng.choice([2, 3, 4, 5, 6, 10, 12, 20, 25, 50, 60, 100]))))
        elif fam == 4:
            k = int(rng.integers(1, 4)); feats.append(('frac', tuple(sorted(rng.choice(len(FRACS), k, replace=False).tolist()))))
        elif fam == 5:
            m = int(rng.choice([2, 3, 9])); feats.append(('dsum', m, int(rng.integers(0, m))))
        elif fam == 6:
            m = 30; k = int(rng.integers(3, 15)); S = tuple(sorted(rng.choice(m, k, replace=False).tolist()))
            feats.append(('mod', m, S))
        else:
            feats.append(('tens', tuple(sorted(rng.choice(10, int(rng.integers(1, 6)), replace=False).tolist()))))
    return feats


def fval(f, v, frac):
    t = f[0]
    if t == 'mod': return (v % f[1]) in f[2]
    if t == 'last': return v >= 10 and (v % 10) in f[1]
    if t == 'lead': return v >= 10 and int(str(v)[0]) in f[1]
    if t == 'div': return v >= f[1] and v % f[1] == 0
    if t == 'frac': return any(FRACS[i] in frac for i in f[1])
    if t == 'dsum': return sum(map(int, str(v))) % f[1] == f[2]
    if t == 'tens': return v >= 10 and (v // 10) % 10 in f[1]
    raise ValueError(f)


def fmatrix(rows, feats):
    vals = sorted(set((r['v'], r['frac']) for r in rows))
    idx = {k: i for i, k in enumerate(vals)}
    T = np.array([[fval(f, v, fr) for f in feats] for v, fr in vals], dtype=np.float32)
    return T[[idx[(r['v'], r['frac'])] for r in rows]]


class Panel:
    """Within-document residual covariance statistic for many features, with stratum permutation null."""
    def __init__(self, rows):
        self.rows = rows
        docs = sorted(set(r['doc'] for r in rows)); self.docs = docs
        di = {d: i for i, d in enumerate(docs)}
        self.g = np.array([di[r['doc']] for r in rows])
        st = sorted(set(r['stratum'] for r in rows)); si = {s: i for i, s in enumerate(st)}
        self.s = np.array([si[r['stratum']] for r in rows])
        self.strata = [np.where(self.s == k)[0] for k in range(len(st))]
        self.nd = np.bincount(self.g, minlength=len(docs)).astype(float)
        self.den = (self.nd * (self.nd - 1)).sum()

    def resid(self, F):
        R = F.copy()
        for ix in self.strata:
            R[ix] -= F[ix].mean(0)
        return R

    def stat(self, F, perm=None):
        R = self.resid(F)
        if perm is not None: R = R[perm]
        G = np.zeros((len(self.docs), F.shape[1]), dtype=np.float64)
        np.add.at(G, self.g, R)
        Q = np.zeros_like(G); np.add.at(Q, self.g, R.astype(np.float64) ** 2)
        num = (G ** 2 - Q).sum(0)
        var = (R.astype(np.float64) ** 2).mean(0) + 1e-12
        return num / self.den / var        # within-document residual correlation

    def perm(self, rng):
        p = np.arange(len(self.rows))
        for ix in self.strata:
            p[ix] = rng.permutation(ix)
        return p

    def doc_scores(self, F, w=None):
        R = self.resid(F)
        if w is not None: R = R * w
        G = np.zeros((len(self.docs), F.shape[1])); np.add.at(G, self.g, R)
        return G / np.sqrt(self.nd)[:, None]


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
