"""LA-15 'count the words like wildlife': shared loaders and field-ecology estimators.

Words (2+ signs) are species, documents are sampling units, sites are habitats.
No reading or sound value is used anywhere; only sign identities, sites, scribes and
publication source (GORILA volume vs later papers, from lineara.xyz imageRightsURL).
"""
import json, re, os, math, random, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(DATA, 'la15')
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Linear A

def _pub_source():
    s = open(os.path.join(DATA, 'LinearAInscriptions.js')).read()
    src = {}
    for m in re.finditer(r'\["([^"]+)",\{(.*?)\n\}\]', s, re.S):
        u = re.search(r'"imageRightsURL": "([^"]*)"', m.group(2))
        u = re.sub(r'#.*', '', u.group(1)) if u else ''
        g = re.search(r'GORILA-Vol(\d)', u)
        src[m.group(1)] = ('G' + g.group(1)) if g else ('blank' if u == '' else 'post')
    return src

# GORILA publication years (vol. 1 1976, 2 1979, 3 1976, 4 1982, 5 1985); 'post' = later papers
PUB_YEAR = {'G1': 1976, 'G3': 1976, 'G2': 1979, 'G4': 1982, 'G5': 1985, 'post': 2000, 'blank': None}


def _logo(s):
    """Logogram / vessel / ligature signs (A 400+ numbers, VS, VAS): not syllabic."""
    return bool(re.fullmatch(r'\*[4-9]\d\d.*', s)) or s in ('VS', 'VAS') or bool(re.search(r'[a-z]', s))


def load_la():
    C = json.load(open(os.path.join(DATA, 'corpus.json')))
    src = _pub_source()
    ids = {d['id'] for d in C}
    docs = []
    for d in C:
        # joins whose components are also listed are dropped (avoid double counting)
        if '+' in d['id']:
            parts = re.split(r'\+', re.sub(r'[ab]$', '', d['id']))
            pre = re.match(r'[A-Z]+[A-Za-z]*?(?=\d)', parts[0])
            pre = pre.group(0) if pre else ''
            comp = [parts[0]] + [p if not p[0].isdigit() else pre + p for p in parts[1:]]
            if all(any(i.startswith(c) for i in ids if i != d['id']) for c in comp):
                continue
        words = [w for w in d['words'] if '-' in w and not any(_logo(x) for x in w.split('-'))]
        signs = [s for t in d['tokens'] if t['t'] == 'word' for s in t['s'] if not _logo(s)]
        docs.append(dict(id=d['id'], site=d['site'] or '?', support=d['support'],
                         scribe=d.get('scribe') or '', pub=src.get(d['id'], 'blank'),
                         words=words, signs=signs))
    return docs

# ---------------------------------------------------------------- Linear B (DAMOS)

_DOT = '̣'


def load_lb():
    docs = []
    for l in open(os.path.join(DATA, 'damos_items.jsonl')):
        x = json.loads(l)
        h, c = x.get('heading') or '', x.get('content') or ''
        if not h:
            continue
        site = h.split()[0]
        sc = re.findall(r'\(([^)]*)\)\s*$', h)
        c = c.replace(_DOT, '')
        words, signs = [], []
        for tok in re.split(r'[\s,/|]+', c):
            tok = tok.strip("'\".⸤⸥⌞⌟")
            if not tok or '[' in tok or ']' in tok or '?' in tok:
                continue
            if re.fullmatch(r'[a-z*0-9₂₃]+(-[a-z*0-9₂₃]+)+', tok) and re.search(r'[a-z]', tok):
                sy = tok.split('-')
                words.append(tok.upper())
                signs += [s.upper() for s in sy]
        docs.append(dict(id=h, site=site, support='tablet', scribe=sc[0] if sc else '',
                         pub='', words=words, signs=signs))
    return docs

# ---------------------------------------------------------------- estimators

def freq_counts(abund):
    """abund: iterable of per-species counts -> dict k -> f_k"""
    return collections.Counter(a for a in abund if a > 0)


def chao1(abund):
    a = [x for x in abund if x > 0]
    S, n = len(a), sum(a)
    f = freq_counts(a)
    f1, f2 = f.get(1, 0), f.get(2, 0)
    k = (n - 1) / n if n > 0 else 1
    if f2 > 0:
        return S + k * f1 * f1 / (2 * f2)
    return S + k * f1 * (f1 - 1) / 2


def chao2(incid, T):
    """incid: per-species number of sampling units containing it; T units"""
    a = [x for x in incid if x > 0]
    S = len(a)
    f = freq_counts(a)
    q1, q2 = f.get(1, 0), f.get(2, 0)
    k = (T - 1) / T
    if q2 > 0:
        return S + k * q1 * q1 / (2 * q2)
    return S + k * q1 * (q1 - 1) / 2


def ace(abund, rare=10):
    a = [x for x in abund if x > 0]
    S_abun = sum(1 for x in a if x > rare)
    rr = [x for x in a if x <= rare]
    S_rare = len(rr)
    n_rare = sum(rr)
    f = freq_counts(rr)
    f1 = f.get(1, 0)
    if n_rare == 0:
        return float(S_abun)
    C = 1 - f1 / n_rare
    if C <= 0:
        return chao1(a)
    g2 = max(S_rare / C * sum(i * (i - 1) * f[i] for i in f) / (n_rare * (n_rare - 1)) - 1, 0)
    return S_abun + S_rare / C + f1 / C * g2


def jack2(incid, T):
    a = [x for x in incid if x > 0]
    f = freq_counts(a)
    q1, q2 = f.get(1, 0), f.get(2, 0)
    return len(a) + q1 * (2 * T - 3) / T - q2 * (T - 2) ** 2 / (T * (T - 1))


def coverage(abund):
    """Chao-Jost sample coverage."""
    a = [x for x in abund if x > 0]
    n = sum(a)
    f = freq_counts(a)
    f1, f2 = f.get(1, 0), f.get(2, 0)
    if n < 2:
        return 0.0
    if f2 > 0:
        r = (n - 1) * f1 / ((n - 1) * f1 + 2 * f2)
    else:
        r = (n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2) if f1 > 1 else 0
    return 1 - f1 / n * r


def extrap_new(abund, m):
    """Shen-Chao-Lin: expected number of NEW species in m further individuals (Chao1 f0)."""
    a = [x for x in abund if x > 0]
    n = sum(a)
    f = freq_counts(a)
    f1 = f.get(1, 0)
    f0 = chao1(a) - len(a)
    if f0 <= 0 or f1 == 0:
        return 0.0
    return f0 * (1 - (1 - f1 / (n * f0 + f1)) ** m)


def rarefy(abund, m):
    """Expected species in a random subsample of m individuals (hypergeometric)."""
    a = np.array([x for x in abund if x > 0], float)
    n = a.sum()
    if m >= n:
        return float(len(a))
    from math import lgamma
    lc = lambda N, k: lgamma(N + 1) - lgamma(k + 1) - lgamma(N - k + 1)
    tot = lc(n, m)
    s = 0.0
    for x in a:
        if n - x >= m:
            s += 1 - math.exp(lc(n - x, m) - tot)
        else:
            s += 1
    return s


def all_estimates(docs, key='words'):
    ab = collections.Counter(w for d in docs for w in d[key])
    inc = collections.Counter(w for d in docs for w in set(d[key]))
    T = sum(1 for d in docs if d[key])
    A = list(ab.values())
    return dict(S=len(ab), n=sum(A), T=T, f1=sum(1 for x in A if x == 1), f2=sum(1 for x in A if x == 2),
                chao1=chao1(A), ace=ace(A), chao2=chao2(list(inc.values()), T),
                jack2=jack2(list(inc.values()), T), cover=coverage(A))

# ---------------------------------------------------------------- null model


def curveball(rows, iters, rng):
    """Fixed-fixed swap null for a presence matrix given as a list of sets (rows=docs).
    Row sizes and column totals are preserved (Strona et al. 2014)."""
    rows = [set(r) for r in rows]
    R = len(rows)
    for _ in range(iters):
        i, j = rng.randrange(R), rng.randrange(R)
        if i == j:
            continue
        a, b = rows[i], rows[j]
        ua, ub = list(a - b), list(b - a)
        if not ua or not ub:
            continue
        pool = ua + ub
        rng.shuffle(pool)
        k = len(ua)
        common = a & b
        rows[i] = common | set(pool[:k])
        rows[j] = common | set(pool[k:])
    return rows


def token_shuffle(docs, rng, key='words'):
    """Break word-site links: permute word tokens across documents, keep doc sizes."""
    pool = [w for d in docs for w in d[key]]
    rng.shuffle(pool)
    out, i = [], 0
    for d in docs:
        k = len(d[key])
        e = dict(d)
        e[key] = pool[i:i + k]
        i += k
        out.append(e)
    return out


def write_rows(path, header, rows):
    with open(path, 'w') as f:
        f.write(header + '\n\n| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            f.write('| ' + ' | '.join(r) + ' |\n')
