"""LA-23 'count the people': shared loaders, entry features, person-like classifier,
capture-recapture estimators.

No sound value or reading is used for Linear A. Linear B (DAMOS) is used only as a
calibration set: the 'personnel name' label is the series-derived list of
tools/dark_loop56_prep.py (first words of D/E/J/N... tablets and all words of
As/An/B/V... lists, minus occupation/transaction words).
"""
import json, re, os, math, collections, zlib, unicodedata
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la23_ckpt')
os.makedirs(CK, exist_ok=True)
C56 = '/home/user/Indus-/data/derived/dark/loop56_corpora/'


def seed(name):
    return zlib.crc32(name.encode()) % 2 ** 31


# ------------------------------------------------------------------ Linear A
def _js_meta():
    s = open(os.path.join(DATA, 'LinearAInscriptions.js')).read()
    meta = {}
    for m in re.finditer(r'\["([^"]+)",\{(.*?)\n\}\]', s, re.S):
        b = m.group(2)
        f = re.search(r'"findspot": "([^"]*)"', b)
        u = re.search(r'"imageRightsURL": "([^"]*)"', b)
        u = re.sub(r'#.*', '', u.group(1)) if u else ''
        g = re.search(r'GORILA-Vol(\d)', u)
        meta[m.group(1)] = dict(findspot=f.group(1) if f else '',
                                pub=('G' + g.group(1)) if g else ('blank' if u == '' else 'post'))
    return meta


def _la_logo_sign(s):
    return bool(re.fullmatch(r'\*[4-9]\d\d.*', s)) or s in ('VS', 'VAS') or bool(re.search(r'[a-z]', s))


def la_docs(site='Haghia Triada', support='Tablet'):
    """Documents as lists of lines; each line a list of tokens ('W', word) / ('L', logo) / ('N', value)."""
    C = json.load(open(os.path.join(DATA, 'corpus.json')))
    meta = _js_meta()
    ids = {d['id'] for d in C}
    out = []
    for d in C:
        if '+' in d['id']:  # joins whose components are also listed are dropped (as in la15)
            parts = re.split(r'\+', re.sub(r'[ab]$', '', d['id']))
            pre = re.match(r'[A-Z]+[A-Za-z]*?(?=\d)', parts[0]); pre = pre.group(0) if pre else ''
            comp = [parts[0]] + [p if not p[0].isdigit() else pre + p for p in parts[1:]]
            if all(any(i.startswith(c) for i in ids if i != d['id']) for c in comp):
                continue
        if site and d['site'] != site:
            continue
        if support and d['support'] != support:
            continue
        lines, cur = [], []
        for t in d['tokens']:
            if t['t'] == 'nl':
                if cur: lines.append(cur)
                cur = []
            elif t['t'] == 'word':
                if any(_la_logo_sign(x) for x in t['s']):
                    cur.append(('L', '-'.join(t['s'])))
                elif len(t['s']) == 1:
                    cur.append(('W1', t['s'][0]))
                else:
                    cur.append(('W', '-'.join(t['s'])))
            elif t['t'] == 'logo':
                cur.append(('L', t['v']))
            elif t['t'] == 'num':
                cur.append(('N', t['v'] + (0.5 if t['frac'] and t['v'] == 0 else 0)))
            elif t['t'] == 'div':
                cur.append(('D', None))
        if cur: lines.append(cur)
        m = meta.get(d['id'], {})
        logos = collections.Counter(re.split(r'[+]', v)[0] for ln in lines for k, v in ln if k == 'L')
        out.append(dict(id=d['id'], site=d['site'], scribe=d['scribe'] or '', findspot=m.get('findspot', ''),
                        pub=m.get('pub', ''), lines=lines,
                        ctype=(logos.most_common(1)[0][0] if logos else 'none')))
    return out


# ------------------------------------------------------------------ Linear B
_DOT = '̣'


def _strip(w):
    return ''.join(c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn')


def lb_docs(sites=('KN', 'PY')):
    out = []
    for l in open(os.path.join(DATA, 'damos_items.jsonl')):
        x = json.loads(l)
        h = x.get('heading') or ''
        m = re.match(r'(KN|PY|TH|MY|TI)\s+([A-Z][a-z]?)', h)
        if not m or m.group(1) not in sites:
            continue
        hand = re.findall(r'\(([^)]*)\)\s*$', h)
        hand = hand[0] if hand else '-'
        lines = []
        for ln in (x.get('content') or '').split('\n'):
            ln = _strip(ln)
            toks = ln.split()
            cur = []
            for t in toks:
                if re.fullmatch(r'\.?v?\.?[0-9]*[A-Za-z]?[0-9]*', t) and t.startswith('.'):
                    continue  # line label
                if t in (',', '/', '|', 'vac.', 'vacat', 'vest.', 'lat.', 'inf.', 'sup.', 'mut.', 'v.', 'v.↓', '↓'):
                    if t == ',' and cur: cur.append(('D', None))
                    continue
                t2 = t.strip(",'\"⸤⸥⌞⌟")
                if '[' in t2 or ']' in t2 or '?' in t2:
                    t3 = t2.replace('[', '').replace(']', '')
                    if re.fullmatch(r'\d+', t3):
                        cur.append(('N', int(t3)))
                    else:
                        cur.append(('X', None))
                    continue
                if re.fullmatch(r'\d+', t2):
                    cur.append(('N', int(t2)))
                elif re.fullmatch(r'[a-z0-9*]+(-[a-z0-9*]+)+', t2) and re.search(r'[a-z]', t2):
                    cur.append(('W', t2.upper()))
                elif re.fullmatch(r'[a-z]{1,3}\d?', t2):
                    cur.append(('W1', t2.upper()))
                elif re.fullmatch(r'[A-Z][A-Z0-9:+*^]*.*', t2) or t2.startswith('*'):
                    if re.fullmatch(r'[SVZTMNLPQ]', t2):
                        continue  # measure letter
                    cur.append(('L', t2))
            if cur:
                lines.append(cur)
        out.append(dict(id=h, site=m.group(1), series=m.group(2), scribe=hand, lines=lines,
                        ctype=m.group(2)[0]))
    return out


def lb_name_label():
    def jl(f):
        return [tuple(y for y in json.loads(l)['seq'] if y) for l in open(C56 + f)]
    S = set('-'.join(x.upper() for x in w) for w in jl('linb_personnel_dedup.jsonl') if len(w) >= 2)
    return S


# ------------------------------------------------------------------ entries and features
def occurrences(docs):
    """Every word occurrence ('W' tokens, 2+ signs) with its slot features."""
    occ = []
    for di, d in enumerate(docs):
        first_word_seen = False
        nent = 0
        for li, line in enumerate(d['lines']):
            for ti, (k, v) in enumerate(line):
                if k != 'W':
                    continue
                # what follows before the next word
                nxt_num, via_logo, numv, nxt_div = False, False, None, False
                rest = line[ti + 1:] + [tok for ln2 in d['lines'][li + 1:] for tok in ln2]
                for k2, v2 in rest:
                    if k2 in ('W', 'W1', 'X'):
                        break
                    if k2 == 'L': via_logo = True
                    if k2 == 'D': nxt_div = True
                    if k2 == 'N':
                        nxt_num = True; numv = v2; break
                prev = [k2 for k2, _ in line[:ti] if k2 != 'D']
                occ.append(dict(doc=di, w=v, line=li, line_init=(len(prev) == 0), doc_init=not first_word_seen,
                                entry=nxt_num, direct=nxt_num and not via_logo, num=numv, div=nxt_div))
                first_word_seen = True
    # document-level context
    dn = collections.Counter(o['doc'] for o in occ if o['entry'])
    dsmall = collections.defaultdict(list)
    for o in occ:
        if o['entry']: dsmall[o['doc']].append(o['num'] is not None and o['num'] <= 5)
    for o in occ:
        o['doc_nent'] = dn[o['doc']]
        L = dsmall[o['doc']]
        o['doc_small'] = (sum(L) / len(L)) if L else 0.0
        o['doc_logo'] = sum(1 for ln in docs[o['doc']]['lines'] for k, _ in ln if k == 'L')
    return occ


FEATS = ['log_occ', 'log_docs', 'rep_per_doc', 'f_entry', 'f_direct', 'f_small', 'f_one', 'med_lognum', 'f_lineinit',
         'f_docinit', 'f_div', 'nsign', 'doc_nent', 'doc_small', 'doc_logo']


def type_features(occ):
    by = collections.defaultdict(list)
    for o in occ:
        by[o['w']].append(o)
    T, X = [], []
    for w, L in by.items():
        n = len(L); docs = len(set(o['doc'] for o in L))
        nums = [o['num'] for o in L if o['entry'] and o['num'] is not None]
        ent = [o for o in L if o['entry']]
        f = dict(log_occ=math.log(n), log_docs=math.log(docs), rep_per_doc=n / docs,
                 f_entry=len(ent) / n, f_direct=sum(o['direct'] for o in L) / n,
                 f_small=(sum(1 for x in nums if x <= 5) / len(nums)) if nums else 0.0,
                 f_one=(sum(1 for x in nums if x == 1) / len(nums)) if nums else 0.0,
                 med_lognum=(math.log1p(float(np.median(nums)))) if nums else -1.0,
                 f_lineinit=sum(o['line_init'] for o in L) / n, f_docinit=sum(o['doc_init'] for o in L) / n,
                 f_div=sum(o['div'] for o in L) / n, nsign=len(w.split('-')),
                 doc_nent=math.log1p(np.mean([o['doc_nent'] for o in L])),
                 doc_small=float(np.mean([o['doc_small'] for o in L])),
                 doc_logo=math.log1p(np.mean([o['doc_logo'] for o in L])))
        T.append(w); X.append([f[k] for k in FEATS])
    return T, np.array(X, float)


# ------------------------------------------------------------------ capture-recapture
def chapman(n1, n2, m):
    return (n1 + 1) * (n2 + 1) / (m + 1) - 1


def chao1(freqs):
    f = collections.Counter(freqs)
    S = len(freqs); f1, f2 = f.get(1, 0), f.get(2, 0)
    n = sum(freqs)
    if f2 > 0:
        return S + (n - 1) / n * f1 * f1 / (2 * f2)
    return S + (n - 1) / n * f1 * (f1 - 1) / 2


def coverage(freqs):
    f = collections.Counter(freqs); n = sum(freqs); f1, f2 = f.get(1, 0), f.get(2, 0)
    if n == 0: return 0.0
    if f2 == 0: return 1 - f1 / n
    return 1 - f1 / n * ((n - 1) * f1 / ((n - 1) * f1 + 2 * f2))


def poisson_glm(X, y, iters=60):
    """Plain IRLS Poisson regression; returns beta, deviance, fitted mu."""
    beta = np.zeros(X.shape[1]); beta[0] = math.log(max(y.mean(), 0.5))
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30); mu = np.exp(eta)
        z = eta + (y - mu) / mu
        W = mu
        A = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        nb = np.linalg.solve(A, X.T @ (W * z))
        if np.max(np.abs(nb - beta)) < 1e-9:
            beta = nb; break
        beta = nb
    mu = np.exp(np.clip(X @ beta, -30, 30))
    dev = 2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-300) / mu), 0) - (y - mu))
    return beta, dev, mu


def loglinear(histories, K, models=('M0', 'Mt', 'Mh', 'Mth', 'Mt2')):
    """Log-linear capture-recapture on K lists. histories: list of tuples of 0/1 (observed, non-zero).
    Mh = Darroch quasi-symmetry (main effects + indicators for number of captures >= 2... K-1).
    Returns dict model -> (N_hat, AIC)."""
    cells = [tuple((i >> j) & 1 for j in range(K)) for i in range(1, 2 ** K)]
    cnt = collections.Counter(histories)
    y = np.array([cnt.get(c, 0) for c in cells], float)
    S = y.sum()
    res = {}
    for mod in models:
        cols = [[1.0] * len(cells)]
        if mod in ('Mt', 'Mth', 'Mt2'):
            for j in range(K): cols.append([c[j] for c in cells])
        else:
            cols.append([sum(c) for c in cells])
        if mod in ('Mh', 'Mth'):
            for r in range(3, K + 1):  # quasi-symmetry: (s choose 2) heterogeneity term, Darroch et al.
                pass
            cols.append([sum(c) * (sum(c) - 1) / 2 for c in cells])
        if mod == 'Mt2':
            for a in range(K):
                for b in range(a + 1, K):
                    cols.append([c[a] * c[b] for c in cells])
        X = np.array(cols, float).T
        if np.linalg.matrix_rank(X) >= len(cells):
            continue
        beta, dev, mu = poisson_glm(X, y)
        n0 = math.exp(min(beta[0], 30))
        ll = float(np.sum(y * np.log(np.maximum(mu, 1e-300)) - mu))
        res[mod] = (S + n0, -2 * ll + 2 * X.shape[1], dev)
    return res


def ztbb_loglik(fk, T, N, a, b):
    """Zero-truncated beta-binomial capture-frequency likelihood with N individuals
    (full multinomial incl. unseen). fk: array length T+1 of counts of individuals caught k times."""
    from scipy.special import gammaln, betaln
    k = np.arange(T + 1)
    lp = gammaln(T + 1) - gammaln(k + 1) - gammaln(T - k + 1) + betaln(k + a, T - k + b) - betaln(a, b)
    S = fk[1:].sum()
    if N < S: return -np.inf
    f0 = N - S
    return (gammaln(N + 1) - gammaln(f0 + 1) - np.sum(gammaln(fk[1:] + 1)) + f0 * lp[0] + np.sum(fk[1:] * lp[1:]))


def mh_bayes(fk, T, rng, n_iter=6000, burn=2000, thin=4, Nmax=None):
    """Bayesian Mh (beta-binomial heterogeneity), prior N ~ 1/N (scale), log a, log b ~ N(0, 2^2).
    Random-walk Metropolis. Returns posterior draws of N."""
    S = int(fk[1:].sum())
    if Nmax is None: Nmax = S * 200
    N, la, lb = int(S * 1.5) + 1, 0.0, 1.0
    cur = ztbb_loglik(fk, T, N, math.exp(la), math.exp(lb)) - math.log(N) - (la ** 2 + lb ** 2) / 8
    out = []
    step = max(2, S // 5)
    for it in range(n_iter):
        N2 = N + int(rng.integers(-step, step + 1))
        la2 = la + rng.normal(0, 0.15); lb2 = lb + rng.normal(0, 0.15)
        if S <= N2 <= Nmax:
            new = ztbb_loglik(fk, T, N2, math.exp(la2), math.exp(lb2)) - math.log(N2) - (la2 ** 2 + lb2 ** 2) / 8
            if math.log(rng.random() + 1e-300) < new - cur:
                N, la, lb, cur = N2, la2, lb2, new
        if it >= burn and (it - burn) % thin == 0:
            out.append(N)
    return np.array(out)


def fk_from_counts(counts, T):
    fk = np.zeros(T + 1)
    for c in counts:
        fk[min(c, T)] += 1
    return fk
