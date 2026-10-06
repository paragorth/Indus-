#!/usr/bin/env python3
"""la66 EVERY LINEAR A WORD HAS A WEIGHT (port of proto-elamite pe52 / pe58 to Linear A).

Arrow in the dark: if Linear A entry words are descriptions (grade, age, sex, unit size, ration class)
rather than names, each word and each single sign in an entry shifts the log quantity written with it by a
fixed amount.  Thousands of random log-linear (multiplicative) and linear-relative (additive) models are
fitted on half the documents, screened on a quarter and re-scored on a reserved quarter, always against a
baseline that knows the exact identity of the entry string (word identity alone).  A feature is a
DESCRIPTOR when its effect is stable across splits and removing it hurts prediction on strings never seen
in training; a frequent feature with no such effect is an IDENTIFIER.

Rows: dict(doc, f: tuple of feature ids, key: identity key, q: quantity (integer part, >0 or 0.5 for
fraction-only), s: stratum (commodity x fraction use), lab: optional truth label per feature).
Feature ids: 'W:<word>' (every word in the entry, multi- or single-sign), 'S:<sign>' (every sign of a
multi-sign word), 'G:<part>' (ligature part / sex-age modifier of the commodity logogram).
No Linear B value of any Linear A sign is an input; Linear B and Ur III are only controls.
"""
import os, sys, json, math, random, re, hashlib
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(ROOT, 'data')
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(DATA, 'la66_ckpt'); os.makedirs(CK, exist_ok=True)
PEDATA = os.path.abspath(os.path.join(ROOT, '..', 'proto-elamite', 'data'))

LA_COMW = {'NI', 'DI-DE-RU', 'DA-SI-*118', '*28B-NU-MA-RE', 'U-*325-ZA', 'TE-TU', '*304', '*308', '*306'}
LA_TOT = {'KU-RO', 'PO-TO-KU-RO', 'KI-RO'}


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ corpora
def _feats(words, lig):
    f = []
    for w in words:
        f.append('W:' + w)
        p = w.split('-')
        if len(p) > 1:
            f += ['S:' + s for s in p]
    f += ['G:' + g for g in lig]
    return tuple(dict.fromkeys(f))


def la_rows(totals=False):
    """Linear A administrative documents (la60 loader, libation supports dropped)."""
    from la60_common import load_la
    out = []
    for d in load_la():
        if d['lib']:
            continue
        com, lig, pend = 'NONE', [], []
        for kind, v, fr in d['toks']:
            if kind == 'NL':
                continue
            if kind == 'L':
                parts = [p.strip('[]').lstrip('*') for p in v.split('+')]
                parts = [p for p in parts if p and p != '?']
                base = parts[0] if parts else '?'
                com = 'VIR' if base.startswith('VIR') else base
                lig = parts[1:]
                continue
            if kind == 'W':
                if v in LA_COMW:
                    com, lig = v, []
                else:
                    pend.append(v)
                continue
            if kind == 'N':
                n = int(v)
                if n == 0 and not fr:
                    pend = []; continue
                tot = any(w in LA_TOT for w in pend)
                if tot and not totals:
                    pend = []; continue
                words = [w for w in pend if '?' not in w]
                q = float(n) if n > 0 else 0.5
                fx = _feats(words, lig)
                out.append(dict(doc=d['id'], f=fx, key=(com,) + tuple(sorted(words)), q=q,
                                s=com + ('|F' if fr else '|I') + ('|0' if n == 0 else ''), com=com,
                                site=d['site'], tot=tot))
                pend = []
    return out


def _strip(t):
    import unicodedata
    return ''.join(ch for ch in unicodedata.normalize('NFD', t) if unicodedata.category(ch) != 'Mn')


LB_SUB = {'T', 'V', 'Z', 'S', 'M', 'N', 'P', 'Q', 'L'}


def lb_rows(series=None):
    """Linear B (DAMOS).  Entries: words since the previous number, the current logogram with its
    modifiers (OVIS:m -> base OVIS, G:m; TELA+TE -> G:TE), integer part q; subunits only as a fraction flag.
    Trailing words of a document (e.g. KN Da .B lines) are attached to the last entry if it has none."""
    fn = os.path.join(CK, 'lb_rows.json')
    if os.path.exists(fn) and series is None:
        R = json.load(open(fn))
        for r in R:
            r['f'] = tuple(r['f']); r['key'] = tuple(r['key'])
        return R
    out = []
    for raw in open(os.path.join(DATA, 'damos_items.jsonl')):
        d = json.loads(raw)
        h = d.get('heading') or ''
        m = re.match(r'^([A-Z]{2,3})\s+([A-Z][a-z]?)', h)
        if not m:
            continue
        did = re.sub(r'\s*\(\S*\)\s*$', '', h).strip()
        com, lig, pend, cur = None, [], [], None
        ents = []
        body = (d.get('content') or '')
        body = re.sub(r'⟦[^⟧]*⟧', ' ', body)
        for ln in body.split('\n'):
            for t in _strip(ln).split():
                s = re.sub(r'[\[\]?!,⌞⌟\'"]', '', t)
                if not s or s.startswith('.') or s in ('/', 'vac', 'vac.', 'X', 'v', 'v.', 'lat.', 'inf.', 'sup.'):
                    continue
                if re.fullmatch(r'\d+', s):
                    v = int(s)
                    if com is None or v == 0:
                        pend = []; continue
                    if cur == 'sub':
                        if ents:
                            ents[-1]['fr'] = True
                        cur = None; continue
                    ents.append(dict(com=com, lig=list(lig), words=list(pend), n=v, fr=False))
                    pend = []; cur = None
                    continue
                if s in LB_SUB:
                    cur = 'sub'; continue
                if re.fullmatch(r'[A-Z*][A-Z0-9*]*([:+][A-Za-z0-9*+:]+)?', s) and not re.fullmatch(r'[a-z]', s):
                    parts = re.split(r'[:+]', s)
                    com = parts[0]; lig = [p for p in parts[1:] if p]
                    cur = None
                    continue
                if re.fullmatch(r'[a-z0-9*]+(-[a-z0-9*]+)*', s):
                    pend.append(s)
        if pend and ents and not ents[-1]['words']:
            ents[-1]['words'] = pend
        for e in ents:
            words = e['words']
            fx = _feats(words, e['lig'])
            out.append(dict(doc=did, f=fx, key=(e['com'],) + tuple(sorted(words)) + tuple(e['lig']),
                            q=float(e['n']), s=e['com'] + ('|F' if e['fr'] else '|I'), com=e['com'],
                            site=m.group(1), series=m.group(1) + ' ' + m.group(2),
                            tot=any(w in ('to-so', 'to-sa', 'to-so-de', 'to-sa-de') for w in words)))
    json.dump(out, open(fn, 'w'))
    return out


def ur_herd_rows():
    """Ur III Drehem livestock (pe52 herd.json): opaque word-level tokens; lab 1 = name/deity token."""
    P = json.load(open(os.path.join(PEDATA, 'pe52_ckpt', 'herd.json')))
    out = []
    for r in P:
        if r['q'] <= 0:
            continue
        w = list(dict.fromkeys(r['w']))
        out.append(dict(doc=r['tab'], f=tuple('W:' + x for x in w), key=tuple(sorted(w)), q=float(r['q']),
                        s=r['sys'] + '|I', com=r['sys'], tot=False))
    return out


def ur_ration_rows(total=True):
    """Ur III distributive ration/fodder lines (pe58 ur3ta.json).  total=True: q = n x rate (what a Linear
    A list would show if it lists group allotments); total=False: q = rate per head."""
    T = json.load(open(os.path.join(PEDATA, 'pe58_ckpt', 'ur3ta.json')))
    out = []
    for t in T:
        w = list(dict.fromkeys(t['grade'] + [x for x in t['after'][:3] if not x[0].isdigit()]))
        q = t['rate'] * (t['n'] if total else 1)
        if q > 0:
            out.append(dict(doc=t['tab'], f=tuple('W:' + x for x in w), key=tuple(sorted(w)), q=float(q),
                            s='C|I', com='C', grade=tuple(t['grade']), n=t['n'], rate=t['rate'], tot=False))
    return out


def opaque(rows, seed=0):
    """replace every feature string by an opaque id (truth map returned)"""
    names = sorted({f for r in rows for f in r['f']})
    rng = random.Random(seed); ids = list(range(len(names))); rng.shuffle(ids)
    mp = {n: n[:2] + 'X%05d' % i for n, i in zip(names, ids)}
    out = []
    for r in rows:
        o = dict(r); o['f'] = tuple(mp[f] for f in r['f']); o['key'] = (r['com'],) + tuple(sorted(o['f']))
        out.append(o)
    return out, mp


def thin(rows, n_target, seed, min_ent=2, by=None):
    """keep whole documents (>= min_ent entries) in random order until n_target entries (LA size)."""
    rng = random.Random(seed)
    g = defaultdict(list)
    for r in rows:
        g[r['doc']].append(r)
    docs = [d for d in g if len(g[d]) >= min_ent]
    rng.shuffle(docs)
    out = []
    for d in docs:
        out += g[d]
        if len(out) >= n_target:
            break
    return out


# ------------------------------------------------------------------ nulls
def null_q_com(rows, seed):
    """quantities shuffled among entries of the same stratum (commodity x fraction use), whole corpus"""
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r['s']].append(i)
    out = [dict(r) for r in rows]
    for idx in by.values():
        qs = [rows[i]['q'] for i in idx]; rng.shuffle(qs)
        for i, q in zip(idx, qs):
            out[i]['q'] = q
    return out


def null_q_doc(rows, seed):
    """quantities shuffled within document x stratum (keeps document scale)"""
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[(r['doc'], r['s'])].append(i)
    out = [dict(r) for r in rows]
    for idx in by.values():
        qs = [rows[i]['q'] for i in idx]; rng.shuffle(qs)
        for i, q in zip(idx, qs):
            out[i]['q'] = q
    return out


def _regroup(words, lig):
    return _feats(words, lig)


def null_words_doc(rows, seed):
    """the WORD bundles (all features of an entry except ligature parts) re-dealt among the entries of the
    same document: a word keeps its document but loses its own quantity."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r['doc']].append(i)
    out = [dict(r) for r in rows]
    for idx in by.values():
        bundles = [tuple(f for f in rows[i]['f'] if not f.startswith('G:')) for i in idx]
        rng.shuffle(bundles)
        for i, b in zip(idx, bundles):
            g = tuple(f for f in rows[i]['f'] if f.startswith('G:'))
            out[i]['f'] = b + g
            out[i]['key'] = (rows[i]['com'],) + tuple(sorted(b + g))
    return out


def null_signs_entry(rows, seed):
    """signs shuffled across the multi-sign words of each entry (sign bag kept, word identities broken)"""
    rng = random.Random(seed)
    out = []
    for r in rows:
        words = [f[2:] for f in r['f'] if f.startswith('W:')]
        g = [f[2:] for f in r['f'] if f.startswith('G:')]
        multi = [w for w in words if '-' in w]
        single = [w for w in words if '-' not in w]
        pool = [s for w in multi for s in w.split('-')]
        rng.shuffle(pool)
        nw, p = [], 0
        for w in multi:
            L = len(w.split('-')); nw.append('-'.join(pool[p:p + L])); p += L
        o = dict(r); o['f'] = _feats(single + nw, g); o['key'] = (r['com'],) + tuple(sorted(single + nw))
        out.append(o)
    return out


def plant(rows, seed, k=12, id_sd=0.35, kinds=('W', 'S'), lo=0.4, hi=1.0, values=None):
    """Linear A skeleton with k planted descriptor features (effects +-U(lo,hi) log, or given values),
    an identity effect N(0,id_sd) per distinct string, document x stratum means and residual sd copied."""
    rng = np.random.default_rng(seed)
    cnt = Counter(f for r in rows for f in r['f'])
    dset = defaultdict(set)
    for r in rows:
        for f in r['f']:
            dset[f].add(r['doc'])
    pool = sorted(f for f in cnt if f[0] in kinds and 8 <= cnt[f] <= 150 and len(dset[f]) >= 4)
    chosen = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    if values is None:
        eff = {f: float(rng.choice([-1, 1]) * rng.uniform(lo, hi)) for f in chosen}
    else:
        eff = {f: float(v) for f, v in zip(chosen, values)}
    grp = defaultdict(list)
    for r in rows:
        grp[(r['doc'], r['s'])].append(math.log(r['q']))
    gm = {g: float(np.mean(v)) for g, v in grp.items()}
    res = [math.log(r['q']) - gm[(r['doc'], r['s'])] for r in rows if len(grp[(r['doc'], r['s'])]) > 1]
    sd = float(np.std(res)) * 0.8
    ide, out = {}, []
    for r in rows:
        if r['key'] not in ide:
            ide[r['key']] = float(rng.normal(0, id_sd))
        y = gm[(r['doc'], r['s'])] + sum(eff.get(f, 0.0) for f in r['f']) + ide[r['key']] + rng.normal(0, sd)
        o = dict(r); o['q'] = float(math.exp(y)); out.append(o)
    return out, eff


# ------------------------------------------------------------------ engine
class Data:
    def __init__(self, rows, mode):
        self.mode = mode
        y = np.array([math.log(r['q']) for r in rows])
        S = [r['s'] for r in rows]
        smu = defaultdict(list)
        for v, s in zip(y, S):
            smu[s].append(v)
        smu = {s: float(np.mean(v)) for s, v in smu.items()}
        yc = y - np.array([smu[s] for s in S])            # stratum-centred (whole corpus; re-centred on train later)
        grp = defaultdict(list)
        for i, r in enumerate(rows):
            grp[r['doc']].append(i)
        ctx = np.zeros(len(rows)); nos = np.zeros(len(rows)); keep = np.ones(len(rows), bool)
        for idx in grp.values():
            if len(idx) < 2:
                nos[idx] = 1; keep[idx] = mode != 'tab'; continue
            sm = yc[idx].sum()
            for i in idx:
                ctx[i] = (sm - yc[i]) / (len(idx) - 1)
        idx = np.where(keep)[0]
        self.rows = [rows[i] for i in idx]
        self.y = y[idx]; self.S = np.array([S[i] for i in idx]); self.ctx = ctx[idx]; self.nos = nos[idx]
        self.doc = np.array([rows[i]['doc'] for i in idx])
        self.key = [rows[i]['key'] for i in idx]
        self.n = len(idx)
        self.docs = sorted(set(self.doc))
        names = sorted({f for r in self.rows for f in r['f']})
        self.fi = {f: j for j, f in enumerate(names)}
        self.fnames = names
        X = np.zeros((self.n, len(names)), np.float32)
        for i, r in enumerate(self.rows):
            for f in r['f']:
                X[i, self.fi[f]] = 1
        self.X = X

    def split(self, seed):
        rng = random.Random(seed)
        d = list(self.docs); rng.shuffle(d)
        n = len(d)
        part = {t: (0 if i < n // 2 else 1 if i < 3 * n // 4 else 2) for i, t in enumerate(d)}
        p = np.array([part[t] for t in self.doc])
        return np.where(p == 0)[0], np.where(p == 1)[0], np.where(p == 2)[0]

    def target(self, tr, ev):
        """stratum means from train; mode 'tab' subtracts the leave-one-out document mean (document effect)"""
        mu = defaultdict(list)
        for i in tr:
            mu[self.S[i]].append(self.y[i])
        mu = {s: float(np.mean(v)) for s, v in mu.items()}
        g = float(np.mean(self.y[tr]))
        t = self.y - np.array([mu.get(s, g) for s in self.S])
        if self.mode == 'tab':
            t = t.copy()
            grp = defaultdict(list)
            for i in range(self.n):
                grp[self.doc[i]].append(i)
            out = np.zeros(self.n)
            for idx in grp.values():
                sm = t[idx].sum()
                for i in idx:
                    out[i] = t[i] - (sm - t[i]) / (len(idx) - 1)
            t = out
        return t[tr], t[ev]


def draw_model(rng):
    return dict(minf=rng.choice([2, 3, 4, 6]), keep=rng.uniform(0.3, 1.0), lam=10 ** rng.uniform(-0.5, 2),
                kinds=rng.choice(['WSG', 'WSG', 'WG', 'SG', 'W', 'S']), thr=0.0 if rng.random() < 0.5 else rng.uniform(0.03, 0.25),
                scale='mult' if rng.random() < 0.7 else 'add', ctx=rng.random() < 0.5)


_PF = {}


def pick_feats(D, tr, m, rng):
    key = (id(D), len(tr), int(tr[0]), int(tr[-1]))
    if key not in _PF:
        _PF.clear()
        cnt = D.X[tr].sum(0)
        nd = np.array([len(set(D.doc[tr][D.X[tr, j] > 0])) if cnt[j] >= 2 else 0 for j in range(D.X.shape[1])])
        _PF[key] = (cnt, nd)
    cnt, nd = _PF[key]
    cols = []
    for j in np.where((cnt >= m['minf']) & (nd >= 2))[0]:
        if D.fnames[j][0] not in m['kinds']:
            continue
        if rng.random() < m['keep']:
            cols.append(j)
    return np.array(cols, int)


def ridge(X, y, lam):
    mu = y.mean()
    if X.shape[1] == 0:
        return mu, np.zeros(0)
    xm = X.mean(0)
    Xc = X - xm
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    b = np.linalg.solve(A, Xc.T @ (y - mu))
    return mu - xm @ b, b


def design(D, idx, cols, m):
    X = D.X[np.ix_(idx, cols)].astype(float) if len(cols) else np.zeros((len(idx), 0))
    if m['ctx'] and D.mode != 'tab':
        X = np.hstack([X, D.ctx[idx, None], D.nos[idx, None]])
    return X


def _fwd(t, m):
    return t if m['scale'] == 'mult' else np.exp(t) - 1.0


def _back(p, m):
    return p if m['scale'] == 'mult' else np.log(np.clip(p + 1.0, 0.05, None))


def fit(D, tr, ttr, m, cols):
    X = design(D, tr, cols, m)
    a, b = ridge(X, _fwd(ttr, m), m['lam'])
    nc = len(cols)
    if m['thr'] > 0 and nc:
        k = np.abs(b[:nc]) >= m['thr']
        cols = cols[k]
        X = design(D, tr, cols, m)
        a, b = ridge(X, _fwd(ttr, m), m['lam'])
    return a, b, cols


def predict(D, idx, a, b, cols, m):
    X = design(D, idx, cols, m)
    return _back(a + X @ b, m), X


def id_baseline(D, tr, ttr, ev):
    s = defaultdict(list)
    for i, v in zip(tr, ttr):
        s[D.key[i]].append(v)
    mu = {k: np.sum(v) / (len(v) + 1.0) for k, v in s.items()}
    pred = np.array([mu.get(D.key[i], 0.0) for i in ev])
    novel = np.array([D.key[i] not in mu for i in ev])
    return pred, novel


class Gram:
    """precomputed sufficient statistics of one training set (features + ctx + nos columns), so each random
    model is a small ridge solve on a sub-Gram (identical to ridge() on the selected columns)."""
    def __init__(self, D, tr, ttr):
        X = np.hstack([D.X[tr].astype(float), D.ctx[tr, None], D.nos[tr, None]])
        self.n = len(tr); self.F = D.X.shape[1]
        self.G = X.T @ X; self.s = X.sum(0)
        self.t = {'mult': ttr, 'add': np.exp(ttr) - 1.0}
        self.Xt = {k: X.T @ v for k, v in self.t.items()}

    def solve(self, cols, m):
        c = np.asarray(cols, int)
        if m['ctx'] and getattr(self, 'mode', 'sys') != 'tab':
            c = np.concatenate([c, [self.F, self.F + 1]])
        y = self.t[m['scale']]; mu = y.mean()
        if len(c) == 0:
            return mu, np.zeros(0)
        xm = self.s[c] / self.n
        A = self.G[np.ix_(c, c)] - self.n * np.outer(xm, xm) + m['lam'] * np.eye(len(c))
        r = self.Xt[m['scale']][c] - self.n * xm * mu
        try:
            b = np.linalg.solve(A, r)
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(A, r, rcond=None)[0]
        return mu - xm @ b, b

    def fit(self, cols, m):
        a, b = self.solve(cols, m)
        nc = len(cols)
        if m['thr'] > 0 and nc:
            k = np.abs(b[:nc]) >= m['thr']
            cols = np.asarray(cols)[k]
            a, b = self.solve(cols, m)
        return a, b, np.asarray(cols, int)


def run_split(D, seed, M, keep_top=0.1):
    rng = random.Random(seed * 7919 + 1)
    A, B, C = D.split(seed)
    tA, tB = D.target(A, B)
    idB, novB = id_baseline(D, A, tA, B)
    mse_idB = float(np.mean((tB - idB) ** 2))
    res = []
    GA = Gram(D, A, tA); GA.mode = D.mode
    for k in range(M):
        m = draw_model(rng)
        cols = pick_feats(D, A, m, random.Random(seed * 7919 + 1 + 100003 * (k + 1)))
        a, b, c2 = GA.fit(cols, m)
        pB, _ = predict(D, B, a, b, c2, m)
        res.append((mse_idB - float(np.mean((tB - pB) ** 2)), k, m))
    res.sort(key=lambda x: -x[0])
    surv = [r for r in res[:max(1, int(keep_top * M))] if r[0] > 0]
    AB = np.concatenate([A, B])
    tAB, tC = D.target(AB, C)
    idC, novC = id_baseline(D, AB, tAB, C)
    mse_idC = float(np.mean((tC - idC) ** 2))
    out = dict(seed=seed, nA=len(A), nC=len(C), novC=int(novC.sum()), mse_idC=mse_idC,
               mse0C=float(np.mean(tC ** 2)), surv=[], abl={})
    # reference: document context only (no word or sign), so word gains are not document-scale gains
    m0 = dict(scale='mult', ctx=True, thr=0.0, lam=1.0)
    a0, b0 = ridge(design(D, AB, np.zeros(0, int), m0), tAB, 1.0)
    p0 = a0 + design(D, C, np.zeros(0, int), m0) @ b0
    out['g_ctx0'] = mse_idC - float(np.mean((tC - p0) ** 2))
    abl = defaultdict(lambda: [0.0, 0])
    GAB = Gram(D, AB, tAB); GAB.mode = D.mode
    for j, (gB, k, m) in enumerate(surv):
        cols = pick_feats(D, AB, m, random.Random(seed * 7919 + 1 + 100003 * (k + 1)))
        a, b, c2 = GAB.fit(cols, m)
        pC, XC = predict(D, C, a, b, c2, m)
        gC = mse_idC - float(np.mean((tC - pC) ** 2))
        gCn = float(np.mean(tC[novC] ** 2) - np.mean((tC[novC] - pC[novC]) ** 2)) if novC.any() else 0.0
        # effects on the log scale: mult -> b; add -> log(1 + b / (1 + a)) (approximate factor)
        if m['scale'] == 'mult':
            eff = b[:len(c2)]
        else:
            eff = np.log(np.clip(1 + b[:len(c2)] / max(1e-3, 1 + a), 0.05, None))
        coef = {D.fnames[c]: float(v) for c, v in zip(c2, eff)}
        out['surv'].append(dict(gB=float(gB), gC=gC, gCn=gCn, coef=coef, scale=m['scale']))
        if j < 3 and m['scale'] == 'mult':
            nov = np.where(novC)[0]
            for jj, c in enumerate(c2):
                rr = nov[XC[nov, jj] > 0]
                if len(rr) == 0:
                    continue
                e1 = (tC[rr] - pC[rr]) ** 2
                e0 = (tC[rr] - (pC[rr] - b[jj])) ** 2
                abl[D.fnames[c]][0] += float(np.sum(e0 - e1)); abl[D.fnames[c]][1] += int(len(rr))
    out['abl'] = {k: tuple(v) for k, v in abl.items()}
    return out


def summarise(D, R, min_split_frac=0.75, cons=0.9, eff=0.15, ablfrac=0.75, nabl=3):
    S = len(R)
    occ = Counter(); nd = defaultdict(set)
    for r, dd in zip(D.rows, D.doc):
        for f in r['f']:
            occ[f] += 1; nd[f].add(dd)
    per = defaultdict(list); pooled = defaultdict(list); ab = defaultdict(list)
    for r in R:
        cs = defaultdict(list)
        for sv in r['surv']:
            for s, v in sv['coef'].items():
                cs[s].append(v); pooled[s].append(v)
        for s, v in cs.items():
            per[s].append(float(np.mean(v)))
        for s, (d, n) in r['abl'].items():
            ab[s].append(d)
    tab = {}
    for s in occ:
        v = per.get(s, [])
        med = float(np.median(pooled[s])) if pooled.get(s) else 0.0
        sg = np.sign(med) if med else 0
        consist = float(np.mean([np.sign(x) == sg for x in v])) if v and sg else 0.0
        a = ab.get(s, [])
        apos = float(np.mean([x > 0 for x in a])) if a else 0.0
        desc = (len(v) >= min_split_frac * S and consist >= cons and abs(med) >= eff and len(a) >= nabl and apos >= ablfrac)
        ident = (not desc) and occ[s] >= 6 and len(nd[s]) >= 3 and (abs(med) < 0.1 or consist < 0.75)
        tab[s] = dict(occ=occ[s], ndoc=len(nd[s]), splits=len(v), med=med, consist=consist,
                      nabl=len(a), ablpos=apos, cls='D' if desc else 'I' if ident else '-')
    return tab


def scores(R):
    top = [r['surv'][0] for r in R if r['surv']]
    allC = [sv['gC'] for r in R for sv in r['surv']]
    return dict(best_gC=float(np.mean([s['gC'] for s in top])) if top else 0.0,
                best_gCn=float(np.mean([s['gCn'] for s in top])) if top else 0.0,
                surv_pos=float(np.mean([x > 0 for x in allC])) if allC else 0.0,
                nsurv=float(np.mean([len(r['surv']) for r in R])),
                mse_idC=float(np.mean([r['mse_idC'] for r in R])), g_ctx0=float(np.mean([r['g_ctx0'] for r in R])), mse0C=float(np.mean([r['mse0C'] for r in R])),
                add_share=float(np.mean([sv['scale'] == 'add' for r in R for sv in r['surv']])) if allC else 0.0)


def run_corpus(rows, mode, S, M, seed0=0):
    D = Data(rows, mode)
    R = [run_split(D, seed0 + s, M) for s in range(S)]
    return D, R


# ------------------------------------------------------------------ factors (pe58 port)
def factor_boot(rows, feat, B=200, seed=0, how='dmean'):
    """within-document log shift of entries carrying feat vs same-stratum siblings without it;
    document bootstrap.  Returns (point, lo, hi, ndocs)."""
    g = defaultdict(list)
    for r in rows:
        g[(r['doc'], r['s'])].append(r)
    diffs = []
    for k, rs in g.items():
        a = [math.log(r['q']) for r in rs if feat in r['f']]
        b = [math.log(r['q']) for r in rs if feat not in r['f']]
        if a and b:
            if how == 'dmean':
                diffs.append(np.mean(a) - np.mean(b))
            else:
                diffs.append(np.median(a) - np.median(b))
    if not diffs:
        return None
    diffs = np.array(diffs)
    rng = np.random.default_rng(seed)
    bs = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(B)]
    return float(diffs.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(diffs)


def lattice_points(pmax=6):
    return np.array(sorted({math.log(p / q) for p in range(1, pmax + 1) for q in range(1, pmax + 1) if p != q}))


LAT = lattice_points()


def lat_dist(x):
    x = np.atleast_1d(np.asarray(x, float))
    return np.min(np.abs(x[:, None] - LAT[None, :]), axis=1)


def lattice_test(logs, n=20000, jit=0.25, seed=0):
    logs = np.asarray(logs, float)
    if len(logs) == 0:
        return dict(obs=None, k=0, p=None)
    obs = float(lat_dist(logs).mean())
    rng = np.random.default_rng(seed)
    J = logs[None, :] + rng.uniform(-jit, jit, (n, len(logs)))
    nj = lat_dist(J.ravel()).reshape(J.shape).mean(1)
    return dict(obs=obs, k=len(logs), null=float(nj.mean()), p=float((np.sum(nj <= obs) + 1) / (n + 1)))
