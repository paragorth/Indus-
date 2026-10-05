"""pe46 THE "NAMES" ARE SERIAL NUMBERS: shared library.

Arrow in the dark: the multi-sign strings at the start of PE entries are not personal names but
administrative codes (account numbers / postcodes): each slot draws on its own small alphabet and
encodes one attribute (office, unit, household, team, serial).

A random SLOT SCHEMA (adapted from voynich/tools/v52_lib.py) cuts each string into K = 2..4 slots by
K-1 cut rules applied left to right from a pointer p:
  ('fix', k)  cut at p+k          ('end', k)  cut at max(p, L-k)
  ('set', S)  cut before the first unit at/after p in the random unit set S
  ('run', S)  cut after the maximal run of S-units starting at p
Each slot keeps its top-N values (random N; alphabet learned on the FIT tablets only) plus OTHER.

Scores of a schema on a set of tokens (strings with tablet + external variables):
  E[s,v]  shuffle-corrected normalised MI between slot s and external variable v
          (tablet-level variables: values permuted across tablets; entry-level: within tablet;
           TABID: tokens permuted across all tablets)
  FS      field purity (v52): sum over variable groups of the best slot's excess over its
          tracking of the other groups -> large only if DIFFERENT slots track DIFFERENT attributes
  OFF/SER office-and-serial signature: per slot, within-tablet same-value rate / random-pair rate
          (log ratio R_s, real values only). OFF = max_s R_s (a slot shared on the tablet: office),
          SER = -min_s R_s (a slot that never repeats on the tablet: serial). OS = min(OFF, SER)
  DEP     conditional dependence of adjacent slots given the tracked variables (low for a code)
  CODE    = FS + OS - max(0, DEP)     (search objective)
"""
import os, sys, json, math, random, re
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, entries, base, header  # noqa: E402

PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe46_ckpt'); os.makedirs(CK, exist_ok=True)
REPO = os.path.abspath(os.path.join(PEROOT, '..', '..'))
DARK = os.path.join(REPO, 'data', 'derived', 'dark', 'loop56_corpora')

_r = json.load(open(os.path.join(DATA, 'res_a_slots.json')))['rows']
FINAL = {x['sign'] for x in _r if x['z_final'] >= 3}

CORE = ['TABID', 'BATCH', 'CLS', 'ORD', 'PREV']
GROUPS = dict(TABID='TAB', BATCH='TAB', HDR='TAB', PUB='TAB', CLS='ENT', QTY='ENT', SYS='ENT',
              ORD='ORD', PREV='NBR')
TABVARS = {'BATCH', 'HDR', 'PUB'}


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


# ---------------------------------------------------------------- corpus container
def make(tokens, name, true=None):
    """tokens: list of dicts {w: tuple units, tab: id, ord: int, CLS, BATCH, ... }"""
    C = dict(name=name, n=len(tokens))
    C['w'] = [tuple(t['w']) for t in tokens]
    tabs = {}
    C['tab'] = np.array([tabs.setdefault(t['tab'], len(tabs)) for t in tokens], dtype=np.int64)
    C['tabnames'] = list(tabs)
    C['vars'] = {}
    for v in GROUPS:
        if v == 'TABID':
            C['vars'][v] = C['tab'].copy(); continue
        if v == 'PREV':
            vals = []
            last = {}
            for t in tokens:
                vals.append(last.get(t['tab'], '^')); last[t['tab']] = t['w'][0] if t['w'] else '-'
        elif v == 'ORD':
            vals = [min(t['ord'], 6) for t in tokens]
        else:
            vals = [t.get(v, '-') for t in tokens]
        m = {}
        C['vars'][v] = np.array([m.setdefault(x, len(m)) for x in vals], dtype=np.int64)
    uc = Counter(u for w in C['w'] for u in w)
    C['units'] = [u for u, _ in uc.most_common(24)]
    C['true'] = true if true is not None else ([t['true'] for t in tokens] if tokens and 'true' in tokens[0] else None)
    return C


def subset(C, mask, name=None):
    D = dict(C); idx = np.where(mask)[0]
    D['w'] = [C['w'][i] for i in idx]; D['tab'] = C['tab'][idx]
    D['vars'] = {k: v[idx] for k, v in C['vars'].items()}
    D['n'] = len(idx); D['name'] = name or C['name']
    if C.get('true') is not None:
        D['true'] = [C['true'][i] for i in idx]
    return D


def tab_split(C, seed):
    rng = np.random.default_rng(seed)
    A = rng.random(C['tab'].max() + 1) < 0.5
    return A[C['tab']]


# ---------------------------------------------------------------- PE
def pe_tokens(minlen=2, middle=True, base_signs=True):
    T = load()
    E = entries(T, require_clean=False, base_signs=base_signs)
    tinfo = {}
    pids = sorted(t['id'] for t in T)
    rank = {p: i for i, p in enumerate(pids)}
    for t in T:
        h = header(t)
        pub = re.sub(r',.*', '', t.get('designation', '') or '')
        tinfo[t['id']] = dict(HDR=(h[0] if h else '-'), PUB=pub, BATCH=rank[t['id']] // 25)
    from pe4_common import value
    out = []; ordc = Counter()
    for e in E:
        s = e['signs']
        if 'x' in s:
            ordc[e['tablet']] += 1; continue
        if middle and base(s[-1]) in FINAL and len(s) >= 2:
            cls, mid = s[-1], tuple(s[:-1])
        else:
            cls, mid = '-', tuple(s)
        o = ordc[e['tablet']]; ordc[e['tablet']] += 1
        if len(mid) < minlen:
            continue
        v = value(e['numerals'], e['system'])
        q = 'NA' if not v else int(math.log2(v))
        ti = tinfo[e['tablet']]
        out.append(dict(w=mid, tab=e['tablet'], ord=o, CLS=cls, SYS=e['system'] or '-', QTY=q,
                        BATCH=ti['BATCH'], HDR=ti['HDR'], PUB=ti['PUB']))
    return out


# ---------------------------------------------------------------- controls
def _pick_tablets(toks, n_target, rng):
    by = defaultdict(list)
    for t in toks:
        by[t['tab']].append(t)
    tabs = list(by); rng.shuffle(tabs)
    out = []
    for tb in tabs:
        if len(by[tb]) < 2:
            continue
        out += by[tb]
        if len(out) >= n_target:
            break
    return out


def _batch(toks):
    pids = sorted({t['tab'] for t in toks})
    rank = {p: i for i, p in enumerate(pids)}
    for t in toks:
        t['BATCH'] = rank[t['tab']] // 25
    return toks


def ur3_herd(n_target, seed=0):
    """POSITIVE (real structured descriptions): Ur III Drehem livestock lines, attribute tokens."""
    C = json.load(open(os.path.join(DATA, 'pe4_controls.json')))
    toks = []; ordc = Counter()
    for r in C['herd']:
        o = ordc[r['t']]; ordc[r['t']] += 1
        a = [x for x in r['attr'] if not re.match(r'^\d', x)]
        if len(a) < 2:
            continue
        toks.append(dict(w=tuple(a), tab=r['t'], ord=o, CLS=r['head']))
    return _batch(_pick_tablets(toks, n_target, random.Random(seed)))


def ur3_names(n_target, seed=0):
    """NEGATIVE: Ur III Drehem personal names (ki PN-ta / giri3 PN), tablet commodity = majority head."""
    C = json.load(open(os.path.join(DATA, 'pe4_controls.json')))
    head = defaultdict(Counter)
    for r in C['herd']:
        head[r['t']][r['head']] += 1
    toks = []; ordc = Counter()
    for r in C['names']:
        o = ordc[r['t']]; ordc[r['t']] += 1
        if len(r['attr']) < 2:
            continue
        cl = head[r['t']].most_common(1)[0][0] if head[r['t']] else '-'
        toks.append(dict(w=tuple(r['attr']), tab=r['t'], ord=o, CLS=cl))
    rng = random.Random(seed); rng.shuffle(toks)
    # names are ~1 per tablet: take tablets with >=2 first, then fill
    by = defaultdict(list)
    for t in toks:
        by[t['tab']].append(t)
    multi = [t for tb in by for t in by[tb] if len(by[tb]) >= 2]
    single = [t for tb in by for t in by[tb] if len(by[tb]) < 2]
    out = (multi + single)[:n_target]
    return _batch(out)


LB_SERIES = {('PY', 'Jn'), ('KN', 'As'), ('PY', 'Cn'), ('PY', 'An'), ('PY', 'En'), ('PY', 'Eo'),
             ('PY', 'Ep'), ('PY', 'Es'), ('KN', 'Ai'), ('PY', 'Ae'), ('KN', 'Ak'), ('PY', 'Aa'),
             ('PY', 'Ad'), ('KN', 'Am'), ('PY', 'Ea'), ('PY', 'Eb'), ('KN', 'B'), ('KN', 'Dv')}


def linb(first_only=True):
    """Linear B entry heads from DAMOS personnel series. first_only: first word (a name, NEGATIVE);
    else all words before the logogram (name + place / office: REAL multi-field designation)."""
    path = os.path.join(REPO, 'other-scripts', 'linear-a', 'data', 'damos_items.jsonl')
    toks = []
    for line in open(path):
        x = json.loads(line)
        if 'heading' not in x:
            continue
        m = re.match(r'(\w+) ([A-Z][a-z]*)', x['heading'])
        if not m or (m.group(1), m.group(2)) not in LB_SERIES:
            continue
        tab = x['heading']; o = 0
        for ln in x['content'].split('\n'):
            ln = re.sub(r'^\s*\.\S+', '', ln)
            ln = re.sub(r'[\[\]⌞⌟̣̀-ͯ\?]', '', ln)
            mm = re.search(r'\b[A-Z][A-Z*+±]{1,}\b', ln)
            if not mm:
                continue
            pre = ln[:mm.start()]
            words = [w.strip() for w in re.split(r'[,/]', pre) if w.strip()]
            words = [w for w in words if re.fullmatch(r'[a-z0-9*\-]+', w.split()[0] if w.split() else '')]
            words = [w.split()[0] for w in words]
            if not words:
                continue
            if first_only:
                w = tuple(s for s in words[0].split('-') if s)
            else:
                w = tuple(s for wd in words for s in wd.split('-') if s)
            o += 1
            if len(w) >= 2:
                toks.append(dict(w=w, tab=tab, ord=o, CLS=mm.group(0), BATCH=m.group(1) + m.group(2)))
    return toks


HTS_SECTIONS = [(1, 5), (6, 14), (15, 15), (16, 24), (25, 27), (28, 38), (39, 40), (41, 43), (44, 46),
                (47, 49), (50, 63), (64, 67), (68, 70), (71, 71), (72, 83), (84, 85), (86, 89), (90, 92),
                (93, 93), (94, 96), (97, 99)]


def hts(n_target, seed=0):
    """POSITIVE (real hierarchical codes in opaque units): HTS tariff numbers (2-digit groups), placed on
    pseudo-invoices: each invoice favours one section (70%). CLS = section. True slots = groups."""
    rng = random.Random(seed)
    codes = []
    for line in open(os.path.join(DARK, 'hts.jsonl')):
        s = json.loads(line)['seq']
        if 3 <= len(s) <= 5:
            codes.append(tuple(s))
    sec = {}
    for c in codes:
        ch = int(c[0]); sec[c] = next(i for i, (a, b) in enumerate(HTS_SECTIONS) if a <= ch <= b)
    bysec = defaultdict(list)
    for c in codes:
        bysec[sec[c]].append(c)
    sym = ['h%02d' % i for i in range(100)]; rng.shuffle(sym)
    op = lambda c: tuple(sym[int(g)] for g in c)
    toks = []; t = 0
    while len(toks) < n_target:
        s = rng.choice(list(bysec)); k = rng.randint(2, 8)
        for o in range(k):
            c = rng.choice(bysec[s]) if rng.random() < 0.7 else rng.choice(codes)
            toks.append(dict(w=op(c), tab='inv%d' % t, ord=o, CLS=sec[c], true=list(c)))
        t += 1
    for x in toks:
        x['BATCH'] = int(x['tab'][3:]) // 25
    return toks


def planted(pe_toks, seed=0, p=0.7):
    """PLANTED code system on the real PE tablet frame, values spelled with PE signs:
    slot1 OFFICE (one per tablet, p), slot2 UNIT (tracks the class sign, p), slot3 SERIAL
    (sequential within tablet from a random start, modulo 12). Values 1-2 signs."""
    rng = random.Random(seed)
    signs = sorted({u for t in pe_toks for u in t['w']})
    rng.shuffle(signs)
    pool = iter(signs)
    mk = lambda n: [tuple(next(pool) for _ in range(rng.choice([1, 1, 2]))) for _ in range(n)]
    OFF, UNIT, SER = mk(10), mk(6), mk(12)
    tabs = sorted({t['tab'] for t in pe_toks})
    toff = {tb: rng.randrange(10) for tb in tabs}; tstart = {tb: rng.randrange(12) for tb in tabs}
    cls = sorted({t['CLS'] for t in pe_toks}); cmap = {c: rng.randrange(6) for c in cls}
    out = []
    for t in pe_toks:
        a = toff[t['tab']] if rng.random() < p else rng.randrange(10)
        b = cmap[t['CLS']] if rng.random() < p else rng.randrange(6)
        c = (tstart[t['tab']] + t['ord']) % 12
        w = OFF[a] + UNIT[b] + SER[c]
        x = dict(t); x['w'] = w; x['true'] = [a, b, c]
        out.append(x)
    return out


# ---------------------------------------------------------------- nulls on token lists
def null_signshuf(toks, seed):
    rng = random.Random(seed); out = []
    for t in toks:
        w = list(t['w']); rng.shuffle(w); x = dict(t); x['w'] = tuple(w); out.append(x)
    return out


def null_tabshuf(toks, seed):
    rng = random.Random(seed); ws = [t['w'] for t in toks]; rng.shuffle(ws)
    return [dict(t, w=w) for t, w in zip(toks, ws)]


def null_markov(toks, seed):
    rng = random.Random(seed)
    cnt = defaultdict(Counter)
    for t in toks:
        s = ('^',) + tuple(t['w']) + ('$',)
        for a, b in zip(s, s[1:]):
            cnt[a][b] += 1
    tab = {a: (list(c), np.cumsum(list(c.values()))) for a, c in cnt.items()}
    out = []
    for t in toks:
        while True:
            w = []; h = '^'
            while len(w) < 12:
                ks, cs = tab[h]; u = ks[int(np.searchsorted(cs, rng.random() * cs[-1], side='right'))]
                if u == '$':
                    break
                w.append(u); h = u
            if len(w) >= 2:
                break
        out.append(dict(t, w=tuple(w)))
    return out


# ---------------------------------------------------------------- schemas
def random_schema(rng, units):
    K = rng.randint(2, 4); rules = []
    for _ in range(K - 1):
        t = rng.choice(['fix', 'fix', 'end', 'set', 'run'])
        if t in ('fix', 'end'):
            rules.append((t, rng.randint(1, 3)))
        else:
            S = tuple(sorted(rng.sample(units[:20], rng.randint(1, 6))))
            rules.append((t, S))
    return dict(rules=rules, N=rng.choice([6, 12, 24, 48]))


def cut(w, rules):
    L = len(w); p = 0; out = []
    for t, a in rules:
        if t == 'fix':
            q = min(L, p + a)
        elif t == 'end':
            q = max(p, L - a)
        elif t == 'set':
            q = L
            for i in range(p, L):
                if w[i] in a:
                    q = i; break
        else:
            q = p
            while q < L and w[q] in a:
                q += 1
        out.append(w[p:q]); p = q
    out.append(w[p:])
    return out


def alphabet(C, sch):
    """Slot value maps learned on corpus C (the fit half)."""
    K = len(sch['rules']) + 1
    cnt = [Counter() for _ in range(K)]
    for w in C['w']:
        for s, v in enumerate(cut(w, sch['rules'])):
            cnt[s][v] += 1
    return [{v: i for i, (v, _) in enumerate(c.most_common(sch['N']))} for c in cnt]


def codes(C, sch, alph):
    """K x n arrays: value index; len(map) = OTHER; -1 marks an empty slot."""
    K = len(alph); out = np.empty((K, C['n']), dtype=np.int64)
    for i, w in enumerate(C['w']):
        for s, v in enumerate(cut(w, sch['rules'])):
            out[s, i] = -1 if not v else alph[s].get(v, len(alph[s]))
    return out


# ---------------------------------------------------------------- information
def H(x):
    c = np.bincount(x); c = c[c > 0]; p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def MI(x, y):
    ny = int(y.max()) + 1
    j = np.bincount(x * ny + y); j = j[j > 0]; p = j / j.sum()
    return H(x) + H(y) - float(-(p * np.log2(p)).sum())


def CMI(x, y, z):
    nx = int(x.max()) + 1; ny = int(y.max()) + 1
    xz = z * nx + x; yz = z * ny + y
    _, xyz = np.unique((z * nx + x) * ny + y, return_inverse=True)
    return H(xz) + H(yz) - H(xyz) - H(z)


def null_vars(C, rng, R=3):
    tab = C['tab']; nt = tab.max() + 1
    order = np.argsort(tab, kind='stable'); b = np.searchsorted(tab[order], np.arange(nt + 1))
    first = order[b[:-1]]
    outs = []
    for _ in range(R):
        perm = rng.permutation(nt); d = {}
        for k, v in C['vars'].items():
            if k == 'TABID':
                d[k] = v[rng.permutation(len(v))]
            elif k in TABVARS:
                d[k] = v[first][perm][tab]
            else:
                nv = v.copy()
                for q in range(nt):
                    ix = order[b[q]:b[q + 1]]
                    if len(ix) > 1:
                        nv[ix] = v[rng.permutation(ix)]
                d[k] = nv
        outs.append(d)
    return outs


def share_ratio(c, tab):
    """Within-tablet same-value pairs vs expectation from the all-pair rate, as a Poisson z
    (real values only: empty and OTHER excluded). None if the expectation is < 3 pairs."""
    ok = c >= 0
    if ok.sum() < 20:
        return None
    c = c[ok]; t = tab[ok]
    vc = np.bincount(c); allp = (vc * (vc - 1)).sum() / (len(c) * (len(c) - 1))
    _, tj = np.unique(t * (c.max() + 1) + c, return_inverse=True)
    jc = np.bincount(tj); tc = np.bincount(t)
    E = (tc * (tc - 1)).sum() * allp / 2.0
    if E < 3:
        return None
    W = (jc * (jc - 1)).sum() / 2.0
    return float((W - E) / math.sqrt(E))


def score(C, sch, alph, nulls, varlist=CORE, dep=True, rng=None):
    X = codes(C, sch, alph); K = X.shape[0]
    cod = [np.where(x < 0, x.max() + 1, x) if (x < 0).any() else x for x in X]  # empty as a value for MI
    Hs = [H(c) for c in cod]
    E = np.zeros((K, len(varlist)))
    for j, v in enumerate(varlist):
        y = C['vars'][v]; hy = H(y)
        if hy < 1e-9:
            continue
        for s in range(K):
            if Hs[s] < 0.3:
                continue
            real = MI(cod[s], y); nm = np.mean([MI(cod[s], nd[v]) for nd in nulls])
            E[s, j] = (real - nm) / max(1e-9, min(Hs[s], hy))
    groups = sorted(set(GROUPS[v] for v in varlist))
    G = np.zeros((K, len(groups)))
    for gi, g in enumerate(groups):
        idx = [j for j, v in enumerate(varlist) if GROUPS[v] == g]
        G[:, gi] = np.clip(E[:, idx].max(1), 0, None)
    FS = 0.0; tgt = {}
    for gi, g in enumerate(groups):
        pure = G[:, gi] - (G.sum(1) - G[:, gi]); s = int(np.argmax(pure))
        if pure[s] > 0:
            FS += pure[s]; tgt[g] = s
    R = []
    for s in range(K):
        x = X[s].copy(); x[x >= len(alph[s])] = -1          # OTHER excluded
        R.append(share_ratio(x, C['tab']) if Hs[s] >= 0.3 else None)
    Rv = [r for r in R if r is not None]
    OFF = max(Rv) if Rv else 0.0; SER = -min(Rv) if len(Rv) >= 2 else 0.0
    OS = max(0.0, min(OFF, SER)) / 5.0 if len(Rv) >= 2 else 0.0
    res = dict(FS=FS, OFF=OFF, SER=SER, OS=OS, R=[None if r is None else round(r, 3) for r in R],
               E=E.round(4).tolist(), groups=groups, varlist=list(varlist), H=[round(h, 3) for h in Hs],
               tgt=tgt, K=K, cover=[float((X[s] >= 0).mean()) for s in range(K)])
    DEP = 0.0
    if dep:
        z = np.zeros(C['n'], dtype=np.int64)
        for gi, g in enumerate(groups):
            idx = [j for j, v in enumerate(varlist) if GROUPS[v] == g and v != 'TABID']
            if not idx:
                continue
            v = varlist[idx[int(np.argmax(E[:, idx].max(0)))]]
            z = z * (int(C['vars'][v].max()) + 1) + C['vars'][v]
        _, z = np.unique(z, return_inverse=True)
        rng = rng or np.random.default_rng(0)
        order = np.argsort(z, kind='stable'); b = np.searchsorted(z[order], np.arange(z.max() + 2))
        dsum = 0.0; m = 0
        for s in range(K - 1):
            a, c = cod[s], cod[s + 1]
            if Hs[s] < 0.3 or Hs[s + 1] < 0.3:
                continue
            real = CMI(a, c, z); cp = c.copy()
            for q in range(len(b) - 1):
                ix = order[b[q]:b[q + 1]]
                if len(ix) > 1:
                    cp[ix] = c[rng.permutation(ix)]
            dsum += (real - CMI(a, cp, z)) / min(Hs[s], Hs[s + 1]); m += 1
        DEP = dsum / max(1, m)
    res['DEP'] = DEP
    res['CODE'] = FS + OS - max(0.0, DEP)
    return res


def search(C, n_schemas, seed, split_seed=0, top=20, varlist=CORE, log=None):
    """Random schemas scored on fit tablets (A), top-k re-scored on held-out tablets (B)."""
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    A = tab_split(C, split_seed); CA = subset(C, A); CB = subset(C, ~A)
    nA = null_vars(CA, nrng, 3); nB = null_vars(CB, nrng, 3)
    res = []
    for i in range(n_schemas):
        sch = random_schema(rng, C['units'])
        al = alphabet(CA, sch)
        r = score(CA, sch, al, nA, varlist, dep=False)
        res.append((r['FS'] + r['OS'], sch))
    res.sort(key=lambda x: -x[0])
    out = []
    for fit, sch in res[:top]:
        al = alphabet(CA, sch)
        ra = score(CA, sch, al, nA, varlist, dep=True)
        rb = score(CB, sch, al, nB, varlist, dep=True)
        out.append(dict(sch=sch, fit=ra, held=rb))
    return out


def summarize(out):
    hb = [o['held']['CODE'] for o in out]; fa = [o['fit']['CODE'] for o in out]
    return dict(held_med=float(np.median(hb)), held_max=float(np.max(hb)), fit_med=float(np.median(fa)),
                held_FS=float(np.median([o['held']['FS'] for o in out])),
                held_OS=float(np.median([o['held']['OS'] for o in out])),
                held_DEP=float(np.median([o['held']['DEP'] for o in out])))


def recovery(C, sch):
    """For corpora with true slots: mean over true slots of the best NMI with a schema slot."""
    if C.get('true') is None:
        return None
    al = alphabet(C, sch); X = codes(C, sch, al)
    T = np.array(C['true'], dtype=object)
    out = []
    for j in range(max(len(t) for t in C['true'])):
        tv = np.array([hash(str(t[j]) if j < len(t) else '-') for t in C['true']])
        _, tv = np.unique(tv, return_inverse=True)
        best = 0
        for s in range(X.shape[0]):
            x = X[s] - X[s].min()
            m = MI(x, tv) / max(1e-9, min(H(x), H(tv))) if H(x) > 0 and H(tv) > 0 else 0
            best = max(best, m)
        out.append(round(best, 3))
    return out


def sch_str(sch):
    parts = []
    for t, a in sch['rules']:
        parts.append(f'{t}{a}' if t in ('fix', 'end') else f'{t}{{{",".join(a)}}}')
    return ' | '.join(parts) + f' N{sch["N"]}'
