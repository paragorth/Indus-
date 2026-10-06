#!/usr/bin/env python3
"""la64 MINOAN NUMBERS MUST OBEY PHYSIOLOGY (port of proto-elamite pe55 to Linear A).

Every written commodity quantity q (integer + fraction signs) times an unknown unit size u_c
(litres per written unit of commodity c) must be a physically possible amount: a person's ration
for a day / month / year, a household's yearly output, or the content of a storage vessel.  Where a
person count n stands next to a commodity quantity, q*u_c/n must be a per-head ration.  Thousands of
random hypotheses (u_c for every commodity key, a value in (0,1) for every fraction sign, treated as
nuisance) are scored by how many entries land inside physical windows at once (robust mixture: each
entry is physical with weight w or unit-free junk), fitted on half the documents, scored frozen on
the other half.

Physical windows (litres; median, log-sd).  Sources (outside every corpus used here):
  grain ration  1.2 l/person/day (0.5-2.5): FAO/WHO/UNU 2004 'Human energy requirements' (2,000-3,200
                kcal/d adults; 3,520 kcal/kg cereal, 0.65 kg/l) -- as pe55.
  wine ration   0.5 l/day (0.15-1.5): Cato, De agri cultura 57 (slaves' wine ~10 quadrantalia/yr ~
                260 l/yr, i.e. 0.7 l/d; less for some), classical military issues 0.3-0.5 l/d.
  oil ration    0.05 l/day (0.015-0.15): Cato, De agri cultura 58 (1 sextarius = 0.55 l per slave per
                month = 0.018 l/d); Foxhall 2007 'Olive cultivation in ancient Greece' (OUP) 20-30
                l/person/yr modern-traditional Mediterranean diets (0.05-0.08 l/d).
  olive ration  0.15 l/day (0.03-0.7): Cato 58 (windfall olives as relish, then salted olives, a
                modius ~8.7 l per slave per few weeks).
  beer (Ur III control only) 2 l/day (1-4): fluid + energy; ancient beer rations of 1-3 l/day.
  generic food  0.5 l/day, wide (log-sd 1.2) for commodities whose identity is not assumed.
  x1 day, x30 month, x360 year.
  household yearly output: grain 2,500 l (1-5 ha x 0.5-1.5 t/ha; Halstead 1981/2014 'Two oxen ahead',
                rainfed Mediterranean barley/wheat); wine 1,000 l (0.2-1 ha vineyard x 2-5 t grapes/ha);
                oil 150 l (30-60 trees x 2-5 kg oil/tree, Foxhall 2007); olives 1,500 l; generic 1,000 l.
                All log-sd 0.9.
  storage vessel: pithos 250 l (log-sd 0.7; Christakis 2005 'Cretan Bronze Age pithoi', INSTAP
                Prehistory Monographs 18, Neopalatial pithoi ~50-1,000 l; Knossos West Magazines ~80,000 l
                in ~400 pithoi; Younger: ZA Zb 3 pithos <= ~1,000 l); jar 20 l (log-sd 0.4;
                Neopalatial oval-mouthed amphorae and transport jars ~12-30 l).
No Linear B value of any Linear A sign is an input.  Commodity CLASSES of GRA, VIN, OLE, OLIV follow
the logograms' pictures (assumption, tested by a class-permutation null); every other key (CYP, NI,
*304, la57 words, ...) is 'generic'.  Linear B units (dry 96 l, liquid 28.8 l: Ventris-Chadwick) and
Ur III sila (~1 l) are used only as the truth of the controls.
"""
import json, os, re, sys, math, hashlib
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la64_ckpt')
os.makedirs(CK, exist_ok=True)
PEDATA = os.path.join(HERE, '..', '..', 'proto-elamite', 'data')
sys.path.insert(0, HERE)


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ physics
DAY = {'grain': (1.2, 0.40), 'wine': (0.5, 0.60), 'oil': (0.05, 0.70), 'olive': (0.15, 0.80),
       'beer': (2.0, 0.40), 'generic': (0.5, 1.20)}
DELIV = {'grain': 2500., 'wine': 1000., 'oil': 150., 'olive': 1500., 'beer': 3000., 'generic': 1000.}
CLASSES = list(DAY)
ROLES = ['rd', 'rm', 'ry', 'deliv', 'pithos', 'jar']


def windows(cls, ruler=None):
    """-> (mu[6], sd[6]) for entry roles of a commodity class.  ruler: optional dict of
    multiplicative distortions {(cls, role): factor} for nonsense rulers."""
    m, s = DAY[cls]
    mu = [m, m * 30, m * 360, DELIV[cls], 250., 20.]
    sd = [s, s, s, 0.9, 0.7, 0.4]
    if ruler is not None:
        mu = [x * ruler.get((cls, r), 1.0) for x, r in zip(mu, ROLES)]
    return np.log(np.array(mu)), np.array(sd)


def head_windows(cls, ruler=None):
    """per-head windows for count/commodity pairs: person d/m/y (+ stock fodder d/m/y for grain)."""
    m, s = DAY[cls]
    mu = [m, m * 30, m * 360]
    sd = [s, s, s]
    names = ['pd', 'pm', 'py']
    if cls == 'grain':
        mu += [0.8, 24., 288., 4.5, 135., 1620.]
        sd += [0.6] * 6
        names += ['sd', 'sm', 'sy', 'ld', 'lm', 'ly']
    if ruler is not None:
        mu = [x * ruler.get((cls, 'h' + n), 1.0) for x, n in zip(mu, names)]
    return np.log(np.array(mu)), np.array(sd), names


LOGU = (math.log(0.01), math.log(1000.0))
BG_SD = 2.0
W_GRID = (0.3, 0.6, 0.9)


# ------------------------------------------------------------------ corpora
LA_CLASS = {'GRA': 'grain', 'VIN': 'wine', 'OLE': 'oil', 'OLIV': 'olive'}
LA_COMW = {'NI', 'DI-DE-RU', 'DA-SI-*118', '*28B-NU-MA-RE', 'U-*325-ZA', 'TE-TU', '*304', '*308', '*306'}
LA_TOT = {'KU-RO', 'PO-TO-KU-RO', 'KI-RO'}
LA_FR_SPLIT = {'JE': ['J', 'E'], 'DD': ['D', 'D']}


def la_base(v):
    b = v.split('+')[0].strip('[]').lstrip('*') if v else ''
    if b in ('OLIV',):
        return 'OLIV'
    return b


def load_la():
    """Linear A administrative documents -> docs of entries.
    entry: dict(c=commodity key, n=int, fr=[sign...], w=word or None, line, tot=bool)
    person counts: dict(n, line, key='VIR', tot)"""
    fn = os.path.join(CK, 'la_docs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    from la60_common import load_la as l60
    keep = {d['id']: d for d in l60() if not d['lib']}
    C = json.load(open(os.path.join(DATA, 'corpus.json')))
    out = []
    for d in C:
        if d['id'] not in keep:
            continue
        line, carry, lastw, last_tok = 0, None, None, None
        E, P = [], []
        line_com = None
        for t in d['tokens']:
            if t['t'] == 'nl':
                line += 1
                line_com, lastw = None, None
                continue
            if t['t'] == 'logo':
                b = la_base(t['v'])
                if b.startswith('VIR'):
                    carry = 'VIR'; line_com = 'VIR'
                elif b in LA_CLASS or b in ('CYP', 'AROM'):
                    carry = b; line_com = b
                last_tok = 'L'
                continue
            if t['t'] == 'word':
                w = '-'.join(t['s'])
                if w in LA_COMW:
                    carry = w; line_com = w
                else:
                    lastw = w
                continue
            if t['t'] == 'num':
                fr = []
                for f in t['frac']:
                    fr += LA_FR_SPLIT.get(f, [f])
                n = int(t['v'])
                if n == 0 and not fr:
                    continue
                tot = lastw in LA_TOT
                if carry == 'VIR':
                    if n >= 1:
                        P.append(dict(n=n, line=line, key='VIR', tot=tot))
                elif carry is not None:
                    E.append(dict(c=carry, n=n, fr=fr, w=None if tot else lastw, line=line, tot=tot))
                lastw = None
        if E:
            out.append(dict(id=d['id'], site=keep[d['id']]['site'], E=E, P=P))
    json.dump(out, open(fn, 'w'))
    return out


def la_class(c):
    return LA_CLASS.get(c, 'generic')


# Linear B: DAMOS transliterations, all sites; dry and liquid subunits kept as hidden 'fraction signs'
LB_CLASS = {'GRA': 'grain', 'HORD': 'grain', 'FAR': 'grain', 'VIN': 'wine', 'OLE': 'oil', 'OLIV': 'olive'}
LB_DRY = {'GRA', 'HORD', 'FAR', 'OLIV', 'NI', 'CYP', 'AROM', 'KAPO', 'CROC'}
LB_LIQ = {'VIN', 'OLE'}
LB_PERS = {'VIR', 'MUL', 'ko-wa', 'ko-wo'}
LB_STOCK = {'OVIS', 'CAP', 'SUS', 'BOS', 'EQU'}
LB_TRUE_U = {'dry': 96.0, 'liq': 28.8}
LB_TRUE_FR = {'dT': 0.1, 'dV': 1 / 60, 'dZ': 1 / 240, 'lS': 1 / 3, 'lV': 1 / 18, 'lZ': 1 / 72}


def _strip(t):
    import unicodedata
    return ''.join(ch for ch in unicodedata.normalize('NFD', t) if unicodedata.category(ch) != 'Mn')


def load_lb():
    fn = os.path.join(CK, 'lb_docs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    out = []
    for raw in open(os.path.join(DATA, 'damos_items.jsonl')):
        d = json.loads(raw)
        h = d.get('heading') or ''
        m = re.match(r'^([A-Z]{2,3})\s', h)
        if not m:
            continue
        E, P = [], []
        for li, ln in enumerate((d.get('content') or '').split('\n')):
            toks = _strip(ln).split()
            cur, lastw, meas, ent = None, None, None, None
            for t in toks:
                s = re.sub(r'[\[\]⟦⟧?!,⌞⌟\'"]', '', t)
                if not s or s.startswith('.') or s in ('/', 'vac', 'vac.'):
                    continue
                b = re.sub(r'[mf]$', '', s.split('+')[0])
                if b in LB_DRY or b in LB_LIQ or b in LB_PERS or b in LB_STOCK:
                    cur, ent, meas = b, None, None
                    continue
                if s in ('T', 'V', 'Z', 'S'):
                    meas = s
                    continue
                if s in ('M', 'N', 'P', 'Q', 'L'):
                    cur, ent, meas = None, None, None
                    continue
                if re.fullmatch(r'\d+', s):
                    v = int(s)
                    if cur is None or v == 0:
                        continue
                    if cur in LB_PERS or cur in LB_STOCK:
                        P.append(dict(n=v, line=li, key=cur, tot=lastw == 'to-so'))
                        continue
                    kind = 'l' if cur in LB_LIQ else 'd'
                    if meas is None:
                        ent = dict(c=cur, n=v, fr=[], w=None if lastw == 'to-so' else lastw, line=li,
                                   tot=lastw == 'to-so', kind=kind)
                        E.append(ent)
                    else:
                        if ent is None:
                            ent = dict(c=cur, n=0, fr=[], w=lastw, line=li, tot=lastw == 'to-so', kind=kind)
                            E.append(ent)
                        ent['fr'] += [kind + meas] * v
                    meas = None
                    continue
                if re.fullmatch(r'[a-z0-9*]+(-[a-z0-9*]+)*', s):
                    lastw = s
                    if s in LB_PERS:
                        cur, ent = s, None
        if E:
            out.append(dict(id=re.sub(r'\s*\(\S*\)\s*$', '', h).strip(), site=m.group(1), E=E, P=P))
    json.dump(out, open(fn, 'w'))
    return out


def lb_class(c):
    return LB_CLASS.get(c, 'generic')


def lb_true_u(c):
    return LB_TRUE_U['liq' if c in LB_LIQ else 'dry']


# Ur III (pe27 extraction; sila hidden; truth ~1 l)
UR_COM = [('kasz', 'beer'), ('i3', 'oil'), ('zu2-lum', 'generic'), ('sze', 'grain'), ('zi3', 'grain'),
          ('dabin', 'grain'), ('ziz2', 'grain'), ('esza', 'grain'), ('ninda', 'grain'), ('gig', 'grain')]
UR_PERS = {'gurusz', 'geme2', 'erin2', 'dumu', 'lu2', 'szu-gi4', 'dumu-munus', 'munus', 'nita2', 'ug3-IL2', 'kinkin2'}
UR_STOCK = {'udu', 'u8', 'masz2', 'ud5', 'sila4', 'udu-nita2', 'gukkal', 'masz', 'kir11', 'gu4', 'ab2', 'amar', 'ansze', 'dusu2'}
UR_CLASS = dict(UR_COM)


def load_ur3():
    fn = os.path.join(CK, 'ur3_docs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    d = json.load(open(os.path.join(PEDATA, 'pe27_ckpt', 'ur3_tabs.json')))
    out = []
    for t in d:
        E, P = [], []
        for q in t['Q']:
            v = float(eval(str(q['v']))) if '/' in str(q['v']) else float(q['v'])
            if v <= 0:
                continue
            if q['sys'] == 'CNT':
                if q['fin'] in UR_PERS or q['fin'] in UR_STOCK:
                    P.append(dict(n=v, line=q['line'], key=q['fin'], tot=False))
                continue
            toks = re.findall(r"[a-z0-9'{}-]+", q['raw'].replace('(', ' ('))
            words = [w for w in q['raw'].split() if not re.match(r'^[\d/]+\(', w)]
            c = 'gen'
            for k, _ in UR_COM:
                if any(w == k or w.startswith(k + '-') for w in words):
                    c = k
                    break
            tot = q['fin'] in ('szunigin', 'sze-bi', 'i3-bi', 'ninda-bi', 'kasz-bi') or 'szunigin' in q['raw']
            n = int(math.floor(v + 1e-9))
            r = round(v - n, 4)
            fr = ['f%g' % r] if r > 0 else []
            rest = [w for w in words if w not in ('sila3', 'gur', c) and not w.startswith(c)]
            E.append(dict(c=c, n=n, fr=fr, w=rest[-1] if rest else None, line=q['line'], tot=tot))
        if E:
            out.append(dict(id=t['id'], site=t['site'], E=E, P=P))
    json.dump(out, open(fn, 'w'))
    return out


def ur_class(c):
    return UR_CLASS.get(c, 'generic')


# ------------------------------------------------------------------ arrays
def build(docs, clsf, min_n=5, keys=None, frkeys=None, pkeys=None):
    """Flatten to arrays.  Entries (not totals) -> E2; pairs (person count, commodity entry on the
    same / next / previous line, else document person total) -> E1."""
    ents = [(di, e) for di, d in enumerate(docs) for e in d['E'] if not e['tot']]
    if keys is None:
        cnt = Counter(e['c'] for _, e in ents)
        keys = sorted(k for k, n in cnt.items() if n >= min_n)
    ki = {k: i for i, k in enumerate(keys)}
    ents = [(di, e) for di, e in ents if e['c'] in ki]
    if frkeys is None:
        fc = Counter(f for _, e in ents for f in e['fr'])
        frkeys = sorted(f for f, n in fc.items() if n >= 2)
    fi = {f: i for i, f in enumerate(frkeys)}
    nE = len(ents)
    F = np.zeros((nE, len(frkeys) + 1))       # last column = rare fraction signs (fixed 0.25)
    n = np.zeros(nE)
    k = np.zeros(nE, int)
    dd = np.zeros(nE, int)
    words = []
    for j, (di, e) in enumerate(ents):
        n[j] = e['n']
        k[j] = ki[e['c']]
        dd[j] = di
        words.append(e['w'])
        for f in e['fr']:
            F[j, fi.get(f, len(frkeys))] += 1
    # pairs
    pr = []
    for di, d in enumerate(docs):
        if not d['P']:
            continue
        cnts = [p for p in d['P'] if not p['tot']]
        totp = [p for p in d['P'] if p['tot']]
        ptot = totp[0]['n'] if totp else sum(p['n'] for p in cnts)
        byl = defaultdict(list)
        for p in cnts:
            byl[p['line']].append(p)
        for j in np.where(dd == di)[0]:
            e = ents[j][1]
            hit = None
            for dl in (0, -1, 1):
                if byl.get(e['line'] + dl):
                    hit = byl[e['line'] + dl][0]
                    break
            if hit is not None:
                pr.append((j, hit['n'], hit['key']))
            elif ptot > 0:
                pr.append((j, ptot, 'DOC'))
    if pkeys is None:
        pc = Counter(p[2] for p in pr)
        pkeys = sorted(x for x, m in pc.items() if m >= 3)
    pki = {x: i for i, x in enumerate(pkeys)}
    pr = [p for p in pr if p[2] in pki]
    return dict(keys=keys, cls=[clsf(x) for x in keys], frkeys=frkeys, n=n, F=F, k=k, doc=dd, words=words,
                pe=np.array([p[0] for p in pr], int), pn=np.array([p[1] for p in pr], float),
                pk=np.array([pki[p[2]] for p in pr], int), pkeys=pkeys, docs=docs)


def subset(A, docset):
    m = np.isin(A['doc'], list(docset))
    idx = np.where(m)[0]
    remap = -np.ones(len(m), int)
    remap[idx] = np.arange(len(idx))
    pm = m[A['pe']] if len(A['pe']) else np.zeros(0, bool)
    return dict(A, n=A['n'][m], F=A['F'][m], k=A['k'][m], doc=A['doc'][m],
                words=[w for w, x in zip(A['words'], m) if x],
                pe=remap[A['pe'][pm]], pn=A['pn'][pm], pk=A['pk'][pm])


# ------------------------------------------------------------------ scoring
def logq(A, theta):
    """theta[H, nfr] -> log quantity [H, E] (rare fraction column fixed at 0.25)."""
    H = theta.shape[0]
    th = np.concatenate([theta, np.full((H, 1), 0.25)], 1)
    q = A['n'][None, :] + th @ A['F'].T
    return np.log(np.maximum(q, 1e-3))


def _mix(x, mu, sd, lb):
    """x[H,E], windows mu[R], sd[R], background lb[E] -> gain matrix per w [W,H,E]"""
    lf = -0.5 * ((x[:, :, None] - mu[None, None, :]) / sd[None, None, :]) ** 2 - np.log(sd)[None, None, :]
    lf = np.logaddexp.reduce(lf, axis=2) - math.log(len(mu))
    g = np.clip(lf - lb[None, :], -50, 50)
    return np.stack([np.log(w * np.exp(g) + 1 - w) for w in W_GRID])


def score(A, logu, theta, ruler=None, med=None, use=('E2', 'E1'), plab=None, return_parts=False):
    """logu[H, K], theta[H, nfr] -> total gain[H] (sum over commodity keys of best-w robust gain).
    E2: entries vs entry windows of their class.  E1: per-head ratio vs head windows, count key labels
    chosen best per (h, key) among {person/stock windows, none} unless plab given."""
    H = logu.shape[0]
    LQ = logq(A, theta)
    if med is None:
        med = np.array([np.median(np.log(np.maximum(A['n'][A['k'] == i], 0.5))) if (A['k'] == i).any() else 0
                        for i in range(len(A['keys']))])
    tot = np.zeros(H)
    parts = {}
    if 'E2' in use:
        for i, c in enumerate(A['keys']):
            m = A['k'] == i
            if not m.any():
                continue
            mu, sd = windows(A['cls'][i], ruler)
            x = LQ[:, m] + logu[:, i:i + 1]
            lb = -0.5 * ((LQ[0, m] - med[i]) / BG_SD) ** 2 - math.log(BG_SD)
            G = _mix(x, mu, sd, lb).sum(2)            # [W, H]
            s = G.max(0)
            parts[c] = s
            tot += s
    if 'E1' in use and len(A['pe']):
        lr = LQ[:, A['pe']] - np.log(A['pn'])[None, :]
        kk = A['k'][A['pe']]
        for j, pkey in enumerate(A['pkeys']):
            mj = A['pk'] == j
            GF = {'p': np.zeros(H), 's': np.zeros(H), 'l': np.zeros(H)}
            for i in np.unique(kk[mj]):
                mm = mj & (kk == i)
                mu, sd, names = head_windows(A['cls'][i], ruler)
                x = lr[:, mm] + logu[:, i:i + 1]
                lbm = np.median(lr[0, mm])
                lb = -0.5 * ((lr[0, mm] - lbm) / BG_SD) ** 2 - math.log(BG_SD)
                for f in GF:
                    sel = [q for q, nm in enumerate(names) if nm[0] == f]
                    if sel:
                        GF[f] += _mix(x, mu[sel], sd[sel], lb).sum(2).max(0)
            if plab is not None:
                best = GF[plab[pkey]] if plab.get(pkey) in GF else np.zeros(H)
            else:
                best = np.maximum(0, np.maximum(GF['p'], np.maximum(GF['s'], GF['l'])))
            parts['P:' + pkey] = best
            parts['PF:' + pkey] = GF
            tot += best
    return (tot, parts) if return_parts else tot


def rand_hyp(rng, H, K, nfr):
    logu = rng.uniform(*LOGU, (H, K))
    theta = np.exp(rng.uniform(math.log(1 / 100), math.log(0.95), (H, nfr)))
    return logu, theta


def search(A, rng, H=6000, ruler=None, chunk=500, **kw):
    """Massive random guessing over (u_c, fraction values).  Because the score is a sum over keys,
    each key's u is also searched independently: for every random theta, the per-key best of the
    H random u values is kept (coordinate-wise best; reported separately)."""
    K, nf = len(A['keys']), len(A['frkeys'])
    logu, theta = rand_hyp(rng, H, K, nf)
    S = np.concatenate([score(A, logu[i:i + chunk], theta[i:i + chunk], ruler, **kw) for i in range(0, H, chunk)])
    return dict(logu=logu, theta=theta, S=S)


def fit_best(A, rng, H=6000, ruler=None, top=0.02, **kw):
    """Random search, then per-key refinement on the top theta (grid of u per key holding others)."""
    R = search(A, rng, H, ruler, **kw)
    b = int(np.argmax(R['S']))
    lu, th = R['logu'][b].copy(), R['theta'][b].copy()
    grid = np.linspace(*LOGU, 241)
    for _ in range(2):
        for i in range(len(A['keys'])):
            L = np.tile(lu, (len(grid), 1))
            L[:, i] = grid
            s = score(A, L, np.tile(th, (len(grid), 1)), ruler, **kw)
            lu[i] = grid[int(np.argmax(s))]
    s0 = float(score(A, lu[None], th[None], ruler, **kw)[0])
    ord_ = np.argsort(-R['S'])[:max(5, int(top * H))]
    return dict(logu=lu, theta=th, S=s0, top_logu=R['logu'][ord_], top_theta=R['theta'][ord_], R=R)


def key_profile(A, i, theta, ruler=None, grid=None, use=('E2', 'E1')):
    """Profile score of key i over a u grid (others irrelevant: score is additive over keys)."""
    if grid is None:
        grid = np.linspace(*LOGU, 241)
    K = len(A['keys'])
    L = np.zeros((len(grid), K))
    L[:, i] = grid
    th = np.tile(theta, (len(grid), 1))
    _, parts = score(A, L, th, ruler, use=use, return_parts=True)
    return grid, parts.get(A['keys'][i], np.zeros(len(grid))).copy()


def ratio_parts(A, logu, theta, ruler=None):
    """per key E2 profile + E1 contributions of that key (pairs keyed by commodity)."""
    return score(A, logu[None], theta[None], ruler, return_parts=True)[1]


def nonsense_ruler(rng, spread=1.5):
    r = {}
    for c in CLASSES:
        for role in ROLES:
            r[(c, role)] = 10 ** rng.uniform(-spread, spread)
        for nm in ['pd', 'pm', 'py', 'sd', 'sm', 'sy', 'ld', 'lm', 'ly']:
            r[(c, 'h' + nm)] = 10 ** rng.uniform(-spread, spread)
    return r


def shuffle_numbers(docs, rng, within=None):
    """Null: quantities (integer + fractions) re-dealt across documents (all entries, or within
    commodity key if within='c').  Person counts are re-dealt among documents too."""
    docs = json.loads(json.dumps(docs))
    if within == 'c':
        groups = defaultdict(list)
        for d in docs:
            for e in d['E']:
                groups[e['c']].append(e)
        for g in groups.values():
            vals = [(e['n'], e['fr']) for e in g]
            rng.shuffle(vals)
            for e, (n, fr) in zip(g, vals):
                e['n'], e['fr'] = n, fr
    else:
        allE = [e for d in docs for e in d['E']]
        vals = [(e['n'], e['fr']) for e in allE]
        rng.shuffle(vals)
        for e, (n, fr) in zip(allE, vals):
            e['n'], e['fr'] = n, fr
    allP = [p for d in docs for p in d['P']]
    pv = [p['n'] for p in allP]
    rng.shuffle(pv)
    for p, v in zip(allP, pv):
        p['n'] = v
    return docs


def split_docs(n, rng):
    p = rng.permutation(n)
    return set(p[:n // 2].tolist()), set(p[n // 2:].tolist())


def thin(docs, rng, n_ent):
    """random documents until about n_ent non-total entries"""
    p = rng.permutation(len(docs))
    out, c = [], 0
    for i in p:
        out.append(docs[i])
        c += sum(not e['tot'] for e in docs[i]['E'])
        if c >= n_ent:
            break
    return out


def heldout(A, rng, H=4000, ruler=None, n_rand=200):
    """fit on half A, freeze, score on half B; compare to random units (null b)."""
    a, b = split_docs(len(A['docs']), rng)
    TA, TB = subset(A, a), subset(A, b)
    f = fit_best(TA, rng, H, ruler)
    sB = float(score(TB, f['logu'][None], f['theta'][None], ruler)[0])
    lu, th = rand_hyp(rng, n_rand, len(A['keys']), len(A['frkeys']))
    th[:] = f['theta']
    sR = score(TB, lu, th, ruler)
    return dict(fit=f, sB=sB, p_units=float((sR >= sB).mean()), sR_med=float(np.median(sR)))
