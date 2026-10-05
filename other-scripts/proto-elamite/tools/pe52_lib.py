"""pe52 EVERY SIGN HAS A WEIGHT: shared library.

Arrow in the dark: if PE entry strings are descriptions ("fattened female sheep of herd X") rather than
names, each sign is a feature that shifts the log quantity written with the string. Thousands of random
log-linear models (random sign subsets, ridge strength, hard-threshold sparsity, pair interactions,
first/last position features, string length) are fitted on half the tablets, screened on a quarter
and re-scored on a reserved quarter. A sign is a DESCRIPTOR when its effect is stable across tablet
splits and it improves held-out prediction for strings never seen in training (string identity cannot
help there). Frequent signs without such an effect are IDENTIFIERS.

Corpora are lists of dicts {tab, w: tuple of tokens, q: positive quantity, sys, lab?}.
Target modes: 'sys'  log q minus the training mean of its number system
              'tab'  log q minus the leave-one-out mean of the other entries of the same tablet and
                     system (tablet effect; entries without a sibling are dropped)
"""
import os, sys, json, math, random, re
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe52_ckpt'); os.makedirs(CK, exist_ok=True)


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')


# ------------------------------------------------------------------ corpora
def pe_corpus(minlen=1):
    from pe1_lib import load, entries, quantity
    T = load()
    out = []
    for e in entries(T):
        q = quantity(e['numerals'])
        if not q or q[1] <= 0:
            continue
        s = [x for x in e['signs'] if x != 'x']
        if 'x' in e['signs'] or len(s) < minlen:
            continue
        out.append(dict(tab=e['tablet'], w=tuple(s), q=float(q[1]), sys=q[0]))
    return out


def _pick_tabs(rows, n_target, rng, min_lines=2):
    by = defaultdict(list)
    for r in rows:
        by[r['tab']].append(r)
    tabs = [t for t in by if len(by[t]) >= min_lines]
    rng.shuffle(tabs)
    out = []
    for t in tabs:
        out += by[t]
        if len(out) >= n_target:
            break
    return out


def herd_corpus(n_target=6500, seed=0):
    """Ur III Drehem livestock lines: head word + attribute tokens, count n."""
    C = json.load(open(os.path.join(DATA, 'pe4_controls.json')))
    rows = []
    for r in C['herd']:
        if not r['n'] or r['n'] <= 0:
            continue
        w = [r['head']] + [a for a in r['attr']]
        rows.append(dict(tab=r['t'], w=tuple(w), q=float(r['n']), sys='K'))
    return _pick_tabs(rows, n_target, random.Random(seed))


def herd2_corpus(n_target=6500, seed=0):
    """Ur III Drehem livestock lines rebuilt from CDLI with labels (name / deity tokens vs other)."""
    return pers_corpus(n_target, seed, fn='herd.json')


def pers_corpus(n_target=6500, seed=0, fn='pers.json'):
    P = json.load(open(os.path.join(CK, fn)))
    rows = [dict(tab=r['tab'], w=tuple(r['w']), q=float(r['q']), sys=r['sys'], lab=tuple(r['lab'])) for r in P]
    return _pick_tabs(rows, n_target, random.Random(seed))


def planted(pe, seed=0, k=14, id_sd=0.35):
    """PE skeleton (real strings, tablets, systems). k planted descriptor signs with effects +-U(0.4,1.0);
    every distinct string also gets an idiosyncratic identity effect N(0, id_sd) (a 'name' quantity that no
    sign carries); tablet x system means and residual noise copied from the real data."""
    rng = np.random.default_rng(seed)
    cnt = Counter(s for r in pe for s in set(r['w']))
    tabs_of = defaultdict(set)
    for r in pe:
        for s in set(r['w']):
            tabs_of[s].add(r['tab'])
    pool = sorted(s for s in cnt if 15 <= cnt[s] <= 250 and len(tabs_of[s]) >= 6)
    chosen = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    eff = {s: float(rng.choice([-1, 1]) * rng.uniform(0.4, 1.0)) for s in chosen}
    grp = defaultdict(list)
    for r in pe:
        grp[(r['tab'], r['sys'])].append(math.log(r['q']))
    gm = {g: float(np.mean(v)) for g, v in grp.items()}
    res = [math.log(r['q']) - gm[(r['tab'], r['sys'])] for r in pe if len(grp[(r['tab'], r['sys'])]) > 1]
    sd = float(np.std(res)) * 0.8
    ide = {}
    out = []
    for r in pe:
        w = r['w']
        if w not in ide:
            ide[w] = float(rng.normal(0, id_sd))
        y = gm[(r['tab'], r['sys'])] + sum(eff.get(s, 0.0) for s in set(w)) + ide[w] + rng.normal(0, sd)
        out.append(dict(tab=r['tab'], w=w, q=float(math.exp(y)), sys=r['sys']))
    return out, eff


# ------------------------------------------------------------------ nulls
def null_qshuf_sys(C, seed):
    """quantities shuffled among entries within number system (whole corpus)"""
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, r in enumerate(C):
        by[r['sys']].append(i)
    out = [dict(r) for r in C]
    for idx in by.values():
        qs = [C[i]['q'] for i in idx]; rng.shuffle(qs)
        for i, q in zip(idx, qs):
            out[i]['q'] = q
    return out


def null_qshuf_tab(C, seed):
    """quantities shuffled within tablet x system (keeps tablet scale)"""
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, r in enumerate(C):
        by[(r['tab'], r['sys'])].append(i)
    out = [dict(r) for r in C]
    for idx in by.values():
        qs = [C[i]['q'] for i in idx]; rng.shuffle(qs)
        for i, q in zip(idx, qs):
            out[i]['q'] = q
    return out


def null_sign_within(C, seed):
    """signs shuffled inside each string (bag unchanged; only order features can notice)"""
    rng = random.Random(seed)
    out = []
    for r in C:
        w = list(r['w']); rng.shuffle(w)
        o = dict(r); o['w'] = tuple(w); out.append(o)
    return out


def null_sign_tab(C, seed):
    """sign tokens re-dealt among the strings of the same tablet (string lengths and tablet pool kept)"""
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, r in enumerate(C):
        by[r['tab']].append(i)
    out = [dict(r) for r in C]
    for idx in by.values():
        pool = [s for i in idx for s in C[i]['w']]; rng.shuffle(pool)
        p = 0
        for i in idx:
            L = len(C[i]['w']); out[i]['w'] = tuple(pool[p:p + L]); p += L
    return out


# ------------------------------------------------------------------ engine
class Data:
    def __init__(self, C, mode):
        y = np.array([math.log(r['q']) for r in C])
        sysv = [r['sys'] for r in C]
        keep = np.ones(len(C), bool)
        if mode == 'tab':
            grp = defaultdict(list)
            for i, r in enumerate(C):
                grp[(r['tab'], r['sys'])].append(i)
            t = np.zeros(len(C))
            for idx in grp.values():
                if len(idx) < 2:
                    keep[idx] = False; continue
                s = y[idx].sum()
                for i in idx:
                    t[i] = y[i] - (s - y[i]) / (len(idx) - 1)
            self.target_fixed = True
        else:
            t = y.copy(); self.target_fixed = False
        # tablet context (for mode 'ctx'): leave-one-out mean log q of the other entries of the same tablet and
        # system, minus the corpus mean of that system; 0 plus a no-sibling flag when there is none
        grp = defaultdict(list)
        for i, r in enumerate(C):
            grp[(r['tab'], r['sys'])].append(i)
        smu = {s_: float(np.mean([y[i] for i in range(len(C)) if sysv[i] == s_])) for s_ in set(sysv)}
        ctx = np.zeros(len(C)); nos = np.zeros(len(C))
        for idx_ in grp.values():
            if len(idx_) < 2:
                nos[idx_] = 1; continue
            sm = y[idx_].sum()
            for i in idx_:
                ctx[i] = (sm - y[i]) / (len(idx_) - 1) - smu[sysv[i]]
        self.mode = mode
        idx = np.where(keep)[0]
        self.ctx = ctx[idx]; self.nos = nos[idx]
        self.rows = [C[i] for i in idx]
        self.t = t[idx]
        self.sys = np.array([sysv[i] for i in idx])
        self.tab = np.array([C[i]['tab'] for i in idx])
        self.bag = [frozenset(C[i]['w']) for i in idx]
        self.first = [C[i]['w'][0] for i in idx]
        self.last = [C[i]['w'][-1] for i in idx]
        self.L = np.array([len(C[i]['w']) for i in idx], float)
        self.key = [tuple(sorted(b)) for b in self.bag]
        self.n = len(idx)
        self.tabs = sorted(set(self.tab))

    def split(self, seed):
        rng = random.Random(seed)
        tabs = list(self.tabs); rng.shuffle(tabs)
        n = len(tabs)
        part = {t: (0 if i < n // 2 else 1 if i < 3 * n // 4 else 2) for i, t in enumerate(tabs)}
        p = np.array([part[t] for t in self.tab])
        return np.where(p == 0)[0], np.where(p == 1)[0], np.where(p == 2)[0]


def center(D, tr, ev):
    """returns targets for train and eval rows (system means from train rows when mode 'sys')"""
    if D.target_fixed:
        return D.t[tr], D.t[ev]
    mu = {s: D.t[tr][D.sys[tr] == s].mean() for s in set(D.sys[tr])}
    g = float(D.t[tr].mean())
    f = lambda idx: D.t[idx] - np.array([mu.get(s, g) for s in D.sys[idx]])
    return f(tr), f(ev)


def draw_model(rng):
    return dict(minf=rng.choice([2, 3, 5, 8]), keep=rng.uniform(0.3, 1.0), lam=10 ** rng.uniform(-1, 2),
                pos=rng.random() < 0.3, npairs=0 if rng.random() < 0.5 else rng.randint(5, 60),
                lenf=rng.random() < 0.5, thr=0.0 if rng.random() < 0.5 else rng.uniform(0.02, 0.25))


def features(D, tr, m, rng):
    dfc = Counter(); dtab = defaultdict(set)
    for i in tr:
        for s in D.bag[i]:
            dfc[s] += 1; dtab[s].add(D.tab[i])
    signs = sorted(s for s in dfc if dfc[s] >= m['minf'] and len(dtab[s]) >= 2 and rng.random() < m['keep'])
    feats = [('S', s) for s in signs]
    if m['pos']:
        fc = Counter(D.first[i] for i in tr); lc = Counter(D.last[i] for i in tr)
        feats += [('F', s) for s in sorted(fc) if fc[s] >= m['minf'] and rng.random() < m['keep']]
        feats += [('E', s) for s in sorted(lc) if lc[s] >= m['minf'] and rng.random() < m['keep']]
    if m['npairs']:
        ss = set(signs); pc = Counter()
        for i in tr:
            b = sorted(D.bag[i] & ss)
            for a in range(len(b)):
                for c in range(a + 1, len(b)):
                    pc[(b[a], b[c])] += 1
        cand = sorted(p for p in pc if pc[p] >= m['minf'])
        rng.shuffle(cand)
        feats += [('P', p) for p in cand[:m['npairs']]]
    if m['lenf']:
        feats.append(('L', None))
    if D.mode == 'ctx':
        feats += [('X', None), ('N', None)]
    return feats


def design(D, idx, feats):
    col = {f: j for j, f in enumerate(feats)}
    X = np.zeros((len(idx), len(feats)))
    for r, i in enumerate(idx):
        b = D.bag[i]
        for s in b:
            j = col.get(('S', s))
            if j is not None:
                X[r, j] = 1
        j = col.get(('F', D.first[i]))
        if j is not None:
            X[r, j] = 1
        j = col.get(('E', D.last[i]))
        if j is not None:
            X[r, j] = 1
        j = col.get(('L', None))
        if j is not None:
            X[r, j] = math.log(D.L[i])
        j = col.get(('X', None))
        if j is not None:
            X[r, j] = D.ctx[i]
        j = col.get(('N', None))
        if j is not None:
            X[r, j] = D.nos[i]
    for f, j in col.items():
        if f[0] == 'P':
            a, c = f[1]
            for r, i in enumerate(idx):
                if a in D.bag[i] and c in D.bag[i]:
                    X[r, j] = 1
    return X


def ridge(X, y, lam):
    mu = y.mean()
    if X.shape[1] == 0:
        return mu, np.zeros(0)
    xm = X.mean(0)
    Xc = X - xm
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    b = np.linalg.solve(A, Xc.T @ (y - mu))
    return mu - xm @ b, b


def fit(D, tr, ttr, m, feats):
    X = design(D, tr, feats)
    a, b = ridge(X, ttr, m['lam'])
    if m['thr'] > 0 and len(b):
        k = (np.abs(b) >= m['thr']) | np.array([f[0] in 'XN' for f in feats])
        feats = [f for f, kk in zip(feats, k) if kk]
        X = X[:, k]
        a, b = ridge(X, ttr, m['lam'])
    return a, b, feats


def id_baseline(D, tr, ttr, ev):
    s = defaultdict(list)
    for i, v in zip(tr, ttr):
        s[D.key[i]].append(v)
    mu = {k: np.sum(v) / (len(v) + 1.0) for k, v in s.items()}   # shrunk to 0 (one pseudo-entry)
    pred = np.array([mu.get(D.key[i], 0.0) for i in ev])
    novel = np.array([D.key[i] not in mu for i in ev])
    return pred, novel


def run_split(D, seed, M, keep_top=0.1):
    """M random models on one A/B/C split. Returns survivors' coefficient table and C-scores."""
    rng = random.Random(seed * 7919 + 1)
    A, B, Cc = D.split(seed)
    tA, tB = center(D, A, B)
    idB, novB = id_baseline(D, A, tA, B)
    mse_idB = np.mean((tB - idB) ** 2)
    res = []
    for k in range(M):
        m = draw_model(rng)
        feats = features(D, A, m, random.Random(seed * 7919 + 1 + 100003 * (k + 1)))
        a, b, f2 = fit(D, A, tA, m, feats)
        pB = a + design(D, B, f2) @ b if len(f2) else np.full(len(B), a)
        gain = mse_idB - np.mean((tB - pB) ** 2)
        gnov = (np.mean(tB[novB] ** 2) - np.mean((tB[novB] - pB[novB]) ** 2)) if novB.any() else 0.0
        res.append((gain, gnov, k, m))
    res.sort(key=lambda x: -x[0])
    surv = [r for r in res[:max(1, int(keep_top * M))] if r[0] > 0]
    AB = np.concatenate([A, B])
    tAB, tC = center(D, AB, Cc)
    idC, novC = id_baseline(D, AB, tAB, Cc)
    mse_idC = np.mean((tC - idC) ** 2)
    out = dict(seed=seed, nA=len(A), nB=len(B), nC=len(Cc), novC=int(novC.sum()), mse_idC=float(mse_idC),
               mse0C=float(np.mean(tC ** 2)), gainsB=[float(r[0]) for r in res], surv=[])
    best_abl = defaultdict(lambda: [0.0, 0])
    for j, (gB, gn, k, m) in enumerate(surv):
        r2 = random.Random(seed * 7919 + 1 + 100003 * (k + 1))
        feats = features(D, AB, m, r2)
        a, b, f2 = fit(D, AB, tAB, m, feats)
        XC = design(D, Cc, f2)
        pC = a + XC @ b if len(f2) else np.full(len(Cc), a)
        gC = mse_idC - np.mean((tC - pC) ** 2)
        gCn = (np.mean(tC[novC] ** 2) - np.mean((tC[novC] - pC[novC]) ** 2)) if novC.any() else 0.0
        coef = {f[1]: float(v) for f, v in zip(f2, b) if f[0] == 'S'}
        out['surv'].append(dict(gB=float(gB), gC=float(gC), gCn=float(gCn), coef=coef))
        if j < 3:
            # ablation (top 3 survivors pooled) on novel C strings: squared-error increase per entry when the sign's effect is removed
            abl = best_abl
            nov = np.where(novC)[0]
            for jj, f in enumerate(f2):
                if f[0] != 'S':
                    continue
                rows_ = nov[XC[nov, jj] > 0]
                if len(rows_) == 0:
                    continue
                e1 = (tC[rows_] - pC[rows_]) ** 2
                e0 = (tC[rows_] - (pC[rows_] - b[jj])) ** 2
                abl[f[1]][0] += float(np.sum(e0 - e1)); abl[f[1]][1] += int(len(rows_))
    out['abl'] = {k: tuple(v) for k, v in best_abl.items()}
    # identity-only gain on C novel strings vs the 'sign' gain: reference
    out['id_vs_mean_C'] = float(np.mean(tC ** 2) - mse_idC)
    return out


def run_corpus(C, mode, S, M, seed0=0):
    D = Data(C, mode)
    return D, [run_split(D, seed0 + s, M) for s in range(S)]


def summarise(D, R, min_split_frac=0.75, cons=0.9, eff=0.15, ablfrac=0.75):
    """Per-sign table + descriptor / identifier classification (frozen rules)."""
    S = len(R)
    occ = Counter(); ntab = defaultdict(set)
    for b, t in zip(D.bag, D.tab):
        for s in b:
            occ[s] += 1; ntab[s].add(t)
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
        desc = (len(v) >= min_split_frac * S and consist >= cons and abs(med) >= eff
                and len(a) >= 3 and apos >= ablfrac)
        ident = (not desc) and occ[s] >= 10 and len(ntab[s]) >= 3 and (abs(med) < 0.1 or consist < 0.75)
        tab[s] = dict(occ=occ[s], ntab=len(ntab[s]), splits=len(v), med=med, consist=consist,
                      nabl=len(a), ablpos=apos, abl=float(np.mean(a)) if a else 0.0,
                      cls='D' if desc else 'I' if ident else '-')
    return tab


def scores(R):
    gC = [sv['gC'] for r in R for sv in r['surv'][:1]]
    gCn = [sv['gCn'] for r in R for sv in r['surv'][:1]]
    allC = [sv['gC'] for r in R for sv in r['surv']]
    nsurv = [len(r['surv']) for r in R]
    rel = [sv['gCn'] / max(1e-9, r['mse0C']) for r in R for sv in r['surv'][:1]]
    return dict(best_gC=float(np.mean(gC)), best_gC_pos=float(np.mean([x > 0 for x in gC])),
                best_gCn=float(np.mean(gCn)), rel_gCn=float(np.mean(rel)), surv_gC_pos=float(np.mean([x > 0 for x in allC])) if allC else 0.0,
                nsurv=float(np.mean(nsurv)), mse_idC=float(np.mean([r['mse_idC'] for r in R])),
                mse0C=float(np.mean([r['mse0C'] for r in R])))


def explained(D, tab, multi_only=True):
    """share of entries / distinct strings whose every sign is a descriptor"""
    Dset = {s for s, v in tab.items() if v['cls'] == 'D'}
    ent = [b for b in D.bag if (len(b) >= 2 or not multi_only)]
    keys = set(tuple(sorted(b)) for b in ent)
    full_e = np.mean([b <= Dset for b in ent]) if ent else 0
    full_k = np.mean([set(k) <= Dset for k in keys]) if keys else 0
    part_k = np.mean([len(set(k) & Dset) / len(k) for k in keys]) if keys else 0
    return dict(n_entries=len(ent), n_strings=len(keys), full_entries=float(full_e),
                full_strings=float(full_k), mean_desc_share=float(part_k), n_desc=len(Dset))
