#!/usr/bin/env python3
"""PE-61 EVERY BUREAUCRACY ON EARTH VOTES ON PROTO-ELAMITE: shared code.

Ports the Linear A la57 pipeline (linear-a/tools/la57_common.py, la57_c2.py) to Proto-Elamite.
Proto-Elamite is reduced to the same value-free form as the seven known administrations:
  doc = {'id','sys','toks': [('T', opaque sign id) | ('N', value, measured_flag) | ('L',)]}
Each PE sign (base form) is one opaque token; the entry numeral follows its signs (PE layout); header lines have
no numeral.  measured_flag follows the la57 proto-cuneiform rule (numeral system not plain SDB, or fractional).
No sign values or anyone's sign readings are used; the pe59 frozen classes are used ONLY to score agreement.
Features are computed on Linear-A-sized draws (5,219 tokens, the size every known system was trained at):
PE tablets are partitioned into disjoint draws, 3 random partitions; a sign's role score = mean over its
occurrences in all draws.
Shuffles (PE only): S1 signs shuffled over all sign slots of the corpus; S2 signs shuffled WITHIN each entry
(line), numerals untouched; S3 line order shuffled within each tablet.
"""
import os, sys, json, random, pickle, hashlib, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PED = os.path.join(HERE, '..', 'data')
CK = os.path.join(PED, 'pe61_ckpt')
os.makedirs(CK, exist_ok=True)
LAT = os.path.join(HERE, '..', '..', 'linear-a', 'tools')
sys.path.insert(0, HERE)
sys.path.insert(1, LAT)
import la57_common as L          # noqa: E402

LA_SIZE = 5219
NSHUF = 12


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def wlog(path, row):
    with open(path, 'a') as f:
        f.write(row.rstrip() + '\n')


# ------------------------------------------------------------------ Proto-Elamite in the value-free form
def pe_docs():
    import pe56_common as P
    out = []
    for t in P.pe_docs():
        toks = []
        for l in t['lines']:
            line = [('T', s) for s in l['s']]
            if l['sys'] is not None:
                v = l['v']
                meas = l['sys'] not in ('SDB',) or (v is not None and abs(v - round(v)) > 1e-9)
                line.append(('N', float(v) if v is not None else None, bool(meas)))
            if line:
                toks.extend(line); toks.append(('L',))
        while toks and toks[-1][0] == 'L':
            toks.pop()
        if any(x[0] == 'T' for x in toks):
            out.append({'id': t['id'], 'sys': 'PE', 'site': t['site'], 'toks': toks})
    return out


def _lines(d):
    ls, cur = [], []
    for x in d['toks']:
        if x[0] == 'L':
            if cur: ls.append(cur)
            cur = []
        else:
            cur.append(x)
    if cur: ls.append(cur)
    return ls


def _join(ls):
    t = []
    for l in ls:
        t.extend(l); t.append(('L',))
    while t and t[-1][0] == 'L':
        t.pop()
    return t


def shuf_global(docs, rng):
    return L.shuffle_types(docs, rng)


def shuf_entry(docs, rng):
    out = []
    for d in docs:
        ls = []
        for l in _lines(d):
            sg = [x for x in l if x[0] == 'T']; rng.shuffle(sg); it = iter(sg)
            ls.append([next(it) if x[0] == 'T' else x for x in l])
        out.append(dict(d, toks=_join(ls)))
    return out


def shuf_lines(docs, rng):
    out = []
    for d in docs:
        ls = _lines(d); rng.shuffle(ls)
        out.append(dict(d, toks=_join(ls)))
    return out


def partitions(docs, npart, tag):
    """npart random partitions of the docs into disjoint draws of ~LA_SIZE tokens."""
    out = []
    for p in range(npart):
        rng = random.Random(seed('pe61-part-%s-%d' % (tag, p)))
        idx = list(range(len(docs))); rng.shuffle(idx)
        tot = sum(L.ntok([d]) for d in docs)
        k = max(1, round(tot / LA_SIZE))
        draws, cur, c = [], [], 0
        target = tot / k
        for i in idx:
            cur.append(docs[i]); c += L.ntok([docs[i]])
            if c >= target and len(draws) < k - 1:
                draws.append(cur); cur, c = [], 0
        if cur: draws.append(cur)
        out.append(draws)
    return out


def featurise(docs, npart, tag, truth=None):
    """list of feature blocks (one per draw) for the docs."""
    blocks = []
    for draws in partitions(docs, npart, tag):
        for d in draws:
            X, types, labs, di = L.features(d, truth(d) if truth else None)
            blocks.append(dict(X=X, types=types, labs=labs))
    return blocks


# ------------------------------------------------------------------ proto-cuneiform re-laid out in PE order
def pc_pe_layout():
    """la57 proto-cuneiform docs with each line's numeral moved AFTER its signs (the PE entry order)."""
    out = []
    for d in L.pc_docs():
        ls = []
        for l in _lines(d):
            ls.append([x for x in l if x[0] == 'T'] + [x for x in l if x[0] == 'N'])
        out.append(dict(d, toks=_join(ls)))
    return out


# ------------------------------------------------------------------ pe59 frozen classes (scoring only)
def pe59_classes():
    R = json.load(open(os.path.join(PED, 'pe59_reading_frozen.json')))['reading']['roles']
    return {'MEASURED': set(R['MEASURED']['signs']), 'COUNTED': set(R['COUNTED']['signs']),
            'PERSON': set(R['PERSON']['signs']), 'M376': {'M376'}, 'M288': {'M288'},
            'PREFIX': set(R['PREFIX']['signs'])}


# ------------------------------------------------------------------ models (la57_c2 specs)
def make(sp):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    f, fam, par = sp
    if fam == 'LR':
        return LogisticRegression(C=par['C'], max_iter=300)
    if fam == 'TREE':
        return DecisionTreeClassifier(max_depth=par['depth'], min_samples_leaf=par['leaf'])
    return GaussianNB()


def spec(rng):
    k = int(rng.integers(2, 16))
    f = np.sort(rng.choice(L.NF, k, replace=False))
    fam = ['LR', 'TREE', 'NB'][int(rng.integers(0, 3))]
    par = dict(C=float(10 ** rng.uniform(-3, 1))) if fam == 'LR' else (
        dict(depth=int(rng.integers(1, 5)), leaf=int(rng.integers(20, 200))) if fam == 'TREE' else {})
    return (f, fam, par)


def fit(sp, Xs, ys):
    """Each training system weighted 1, classes balanced within each system (as la57_c2)."""
    X = np.vstack(Xs); y = np.concatenate(ys)
    if y.all() or not y.any():
        return None
    w = []
    for yy in ys:
        n1 = max(1, yy.sum()); n0 = max(1, len(yy) - yy.sum())
        w.append(np.where(yy, 0.5 / n1, 0.5 / n0))
    w = np.concatenate(w); w = w / w.mean()
    m = make(sp)
    m.fit(X[:, sp[0]], y, sample_weight=w)
    return m


def score(m, sp, X):
    return m.predict_proba(X[:, sp[0]])[:, 1]


def known():
    """la57 feature draws of the known systems (labelled rows only) and the eligible systems per role."""
    F, sysd = L.load()
    sysd['KH']['kh'] = True
    Xk, rows = {}, {}
    for k in L.KNOWN:
        r = np.array([i for i, l in enumerate(sysd[k]['labs']) if l is not None])
        rows[k] = r
        Xk[k] = sysd[k]['X'][r]
    elig = {}
    for role in L.ROLES:
        elig[role] = [k for k in L.KNOWN
                      if sum(1 for l in sysd[k]['labs'] if l == role) >= 30
                      and sum(1 for l in sysd[k]['labs'] if l is not None and l != role) >= 30]
    return F, sysd, Xk, elig


def sign_table(blocks, scorefns, minocc=3):
    """mean rank-normalised score per sign (rank within each draw, averaged over models, then over occurrences)."""
    from scipy.stats import rankdata
    acc, cnt = collections.defaultdict(float), collections.Counter()
    for b in blocks:
        if not len(b['X']):
            continue
        R = np.mean([rankdata(f(b['X'])) / len(b['X']) for f in scorefns], 0)
        for t, r in zip(b['types'], R):
            acc[t] += r; cnt[t] += 1
    return {t: acc[t] / cnt[t] for t in acc if cnt[t] >= minocc}, cnt


def auc_sets(tab, pos, neg=None):
    ts = list(tab)
    s = np.array([tab[t] for t in ts])
    y = np.array([t in pos for t in ts])
    if neg is not None:
        keep = np.array([t in pos or t in neg for t in ts])
        s, y = s[keep], y[keep]
    return L.auc(s, y)
