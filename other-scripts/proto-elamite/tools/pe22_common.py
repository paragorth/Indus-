"""PE-22 'count the unseen signs and predict the next tablets': shared loaders.

Signs are species, tablets are sampling units, sites are habitats, publication dates order
the sampling. No sign reading is used anywhere; only sign identities (CDLI M-numbers or
proto-cuneiform sign names), findspot (CDLI provenience) and CDLI publication_date.

Estimators and prediction models are reused from the Linear A census (la15):
other-scripts/linear-a/tools/la15_common.py and la15_c3.py (Chao1, ACE, Chao2, jack2,
coverage, community bootstrap EXCH, Zipf-Mandelbrot ABC, habitat novelty HAB).
"""
import json, os, re, sys, collections, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe22_ckpt')
LOOPS = os.path.join(HERE, '..', 'loops')
os.makedirs(CK, exist_ok=True)
LA_TOOLS = os.path.join(HERE, '..', '..', 'linear-a', 'tools')
sys.path.insert(0, LA_TOOLS)
import la15_common as L  # noqa: E402

chao1, chao2, ace, jack2, coverage = L.chao1, L.chao2, L.ace, L.jack2, L.coverage

YR = re.compile(r'(1[89]\d\d|20\d\d)')


def year_of(s):
    m = YR.findall(s or '')
    return int(m[0]) if m else None


def site_short(p):
    p = p or ''
    for k, v in (('Susa', 'Susa'), ('Malyan', 'Malyan'), ('Yahya', 'Yahya'), ('Sialk', 'Sialk'),
                 ('Sofalin', 'Sofalin'), ('Ozbaki', 'Ozbaki'), ('Sokhta', 'ShahriSokhta'),
                 ('Ghazir', 'Ghazir'), ('Chogha', 'ChoghaMish'), ('Larsa', 'Larsa'), ('Uruk', 'Uruk'),
                 ('Jemdet', 'JemdetNasr'), ('Uqair', 'Uqair')):
        if k in p:
            return v
    return 'unprov' if (p == '' or 'uncertain' in p) else p.split(' (')[0]


def base(s):
    """drop variant suffixes (~a, ~2 ...) also inside compounds"""
    return re.sub(r'~[A-Za-z0-9]+', '', s)


def ok_sign(s):
    return s not in ('x', 'X', 'n', '') and not re.search(r'(^|\+)X(\||\+|$)|\+x\+', s) and '...' not in s


def tablet_units(d):
    """per tablet: base-sign tokens, graph tokens, entry strings (>=2 base signs, no x)"""
    bs, gs, ents, bg = [], [], [], []
    for l in d['lines']:
        sg = l['signs']
        good = [s for s in sg if ok_sign(s)]
        bs += [base(s) for s in good]
        gs += good
        if len(sg) >= 2 and len(good) == len(sg) and not l.get('lacuna'):
            ents.append(' '.join(base(s) for s in sg))
        for a, b in zip(sg, sg[1:]):
            if ok_sign(a) and ok_sign(b):
                bg.append(base(a) + ' ' + base(b))
    return bs, gs, ents, bg


def load_pe():
    C = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
    cat = json.load(open(os.path.join(CK, 'catalog.json')))
    docs = []
    for d in C:
        c = cat.get(d['id'], {})
        y = year_of(c.get('pubdate', ''))
        pub = c.get('pub', d.get('designation', ''))
        site = site_short(c.get('prov', d.get('provenience', '')))
        if site == 'unprov':
            dn = d.get('designation', '') + ' ' + pub
            site = 'Susa' if 'RA 050' in dn else 'Sofalin' if 'TSF' in dn else ('Ozbaki' if 'Ozbaki' in dn else
                   ('Yahya' if 'Yahya' in dn else ('Sialk' if 'Sialk' in dn else
                    ('ShahriSokhta' if 'Sokhta' in dn else ('Ghazir' if 'Ghazir' in dn else
                     ('ChoghaMish' if 'Chogha' in dn else 'unprov'))))))
        bs, gs, ents, bg = tablet_units(d)
        docs.append(dict(id=d['id'], site=site, year=y, pub=pub, signs=bs, graphs=gs, words=ents, bigr=bg,
                         h=c.get('h', ''), w=c.get('w', '')))
    return docs


def pe_split(docs):
    """PRE-REGISTERED (4 Oct 2026, before any held-out counts were computed):
    train  = published up to 1999, plus undated old publications (MDP 06 joins, Shahr-i Sokhta,
             Ghazir, Chogha Mish);
    recent = publication year >= 2000 (TCL 32, 2019) or unpublished CDLI entries (Susa, Malyan),
             Tepe Sofalin (TSF) and Ozbaki. PE-inscribed seals from Larsa are dropped (not tablets)."""
    tr, ho = [], []
    for d in docs:
        if d['site'] == 'Larsa':
            continue
        p = d['pub']
        recent = (d['year'] is not None and d['year'] >= 2000) or p.startswith('unpublished') \
            or d['site'] in ('Sofalin', 'Ozbaki')
        (ho if recent else tr).append(d)
    return tr, ho


def load_pc():
    C = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    cat = json.load(open(os.path.join(CK, 'catalog.json')))
    docs = []
    for d in C:
        c = cat.get(d['id'], {})
        bs, gs, ents, bg = tablet_units(d)
        docs.append(dict(id=d['id'], site=site_short(c.get('prov', d.get('provenience', ''))),
                         year=year_of(c.get('pubdate', '')), pub=c.get('pub', ''), signs=bs, graphs=gs,
                         words=ents, bigr=bg))
    return docs


def iv(v):
    v = np.asarray(v, float)
    return [float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def estimates(docs, key):
    ab = collections.Counter(w for d in docs for w in d[key])
    inc = collections.Counter(w for d in docs for w in set(d[key]))
    T = sum(1 for d in docs if d[key])
    A = list(ab.values())
    return dict(S=len(ab), n=sum(A), T=T, f1=A.count(1), f2=A.count(2),
                chao1=chao1(A), ace=ace(A), chao2=chao2(list(inc.values()), T),
                jack2=jack2(list(inc.values()), T), cover=coverage(A))


def community(A):
    A = np.array([x for x in A if x > 0], float)
    n = A.sum()
    C = coverage(A)
    f0 = max(int(round(chao1(A) - len(A))), 0)
    p = A / n * C
    if f0 > 0:
        p = np.concatenate([p, np.full(f0, (1 - C) / f0)])
    return p / p.sum(), len(A)


def exch_new(A, m, B, rng):
    p, S = community(A)
    n = int(sum(A))
    out = np.empty(B, int)
    for b in range(B):
        x = rng.multinomial(n, p)
        pb, Sb = community(x[x > 0])
        y = rng.multinomial(m, pb)
        out[b] = int((y[Sb:] > 0).sum())
    return out


def zm(S, a, q):
    i = np.arange(1, S + 1, dtype=float)
    w = (i + q) ** (-a)
    return w / w.sum()


def summ(A):
    A = np.asarray(A)
    A = A[A > 0]
    return np.array([len(A), (A == 1).sum(), (A == 2).sum(), (A == 3).sum(), A.max()], float)


def abc_fit(A, rng, n_sims=4000, keep=120):
    n = int(sum(A))
    so = summ(A)
    sc = np.array([so[0], so[1], max(so[2], 5), max(so[3], 5), max(so[4], 5)])
    sims = []
    for _ in range(n_sims):
        S = int(np.exp(rng.uniform(np.log(so[0]), np.log(so[0] * 60))))
        a, q = rng.uniform(0.3, 1.6), rng.uniform(0, 30)
        x = rng.multinomial(n, zm(S, a, q))
        sims.append((float(np.sqrt((((summ(x) - so) / sc) ** 2).sum())), S, a, q))
    sims.sort()
    return sims[:keep]


def abc_new(A, m, rng, post=None):
    n = int(sum(A))
    post = post or abc_fit(A, rng)
    out = []
    for d, S, a, q in post:
        p = zm(S, a, q)
        x = rng.multinomial(n, p)
        y = rng.multinomial(m, p)
        out.append(int(((y > 0) & (x == 0)).sum()))
    return np.array(out), [S for _, S, _, _ in post]


def hab_new(train, held_sites_tokens, key, B, rng, k0=20.0):
    """la15 habitat novelty: per-site rate of tokens unique to one training tablet;
    an unseen site gets the pooled non-majority-site rate."""
    tokc = collections.Counter(w for d in train for w in d[key])
    new_s, tot_s = collections.Counter(), collections.Counter()
    for d in train:
        own = collections.Counter(d[key])
        for w, c in own.items():
            new_s[d['site']] += c if tokc[w] == c else 0
            tot_s[d['site']] += c
    big = max(tot_s, key=tot_s.get)
    pn = sum(new_s[s] for s in tot_s if s != big)
    pt = sum(tot_s[s] for s in tot_s if s != big)
    r0 = pn / pt if pt else new_s[big] / tot_s[big]
    out = np.zeros(B)
    for s, m in held_sites_tokens.items():
        a = new_s.get(s, 0) + k0 * r0
        b = tot_s.get(s, 0) - new_s.get(s, 0) + k0 * (1 - r0)
        r = rng.beta(a, b, size=B)
        out += rng.binomial(m, r)
    return out


def predict_new(train, held_meta, key, rng, B=600, abc=True):
    A = list(collections.Counter(w for d in train for w in d[key]).values())
    m = sum(k for _, k in held_meta)
    hs = collections.Counter()
    for s, k in held_meta:
        hs[s] += k
    P = dict(m=m, EXCH=iv(exch_new(A, m, B, rng)), HAB=iv(hab_new(train, hs, key, B, rng)))
    if abc:
        P['ABC'] = iv(abc_new(A, m, rng)[0])
    return P


def truth_new(train, held, key):
    seen = set(w for d in train for w in d[key])
    return len(set(w for d in held for w in d[key]) - seen)


def scored(P, truth):
    return {k: dict(pred=v, truth=truth, inside=bool(v[1] <= truth <= v[2]), err=v[0] - truth)
            for k, v in P.items() if k != 'm'}


def known_scores(train, held_shell, key, lam=0.5):
    """P(known type appears at least once in the held-out tablets): exchangeable vs habitat mix."""
    tokc = collections.Counter(w for d in train for w in d[key])
    N = sum(tokc.values())
    site_c = collections.defaultdict(collections.Counter)
    for d in train:
        site_c[d['site']].update(d[key])
    ms = collections.Counter()
    for d in held_shell:
        ms[d['site']] += d['m']
    sc_ex, sc_hab, ecount = {}, {}, {}
    for w, c in tokc.items():
        f = c / N
        le, lh, ec = 0.0, 0.0, 0.0
        for s, md in ms.items():
            Ns = sum(site_c[s].values())
            fs = site_c[s][w] / Ns if Ns else f
            ph = lam * fs + (1 - lam) * f
            le += md * np.log1p(-min(f, 0.999999))
            lh += md * np.log1p(-min(ph, 0.999999))
            ec += md * f
        sc_ex[w], sc_hab[w], ecount[w] = 1 - np.exp(le), 1 - np.exp(lh), ec
    return sc_ex, sc_hab, ecount


def auc(scores, present):
    from scipy.stats import rankdata
    keys = list(scores)
    pos = np.array([scores[k] for k in keys if k in present])
    neg = np.array([scores[k] for k in keys if k not in present])
    if len(pos) == 0 or len(neg) == 0:
        return None
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=float).encode()).hexdigest()


def write_rows(path, header, rows):
    with open(path, 'w') as f:
        f.write(header + '\n\n| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            f.write('| ' + ' | '.join(r) + ' |\n')
