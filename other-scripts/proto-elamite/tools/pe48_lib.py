"""pe48 SIGNS AS GOODS IN A TRADE NETWORK: shared library.

Arrow in the dark: treat every sign as a commodity moving on a trade network. Each sign is either
TRAVELLING (same per-token rate at every site, the administrative toolkit) or HOMED at one site with a
gravity / diffusion fall-off  g_h(j) = eps + (1-eps) * exp(-D[h,j] / L) * (M_j / M_h)^b,
D = great-circle or route distance. Thousands of random parameter sets theta = (L, eps, b, metric,
penalty) are fitted; each sign picks its class by penalised training likelihood; theta is scored by
held-out Bernoulli log-likelihood of sign presence on held-out tablets (folds stratified by site),
reported separately for outpost tablets. Survivors (top theta) give each sign a travel posterior.

Corpora: PE (CDLI), Ur III (CDLI, by province), Linear B (DAMOS, by site). Controls use PE-shaped
draws (one hub of 1,500 documents and outposts of 27/22/12/11/1/1 documents).
"""
import os, sys, json, math, re, random, hashlib
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa: E402

PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe48_ckpt'); os.makedirs(CK, exist_ok=True)
REPO = os.path.abspath(os.path.join(PEROOT, '..', '..'))

# ------------------------------------------------------------------ geography
COORD = {
    # PE
    'Susa': (32.19, 48.25), 'Chogha Mish': (32.22, 48.55), 'Ghazir': (31.30, 49.60),
    'Malyan': (30.01, 52.41), 'Yahya': (28.33, 56.87), 'Sialk': (33.97, 51.40),
    'Sofalin': (35.30, 51.72), 'Ozbaki': (35.95, 50.58), 'Shahr-i Sokhta': (30.60, 61.33),
    'Godin': (34.52, 48.07),
    # Ur III
    'Girsu': (31.55, 46.17), 'Umma': (31.66, 45.88), 'Puzriš-Dagan': (32.10, 45.24),
    'Nippur': (32.13, 45.23), 'Ur': (30.96, 46.10), 'Garšana': (31.75, 45.95),
    'Irisagrig': (32.40, 45.40),
    # Linear B
    'KN': (35.30, 25.16), 'PY': (37.03, 21.69), 'TH': (38.32, 23.32), 'MY': (37.73, 22.76),
    'TI': (37.60, 22.80), 'KH': (35.51, 24.02),
}
# route graphs (edges; length = great-circle of the edge, or of the via-point path)
ROUTES = {
    'PE': [('Susa', 'Chogha Mish'), ('Susa', 'Ghazir'), ('Ghazir', 'Malyan'), ('Malyan', 'Yahya'),
           ('Yahya', 'Shahr-i Sokhta'), ('Susa', 'Godin'), ('Godin', 'Ozbaki'), ('Ozbaki', 'Sofalin'),
           ('Sofalin', 'Sialk'), ('Sialk', 'Malyan')],
    'UR3': [('Ur', 'Girsu'), ('Girsu', 'Umma'), ('Umma', 'Garšana'), ('Umma', 'Nippur'),
            ('Nippur', 'Puzriš-Dagan'), ('Nippur', 'Irisagrig')],
    'LB': [('KN', 'KH'), ('KH', 'PY'), ('PY', 'MY'), ('MY', 'TI'), ('MY', 'TH')],
}


def gc(a, b):
    (la1, lo1), (la2, lo2) = COORD[a], COORD[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    dl = math.radians(lo2 - lo1); dp = p2 - p1
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def dist_mats(sites, corpus):
    n = len(sites)
    G = np.array([[gc(a, b) for b in sites] for a in sites])
    # route: Floyd-Warshall on the route graph over all sites mentioned in it
    nodes = sorted({x for e in ROUTES[corpus] for x in e} | set(sites))
    ix = {s: i for i, s in enumerate(nodes)}
    R = np.full((len(nodes), len(nodes)), np.inf); np.fill_diagonal(R, 0)
    for a, b in ROUTES[corpus]:
        R[ix[a], ix[b]] = R[ix[b], ix[a]] = gc(a, b)
    for k in range(len(nodes)):
        R = np.minimum(R, R[:, [k]] + R[[k], :])
    Rs = np.array([[R[ix[a], ix[b]] for b in sites] for a in sites])
    Rs[~np.isfinite(Rs)] = G[~np.isfinite(Rs)] * 1.5
    return {'gc': G, 'route': Rs}


# ------------------------------------------------------------------ corpora
PE_SITE = {'Susa (mod. Shush)': 'Susa', 'Susa (mod. Shush) ?': 'Susa',
           'uncertain (mod. Tepe Yahya)': 'Yahya', 'Anšan (mod. Tell Malyan)': 'Malyan',
           'uncertain (mod. Tepe Sialk)': 'Sialk', 'uncertain (mod. Tepe Sofalin)': 'Sofalin',
           'uncertain (mod. Chogha Mish)': 'Chogha Mish', 'uncertain (mod. Shahr-i Sokhta)': 'Shahr-i Sokhta',
           'uncertain (mod. Ghazir)': 'Ghazir', 'uncertain (mod. Ozbaki)': 'Ozbaki'}


def pe_docs():
    """List of dict(id, site, toks=list of base-sign tokens, roles=list of (sign, role))."""
    out = []
    for t in load():
        site = PE_SITE.get(t['provenience'])
        if not site:
            continue
        toks, roles = [], []
        for i, l in enumerate(t['lines']):
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            if not sg:
                continue
            if i == 0 and not l['numerals']:
                role = 'HEAD'
            elif l['numerals'] and len(sg) == 1:
                role = 'SOLO'
            elif len(sg) >= 3:
                role = 'STRING'
            else:
                role = 'OTHER'
            toks += sg
            roles += [(s, role) for s in sg]
        if toks:
            out.append(dict(id=t['id'], site=site, toks=toks, roles=roles))
    return out


UR_T = set('''udu gu4 sila4 sze kasz ninda masz2 i3 tug2 u8 ab2 siki zi3 gurusz geme2 dumu ugula engar sukkal
maszkim ensi2 lugal dub-sar szabra sanga nu-banda3 sipa muhaldim ku3-babbar zu2-lum gesz naga amar ud5
masz2-gal niga ma2 a-sza3 lu2-kin-gi4-a gu2 ga kusz anszu dur3 ninda-gal munus kaskal gu4-niga udu-niga
gada sa10 zabar urudu na4 ku6 mun gazi sum dabin eme5 szah2 uz-tur tu-gur4{muszen} gi'''.split())
UR_MONTH_STOP = {'u4', 'diri', 'mu', 'ki', 'x', 'iti', 'sza3', 'ta', 'ba-zal'}


def ur3_docs():
    fn = os.path.join(CK, 'ur3_bags.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    d = json.load(open(os.path.join(DATA, 'pe38_ckpt', 'ur3_docs.json')))
    out = []
    names, months = Counter(), Counter()
    for x in d:
        toks = []
        for l in x['lines']:
            t = l['toks']
            for i, w in enumerate(t):
                if not w or '(' in w or w in ('x',):
                    continue
                toks.append(w)
                if i > 0 and t[i - 1] in ('kiszib3', 'giri3'):
                    names[w] += 1
                if i + 2 < len(t) and t[i + 1] == 'szu' and t[i + 2] == 'ba-ti':
                    names[w] += 1
                if i > 0 and t[i - 1] == 'iti' and w not in UR_MONTH_STOP:
                    months[w] += 1
        if toks and x['site'] not in ('', 'uncertain'):
            out.append(dict(id=x['id'], site=x['site'], toks=toks))
    bad = UR_T | {'ensi2-ka', 'nam-sza3-tam', 'nu-tuku', 'lugal', 'arad2', 'dumu', 'du11-ga'}
    gold = {}
    for w, n in names.items():
        if n >= 3 and '-' in w and 'x' not in w.split('-') and '...' not in w and w not in bad and not any(w.startswith(b + '-') for b in ('ensi2', 'ugula', 'dub')):
            gold[w] = 'N'
    for w, n in months.items():
        if n >= 3 and w not in gold and w not in bad:
            gold[w] = 'M'
    for w in UR_T:
        gold[w] = 'T'
    res = dict(docs=out, gold=gold)
    json.dump(res, open(fn, 'w'))
    return res


LB_T_WORDS = {'e-qe-ta', 'te-re-ta', 'ka-ke-u', 'qa-si-re-u', 'ko-re-te', 'do-e-ro', 'do-e-ra',
              'i-je-re-ja', 'ka-ra-wi-po-ro', 'ra-wa-ke-ta', 'wa-na-ka', 'ke-ro-si-ja', 'po-ro-ko-re-te',
              'mo-ro-pa2', 'te-o-jo', 'do-e-ro-i', 'ke-ke-me-na', 'ki-ti-me-na', 'o-na-to', 'to-so',
              'to-sa', 'pa-ro', 'o-pe-ro', 'a-pu-do-si', 'o-u-di-do-si', 'ku-su', 'pe-mo', 'pe-ma',
              'e-ra3-wo', 'e-ra-wo', 'me-ri', 'ku-pa-ro', 'ko-ri-ja-do-no', 'ka-po', 'a-ke-ra2-te',
              'pa-ra-ku', 'ki-to', 'pa-we-a', 'pa-we-a2', 'ri-no', 'a-ra-ka-te-ja', 'ra-pi-ti-ra2',
              'ka-ke-we', 'ra-pte-re', 'po-me', 'po-me-ne', 'su-qo-ta-o', 'a3-ki-pa-ta', 'o-pi',
              'tu-na-no', 'ko-wo', 'ko-wa', 'ko-wo-i', 'ko-wa-i', 'ka-ka', 'e-re-ta', 'e-re-ta-o'}
LB_EDIT = {'mut', 'inf', 'sup', 'vac', 'vacat', 'supra', 'infra', 'sigillum', 'vest', 'vestigia', 'deest',
           'lat', 'dex', 'sin', 'verso', 'recto', 'margo', 'prior', 'pars', 'post', 'qs'}


def lb_docs():
    fn = os.path.join(CK, 'lb_bags.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    out, first = [], Counter()
    for line in open(os.path.join(REPO, 'other-scripts', 'linear-a', 'data', 'damos_items.jsonl')):
        x = json.loads(line)
        h = x.get('heading') or ''
        site = h.split(' ')[0]
        if site not in ('KN', 'PY', 'TH', 'MY', 'TI', 'KH'):
            continue
        toks = []
        for ln in (x.get('content') or '').split('\n'):
            ln = re.sub(r'^\s*\.\S+', '', ln)
            ln = re.sub(r'[\[\]⌞⌟̣̀-ͯ\?]', '', ln)
            ws = re.findall(r"[A-Za-z*0-9+±\-]+", ln)
            lt = []
            for w in ws:
                if re.fullmatch(r'[a-z0-9*]+(?:-[a-z0-9*]+)+', w) and w.split('-')[0] not in LB_EDIT:
                    lt.append(w)
                elif re.fullmatch(r'\*?[A-Z][A-Z+±*0-9]{1,}', w) and not re.fullmatch(r'[A-Z]\d*', w):
                    lt.append(w)
            if lt and lt[0].islower() and any(not t.islower() for t in lt[1:]):
                first[lt[0]] += 1
            toks += lt
        if toks:
            out.append(dict(id=h, site=site, toks=toks))
    gold = {}
    for w, n in first.items():
        if n >= 2 and w not in LB_T_WORDS:
            gold[w] = 'N'
    for d in out:
        for t in d['toks']:
            if not t.islower():
                gold[t] = 'T'
    for w in LB_T_WORDS:
        gold[w] = 'T'
    res = dict(docs=out, gold=gold)
    json.dump(res, open(fn, 'w'))
    return res


PE_SHAPE = [27, 22, 12, 11, 1, 1, 1]   # outposts of the PE corpus, after the hub (1,502)


def pe_shaped(docs, hub, outposts, rng, n_hub=1502, shape=PE_SHAPE):
    """Draw a PE-shaped sample: n_hub documents from the hub, shape[i] from outposts[i]."""
    by = defaultdict(list)
    for d in docs:
        by[d['site']].append(d)
    out = rng.sample(by[hub], min(n_hub, len(by[hub])))
    for s, k in zip(outposts, shape):
        out += rng.sample(by[s], min(k, len(by[s])))
    return out


# ------------------------------------------------------------------ the model
class Data:
    def __init__(self, docs, corpus, min_docs=3, mass=None, max_vocab=600, keep=('PLANT', 'ROUTE')):
        self.sites = sorted({d['site'] for d in docs}, key=lambda s: -sum(x['site'] == s for x in docs))
        self.hub = self.sites[0]
        si = {s: i for i, s in enumerate(self.sites)}
        df = Counter(t for d in docs for t in set(d['toks']))
        top = [t for t, n in df.most_common() if n >= min_docs][:max_vocab]
        self.vocab = sorted(set(top) | {t for t in keep if df.get(t, 0) >= 1})
        vi = {t: i for i, t in enumerate(self.vocab)}
        D, S = len(docs), len(self.vocab)
        self.Y = np.zeros((D, S), bool)
        self.C = np.zeros((D, S), np.float32)
        for i, d in enumerate(docs):
            for t in d['toks']:
                j = vi.get(t)
                if j is not None:
                    self.Y[i, j] = True; self.C[i, j] += 1
        self.L = np.array([max(1, len(d['toks'])) for d in docs], float)
        self.site = np.array([si[d['site']] for d in docs])
        self.docs = docs
        self.Dm = dist_mats(self.sites, corpus)
        cnt = np.bincount(self.site, minlength=len(self.sites)).astype(float)
        self.M = cnt if mass is None else np.array([mass[s] for s in self.sites], float)
        self.outpost = self.site != 0


def gmat(dat, th):
    """G[c, j]: class c = 0 travelling; c = 1 + e*H + h homed at site h with floor level e
    (floors theta.eps * 4^-e, e = 0..NEPS-1: each sign picks its own contrast)."""
    D = dat.Dm[th['metric']]
    H = len(dat.sites)
    rows = [np.ones((1, H))]
    for e in range(NEPS):
        ep = th['eps'] * 4.0 ** (-e)
        g = ep + (1 - ep) * np.exp(-D / th['L'])
        rows.append(g * (dat.M[None, :] / dat.M[:, None]) ** th['b'])
    return np.vstack(rows)


NEPS = 4


def home_of(cls, H):
    """Collapse raw class index to 0 = travelling, 1 + h = homed at site h."""
    return np.where(cls == 0, 0, 1 + (cls - 1) % H)


def _ll(Y, rate):
    rate = np.maximum(rate, 1e-12)
    return np.where(Y, np.log(-np.expm1(-rate)), -rate)


def _groups(dat, mask, Y):
    """Group docs of mask by (site, length): returns site, length, count, presence sums (G, S)."""
    idx = np.where(mask)[0]
    key = dat.site[idx] * 100000 + dat.L[idx].astype(int)
    uk, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv).astype(float)
    K = np.zeros((len(uk), Y.shape[1]))
    np.add.at(K, inv, Y[idx].astype(float))
    return (uk // 100000).astype(int), (uk % 100000).astype(float), n, K, inv


def _llg(n, K, rate):
    rate = np.maximum(rate, 1e-12)
    return K * np.log(-np.expm1(-rate)) - (n[:, None] - K) * rate


def fit_eval(dat, th, train, test, Yover=None, return_cls=False):
    """Fit sign classes on train docs (bool mask), return held-out ll per test doc (model - travelling)."""
    Y = dat.Y if Yover is None else Yover
    G = gmat(dat, th)                                       # (K, H)
    Ltr = dat.L[train]; str_ = dat.site[train]
    ntok = (dat.C if Yover is None else Y.astype(np.float32))[train].sum(0) + 0.5   # (S,)
    denom = np.array([(Ltr * G[c, str_]).sum() for c in range(G.shape[0])])          # (K,)
    q = ntok[None, :] / denom[:, None]                                                # (K, S)
    gs, gl, gn, gK, _ = _groups(dat, train, Y)
    Kc = G.shape[0]
    tll = np.zeros((Kc, Y.shape[1]))
    for c in range(Kc):
        rate = (gl * G[c, gs])[:, None] * q[c][None, :]
        tll[c] = _llg(gn, gK, rate).sum(0)
    tll[1:] -= th['pen']
    cls = tll.argmax(0)                                     # (S,)
    if return_cls:
        return cls, tll, q
    Lte = dat.L[test]; ste = dat.site[test]
    gsel = G[cls][:, ste].T                                 # (Dte, S)
    rate = Lte[:, None] * gsel * q[cls, np.arange(len(cls))][None, :]
    llm = _ll(Y[test], rate).sum(1)
    rate0 = Lte[:, None] * q[0][None, :]
    ll0 = _ll(Y[test], rate0).sum(1)
    return llm - ll0


def folds(dat, k, rng):
    """Stratified by site: each site's docs split into k folds (sites with 1 doc go to one fold)."""
    f = np.zeros(len(dat.site), int)
    for s in range(len(dat.sites)):
        idx = np.where(dat.site == s)[0].copy()
        rng.shuffle(idx)
        off = rng.integers(k)
        for r, i in enumerate(idx):
            f[i] = (r + off) % k
    return f


def rand_theta(rng):
    return dict(L=float(np.exp(rng.uniform(np.log(30), np.log(5000)))),
                eps=float(rng.uniform(0.02, 0.8)), b=float(rng.uniform(-0.5, 0.5)),
                metric=str(rng.choice(['gc', 'route'])), pen=float(rng.uniform(0, 8)))


def cv_score(dat, th, fo, k, Yover=None):
    """Mean held-out gain (nats/doc) over travelling baseline: (all, outposts)."""
    g = np.zeros(len(dat.site))
    for f in range(k):
        te = fo == f
        g[te] = fit_eval(dat, th, ~te, te, Yover)
    return float(g.mean()), float(g[dat.outpost].mean())


def search(dat, n_theta, seed, k=5, Yover=None):
    rng = np.random.default_rng(seed)
    fo = folds(dat, k, rng)
    res = []
    for _ in range(n_theta):
        th = rand_theta(rng)
        a, o = cv_score(dat, th, fo, k, Yover)
        res.append((th, a, o))
    return res


def travel_posterior(dat, thetas, Yover=None):
    """Fraction of theta assigning each sign to each class (fit on all docs), and the mean travel LLR
    (travelling ll minus best homed ll, before penalty). Returns (P (S, K), llr (S,))."""
    K = len(dat.sites) + 1
    P = np.zeros((len(dat.vocab), K)); llr = np.zeros(len(dat.vocab))
    allm = np.ones(len(dat.site), bool)
    for th in thetas:
        cls, tll, _ = fit_eval(dat, th, allm, allm, Yover, return_cls=True)
        P[np.arange(len(cls)), home_of(cls, len(dat.sites))] += 1
        llr += tll[0] - (tll[1:] + th['pen']).max(0)
    n = max(1, len(thetas))
    return P / n, llr / n


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv, kind='mergesort'), kind='mergesort').astype(float)
    # average ties
    from scipy.stats import rankdata
    r = rankdata(allv)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def strat_auc(score, freq, is_pos, is_neg, nbin=5):
    """AUC within doc-frequency quantile bins, averaged with weights n_pos*n_neg."""
    idx = np.where(is_pos | is_neg)[0]
    if len(idx) < 4:
        return float('nan')
    qs = np.quantile(freq[idx], np.linspace(0, 1, nbin + 1))
    b = np.clip(np.searchsorted(qs, freq, side='right') - 1, 0, nbin - 1)
    num = den = 0.0
    for k in range(nbin):
        p = score[is_pos & (b == k)]; n = score[is_neg & (b == k)]
        if len(p) and len(n):
            w = len(p) * len(n); num += auc(p, n) * w; den += w
    return num / den if den else float('nan')


def plant(docs, site_rates, tok, rng):
    """Add token `tok` to a fraction site_rates[site] of documents at each site (copy)."""
    out = []
    for d in docs:
        d2 = dict(d)
        if rng.random() < site_rates.get(d['site'], 0.0):
            d2['toks'] = list(d['toks']) + [tok]
        out.append(d2)
    return out


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]
