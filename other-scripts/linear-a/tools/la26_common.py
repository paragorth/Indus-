"""LA-26 'The rings know who talked': the seal-ring network as text-independent data.

Ring / seal identities across sites are taken as DATA from published seal identifications
(CMS numbers; compiled in Montecchi, Proceedings 12th Int. Cretological Congress, 2018,
https://flore.unifi.it/retrieve/e398c382-48d3-179a-e053-3705fe0a4cff/Montecchi_12ICCS.pdf ;
CMS records checked to exist on Arachne, https://arachne.dainst.org). No interpretation of the
rings (who held them, 'Knossian' etc.) is used: only which seal face was impressed at which site.

Linear A texts: data/corpus.json (lineara.xyz). No sound value or reading is used.
"""
import json, os, re, math, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la26_ckpt')
os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(HERE, '..', 'loops')

# ---------------------------------------------------------------- seal network (data)
# site codes: HT Hagia Triada, KN Knossos, ZA Zakros, KH Khania, GO Gournia, TH Akrotiri (Thera),
# SK Sklavokambos (no Linear A text in the corpus).
SEALS = [
    # id, kind, {site: number of impressed documents}, CMS refs, flags
    dict(id='R15', kind='ring', sites={'HT': 5, 'KN': 2}, cms='II.6 15 = II.8 279'),
    dict(id='R19', kind='ring', sites={'HT': 2, 'SK': 4, 'TH': 3}, cms='II.6 19 = II.6 260 = V S3.2 391'),
    dict(id='R44', kind='ring', sites={'HT': 1, 'SK': 1, 'GO': 1}, cms='II.6 44 = 162 = 255'),
    dict(id='R43', kind='ring', sites={'HT': 3, 'SK': 2, 'GO': 1, 'ZA': 1}, cms='II.6 43 = 259 = 161 = II.7 39'),
    dict(id='R71', kind='ring', sites={'HT': 1, 'ZA': 2, 'KN': 3}, cms='II.7 71 = II.8,2 298 (+ HT nodule Pig. 71980)',
         disputed='HT'),   # CMS suggests the HT nodule is really from Zakros
    dict(id='L117', kind='seal', sites={'HT': 1, 'SK': 1}, cms='Levi 117 = Marinatos 8 (countermark)'),
    dict(id='S68', kind='stone', sites={'HT': 3, 'ZA': 1}, cms='II.6 68 = II.7 45', disputed='ZA'),
    dict(id='LA41', kind='lookalike', sites={'HT': 1, 'SK': 1, 'ZA': 1, 'KH': 1, 'TH': 1},
         cms='II.6 41 ~ II.6 258 ~ II.7 36 ~ V S1A 171 ~ V S3.2 392'),
    dict(id='Z104', kind='style', sites={'HT': 1, 'ZA': 1}, cms='II.6 104-105 (Zakro Master style at HT)'),
]

SITE_NAME = {'HT': 'Haghia Triada', 'KN': 'Knossos', 'ZA': 'Zakros', 'KH': 'Khania', 'GO': 'Gournia',
             'TH': 'Thera', 'PH': 'Phaistos', 'MA': 'Malia', 'TY': 'Tylissos', 'SK': None}
COORD = {'HT': (35.059, 24.792), 'KN': (35.298, 25.163), 'ZA': (35.098, 26.261), 'KH': (35.517, 24.018),
         'GO': (35.108, 25.794), 'TH': (36.351, 25.404), 'PH': (35.051, 24.814), 'MA': (35.293, 25.492),
         'TY': (35.299, 25.020), 'SK': (35.330, 24.870)}
NODES_A = ['HT', 'KN', 'ZA', 'KH', 'GO', 'TH']          # sites inside the documented seal system
NODES_B = NODES_A + ['PH', 'MA', 'TY']                  # + sealing sites with no shared seal recorded
NODES_C = ['HT', 'KN', 'ZA', 'KH']                      # sites with >= 50 documents


def km(a, b):
    la1, lo1 = map(math.radians, a); la2, lo2 = map(math.radians, b)
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def seal_incidence(version='core', nodes=None):
    """Seal x site 0/1 incidence (list of site sets). version:
    core   = same ring / same seal (R*, L117, S68)
    strict = core with disputed attributions removed
    broad  = core + look-alike + style"""
    out = []
    for s in SEALS:
        if s['kind'] in ('lookalike', 'style') and version != 'broad':
            continue
        st = set(s['sites'])
        if version == 'strict' and s.get('disputed'):
            st.discard(s['disputed'])
        out.append(st)
    return out


def ring_matrix(inc, nodes, two_step=False):
    """Weighted site x site matrix: number of seals shared. two_step adds paths through SK."""
    n = len(nodes); W = np.zeros((n, n))
    idx = {s: i for i, s in enumerate(nodes)}
    for st in inc:
        L = [idx[s] for s in st if s in idx]
        for a in L:
            for b in L:
                if a != b:
                    W[a, b] += 1
    if two_step:
        sk = [st for st in inc if 'SK' in st]
        for s1 in sk:
            for s2 in sk:
                if s1 is s2:
                    continue
                for a in s1:
                    for b in s2:
                        if a in idx and b in idx and a != b and W[idx[a], idx[b]] == 0:
                            W[idx[a], idx[b]] = 0.5
                            W[idx[b], idx[a]] = 0.5
    return W


def dist_matrix(nodes):
    n = len(nodes); D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = km(COORD[nodes[i]], COORD[nodes[j]])
    return D


def curveball_inc(inc, all_sites, rng, iters=200):
    """Rewire the seal x site bipartite graph keeping seal sizes and site degrees."""
    rows = [set(r) for r in inc]
    R = len(rows)
    for _ in range(iters):
        i, j = rng.integers(R), rng.integers(R)
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
        rows[i] = common | set(pool[:k]); rows[j] = common | set(pool[k:])
    return rows

# ---------------------------------------------------------------- Linear A documents

ADMIN = {'Tablet', 'Roundel', 'Nodule', 'Lames (short thin tablet)', 'Sealing', '3-sided bar', '4-sided bar', 'Label'}


def _logo(s):
    return bool(re.fullmatch(r'\*[4-9]\d\d.*', s)) or s in ('VS', 'VAS') or bool(re.search(r'[a-z]', s))


def load_docs(nodes, admin_only=False):
    C = json.load(open(os.path.join(DATA, 'corpus.json')))
    inv = {v: k for k, v in SITE_NAME.items() if v}
    docs = []
    for d in C:
        code = inv.get(d['site'])
        if code not in nodes:
            continue
        if admin_only and d['support'] not in ADMIN:
            continue
        words = [w for w in d['words'] if '-' in w and not any(_logo(x) for x in w.split('-'))]
        singles = [w for w in d['words'] if '-' not in w and not _logo(w)]
        signs = [s for t in d['tokens'] if t['t'] == 'word' for s in t['s'] if not _logo(s)]
        logos = [t['v'] for t in d['tokens'] if t['t'] == 'logo']
        # logogram variants: ligatures / modified forms only (contain + or a lowercase / digit modifier)
        lvar = [v for v in logos if '+' in v and '[' not in v]
        cls = []
        for t in d['tokens']:
            if t['t'] == 'word':
                c = 'W' if len(t['s']) > 1 else 'S'
            elif t['t'] == 'num':
                c = 'F' if t.get('frac') and not t.get('v') else 'N'
            elif t['t'] == 'logo':
                c = 'L'
            elif t['t'] == 'nl':
                c = '/'
            elif t['t'] == 'div':
                c = '|'
            else:
                c = '?'
            cls.append(c)
        struct = ['^' + cls[0]] if cls else []
        struct += [a + b for a, b in zip(cls, cls[1:])]
        docs.append(dict(id=d['id'], site=code, support=d['support'], words=words, singles=singles,
                         signs=signs, logos=logos, lvar=lvar, struct=struct))
    return docs


def doc_matrix(docs, key, vocab=None):
    if vocab is None:
        vocab = sorted({x for d in docs for x in d[key]})
    vi = {v: i for i, v in enumerate(vocab)}
    M = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for r, d in enumerate(docs):
        for x in d[key]:
            if x in vi:
                M[r, vi[x]] += 1
    return M, vocab


def site_onehot(labels, nodes):
    idx = {s: i for i, s in enumerate(nodes)}
    O = np.zeros((len(nodes), len(labels)), dtype=np.float32)
    for r, s in enumerate(labels):
        O[idx[s], r] = 1
    return O


def shared_types(S):
    P = (S > 0).astype(np.float32)
    return P @ P.T


def jsd_sim(S):
    """1 - Jensen-Shannon divergence (bits) between site profiles."""
    n = S.shape[0]
    tot = S.sum(1, keepdims=True); tot[tot == 0] = 1
    P = S / tot
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = 0.5 * (P[i] + P[j])
            def kl(p):
                k = p > 0
                return float((p[k] * np.log2(p[k] / m[k])).sum())
            out[i, j] = out[j, i] = 1 - 0.5 * (kl(P[i]) + kl(P[j]))
    return out


MEASURES = {
    'words': ('words', shared_types),        # shared word types (2+ signs)
    'signs': ('signs', jsd_sim),             # syllabic sign profile similarity
    'lvar': ('lvar', shared_types),          # shared logogram variants / ligatures
    'struct': ('struct', jsd_sim),           # entry-structure (token class bigram) similarity
}


def perm_within_support(labels, supports, rng):
    lab = np.array(labels, dtype=object)
    sup = np.array(supports, dtype=object)
    out = lab.copy()
    for s in set(supports):
        k = np.where(sup == s)[0]
        out[k] = lab[rng.permutation(k)]
    return list(out)


def excess_z(docs, nodes, key, fn, nperm, rng, M=None):
    """Observed site-pair similarity standardized against document-label permutations within
    support type (controls site size and support mix)."""
    if M is None:
        M, _ = doc_matrix(docs, key)
    labels = [d['site'] for d in docs]; sups = [d['support'] for d in docs]
    obs = fn(site_onehot(labels, nodes) @ M)
    sims = np.zeros((nperm,) + obs.shape)
    for p in range(nperm):
        sims[p] = fn(site_onehot(perm_within_support(labels, sups, rng), nodes) @ M)
    mu, sd = sims.mean(0), sims.std(0)
    sd[sd == 0] = np.inf
    return (obs - mu) / sd, obs, mu


def upper(X):
    i, j = np.triu_indices(X.shape[0], 1)
    return X[i, j]


def resid_on(y, xs):
    X = np.column_stack([np.ones_like(y)] + xs)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def ring_stat(Z, R, D, valid=None):
    """Partial association of ring linkage with excess similarity, distance removed:
    Spearman-free version = mean residual on ring pairs minus mean residual on non-ring pairs."""
    z = upper(Z); r = upper(R) > 0; d = np.log(upper(D))
    ok = np.isfinite(z) if valid is None else (np.isfinite(z) & upper(valid).astype(bool))
    z, r, d = z[ok], r[ok], d[ok]
    if r.all() or not r.any():
        return np.nan
    e = resid_on(z, [d])
    return float(e[r].mean() - e[~r].mean())


def write_rows(path, header, rows):
    with open(path, 'w') as f:
        f.write(header + '\n\n| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            f.write('| ' + ' | '.join(r) + ' |\n')
