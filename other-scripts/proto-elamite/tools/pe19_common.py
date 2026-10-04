"""pe19 shared code: THE SCRIPT AGED. Blind seriation of tablets by evolving traits.

Fitting code (features, seriation) NEVER reads data/pe19_hidden.json; only the
scoring functions at the bottom do, and only after an order is frozen + hashed.

Traits per tablet (binary):
  B:  base sign present (variant suffix stripped; compound components and the
      compound itself, stripped)
  V:  variant form present, only for bases attested with >= 2 forms
  N:  numeral code present (N01, N14, N39B, N30C@b ...)
  F:  format bins (lines, entry length, multi-sign share, reverse, header,
      compounds, distinct numeral codes per line)
Seriation methods: CA axis 1, spectral (Fiedler vector of a kNN cosine graph),
2-opt TSP path, and a unimodal latent-trait model ('each trait is born, peaks,
dies'), fitted with torch, restarts chosen by training loss only.
Orientation rule (fixed before scoring): the late end is the end where the mean
number of signs per entry is larger (writing grows from single-sign notations).
"""
import hashlib, json, math, os, re
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe19_ckpt')
os.makedirs(CK, exist_ok=True)


def base_of(s):
    s = s.strip('|')
    return re.sub(r'~[a-z0-9]+', '', s)


def comps(s):
    return [c for c in re.split(r'[+x.&]', s.strip('|')) if c and not c.startswith(('1(', '2(', '3('))]


def load_pe():
    return json.load(open(os.path.join(DATA, 'pe_corpus.json')))


def load_pc(site='Uruk (mod. Warka)'):
    T = json.load(open(os.path.join(DATA, 'pe2_pc_corpus.json')))
    return [t for t in T if site is None or t['provenience'] == site]


def bin_(v, edges):
    for i, e in enumerate(edges):
        if v <= e:
            return i
    return len(edges)


def tablet_traits(t):
    B, V, N, F = set(), set(), set(), set()
    nl = len(t['lines'])
    ent_len, multi, ndist = [], 0, []
    comp = rev = 0
    for l in t['lines']:
        sg = [s for s in l['signs'] if s.lower() != 'x' and not s.startswith('X')]
        for s in sg:
            B.add(base_of(s))
            if '~' in s:
                V.add(s.strip('|'))
            if '+' in s or '|' in s:
                comp = 1
                for c in comps(s):
                    B.add(base_of(c))
        for k, c in l['numerals']:
            N.add(c)
        if l['numerals'] and sg:
            ent_len.append(len(sg)); multi += len(sg) > 1
        if l['numerals']:
            ndist.append(len(set(c for _, c in l['numerals'])))
        if l['surface'] == 'reverse':
            rev = 1
    F.add('F_lines%d' % bin_(nl, [1, 3, 7, 15]))
    if ent_len:
        F.add('F_elen%d' % bin_(np.mean(ent_len), [1.0, 1.5, 2.5]))
        F.add('F_multi%d' % bin_(multi / len(ent_len), [0, 0.34, 0.67]))
    else:
        F.add('F_noentry')
    F.add('F_rev%d' % rev)
    F.add('F_comp%d' % comp)
    first = t['lines'][0] if t['lines'] else None
    F.add('F_hdr%d' % int(bool(first and first['signs'] and not first['numerals'])))
    if ndist:
        F.add('F_ndist%d' % bin_(max(ndist), [1, 2, 3]))
    return B, V, N, F, (np.mean(ent_len) if ent_len else 0.0)


def build_matrix(T, sets='BVNF', min_tab=3, min_traits=2):
    """Return ids, X (n x m float32 0/1), trait names, entry-length vector."""
    rows, elen = {}, {}
    vforms = defaultdict(set)
    tt = {}
    for t in T:
        B, V, N, F, el = tablet_traits(t)
        tt[t['id']] = (B, V, N, F); elen[t['id']] = el
        for v in V:
            vforms[base_of(v)].add(v)
    multi = {b for b, f in vforms.items() if len(f) >= 2}
    for pid, (B, V, N, F) in tt.items():
        tr = set()
        if 'B' in sets: tr |= {'B_' + b for b in B}
        if 'V' in sets: tr |= {'V_' + v for v in V if base_of(v) in multi}
        if 'N' in sets: tr |= {'N_' + n for n in N}
        if 'F' in sets: tr |= F
        rows[pid] = tr
    cnt = Counter(x for tr in rows.values() for x in tr)
    names = sorted(x for x, c in cnt.items() if c >= min_tab)
    idx = {x: i for i, x in enumerate(names)}
    ids = [p for p in rows if sum(x in idx for x in rows[p]) >= min_traits]
    X = np.zeros((len(ids), len(names)), np.float32)
    for i, p in enumerate(ids):
        for x in rows[p]:
            if x in idx:
                X[i, idx[x]] = 1
    return ids, X, names, np.array([elen[p] for p in ids])


# ---------------- seriation methods (blind) ----------------
def ca_scores(X):
    P = X / X.sum()
    r, c = P.sum(1), P.sum(0)
    keep = c > 0
    P, c = P[:, keep], c[keep]
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    return U[:, 0] / np.sqrt(r)


def spectral_scores(X, k=10):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import laplacian, connected_components
    from scipy.sparse.linalg import eigsh
    idf = np.log(X.shape[0] / (1 + X.sum(0)))
    Y = X * idf
    Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    S = Y @ Y.T
    np.fill_diagonal(S, 0)
    n = len(S)
    nb = np.argpartition(-S, k, axis=1)[:, :k]
    rr = np.repeat(np.arange(n), k)
    W = csr_matrix((np.maximum(S[rr, nb.ravel()], 1e-4), (rr, nb.ravel())), shape=(n, n))
    W = W.maximum(W.T)
    ncomp, lab = connected_components(W)
    if ncomp > 1:  # connect components weakly to keep a single ordering
        W = W + csr_matrix(np.full((n, n), 1e-5))
    L = laplacian(W, normed=True)
    vals, vecs = eigsh(L, k=3, sigma=-1e-3, which='LM')
    o = np.argsort(vals)
    return vecs[:, o[1]]


def jaccard_dist(X):
    inter = X @ X.T
    sz = X.sum(1)
    uni = sz[:, None] + sz[None, :] - inter
    return 1 - inter / np.maximum(uni, 1)


def tsp_order(D, seed_order, passes=40):
    """Open-path 2-opt from a seed order (vectorised over j)."""
    o = np.array(seed_order)
    n = len(o)
    for _ in range(passes):
        improved = False
        for i in range(0, n - 2):
            a, b = o[i], o[i + 1]
            j = np.arange(i + 2, n)
            c = o[j]
            d = np.where(j + 1 < n, o[np.minimum(j + 1, n - 1)], -1)
            old = D[a, b] + np.where(d >= 0, D[c, np.maximum(d, 0)], 0)
            new = D[a, c] + np.where(d >= 0, D[b, np.maximum(d, 0)], 0)
            g = old - new
            k = int(np.argmax(g))
            if g[k] > 1e-9:
                jj = j[k]
                o[i + 1:jj + 1] = o[i + 1:jj + 1][::-1]
                improved = True
        if not improved:
            break
    pos = np.empty(n); pos[o] = np.arange(n)
    return pos


def unimodal_fit(X, init, steps=600, seed=0, lr=0.05):
    """Latent 1-D time x_i; P(trait j) = sigmoid(a_j - (x_i-u_j)^2 * w_j).
    Returns (x, loss). Blind: only the trait matrix is used."""
    import torch
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(X)
    n, m = X.shape
    x0 = torch.tensor((init - init.mean()) / (init.std() + 1e-9), dtype=torch.float32)
    x = x0.clone().requires_grad_(True)
    p = Xt.mean(0).clamp(1e-3, 0.999)
    a = torch.log(p / (1 - p)).clone().requires_grad_(True)
    u = (torch.randn(m, generator=g) * 0.5).requires_grad_(True)
    lw = torch.full((m,), -1.0).requires_grad_(True)
    opt = torch.optim.Adam([x, a, u, lw], lr=lr)
    for it in range(steps):
        opt.zero_grad()
        xs = (x - x.mean()) / (x.std() + 1e-6)
        z = a[None, :] - (xs[:, None] - u[None, :]) ** 2 * torch.exp(lw)[None, :]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Xt, reduction='mean')
        loss.backward(); opt.step()
    xs = ((x - x.mean()) / (x.std() + 1e-6)).detach().numpy()
    return xs, float(loss)


def orient(score, elen):
    """Fixed rule: late end = longer entries."""
    r = np.corrcoef(rankdata(score), elen)[0, 1]
    return score if r >= 0 else -score


def rankdata(v):
    from scipy.stats import rankdata as rd
    return rd(v)


def freeze(name, ids, score, extra=None):
    order = [ids[i] for i in np.argsort(score, kind='stable')]
    h = hashlib.sha256(json.dumps(order).encode()).hexdigest()[:16]
    json.dump({'order': order, 'hash': h, 'extra': extra or {}},
              open(os.path.join(CK, 'frozen_%s.json' % name), 'w'))
    return h


# ---------------- scoring (reads hidden labels) ----------------
def load_hidden():
    return json.load(open(os.path.join(DATA, 'pe19_hidden.json')))


def auc(pos_early, pos_late):
    a, b = np.array(pos_early), np.array(pos_late)
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    return float(((a[:, None] < b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def score_order(order, H):
    """Order: list of ids from early to late. Returns rho (Susa levels), AUC early
    sites, AUC Malyan (level 3 earlier than 2), n's, and a combined stat."""
    from scipy.stats import spearmanr
    n = len(order)
    pos = {p: i / max(n - 1, 1) for i, p in enumerate(order)}
    sl = [(pos[p], v) for p, v in H['susa_level'].items() if p in pos]
    rho = spearmanr([a for a, _ in sl], [b for _, b in sl])[0] if len(sl) > 3 else float('nan')
    early = [pos[p] for p in H['early_site'] if p in pos]
    rest = [pos[p] for p in pos if p not in H['early_site']]
    a_e = auc(early, rest)
    m0 = [pos[p] for p, v in H['malyan_level'].items() if p in pos and v == 0]
    m1 = [pos[p] for p, v in H['malyan_level'].items() if p in pos and v == 1]
    a_m = auc(m0, m1)
    comb = np.nansum([rho, 2 * (a_e - 0.5), 2 * (a_m - 0.5)])
    return {'rho_susa': rho, 'n_susa': len(sl), 'auc_early': a_e, 'n_early': len(early),
            'auc_malyan': a_m, 'n_mal': [len(m0), len(m1)], 'comb': float(comb)}


def perm_null(ids, H, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        o = list(rng.permutation(ids))
        out.append(score_order(o, H))
    return out
