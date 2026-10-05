#!/usr/bin/env python3
"""LA-59 'human ears are the answer key': shared helpers.

Human confusion keys (measured outside any script):
  consonants: Miller & Nicely (1955) 16 English consonants in noise, symmetrised similarities as distributed
              in the MAPCLUS archive (netlib.org/mds/mapclus.shar; Shepard 1972 normalisation). Secondary key:
              the Hubert (1972) normalisation in the R package semds (data 'Miller').
  vowels:     F1/F2 of five-vowel systems (Greek, Fourakis et al. 1999; Spanish, Bradlow 1995; Hebrew, Aronson
              et al. 1996; as distributed in the R package phonTools), Bark distance -> similarity.
Two extra consonant states: 0 (no onset) and X (a sonorant / anything outside the 16), joined to every
consonant at the 10th percentile Miller-Nicely similarity.

Kernel K[(c,v),(c',v')] = SC[c,c'] * SV[v,v'] + eps, diagonals 1.
Model of an alternation graph: P(edge a-b) proportional to d_a d_b K(s_a, s_b). Gain (nats per edge) of an
assignment s over the degree-only model: sum w_ab log K_ab / W - log( sum d_a d_b K_ab / sum d_a d_b ).
No Linear B value is used for Linear A; LB values enter only the Linear B control's scoring.
"""
import os, sys, re, json, collections, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la32_common import la_words, lb_words, one_sign_pairs, lb_cv, DATA, LOOPS

CK = os.path.join(DATA, 'la59_ckpt')
os.makedirs(CK, exist_ok=True)

MN_LABELS = ['p', 't', 'k', 'f', 'T', 's', 'S', 'b', 'd', 'g', 'v', 'D', 'z', 'Z', 'm', 'n']
# features of the 16: place (lab, cor, dor), manner (stop, fric, nas), voice (0/1)
PLACE = dict(p='lab', b='lab', f='lab', v='lab', m='lab', t='cor', d='cor', T='cor', D='cor', s='cor', z='cor',
             S='cor', Z='cor', n='cor', k='dor', g='dor')
MANNER = dict(p='stop', t='stop', k='stop', b='stop', d='stop', g='stop', f='fric', T='fric', s='fric', S='fric',
              v='fric', D='fric', z='fric', Z='fric', m='nas', n='nas')
VOICE = dict(p=0, t=0, k=0, f=0, T=0, s=0, S=0, b=1, d=1, g=1, v=1, D=1, z=1, Z=1, m=1, n=1)
VOWELS = ['a', 'e', 'i', 'o', 'u']
C_LABELS = MN_LABELS + ['0', 'X']


def mn_matrix(which='mapclus'):
    if which == 'mapclus':
        txt = open(os.path.join(CK, 'mapclus.shar')).read().split('MILLER-NICELY DATA FOR CONFUSIONS BETWEEN 16 CONSONANTS')[1]
        lines = txt.split('\n')[3:18]
        S = np.eye(16)
        for i, l in enumerate(lines):
            vals = [float(x) for x in l[:76].split()]
            assert len(vals) == i + 1, (i, l)
            for j, v in enumerate(vals):
                S[i + 1, j] = S[j, i + 1] = v
        return S
    import pyreadr
    D = pyreadr.read_r(os.path.join(CK, 'semds/data/Miller.rda'))['Miller'].values.astype(float)
    Sim = 1 - (D + D.T) / 2
    # rescale off-diagonal to the mapclus range so the two keys are comparable
    off = ~np.eye(16, dtype=bool)
    m = mn_matrix('mapclus')
    r = np.argsort(np.argsort(Sim[off]))
    S = np.eye(16)
    S[off] = np.sort(m[off])[r]
    return (S + S.T) / 2


def consonant_key(which='mapclus'):
    S16 = mn_matrix(which)
    off = S16[~np.eye(16, dtype=bool)]
    lo = float(np.quantile(off, 0.10))
    S = np.full((18, 18), lo)
    S[:16, :16] = S16
    np.fill_diagonal(S, 1.0)
    return S


def bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def vowel_key(scale_nearest=0.30):
    import pyreadr
    F = []
    for nm in ('f99', 'b95', 'a96'):
        d = pyreadr.read_r(os.path.join(CK, 'phonTools/data/%s.rda' % nm))[nm]
        d = d[d['sex'].astype(str).isin(['m', '1', '1.0'])]
        F.append(np.array([[bark(float(d[d.vowel == v].f1.iloc[0])), bark(float(d[d.vowel == v].f2.iloc[0]))] for v in VOWELS]))
    F = np.mean(F, 0)
    D = np.sqrt(((F[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    off = D[~np.eye(5, dtype=bool)]
    tau = off.min() / -np.log(scale_nearest)       # nearest vowel pair gets similarity scale_nearest (a priori, not fitted)
    S = np.exp(-D / tau)
    np.fill_diagonal(S, 1.0)
    return S, D


def kernel(SC, SV, eps=0.005):
    return np.kron(SC, SV) + eps     # state index = c * 5 + v


def shuffle_offdiag(S, rng, nfix=None):
    """random symmetric matrix with the same diagonal and the same off-diagonal values (human structure destroyed)."""
    n = S.shape[0]
    iu = np.triu_indices(n, 1)
    v = S[iu].copy(); rng.shuffle(v)
    T = np.eye(n) * np.diag(S)
    T[iu] = v; T.T[iu] = v
    return T


# ------------------------------------------------------------------ alternation graphs
def edges_to_graph(E, signs=None, weight='type'):
    W = collections.Counter()
    for e in E:
        if e['a'] == e['b']:
            continue
        W[tuple(sorted((e['a'], e['b'])))] += 1
    if signs is None:
        signs = sorted({s for k in W for s in k})
    idx = {s: i for i, s in enumerate(signs)}
    A = np.zeros((len(signs), len(signs)))
    for (a, b), w in W.items():
        if a in idx and b in idx:
            A[idx[a], idx[b]] += w; A[idx[b], idx[a]] += w
    return signs, A


def ensemble_graph(words, n_settings, rng, groups=('site', None), doc_frac=0.8, keep_settings=False):
    """Consensus alternation graph over many random extraction settings (min length, position class,
    grouping, rare/frequent filter, document subsample). Weight = mean count over settings."""
    docs = sorted({r['doc'] for r in words})
    acc = collections.Counter(); settings = []
    for k in range(n_settings):
        minlen = int(rng.choice([2, 2, 3, 3, 4]))
        pos = str(rng.choice(['all', 'all', 'final', 'nonfinal']))
        grp = groups[rng.integers(len(groups))]
        mintok = int(rng.choice([1, 1, 2]))
        keep = set(rng.choice(docs, int(len(docs) * doc_frac), replace=False))
        ws = [r for r in words if r['doc'] in keep]
        E = one_sign_pairs(ws, minlen, grp, pos)
        if mintok > 1:
            E = [e for e in E if max(e['n1'], e['n2']) >= mintok]
        for e in E:
            if e['a'] != e['b']:
                acc[tuple(sorted((e['a'], e['b'])))] += 1
        settings.append((minlen, pos, grp, mintok))
    W = {k: v / n_settings for k, v in acc.items()}
    return (W, settings) if keep_settings else W


def graph_from_W(W, min_deg=0.0, signs=None):
    if signs is None:
        deg = collections.Counter()
        for (a, b), w in W.items():
            deg[a] += w; deg[b] += w
        signs = sorted(s for s in deg if deg[s] >= min_deg)
    idx = {s: i for i, s in enumerate(signs)}
    A = np.zeros((len(signs), len(signs)))
    for (a, b), w in W.items():
        if a in idx and b in idx:
            A[idx[a], idx[b]] += w; A[idx[b], idx[a]] += w
    return signs, A


# ------------------------------------------------------------------ scoring and fitting
# Mixture model: P(edge a-b) = lam * d_a d_b / D + (1 - lam) * d_a d_b K_ab / Z  (lam = share of alternations
# unrelated to sound). Gain = mean over edge weight of log(lam + (1 - lam) K_ab D / Z), >= 0 at lam = 1.
LAMS = np.array([0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.92, 0.97, 1.0])

def _parts(A, s, K, d=None):
    d = A.sum(1) if d is None else d
    iu = np.triu_indices(len(s), 1)
    dd = np.outer(d, d)[iu]
    Kv = K[s[iu[0]], s[iu[1]]]
    return A[iu], Kv, dd

def gain(A, s, K, d=None, lam=None, ret_lam=False):
    w, Kv, dd = _parts(A, s, K, d)
    Wt = w.sum()
    if Wt <= 0:
        return (0.0, 1.0) if ret_lam else 0.0
    r = Kv * dd.sum() / (dd * Kv).sum()
    lams = LAMS if lam is None else np.array([lam])
    g = np.array([(w * np.log(l + (1 - l) * r + 1e-300)).sum() / Wt for l in lams])
    k = int(np.argmax(g))
    return (float(g[k]), float(lams[k])) if ret_lam else float(g[k])

def gain_heldout(Atr_s, Ate, s, K, lam):
    """score held-out edges Ate with the train-fitted assignment and lam; degrees from the held-out graph."""
    return gain(Ate, s, K, lam=lam)

def gain_batch(A, S, K):
    d = A.sum(1); iu = np.triu_indices(A.shape[0], 1)
    w = A[iu]; nz = w > 0; dd = np.outer(d, d)[iu]; Wt = w.sum(); D = dd.sum()
    out = np.empty(len(S))
    for k in range(0, len(S), 256):
        blk = S[k:k + 256]
        Kv = K[blk[:, iu[0]], blk[:, iu[1]]]
        Z = (Kv * dd).sum(1)
        r = Kv[:, nz] * (D / Z)[:, None]
        g = np.stack([(np.log(l + (1 - l) * r + 1e-300) * w[nz]).sum(1) / Wt for l in LAMS], 1)
        out[k:k + 256] = g.max(1)
    return out

def anneal(A, K, rng, sweeps=40, T0=3.0, T1=0.03, init=None, allowed=None, lam=None):
    """Gibbs annealing on the total log-likelihood of the mixture model; lam re-fitted after each sweep
    (or fixed). Exact conditional for each sign (Z recomputed for every candidate state)."""
    n = A.shape[0]
    d = A.sum(1)
    states = np.arange(K.shape[0]) if allowed is None else np.asarray(allowed)
    s = rng.choice(states, n) if init is None else init.copy()
    iu = np.triu_indices(n, 1)
    D = np.outer(d, d)[iu].sum()
    nzp = [(i, j) for i, j in zip(*iu) if A[i, j] > 0]
    ei = np.array([p[0] for p in nzp]); ej = np.array([p[1] for p in nzp]); ew = A[ei, ej]
    l = 0.85 if lam is None else lam
    for sw in range(sweeps):
        T = T0 * (T1 / T0) ** (sw / max(1, sweeps - 1))
        for a in rng.permutation(n):
            mask = np.ones(n, bool); mask[a] = False
            col = K[np.ix_(states, s[mask])]                       # (S x n-1)
            Kab = K[s, :][:, s]
            Z = (np.outer(d, d) * Kab)[iu].sum()
            zold = (K[s[a], s[mask]] * d[mask]).sum() * d[a]
            Zc = Z - zold + col @ d[mask] * d[a]                   # (S,)
            oth = (ei != a) & (ej != a)
            Ko = K[s[ei[oth]], s[ej[oth]]]
            f_oth = (np.log(l + (1 - l) * Ko[None, :] * (D / Zc)[:, None]) * ew[oth][None, :]).sum(1)
            nb = np.where(A[a] > 0)[0]
            Kn = K[np.ix_(states, s[nb])]
            f_a = (np.log(l + (1 - l) * Kn * (D / Zc)[:, None]) * A[a, nb][None, :]).sum(1)
            obj = f_oth + f_a
            p = np.exp((obj - obj.max()) / T); p /= p.sum()
            s[a] = states[rng.choice(len(states), p=p)]
        if lam is None:
            g, l = gain(A, s, K, ret_lam=True)
            l = min(max(l, 0.2), 0.97)
    g, l = gain(A, s, K, lam=lam, ret_lam=True)
    return s, g, l


def best_of(A, K, rng, restarts=6, sweeps=60):
    best = None
    for r in range(restarts):
        s, g, l = anneal(A, K, rng, sweeps)
        if best is None or g > best[1]:
            best = (s, g, l)
    return best


def cv(state):
    return C_LABELS[state // 5], VOWELS[state % 5]


def rewire(A, rng, nswap=None):
    """degree-preserving rewiring of a weighted graph by stub shuffling of unit edges (weights rounded to 1/100)."""
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    w = np.round(A[iu] * 100).astype(int)
    stubs = []
    for (i, j, k) in zip(iu[0], iu[1], w):
        if k:
            stubs += [i, j] * 0 + [i] * k + [j] * k
    stubs = np.array(stubs)
    m = len(stubs) // 2
    for _ in range(50):
        rng.shuffle(stubs)
        a, b = stubs[:m], stubs[m:2 * m]
        if (a == b).mean() < 0.02:
            break
    B = np.zeros_like(A)
    ok = a != b
    np.add.at(B, (a[ok], b[ok]), 0.01); np.add.at(B, (b[ok], a[ok]), 0.01)
    return B


def sample_planted(d, s_true, K, n_edges, rng, share=1.0):
    """plant an alternation graph: edges drawn with P ~ d_a d_b K(s_a,s_b) (share) else ~ d_a d_b."""
    n = len(d); iu = np.triu_indices(n, 1)
    dd = np.outer(d, d)[iu]
    pk = dd * K[np.ix_(s_true, s_true)][iu]; pk /= pk.sum()
    p0 = dd / dd.sum()
    p = share * pk + (1 - share) * p0
    c = rng.multinomial(n_edges, p)
    B = np.zeros((n, n)); B[iu] = c; B.T[iu] = c
    return B


def co_assign_agreement(s_fit, truth, comp):
    """AUC that 'same fitted component' predicts 'same true label' over sign pairs with defined truth."""
    idx = [i for i, t in enumerate(truth) if t is not None]
    same_t = []; same_f = []
    for i, j in itertools.combinations(idx, 2):
        same_t.append(truth[i] == truth[j]); same_f.append(comp(s_fit[i]) == comp(s_fit[j]))
    same_t = np.array(same_t); same_f = np.array(same_f)
    if same_t.sum() == 0 or (~same_t).sum() == 0:
        return np.nan
    # Matthews-like: P(same fit | same truth) - P(same fit | diff truth)
    return float(same_f[same_t].mean() - same_f[~same_t].mean())


def split_W(W, rng, frac=0.5):
    tr, te = {}, {}
    for k, w in W.items():
        (tr if rng.random() < frac else te)[k] = w
    return tr, te
