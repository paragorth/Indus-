#!/usr/bin/env python3
"""LA-43 'do the values predict words not yet seen?': shared helpers.

A sound-aware sign-bigram word model is trained on one part of a corpus (one site, or the early publications)
and asked to predict the words of another part (other sites, later publications). 'Sound-aware' means the model
pools statistics between signs that share a consonant (row) or a vowel (column) under a value map:

  P(t | c) = w_K  KN(t | c)                         interpolated absolute-discount sign bigram
           + w_U  U(t)                              sign unigram
           + w_RR P(row t | row c) E_row(t)         consonant -> consonant
           + w_CC P(col t | col c) E_col(t)         vowel -> vowel
           + w_VR P(row t | col c) E_row(t)         previous vowel -> next consonant
           + w_RC P(col t | row c) E_col(t)         previous consonant -> next vowel
weights by EM on a development fifth of the training documents. Word boundaries are a class of their own;
unvalued signs (logograms, *-signs) are singleton rows and columns. The 'none' baseline has only K and U.

Scores on the test part (new word types only, each once):
  bits   mean -log2 P per symbol (signs + end of word)
  mask   mean -log2 of the true sign when one sign is masked and every sign is a candidate
         (score = log P(x | prev) + log P(next | x), normalised over candidates)
  mrr    mean reciprocal rank of the masked true sign
  auc    real word vs its sign-permuted decoys (20 per word, fixed seed)
Relabelings (from la38): R2a vowels permuted inside rows, R2b consonants inside columns, R3 all values.
Gain = score(none) - score(map); the map's gain is compared with the gains of the relabeled maps.
"""
import os, sys, json, hashlib, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import la38_common as L38
import la32_common as C32
import la15_common as L15

LA = os.path.join(HERE, '..')
CK = os.path.join(LA, 'data', 'la43_ckpt')
LOOPS = os.path.join(LA, 'loops')
os.makedirs(CK, exist_ok=True)
COMPS = ['K', 'U', 'RR', 'CC', 'VR', 'RC']
SCORES = ['bits', 'mask', 'mrr', 'auc']


# ---------------------------------------------------------------- corpora: list of (word tuple, doc, group)
def la_units():
    src = L15._pub_source()
    out = []
    for r in C32.la_words():
        w = tuple(s for s in r['w'] if s)
        if not w:
            continue
        out.append(dict(w=w, doc=r['doc'], site=r['site'], pub=src.get(r['doc'], 'blank')))
    return out


def lb_units():
    return [dict(w=tuple(s.upper() for s in r['w']), doc=r['doc'], site=r['site']) for r in C32.lb_words()]


def draw_docs(units, ntok, seed):
    """whole documents in random order until ntok sign tokens."""
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for u in units:
        by[u['doc']].append(u)
    docs = sorted(by); rng.shuffle(docs)
    out, n = [], 0
    for d in docs:
        out += by[d]; n += sum(len(u['w']) for u in by[d])
        if n >= ntok:
            break
    return out


# ---------------------------------------------------------------- the value grid
class Grid:
    """sign inventory (train + test), true (C,V) map, label arrays usable by la38 relabel()."""

    def __init__(self, signs, values=None):
        self.signs = sorted(signs)
        self.ix = {s: i for i, s in enumerate(self.signs)}
        vals = {s: (values.get(s) if values is not None else L38.cv_of(s)) for s in self.signs}
        self.valued = np.array([i for i, s in enumerate(self.signs) if vals[s] is not None])
        tv = [vals[self.signs[i]] for i in self.valued]
        self.true = tv
        self.Clab = sorted({v[0] for v in tv}); self.Vlab = sorted({v[1] for v in tv})
        self.C0 = np.array([self.Clab.index(v[0]) for v in tv]); self.V0 = np.array([self.Vlab.index(v[1]) for v in tv])
        self.S = len(self.signs)

    def classes(self, Cv, Vv):
        """full row/col class vectors (S+1,) from labels of valued signs; boundary = last index."""
        S = self.S; nC = len(self.Clab); nV = len(self.Vlab)
        row = np.empty(S + 1, int); col = np.empty(S + 1, int)
        un = np.setdiff1d(np.arange(S), self.valued)
        row[self.valued] = Cv; col[self.valued] = Vv
        row[un] = nC + np.arange(len(un)); col[un] = nV + np.arange(len(un))
        row[S] = nC + len(un); col[S] = nV + len(un)
        return row, col

    def none_classes(self):
        S = self.S
        return np.arange(S + 1), np.arange(S + 1)

    def relabels(self, kind, N, seed):
        rng = np.random.default_rng(seed)
        return L38.relabel(self, kind, N, rng)


def bigrams(words, ix, S):
    """count matrix (S+1, S+1), index S = boundary (context BOS, target EOS); signs not in ix are skipped."""
    B = np.zeros((S + 1, S + 1))
    for w in words:
        seq = [S] + [ix[s] for s in w if s in ix] + [S]
        for a, b in zip(seq[:-1], seq[1:]):
            B[a, b] += 1
    return B


# ---------------------------------------------------------------- the model
def kn_bigram(B, d=0.75):
    S1 = B.shape[0]
    uni = B.sum(0) + 0.5; uni[:] = uni / uni.sum()
    rs = B.sum(1, keepdims=True)
    nz = (B > 0).sum(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.where(rs > 0, (np.maximum(B - d, 0) + d * nz * uni[None]) / np.where(rs > 0, rs, 1), uni[None])
    return P, np.broadcast_to(uni, (S1, S1))


def class_comp(B, ctx_cls, tgt_cls, alpha=0.5, beta=0.1):
    """P(cls t | cls' c) * E(t | cls t)."""
    n1 = ctx_cls.max() + 1; n2 = tgt_cls.max() + 1
    O1 = np.zeros((len(ctx_cls), n1)); O1[np.arange(len(ctx_cls)), ctx_cls] = 1
    O2 = np.zeros((len(tgt_cls), n2)); O2[np.arange(len(tgt_cls)), tgt_cls] = 1
    M = O1.T @ B @ O2
    T = (M + alpha) / (M + alpha).sum(1, keepdims=True)
    u = B.sum(0) + beta
    tot = np.bincount(tgt_cls, weights=u, minlength=n2)
    E = u / tot[tgt_cls]
    return T[ctx_cls][:, tgt_cls] * E[None, :]


_KN = {}


def components(B, row, col, sound=True):
    key = id(B)
    if key not in _KN:
        _KN[key] = (B, kn_bigram(B))
    K, U = _KN[key][1]
    if not sound:
        return [K, U]
    return [K, U, class_comp(B, row, row), class_comp(B, col, col), class_comp(B, col, row), class_comp(B, row, col)]


def em_weights(comps, Bdev, iters=30):
    a, b = np.nonzero(Bdev); n = Bdev[a, b]
    X = np.stack([c[a, b] for c in comps])          # (k, m)
    w = np.full(len(comps), 1.0 / len(comps))
    for _ in range(iters):
        R = w[:, None] * X; R /= R.sum(0, keepdims=True)
        w = (R * n).sum(1); w /= w.sum()
    return w


def fit(Btr_sub, Bdev, Btr, row, col, sound=True):
    w = em_weights(components(Btr_sub, row, col, sound), Bdev)
    comps = components(Btr, row, col, sound)
    P = sum(wi * c for wi, c in zip(w, comps))
    return P, w


# ---------------------------------------------------------------- test material and scores
class Test:
    def __init__(self, test_words, train_types, ix, S, seed=4343, ndec=20):
        rng = np.random.default_rng(seed)
        types = sorted({w for w in test_words if w not in train_types and all(s in ix for s in w)})
        self.types = types
        self.B = bigrams(types, ix, S)
        self.nsym = self.B.sum()
        tri = []
        for w in types:
            seq = [S] + [ix[s] for s in w] + [S]
            for p in range(1, len(seq) - 1):
                tri.append((seq[p - 1], seq[p], seq[p + 1]))
        self.tri = np.array(tri, int).reshape(-1, 3)
        # decoys: sign permutations of the word (distinct from it)
        real, dec, own = [], [], []
        for k, w in enumerate(types):
            if len(w) < 2 or len(set(w)) < 2:
                continue
            seq = [ix[s] for s in w]
            for _ in range(ndec):
                for _t in range(20):
                    p = list(rng.permutation(seq))
                    if p != seq:
                        break
                dec.append(p); own.append(len(real))
            real.append(seq)
        self.real = real; self.dec = dec; self.own = np.array(own, int)
        S1 = S + 1

        def pairs(seqs):
            a, b, g = [], [], []
            for k, q in enumerate(seqs):
                q = [S] + q + [S]
                a += q[:-1]; b += q[1:]; g += [k] * (len(q) - 1)
            return np.array(a, int), np.array(b, int), np.array(g, int)
        self.ra, self.rb, self.rg = pairs(real)
        self.da, self.db, self.dg = pairs(dec)
        self.S = S

    def digest(self):
        h = hashlib.sha256()
        h.update(json.dumps(['-'.join(w) for w in self.types]).encode())
        h.update(json.dumps([[int(x) for x in d] for d in self.dec]).encode())
        return h.hexdigest()[:16]

    def score(self, P):
        lP = np.log2(P)
        out = {}
        out['bits'] = float(-(self.B * lP).sum() / self.nsym)
        S = self.S
        if len(self.tri):
            M = lP[self.tri[:, 0], :S] + lP[:S, self.tri[:, 2]].T   # (n, S) candidates
            mx = M.max(1, keepdims=True)
            lse = mx[:, 0] + np.log2(np.exp2(M - mx).sum(1))
            tr = M[np.arange(len(M)), self.tri[:, 1]]
            out['mask'] = float(-(tr - lse).mean())
            rank = (M > tr[:, None]).sum(1) + 0.5 * ((M == tr[:, None]).sum(1) - 1) + 1
            out['mrr'] = -float((1.0 / rank).mean())     # negated: lower = better for every score
        else:
            out['mask'] = out['mrr'] = np.nan
        if len(self.real):
            lr = np.bincount(self.rg, weights=lP[self.ra, self.rb], minlength=len(self.real))
            ld = np.bincount(self.dg, weights=lP[self.da, self.db], minlength=len(self.dec))
            r = lr[self.own]
            out['auc'] = -float(((r > ld) + 0.5 * (r == ld)).mean())   # negated
        else:
            out['auc'] = np.nan
        return out


# ---------------------------------------------------------------- one experiment
class Experiment:
    """train/test split with dev fifth; true map, none, relabeled maps."""

    def __init__(self, train, test, values=None, seed=0, grid_signs=None):
        signs = set(s for u in train for s in u['w']) | set(s for u in test for s in u['w'])
        self.grid = G = Grid(signs if grid_signs is None else grid_signs, values)
        S = G.S
        rng = np.random.default_rng(1000 + seed)
        docs = sorted({u['doc'] for u in train}); rng.shuffle(docs)
        dev = set(docs[:max(1, len(docs) // 5)])
        trw = [u['w'] for u in train]
        self.Btr = bigrams(trw, G.ix, S)
        self.Bsub = bigrams([u['w'] for u in train if u['doc'] not in dev], G.ix, S)
        self.Bdev = bigrams([u['w'] for u in train if u['doc'] in dev], G.ix, S)
        self.test = Test([u['w'] for u in test], set(trw), G.ix, S)
        self.ntr = int(self.Btr.sum()); self.nte_types = len(self.test.types)

    def run_map(self, Cv, Vv, sound=True):
        row, col = self.grid.classes(Cv, Vv)
        P, w = fit(self.Bsub, self.Bdev, self.Btr, row, col, sound)
        return P, w

    def frozen(self):
        """predictions of the true map and of 'none', hashed before any score is computed."""
        G = self.grid
        Pt, wt = self.run_map(G.C0, G.V0)
        Pn, wn = self.run_map(G.C0, G.V0, sound=False)
        h = hashlib.sha256(np.round(Pt, 12).tobytes() + np.round(Pn, 12).tobytes()).hexdigest()[:16]
        return Pt, wt, Pn, wn, h

    def null(self, kind, N, seed):
        G = self.grid
        Cn, Vn = G.relabels(kind, N, seed)
        out = {k: np.empty(N) for k in SCORES}
        for i in range(N):
            P, _ = self.run_map(Cn[i], Vn[i])
            s = self.test.score(P)
            for k in SCORES:
                out[k][i] = s[k]
        return out


def summarise(sn, st, null):
    """gains (none - map; positive = map helps) and position of the true map among relabelings (lower score better)."""
    res = {}
    for k in SCORES:
        x = null[k]; o = st[k]
        if np.isnan(o):
            continue
        res[k] = dict(true=o, none=sn[k], gain=sn[k] - o, null_gain=float(np.mean(sn[k] - x)),
                      z=float((x.mean() - o) / (x.std() + 1e-12)), p=float(((x <= o + 1e-12).sum() + 1) / (len(x) + 1)))
    return res


def fmt(res, keys=SCORES):
    out = []
    for k in keys:
        if k in res:
            r = res[k]
            out.append(f"{k} gain {r['gain']:+.3f} (null {r['null_gain']:+.3f}) z{r['z']:+.1f} P{r['p']:.3g}")
    return '; '.join(out)
