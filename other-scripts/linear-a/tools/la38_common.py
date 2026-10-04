#!/usr/bin/env python3
"""LA-38 'are the borrowed values better than chance?': shared helpers.

A corpus is a list of words; each sign carries a (consonant, vowel) value. The value map is scored with
language-independent phonotactic measures and compared against relabelings of the same map:
  R1  consonant labels permuted among rows, vowel labels among columns (grid partition kept)
  R2a vowels permuted among the signs of each consonant row (consonants kept)
  R2b consonants permuted among the signs of each vowel column (vowels kept)
  R3  (C, V) values permuted among all valued signs
Measures (each oriented so that larger = more natural; direction fixed a priori except sameV):
  ocpC    -log(obs/exp) adjacent syllables with the same consonant (pure vowels not counted)
  ocpP    -log(obs/exp) adjacent syllables with consonants of the same place class
  sameV   -log(obs/exp) adjacent syllables with the same vowel (reported only; not in any composite unless it passes on LB)
  vinit   log ratio of the pure-vowel share word-initially vs elsewhere (onset principle)
  son     medial minus initial onset sonority (domain-initial strengthening)
  freqC   Spearman(row token count, cross-linguistic inventory frequency of the consonant)
  freqV   Spearman(column token count, cross-linguistic inventory frequency of the vowel)
  comp    -(bits per phoneme) of the C/V phoneme stream under a plug-in bigram model
  la21    AUC of the blind la21 same-row probability for pairs sharing a consonant (if rows available)
Linear B values enter only as the hypothesis under test (LA) or as the truth of the controls (LB, Cypriot).
"""
import os, sys, json, collections, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import la32_common as C

LA = os.path.join(HERE, '..')
CK = os.path.join(LA, 'data', 'la38_ckpt')
LOOPS = os.path.join(LA, 'loops')
os.makedirs(CK, exist_ok=True)

PLACE = dict(p='lab', m='lab', w='lab', b='lab', ph='lab', t='cor', d='cor', n='cor', s='cor', r='cor', l='cor',
             z='cor', tj='cor', rj='cor', k='dor', q='dor', g='dor', x='dor', j='pal', h='glo')
SON = dict(p=1, t=1, k=1, q=1, d=1, b=1, g=1, x=1, ph=1, tj=1, s=2, z=2, h=2, m=3, n=3, r=4, l=4, rj=4, j=5, w=5)
SON[''] = 6
# approximate share of phoneme inventories containing the segment (cross-linguistic inventory data, PHOIBLE-style)
XFREQ_C = dict(m=.96, k=.90, j=.90, p=.86, w=.82, n=.78, t=.68, s=.67, l=.68, b=.63, h=.63, g=.57, d=.45, r=.44,
               tj=.40, z=.30, ph=.20, q=.12, x=.10, rj=.05)
XFREQ_V = dict(i=.92, u=.88, a=.86, e=.61, o=.60)
MEAS = ['ocpC', 'ocpP', 'sameV', 'vinit', 'son', 'freqC', 'freqV', 'comp', 'la21']


# ---------------------------------------------------------------- values
def cv_of(s):
    """(C, V) for a sign name (LA upper or LB lower case); None = unvalued (logograms, *-signs, pa3, diphthongs)."""
    v = C.lb_cv(s.lower())
    if v is None or (v[0] and v[0] not in PLACE) or len(v[1]) != 1 or v[1] not in 'aeiou':
        return None   # pa3, diphthongs, au, and cluster signs (nwa, dwe, pte, ...) are unvalued
    return v


# ---------------------------------------------------------------- corpora
def la_words():
    return [r['w'] for r in C.la_words() if len(r['w']) >= 2]


def lb_units():
    return [r for r in C.lb_words() if len(r['w']) >= 2]


def lb_words():
    return [tuple(s.upper() for s in r['w']) for r in lb_units()]


def lb_draw(ntok, seed):
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for r in lb_units():
        by[r['doc']].append(tuple(s.upper() for s in r['w']))
    docs = sorted(by); rng.shuffle(docs)
    out, n = [], 0
    for d in docs:
        for w in by[d]:
            out.append(w); n += sum(cv_of(s) is not None for s in w)
        if n >= ntok:
            break
    return out


def cyp_words():
    return [tuple(s.upper() for s in w) for w in json.load(open(os.path.join(CK, 'cypriot_idalion.json'))) if len(w) >= 2]


def shuffle_words(words, seed):
    rng = np.random.default_rng(seed)
    out = []
    for w in words:
        w = list(w); rng.shuffle(w); out.append(tuple(w))
    return out


# ---------------------------------------------------------------- sufficient statistics
class Corpus:
    """counts over valued signs: u (all), I (true word-initial), F (true word-final), B (adjacent valued pairs)."""

    def __init__(self, words, values=None, name=''):
        self.name = name
        vals = {}
        for w in words:
            for s in w:
                if s not in vals:
                    v = (values or {}).get(s) if values is not None else cv_of(s)
                    if values is None:
                        v = cv_of(s)
                    vals[s] = v
        self.signs = sorted([s for s, v in vals.items() if v is not None])
        idx = {s: i for i, s in enumerate(self.signs)}
        S = len(self.signs)
        self.u = np.zeros(S); self.I = np.zeros(S); self.F = np.zeros(S); self.B = np.zeros((S, S))
        for w in words:
            ids = [idx.get(s) for s in w]
            for p, i in enumerate(ids):
                if i is None:
                    continue
                self.u[i] += 1
                if p == 0:
                    self.I[i] += 1
                if p == len(ids) - 1:
                    self.F[i] += 1
                if p + 1 < len(ids) and ids[p + 1] is not None:
                    self.B[i, ids[p + 1]] += 1
        self.true = [vals[s] for s in self.signs]
        self.Clab = sorted({v[0] for v in self.true}); self.Vlab = sorted({v[1] for v in self.true})
        self.C0 = np.array([self.Clab.index(v[0]) for v in self.true])
        self.V0 = np.array([self.Vlab.index(v[1]) for v in self.true])
        self.ntok = self.u.sum()
        self.rows21 = None

    def set_la21(self, signs, P):
        """blind same-row probabilities for the signs present in both."""
        ix = {s.upper(): i for i, s in enumerate(signs)}
        keep = [i for i, s in enumerate(self.signs) if s.upper() in ix]
        a, b = np.triu_indices(len(keep), 1)
        self.r21_i = np.array(keep)[a]; self.r21_j = np.array(keep)[b]
        pv = np.array([P[ix[self.signs[i].upper()], ix[self.signs[j].upper()]] for i, j in zip(self.r21_i, self.r21_j)])
        from scipy.stats import rankdata
        self.r21_rank = rankdata(pv)
        self.rows21 = True


# ---------------------------------------------------------------- relabelings
def relabel(corp, kind, N, rng, base=None):
    """arrays (N, S) of consonant and vowel label indices."""
    C0, V0 = (corp.C0, corp.V0) if base is None else base
    S = len(C0); nC, nV = len(corp.Clab), len(corp.Vlab)
    if kind == 'R1':
        pc = np.argsort(rng.random((N, nC)), 1); pv = np.argsort(rng.random((N, nV)), 1)
        return np.take_along_axis(pc, np.broadcast_to(C0, (N, S)), 1), np.take_along_axis(pv, np.broadcast_to(V0, (N, S)), 1)
    if kind == 'R3':
        perm = np.argsort(rng.random((N, S)), 1)
        return C0[perm], V0[perm]
    if kind in ('R2a', 'R2b'):
        grp = C0 if kind == 'R2a' else V0
        mov = V0 if kind == 'R2a' else C0
        out = np.broadcast_to(mov, (N, S)).copy()
        for g in np.unique(grp):
            ii = np.where(grp == g)[0]
            if len(ii) > 1:
                perm = np.argsort(rng.random((N, len(ii))), 1)
                out[:, ii] = mov[ii][perm]
        return (np.broadcast_to(C0, (N, S)).copy(), out) if kind == 'R2a' else (out, np.broadcast_to(V0, (N, S)).copy())
    raise ValueError(kind)


# ---------------------------------------------------------------- measures (vectorised over labelings)
def _spearman_rows(X, y):
    """Spearman between each row of X (N, k) and fixed y (k,), ignoring nothing."""
    from scipy.stats import rankdata
    rx = np.apply_along_axis(rankdata, 1, X); ry = rankdata(y)
    rx = rx - rx.mean(1, keepdims=True); ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum()); den[den == 0] = np.inf
    return (rx * ry).sum(1) / den


def _ent_bits(T):
    """code length (bits) of transitions in count tensor T (N, rows, cols) under plug-in conditionals."""
    rt = T.sum(2, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        L = np.where(T > 0, T * np.log2(np.where(rt > 0, rt, 1) / np.where(T > 0, T, 1)), 0.0)
    return L.sum((1, 2))


def measures(corp, Cn, Vn):
    """dict measure -> (N,) array."""
    Cn = np.atleast_2d(Cn); Vn = np.atleast_2d(Vn)
    N, S = Cn.shape
    nC, nV = len(corp.Clab), len(corp.Vlab)
    Cl = np.array(corp.Clab, dtype=object)
    empty = corp.Clab.index('') if '' in corp.Clab else -1
    place_of = np.array([['lab', 'cor', 'dor', 'pal', 'glo'].index(PLACE[c]) if c else -1 for c in corp.Clab])
    son_of = np.array([SON[c] for c in corp.Clab], float)
    B = corp.B; tot = B.sum()
    rowsum = B.sum(1); colsum = B.sum(0)
    out = {}
    OC = np.zeros((N, S, nC)); OC[np.arange(N)[:, None], np.arange(S)[None, :], Cn] = 1
    OV = np.zeros((N, S, nV)); OV[np.arange(N)[:, None], np.arange(S)[None, :], Vn] = 1
    # consonant-level pair counts (N, nC, nC)
    BC = np.einsum('nsc,st,ntd->ncd', OC, B, OC, optimize=True)
    outC = np.einsum('nsc,s->nc', OC, rowsum); inC = np.einsum('nsc,s->nc', OC, colsum)
    mask = np.ones(nC, bool)
    if empty >= 0:
        mask[empty] = False
    obs = (BC.diagonal(axis1=1, axis2=2) * mask).sum(1)
    exp = (outC[:, mask] * inC[:, mask]).sum(1) / tot
    out['ocpC'] = -np.log((obs + .5) / (exp + .5))
    P = np.zeros((nC, 5));
    for c in range(nC):
        if place_of[c] >= 0:
            P[c, place_of[c]] = 1
    BP = np.einsum('ncd,cp,dq->npq', BC, P, P); oP = np.einsum('nc,cp->np', outC, P); iP = np.einsum('nc,cp->np', inC, P)
    obs = BP.diagonal(axis1=1, axis2=2).sum(1); exp = (oP * iP).sum(1) / tot
    out['ocpP'] = -np.log((obs + .5) / (exp + .5))
    BV = np.einsum('nsv,st,ntw->nvw', OV, B, OV, optimize=True)
    outV = np.einsum('nsv,s->nv', OV, rowsum); inV = np.einsum('nsv,s->nv', OV, colsum)
    obs = BV.diagonal(axis1=1, axis2=2).sum(1); exp = (outV * inV).sum(1) / tot
    out['sameV'] = -np.log((obs + .5) / (exp + .5))
    # onset principle
    if empty >= 0:
        isV = (Cn == empty).astype(float)
        nonI = corp.u - corp.I
        a = isV @ corp.I + .5; b = corp.I.sum() + 1; c = isV @ nonI + .5; d = nonI.sum() + 1
        out['vinit'] = np.log((a / b) / (c / d))
    else:
        out['vinit'] = np.zeros(N)
    so = son_of[Cn]
    nonI = corp.u - corp.I
    out['son'] = (so @ nonI) / nonI.sum() - (so @ corp.I) / corp.I.sum()
    # inventory-frequency agreement
    tokC = np.einsum('nsc,s->nc', OC, corp.u); tokV = np.einsum('nsv,s->nv', OV, corp.u)
    cm = [i for i, c in enumerate(corp.Clab) if c in XFREQ_C]
    out['freqC'] = _spearman_rows(tokC[:, cm], np.array([XFREQ_C[corp.Clab[i]] for i in cm]))
    out['freqV'] = _spearman_rows(tokV, np.array([XFREQ_V[v] for v in corp.Vlab]))
    # compressibility of the phoneme stream: symbols 0..nC-1 consonants, nC..nC+nV-1 vowels, last '#'
    K = nC + nV + 1; H = K - 1
    T = np.zeros((N, K, K))
    # within-syllable C -> V (pure vowels emit no consonant)
    CV = np.einsum('nsc,s,nsv->ncv', OC, corp.u, OV)
    if empty >= 0:
        CV[:, empty, :] = 0
    T[:, :nC, nC:nC + nV] = CV
    # V -> next onset: consonant, or the vowel itself for a pure-vowel sign
    VC = np.einsum('nsv,st,ntc->nvc', OV, B, OC, optimize=True)
    if empty >= 0:
        e = (Cn == empty).astype(float)
        VV = np.einsum('nsv,st,nt,ntw->nvw', OV, B, e, OV, optimize=True)
        VC[:, :, empty] = 0
        T[:, nC:nC + nV, nC:nC + nV] += VV
    T[:, nC:nC + nV, :nC] += VC
    # word start -> first onset ; last vowel -> '#'
    Ic = np.einsum('nsc,s->nc', OC, corp.I)
    if empty >= 0:
        Iv = np.einsum('nsv,s,ns->nv', OV, corp.I, (Cn == empty).astype(float))
        Ic[:, empty] = 0
        T[:, H, nC:nC + nV] += Iv
    T[:, H, :nC] += Ic
    T[:, nC:nC + nV, H] += np.einsum('nsv,s->nv', OV, corp.F)
    nph = T.sum((1, 2))
    out['comp'] = -_ent_bits(T) / nph
    if corp.rows21 is not None:
        same = (Cn[:, corp.r21_i] == Cn[:, corp.r21_j])
        n1 = same.sum(1); n0 = same.shape[1] - n1
        out['la21'] = ((same * corp.r21_rank).sum(1) - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)
    else:
        out['la21'] = np.full(N, np.nan)
    return out


def null_measures(corp, kind, N, seed, batch=2000, base=None):
    rng = np.random.default_rng(seed)
    acc = collections.defaultdict(list)
    for b0 in range(0, N, batch):
        Cn, Vn = relabel(corp, kind, min(batch, N - b0), rng, base)
        for k, v in measures(corp, Cn, Vn).items():
            acc[k].append(v)
    return {k: np.concatenate(v) for k, v in acc.items()}


def compare(obs, null, use):
    """per-measure z and upper-tail P (share of null >= obs), plus composite of the 'use' measures."""
    res = {}
    zs_obs, zs_null = [], []
    for k in MEAS:
        o = float(obs[k][0]) if np.ndim(obs[k]) else float(obs[k]); x = null[k]
        if np.isnan(o):
            continue
        mu, sd = x.mean(), x.std()
        if sd < 1e-12:
            res[k] = dict(obs=o, z=np.nan, p=np.nan, inv=True); continue
        res[k] = dict(obs=o, z=(o - mu) / sd, p=float(((x >= o - 1e-12).sum() + 1) / (len(x) + 1)), inv=False)
        if k in use:
            zs_obs.append((o - mu) / sd); zs_null.append((x - mu) / sd)
    if zs_obs:
        co = np.mean(zs_obs); cn = np.mean(zs_null, 0)
        res['COMP'] = dict(obs=co, z=(co - cn.mean()) / cn.std(), p=float(((cn >= co - 1e-12).sum() + 1) / (len(cn) + 1)), inv=False)
    return res


def fmt_res(res):
    parts = []
    for k, r in res.items():
        if r.get('inv'):
            parts.append(f"{k} inv")
        else:
            parts.append(f"{k} z{r['z']:+.1f} P{r['p']:.3g}")
    return '; '.join(parts)


def la21_for(tag):
    r = C.la21_rows()
    s, P, n = r[tag]
    return s, P


def build(tag, words):
    corp = Corpus(words, name=tag)
    return corp
