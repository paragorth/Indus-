#!/usr/bin/env python3
"""LA-42 'complete the grid': shared helpers.

Place a sign with no value in the consonant x vowel grid of the valued signs. Candidates: every existing consonant row
(pure vowels included) x every vowel column, plus a NEW row ('@') x every vowel. Each candidate is scored by
  phonotactic (la38-style, language independent): local OCP counts (same consonant / same place with neighbours),
      onset principle (pure vowel word-initial vs medial), phoneme-stream code length (bits, plug-in bigram),
      same-vowel adjacency;
  grid structure: empty-cell indicator, log row size, log column size, new-row indicator;
  near-interchangeability (la37-style, no values): mean context similarity (PPMI cosine + one-slot swaps,
      rank-normalised) of the sign with the members of the candidate row and of the candidate column;
  blind la21 rows (optional, LA and full-LB runs only): mean same-row probability with the candidate row's members.
A conditional-logit (softmax over candidates) combines the features; weights are fitted on Linear B leave-one-out
cases only and then frozen for Linear A. Linear B values are the hypothesis for LA's known signs (la38) and the truth
in the LB control. Published proposals for the unknown signs are never an input.
"""
import os, sys, json, collections
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import la32_common as C
import la38_common as L
import la37_common as L37

LA = os.path.join(HERE, '..')
CK = os.path.join(LA, 'data', 'la42_ckpt')
LOOPS = os.path.join(LA, 'loops')
os.makedirs(CK, exist_ok=True)

NEW = '@'
L.PLACE[NEW] = 'glo'   # only for la38.measures (unused place feature there); local place map below
PL = dict(L.PLACE); PL[NEW] = 'new'
L.SON[NEW] = 2
PLACES = ['lab', 'cor', 'dor', 'pal', 'glo', 'new']
VOW = 'aeiou'
HID = 'HID'
DROP = 'XX'       # thinned-away tokens of the hidden sign: unvalued, word stream breaks there

FEATS = ['bits', 'ocpC', 'ocpP', 'vI', 'vM', 'sameV', 'empty', 'lrow', 'lcol', 'new', 'simC', 'simV', 'r21']


# ---------------------------------------------------------------- corpora
def la_units():
    return [r for r in C.la_words() if len(r['w']) >= 2]


def lb_units():
    return [dict(r, w=tuple(s.upper() for s in r['w'])) for r in C.lb_words() if len(r['w']) >= 2]


def lb_draw(units, ntok, seed):
    rng = np.random.default_rng(seed)
    by = collections.defaultdict(list)
    for r in units:
        by[r['doc']].append(r['w'])
    docs = sorted(by); rng.shuffle(docs)
    out, n = [], 0
    for d in docs:
        for w in by[d]:
            out.append(w); n += sum(L.cv_of(s) is not None for s in w)
        if n >= ntok:
            break
    return out


def known_values(words):
    return {s: L.cv_of(s) for w in words for s in w if L.cv_of(s) is not None}


# ---------------------------------------------------------------- similarity (la37-style, value-free)
def _rank01(M):
    n = M.shape[0]; iu = np.triu_indices(n, 1)
    v = M[iu]; r = v.argsort().argsort() / max(len(v) - 1, 1)
    R = np.zeros_like(M); R[iu] = r; R = R + R.T
    return R


def sim_matrix(words, fmin=3):
    cnt = collections.Counter(s for w in words for s in w)
    alph = sorted([s for s, n in cnt.items() if n >= fmin and s != DROP])
    idx = {s: i for i, s in enumerate(alph)}
    mw = L37.map_rare(words, idx)
    P = L37.ppmi_rows(L37._ctx_fast(mw, idx, None))
    cos = P @ P.T
    sw = L37.swap_matrix(mw, idx)
    S = 0.5 * _rank01(cos) + 0.5 * _rank01(sw + 1e-6 * cos)
    return alph, S


# ---------------------------------------------------------------- case construction
def make_case(words, values, h, keep=None, seed=0, r21=None, sim=None):
    """features (ncand, nfeat), candidate list, and the index of the true value (if h is valued).
    words: tuples; values: sign -> (C, V) for valued signs (h may be in it: it is hidden here).
    keep: thin h to this many tokens (others become DROP). r21: (signs, P) blind same-row matrix or None."""
    rng = np.random.default_rng(seed)
    pos = [(i, j) for i, w in enumerate(words) for j, s in enumerate(w) if s == h]
    keepset = set(range(len(pos))) if keep is None or keep >= len(pos) else set(rng.choice(len(pos), keep, replace=False))
    ww = [list(w) for w in words]
    for t, (i, j) in enumerate(pos):
        ww[i][j] = HID if t in keepset else DROP
    ww = [tuple(w) for w in ww]
    vals = {s: v for s, v in values.items() if s != h}
    vals[HID] = (NEW, 'a')
    corp = L.Corpus(ww, values=vals)
    if HID not in corp.signs:
        return None
    hi = corp.signs.index(HID)
    Clab, Vlab = corp.Clab, corp.Vlab
    assert ''.join(Vlab) == VOW, Vlab
    cands = [(c, v) for c in Clab for v in Vlab]
    N = len(cands); S = len(corp.signs)
    Cn = np.tile(corp.C0, (N, 1)); Vn = np.tile(corp.V0, (N, 1))
    Cn[:, hi] = [Clab.index(c) for c, v in cands]; Vn[:, hi] = [Vlab.index(v) for c, v in cands]
    m = L.measures(corp, Cn, Vn)
    # total phoneme-stream code length (bits) = -comp * number of phoneme transitions
    T_n = corp.u.sum() * 2 + corp.I.sum() + corp.F.sum()  # approx transition count, constant over candidates
    bits = m['comp'] * T_n
    # local counts around HID
    others = [i for i in range(S) if i != hi]
    oc = {corp.signs[i]: corp.true[i] for i in others}
    nb = collections.Counter()  # neighbour values
    for i in others:
        nb[corp.true[i]] += corp.B[hi, i] + corp.B[i, hi]
    selfpair = corp.B[hi, hi]
    nI = corp.I[hi]; nM = corp.u[hi] - corp.I[hi]
    rowsz = collections.Counter(v[0] for v in oc.values()); colsz = collections.Counter(v[1] for v in oc.values())
    cells = set(oc.values())
    # similarity features
    if sim is None:
        alph, Sm = sim_matrix(ww)
    else:
        alph, Sm = sim
    simrow = {}
    if HID in alph:
        a = alph.index(HID)
        for s, v in oc.items():
            if s in alph:
                simrow[s] = Sm[a, alph.index(s)]
    allmean = np.mean(list(simrow.values())) if simrow else 0.5
    def msim(pred):
        x = [simrow[s] for s, v in oc.items() if s in simrow and pred(v)]
        return np.mean(x) if x else allmean
    simC = {c: (msim(lambda v, c=c: v[0] == c) if c != NEW else allmean) for c in Clab}
    simV = {v: msim(lambda x, v=v: x[1] == v) for v in Vlab}
    # la21
    r21f = {c: 0.0 for c in Clab}
    if r21 is not None:
        rs, RP = r21
        up = [x.upper() for x in rs]
        if h.upper() in up:
            a = up.index(h.upper())
            pr = {s: RP[a, up.index(s.upper())] for s in oc if s.upper() in up}
            base = np.mean(list(pr.values())) if pr else 0
            for c in Clab:
                x = [p for s, p in pr.items() if oc[s][0] == c]
                r21f[c] = (np.mean(x) - base) if (x and c != NEW) else 0.0
    F = np.zeros((N, len(FEATS)))
    for k, (c, v) in enumerate(cands):
        pc = PL.get(c, 'x') if c else None
        ocpC = sum(n for (c2, v2), n in nb.items() if c and c2 == c) + (2 * selfpair if c else 0)
        ocpP = sum(n for (c2, v2), n in nb.items() if c and c2 and PL.get(c2) == pc) + (2 * selfpair if c else 0)
        sameV = sum(n for (c2, v2), n in nb.items() if v2 == v) + 2 * selfpair
        F[k] = [bits[k], ocpC, ocpP, nI * (c == ''), nM * (c == ''), sameV,
                float((c, v) not in cells and c != NEW), np.log1p(rowsz.get(c, 0)), np.log1p(colsz.get(v, 0)),
                float(c == NEW), simC[c], simV[v], r21f[c]]
    F[:, 0] -= F[:, 0].mean()
    F[:, 10] -= F[:, 10].mean(); F[:, 11] -= F[:, 11].mean()
    truth = None
    if h in values:
        tc, tv = values[h]
        if tc not in Clab or tc == NEW:
            tc = NEW
        truth = cands.index((tc, tv))
    return dict(F=F, cands=cands, truth=truth, sign=h, ntok=int(corp.u[hi]))


# ---------------------------------------------------------------- conditional logit
def fit(cases, use, l2=0.05):
    cols = [FEATS.index(f) for f in use]
    Xs = [c['F'][:, cols] for c in cases]; ys = [c['truth'] for c in cases]
    sd = np.concatenate(Xs).std(0) + 1e-9
    Xs = [x / sd for x in Xs]
    def nll(w):
        tot = 0.0; g = np.zeros_like(w)
        for x, y in zip(Xs, ys):
            s = x @ w; s = s - s.max(); p = np.exp(s); p /= p.sum()
            tot -= np.log(p[y] + 1e-300); g -= x[y] - p @ x
        return tot / len(Xs) + l2 * (w ** 2).sum(), g / len(Xs) + 2 * l2 * w
    r = minimize(nll, np.zeros(len(cols)), jac=True, method='L-BFGS-B')
    return dict(use=use, w=r.x / sd)


def predict(model, case):
    cols = [FEATS.index(f) for f in model['use']]
    s = case['F'][:, cols] @ model['w']; s = s - s.max(); p = np.exp(s); p /= p.sum()
    return p


def evaluate(model, cases):
    out = collections.defaultdict(list)
    for c in cases:
        p = predict(model, c); o = np.argsort(-p); t = c['truth']
        cand = c['cands']
        rank = int(np.where(o == t)[0][0])
        tc, tv = cand[t]
        # row and column marginal ranks
        rows = sorted({x[0] for x in cand}); pr = {r_: p[[i for i, x in enumerate(cand) if x[0] == r_]].sum() for r_ in rows}
        pv = {v: p[[i for i, x in enumerate(cand) if x[1] == v]].sum() for v in VOW}
        out['top1'].append(rank == 0); out['top3'].append(rank < 3)
        out['row1'].append(max(pr, key=pr.get) == tc); out['col1'].append(max(pv, key=pv.get) == tv)
        rr = sorted(pr, key=lambda k: -pr[k]); out['row3'].append(tc in rr[:3])
        out['lp'].append(np.log(p[t] + 1e-300)); out['pmax'].append(p.max())
        out['chance1'].append(1 / len(cand)); out['chancerow'].append(1 / len(rows))
    return {k: float(np.mean(v)) for k, v in out.items()} | dict(n=len(cases))


def fmt_eval(e):
    return (f"n {e['n']}: cell top-1 {e['top1']:.2f} top-3 {e['top3']:.2f} (chance {e['chance1']:.3f} / {3*e['chance1']:.3f}); "
            f"row top-1 {e['row1']:.2f} top-3 {e['row3']:.2f} (chance {e['chancerow']:.2f}); column top-1 {e['col1']:.2f} (chance 0.20); "
            f"mean logP(true) {e['lp']:.2f} (uniform {np.log(e['chance1']):.2f})")


def la21(tag):
    r = C.la21_rows()
    s, P, n = r[tag]
    return s, P
