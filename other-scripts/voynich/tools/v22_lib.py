"""v22 EVERY PARAGRAPH RUNS THE SAME PROGRAM: shared library.

A unit (paragraph or page) is a token sequence. Each token is coded by three word-class factors
(first unit, last unit, length bin) plus its line position lp (0 line-initial, 1 mid, 2 line-final)
and its line index inside the unit. Every model is conditioned on lp, so line-edge effects are
absorbed by all models alike; what is left to explain is progression through the unit.

Models (all factorised emissions, Dirichlet-smoothed):
  M0    position-free           P(f_k | lp)
  REL   rigid relative bins     P(f_k | bin(i/n), lp)
  LIN   linear drift            P(f_k | lp, r) ~ P(f_k|lp) * exp(b_kv (r - .5))
  HBT   head/body/tail          P(f_k | first line / last line / middle rel-bin, lp)
  LIX   line-index bins         P(f_k | line 0,1,2,3,4-5,6-8,9+, lp)
  HMM   S-stage left-to-right program (stay / +1 / +2), start in stage 0, end anywhere,
        emissions P(f_k | stage, lp); EM with random-segmentation restarts.
"""
import os, sys, math, random, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
import v21_lib as V21

LOOPS = os.path.join(vlib.ROOT, 'loops')
CK = os.path.join(vlib.DATA, 'v22_ckpt'); os.makedirs(CK, exist_ok=True)
ALPHA = 0.5
LN2 = math.log(2)


# ------------------------------------------------------------------ corpora
def voynich_pages(name='ZL3b', minw=40):
    """Like v21 voynich_pages, keeping Currier language and illustration type."""
    lines = vlib.load_voynich(name, drop_uncertain=True)
    pages, order, cur = {}, [], None
    for L in lines:
        f = L['folio']
        if f not in pages:
            pages[f] = {'id': f, 'sec': (L.get('illus') or 'x') + (L.get('lang') or 'x'), 'lang': L.get('lang') or 'x',
                        'illus': L.get('illus') or 'x', 'paras': []}
            order.append(f); cur = None
        ws = [V21.U(w) for w in L['words']]
        ws = [w for w in ws if w and '?' not in w]
        if not ws: continue
        if L['para_start'] or cur is None:
            cur = []; pages[f]['paras'].append(cur)
        cur.append(ws)
        if L.get('para_end'): cur = None
    out = [pages[f] for f in order]
    return [p for p in out if sum(len(l) for pa in p['paras'] for l in pa) >= minw]


def herbal_pages(which):
    C = V21.latin_herbal() if which == 'LA' else V21.italian_herbal()
    for p in C: p['lang'] = p['sec']
    return C


def units(C, level='para', minw=12):
    """-> list of dicts {pid, lang, lines: [[w..]..]}"""
    out = []
    for p in C:
        if level == 'para':
            for pa in p['paras']:
                if sum(len(l) for l in pa) >= minw:
                    out.append({'pid': p['id'], 'lang': p.get('lang', 'x'), 'lines': [list(l) for l in pa]})
        else:
            ls = [list(l) for pa in p['paras'] for l in pa]
            if sum(len(l) for l in ls) >= minw:
                out.append({'pid': p['id'], 'lang': p.get('lang', 'x'), 'lines': ls})
    return out


# ------------------------------------------------------------------ coding
class Coder:
    """Joint word class = (first unit group, last unit group): ONE categorical variable, so that no model can gain
    by capturing dependence between factors of the same word (the cycle-1 artefact of factorised emissions).
    joint=False restores the factorised (first, last, length-bin) coding."""
    def __init__(self, U, nfirst=7, nlast=5, nlen=5, joint=True):
        toks = [w for u in U for l in u['lines'] for w in l]
        self.joint = joint
        self.first = [a for a, _ in Counter(w[0] for w in toks).most_common(nfirst)]
        self.last = [a for a, _ in Counter(w[-1] for w in toks).most_common(nlast)]
        L = np.array([len(w) for w in toks])
        qs = np.quantile(L, np.linspace(0, 1, nlen + 1)[1:-1])
        self.lcut = sorted(set(qs.tolist()))
        nf, nl = len(self.first) + 1, len(self.last) + 1
        self.sizes = [nf * nl] if joint else [nf, nl, len(self.lcut) + 1]

    def word(self, w):
        f = self.first.index(w[0]) if w[0] in self.first else len(self.first)
        l = self.last.index(w[-1]) if w[-1] in self.last else len(self.last)
        if self.joint: return (f * (len(self.last) + 1) + l,)
        n = int(np.searchsorted(self.lcut, len(w), side='right'))
        return f, l, n

    def unit(self, u):
        F, lp, li = [], [], []
        for k, line in enumerate(u['lines']):
            for j, w in enumerate(line):
                F.append(self.word(w))
                lp.append(0 if j == 0 else (2 if j == len(line) - 1 else 1))
                li.append(k)
        n = len(F)
        return {'F': np.array(F, np.int64).reshape(n, -1), 'lp': np.array(lp, np.int64), 'li': np.array(li, np.int64),
                'r': (np.arange(n) + 0.5) / n, 'nl': len(u['lines'])}


def code_all(U, coder):
    return [coder.unit(u) for u in U]


# ------------------------------------------------------------------ rigid models
def _bins_rel(c, B):
    return np.minimum((c['r'] * B).astype(np.int64), B - 1)


def _bins_lix(c, B=None):
    m = np.array([0, 1, 2, 3, 4, 4, 5, 5, 5])
    return np.where(c['li'] < 9, m[np.minimum(c['li'], 8)], 6)


def _bins_hbt(c, B=3):
    nl = c['nl']
    b = np.where(c['li'] == 0, 0, np.where(c['li'] == nl - 1, B + 1, 0))
    mid = (c['li'] > 0) & (c['li'] < nl - 1)
    if mid.any():
        rr = (c['li'][mid] - 1 + 0.5) / max(1, nl - 2)
        b[mid] = 1 + np.minimum((rr * B).astype(np.int64), B - 1)
    return b


BINNERS = {'REL': _bins_rel, 'LIX': _bins_lix, 'HBT': _bins_hbt}


def nbins(kind, B):
    return B if kind == 'REL' else (7 if kind == 'LIX' else B + 2)


def fit_binned(train, sizes, kind, B):
    nb = nbins(kind, B)
    E = [np.full((nb, 3, s), ALPHA) for s in sizes]
    for c in train:
        b = BINNERS[kind](c, B)
        for k in range(len(E)): np.add.at(E[k], (b, c['lp'], c['F'][:, k]), 1)
    return [np.log(e / e.sum(-1, keepdims=True)) for e in E]


def ll_binned(E, test, kind, B):
    tot, n = 0.0, 0
    for c in test:
        b = BINNERS[kind](c, B)
        for k in range(len(E)): tot += E[k][b, c['lp'], c['F'][:, k]].sum()
        n += len(c['lp'])
    return tot, n


def fit_m0(train, sizes):
    E = [np.full((3, s), ALPHA) for s in sizes]
    for c in train:
        for k in range(len(E)): np.add.at(E[k], (c['lp'], c['F'][:, k]), 1)
    return [np.log(e / e.sum(-1, keepdims=True)) for e in E]


def ll_m0(E, test):
    tot, n = 0.0, 0
    for c in test:
        for k in range(len(E)): tot += E[k][c['lp'], c['F'][:, k]].sum()
        n += len(c['lp'])
    return tot, n


def fit_lin(train, sizes, iters=800, lr=0.5, l2=1.0, nb=40):
    """log P(v | lp, r) = a[lp,v] + b[v]*(r-.5) - logZ ; gradient ascent on counts aggregated in nb r-bins."""
    out = []
    rc = (np.arange(nb) + 0.5) / nb - 0.5
    for k, s in enumerate(sizes):
        Y = np.zeros((3, nb, s))
        for c in train:
            np.add.at(Y, (c['lp'], np.minimum((c['r'] * nb).astype(int), nb - 1), c['F'][:, k]), 1)
        Nlr = Y.sum(-1); N = Nlr.sum()
        a = np.log(Y.sum(1) + 0.5); a -= a.mean(1, keepdims=True); b = np.zeros(s)
        for it in range(iters):
            z = a[:, None, :] + rc[None, :, None] * b[None, None, :]
            z -= z.max(-1, keepdims=True); p = np.exp(z); p /= p.sum(-1, keepdims=True)
            g = Y - Nlr[..., None] * p
            a += lr * g.sum(1) / N * 3
            b += lr * 8 * ((g * rc[None, :, None]).sum((0, 1)) - l2 * b) / N
        out.append((a, b))
    return out


def ll_lin(P, test):
    tot, n = 0.0, 0
    for c in test:
        r = c['r'] - 0.5
        for k, (a, b) in enumerate(P):
            z = a[c['lp']] + r[:, None] * b[None, :]
            m = z.max(1, keepdims=True)
            lz = (m[:, 0] + np.log(np.exp(z - m).sum(1)))
            tot += (z[np.arange(len(r)), c['F'][:, k]] - lz).sum()
        n += len(c['lp'])
    return tot, n


# ------------------------------------------------------------------ HMM program
@njit(cache=True)
def _fb_core(Bs, mrow, offs, A, S, want_post):
    """Scaled forward-backward on concatenated sequences. Bs = exp(logB - rowmax), A[s,d] d=0 stay,1 +1,2 +2.
    Start in state 0. End anywhere. Returns total loglik, gamma (T,S), xi counts (S,3)."""
    T = Bs.shape[0]
    gam = np.zeros((T, S)) if want_post else np.zeros((1, S))
    xic = np.zeros((S, 3))
    tot = 0.0
    al = np.zeros((T, S)); z_ = np.ones(T)
    be = np.zeros((T, S)) if want_post else np.zeros((1, S))
    for q in range(len(offs) - 1):
        a0, a1 = offs[q], offs[q + 1]
        for t in range(a0, a1):
            if t == a0:
                al[t, 0] = Bs[t, 0]
            else:
                for s in range(S):
                    v = al[t - 1, s] * A[s, 0]
                    if s >= 1: v += al[t - 1, s - 1] * A[s - 1, 1]
                    if s >= 2: v += al[t - 1, s - 2] * A[s - 2, 2]
                    al[t, s] = v * Bs[t, s]
            z = 0.0
            for s in range(S): z += al[t, s]
            for s in range(S): al[t, s] /= z
            z_[t] = z
            tot += math.log(z) + mrow[t]
        if not want_post: continue
        for s in range(S): be[a1 - 1, s] = 1.0
        for t in range(a1 - 2, a0 - 1, -1):
            iz = 1.0 / z_[t + 1]
            for s in range(S):
                v = A[s, 0] * Bs[t + 1, s] * be[t + 1, s]
                if s + 1 < S: v += A[s, 1] * Bs[t + 1, s + 1] * be[t + 1, s + 1]
                if s + 2 < S: v += A[s, 2] * Bs[t + 1, s + 2] * be[t + 1, s + 2]
                be[t, s] = v * iz
        for t in range(a0, a1):
            for s in range(S): gam[t, s] = al[t, s] * be[t, s]
            if t + 1 < a1:
                iz = 1.0 / z_[t + 1]
                for s in range(S):
                    for d in range(3):
                        if s + d < S:
                            xic[s, d] += al[t, s] * A[s, d] * Bs[t + 1, s + d] * be[t + 1, s + d] * iz
    return tot, gam, xic


def _fb(logB, offs, logA, S, want_post):
    logB = np.ascontiguousarray(logB)
    m = logB.max(1)
    Bs = np.exp(logB - m[:, None])
    return _fb_core(Bs, m, offs, np.exp(logA), S, want_post)


@njit(cache=True)
def _viterbi(logB, a0, a1, logA, S):
    n = a1 - a0
    D = np.full((n, S), -1e300); bp = np.zeros((n, S), np.int64)
    D[0, 0] = logB[a0, 0]
    for t in range(1, n):
        for s in range(S):
            best = -1e300; bs = 0
            for d in range(3):
                if s - d >= 0:
                    v = D[t - 1, s - d] + logA[s - d, d]
                    if v > best: best = v; bs = s - d
            D[t, s] = best + logB[a0 + t, s]; bp[t, s] = bs
    path = np.zeros(n, np.int64)
    s = 0; m = -1e300
    for k in range(S):
        if D[n - 1, k] > m: m = D[n - 1, k]; s = k
    for t in range(n - 1, -1, -1):
        path[t] = s; s = bp[t, s]
    return path


class Seq:
    def __init__(self, codes):
        self.F = np.concatenate([c['F'] for c in codes]); self.lp = np.concatenate([c['lp'] for c in codes])
        self.offs = np.cumsum([0] + [len(c['lp']) for c in codes]).astype(np.int64)
        self.T = len(self.lp)


def _logB(seq, E):
    return sum(E[k][:, seq.lp, seq.F[:, k]].T for k in range(len(E)))


def _norm_A(cnt, S):
    A = cnt + 0.1
    for s in range(S):
        if s + 1 >= S: A[s, 1] = 1e-12
        if s + 2 >= S: A[s, 2] = 1e-12
    A = A / A.sum(1, keepdims=True)
    return np.log(A)


def _m_step(seq, gam, xic, sizes, S):
    E = []
    for k, s in enumerate(sizes):
        e = np.full((S, 3, s), ALPHA)
        idx = seq.lp * s + seq.F[:, k]
        for st in range(S):
            e[st] += np.bincount(idx, weights=gam[:, st], minlength=3 * s).reshape(3, s)
        E.append(np.log(e / e.sum(-1, keepdims=True)))
    return E, _norm_A(xic, S)


def _init_random(seq, sizes, S, rng):
    """Random elastic segmentation of each unit into S stages (Dirichlet durations), hard counts."""
    nU = len(seq.offs) - 1
    d = rng.dirichlet(np.ones(S) * 2.0, size=nU)
    cum = np.cumsum(d, 1)
    lens = np.diff(seq.offs)
    uid = np.repeat(np.arange(nU), lens)
    rel = (np.arange(seq.T) - seq.offs[uid] + 0.5) / lens[uid]
    st = np.minimum((rel[:, None] > cum[uid]).sum(1), S - 1)
    gam = np.zeros((seq.T, S)); gam[np.arange(seq.T), st] = 1
    xic = np.zeros((S, 3)); xic[:, 0] = 10; xic[:, 1] = 1
    return _m_step(seq, gam, xic, sizes, S)


def fit_hmm(seq, sizes, S, restarts, rng, iters=15, polish=25, top=2):
    """Many random restarts with short EM; the best few polished. Returns (E, logA, trainLL)."""
    cands = []
    for r in range(restarts):
        E, lA = _init_random(seq, sizes, S, rng)
        ll = -1e300
        for it in range(iters):
            ll, gam, xic = _fb(_logB(seq, E), seq.offs, lA, S, True)
            E, lA = _m_step(seq, gam, xic, sizes, S)
        cands.append((ll, E, lA))
    cands.sort(key=lambda x: -x[0])
    best = None
    for ll, E, lA in cands[:top]:
        for it in range(polish):
            ll, gam, xic = _fb(_logB(seq, E), seq.offs, lA, S, True)
            E, lA = _m_step(seq, gam, xic, sizes, S)
        if best is None or ll > best[2]: best = (E, lA, ll)
    return best


def ll_hmm(model, seq, S):
    E, lA, _ = model
    ll, _, _ = _fb(_logB(seq, E), seq.offs, lA, S, False)
    return ll, seq.T


def ll_poe(model, ER, E0, test, seq, S, kind='HBT', B=2):
    """Product of experts: stage table x rigid-bin table / position-free table, renormalised over classes for every
    (stage, bin, lp). Measures what the elastic program adds on top of rigid position. Single-factor coding only."""
    E, lA, _ = model
    Eh, Er, Em = E[0], ER[0], E0[0]            # (S,3,C), (nb,3,C), (3,C)
    Z = Eh[:, None] + Er[None] - Em[None, None]  # (S,nb,3,C)
    mx = Z.max(-1, keepdims=True); logZ = (mx + np.log(np.exp(Z - mx).sum(-1, keepdims=True)))[..., 0]
    b = np.concatenate([BINNERS[kind](c, B) for c in test])
    lB = (Eh[:, seq.lp, seq.F[:, 0]].T + Er[b, seq.lp, seq.F[:, 0]][:, None] - Em[seq.lp, seq.F[:, 0]][:, None]
          - logZ[:, b, seq.lp].T)
    ll, _, _ = _fb(lB, seq.offs, lA, S, False)
    return ll


def viterbi_paths(model, seq, S):
    E, lA, _ = model
    lB = _logB(seq, E)
    return [_viterbi(lB, seq.offs[q], seq.offs[q + 1], lA, S) for q in range(len(seq.offs) - 1)]



# ------------------------------------------------------------------ order-free mixture (heterogeneity baseline)
def _counts(codes, sizes):
    """(nU, sum_k 3*s_k) count matrix of (lp, class) per unit."""
    X = np.zeros((len(codes), sum(3 * s for s in sizes)))
    for i, c in enumerate(codes):
        o = 0
        for k, s in enumerate(sizes):
            X[i, o:o + 3 * s] = np.bincount(c['lp'] * s + c['F'][:, k], minlength=3 * s); o += 3 * s
    return X


def _logE_flat(E):
    return np.concatenate([e.reshape(e.shape[0], -1) for e in E], 1)


def _unit_ll(codes, E, K):
    sizes = [e.shape[-1] for e in E]
    return _counts(codes, sizes) @ _logE_flat(E).T


def fit_mix(train, sizes, K, rng, restarts=6, iters=60):
    X = _counts(train, sizes)
    best = None
    for r in range(restarts):
        resp = rng.dirichlet(np.ones(K), size=len(train))
        for it in range(iters):
            M = resp.T @ X
            E, o = [], 0
            for s in sizes:
                e = M[:, o:o + 3 * s].reshape(K, 3, s) + ALPHA; o += 3 * s
                E.append(np.log(e / e.sum(-1, keepdims=True)))
            pi = resp.mean(0) + 1e-3; pi /= pi.sum()
            ul = X @ _logE_flat(E).T + np.log(pi)
            m = ul.max(1, keepdims=True); resp = np.exp(ul - m); z = resp.sum(1, keepdims=True); resp /= z
            ll = (m[:, 0] + np.log(z[:, 0])).sum()
        if best is None or ll > best[2]: best = (E, pi, ll)
    return best


def ll_mix(model, test, K):
    E, pi, _ = model
    ul = _unit_ll(test, E, K) + np.log(pi)
    m = ul.max(1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(ul - m).sum(1))).sum(), sum(len(c['lp']) for c in test)


# ------------------------------------------------------------------ evaluation
def folds_by_unit(U, nfold, seed):
    pids = sorted(set(u['pid'] for u in U))
    rng = random.Random(seed); rng.shuffle(pids)
    f = {p: i % nfold for i, p in enumerate(pids)}
    return np.array([f[u['pid']] for u in U])


MIXK = [2, 4, 8]
RIGID = [('REL', 2), ('REL', 3), ('REL', 4), ('REL', 6), ('REL', 8), ('LIX', 0), ('HBT', 2), ('HBT', 3)]


def evaluate(codes, sizes, fold, Slist, restarts, seed, nfold=5):
    """Held-out bits/token gains over M0 for rigid models, LIN and HMM(S). Returns dict."""
    rng = np.random.default_rng(seed)
    res = {'M0': [0.0, 0]}
    acc = {}
    for f in range(nfold):
        tr = [c for c, g in zip(codes, fold) if g != f]; te = [c for c, g in zip(codes, fold) if g == f]
        if not te: continue
        E0 = fit_m0(tr, sizes); l0, n = ll_m0(E0, te)
        res['M0'][0] += l0; res['M0'][1] += n
        for kind, B in RIGID:
            E = fit_binned(tr, sizes, kind, B); l, _ = ll_binned(E, te, kind, B)
            acc.setdefault(f'{kind}{B}', 0.0); acc[f'{kind}{B}'] += l - l0
        P = fit_lin(tr, sizes); l, _ = ll_lin(P, te)
        acc.setdefault('LIN', 0.0); acc['LIN'] += l - l0
        for K in MIXK:
            mm = fit_mix(tr, sizes, K, rng); l, _ = ll_mix(mm, te, K)
            acc.setdefault(f'MIX{K}', 0.0); acc[f'MIX{K}'] += l - l0
        stq, steq = Seq(tr), Seq(te)
        ER = fit_binned(tr, sizes, 'HBT', 2); lR, _ = ll_binned(ER, te, 'HBT', 2)
        for S in Slist:
            m = fit_hmm(stq, sizes, S, restarts, rng)
            l, _ = ll_hmm(m, steq, S)
            acc.setdefault(f'H{S}', 0.0); acc[f'H{S}'] += l - l0
            if len(sizes) == 1:
                lp_ = ll_poe(m, ER, E0, te, steq, S)
                acc.setdefault(f'POE{S}', 0.0); acc[f'POE{S}'] += lp_ - l0
    N = res['M0'][1]
    out = {k: v / N / LN2 * 1000 for k, v in acc.items()}   # millibits per token
    out['N'] = N
    rig = [k for k in out if k[:3] in ('REL', 'LIX', 'HBT', 'LIN')]
    hm = [k for k in out if k.startswith('H') and k[1:].isdigit()]
    out['best_rigid'] = max(out[k] for k in rig); out['best_rigid_k'] = max(rig, key=lambda k: out[k])
    out['best_hmm'] = max(out[k] for k in hm); out['best_hmm_k'] = max(hm, key=lambda k: out[k])
    out['delta'] = out['best_hmm'] - out['best_rigid']
    out['best_mix'] = max(out[k] for k in out if k.startswith('MIX'))
    pk = [k for k in out if k.startswith('POE')]
    if pk: out['best_poe'] = max(out[k] for k in pk); out['best_poe_k'] = max(pk, key=lambda k: out[k])
    return out


# ------------------------------------------------------------------ nulls and plants
def shuffle_lines(U, rng, keep_head=False):
    out = []
    for u in U:
        ls = [list(l) for l in u['lines']]
        if keep_head:
            rest = ls[1:]; rng.shuffle(rest); ls = ls[:1] + rest
        else:
            rng.shuffle(ls)
        out.append(dict(u, lines=ls))
    return out


def rotate_lines(U, rng):
    """Cyclic rotation of a unit's lines by a random offset (>=1 when possible): keeps every line-to-line adjacency but
    one, so local continuity survives while each stretch moves to another place in the unit."""
    out = []
    for u in U:
        ls = [list(l) for l in u['lines']]; n = len(ls)
        k = rng.randrange(1, n) if n > 1 else 0
        out.append(dict(u, lines=ls[k:] + ls[:k]))
    return out


def shuffle_inline(U, rng):
    """Shuffle the middle words of every line (line-initial and line-final words stay): kills word-to-word
    order inside lines, keeps which words sit on which line."""
    out = []
    for u in U:
        ls = []
        for l in u['lines']:
            l = list(l)
            if len(l) > 3:
                mid = l[1:-1]; rng.shuffle(mid); l = [l[0]] + mid + [l[-1]]
            ls.append(l)
        out.append(dict(u, lines=ls))
    return out


def shuffle_words_lp(U, rng):
    """Shuffle words within unit among slots of the same line position (initial / mid / final)."""
    out = []
    for u in U:
        slots = {0: [], 1: [], 2: []}
        for k, l in enumerate(u['lines']):
            for j, w in enumerate(l):
                slots[0 if j == 0 else (2 if j == len(l) - 1 else 1)].append(w)
        for v in slots.values(): rng.shuffle(v)
        it = {k: iter(v) for k, v in slots.items()}
        ls = [[next(it[0 if j == 0 else (2 if j == len(l) - 1 else 1)]) for j in range(len(l))] for l in u['lines']]
        out.append(dict(u, lines=ls))
    return out


def plant_program(U, rng, S=4, rho=0.2, tilt=1.0, conc=2.0):
    """Elastic planted program: each unit is cut into S stages with Dirichlet(conc) durations; inside stage k a
    fraction rho of words is replaced by a corpus word (same line position) whose first unit is drawn from a
    stage-tilted distribution. Returns (planted units, true stage arrays)."""
    pool = {0: [], 1: [], 2: []}
    for u in U:
        for l in u['lines']:
            for j, w in enumerate(l): pool[0 if j == 0 else (2 if j == len(l) - 1 else 1)].append(w)
    byf = {lp: {} for lp in pool}
    for lp, ws in pool.items():
        for w in ws: byf[lp].setdefault(w[0], []).append(w)
    keys = sorted(set(k for lp in byf for k in byf[lp]))
    base = Counter(w[0] for ws in pool.values() for w in ws)
    keys = [k for k, _ in base.most_common(10)]
    tilts = rng.normal(0, tilt, (S, len(keys)))
    probs = []
    for s in range(S):
        p = np.array([base[k] for k in keys], float) * np.exp(tilts[s]); probs.append(p / p.sum())
    out, truth = [], []
    for u in U:
        n = sum(len(l) for l in u['lines'])
        d = rng.dirichlet(np.ones(S) * conc)
        cut = np.floor(np.cumsum(d) * n).astype(int)
        st = np.minimum(np.searchsorted(cut, np.arange(n), side='right'), S - 1)
        ls, t = [], 0
        for l in u['lines']:
            nl = []
            for j, w in enumerate(l):
                lp = 0 if j == 0 else (2 if j == len(l) - 1 else 1)
                if rng.random() < rho:
                    for _ in range(5):
                        k = keys[rng.choice(len(keys), p=probs[st[t]])]
                        if byf[lp].get(k): w = byf[lp][k][rng.integers(len(byf[lp][k]))]; break
                nl.append(w); t += 1
            ls.append(nl)
        out.append(dict(u, lines=ls)); truth.append(st)
    return out, truth


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


def save(name, obj):
    with open(os.path.join(CK, name), 'w') as f: json.dump(obj, f, indent=1, default=float)


def load(name):
    p = os.path.join(CK, name)
    return json.load(open(p)) if os.path.exists(p) else None


def row(fn, rid, method, result, verdict):
    with open(os.path.join(LOOPS, fn), 'a') as f:
        f.write(f'| {rid} | {method} | {result} | {verdict} |\n')
