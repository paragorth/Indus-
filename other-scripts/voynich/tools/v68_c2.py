"""v68 cycle 2: is the word-to-word coupling a ONE-DIMENSIONAL continuity (a melody's pitch line)?

Each word type gets an entry class s(w) and an exit class e(w) in 0..K-1.
  free model     P(v|w) = P(v|s_v) * P(s_v | e_w)                (P(s|e) any K x K table)
  pitch model    P(v|w) = P(v|s_v) * pi(s) k(s - e) / Z_e          (Toeplitz: depends only on the step)
A melody written syllable by syllable makes the coupling a function of the pitch step (entry note of
the next syllable minus exit note of this one), so the pitch model should recover almost all of the
free model's gain (ratio R near 1).  Arbitrary categorical coupling (syntax, spelling junctions) needs
the free table.  Both are fitted on half the folios (exchange algorithm, random restarts) and scored on
the other half: gain = bits/bigram over the unigram model.
Calibration: chant coded with an opaque code (true entry/exit pitches known -> recovery checked);
Latin and German coded the same way; within-line word shuffle of each (coupling destroyed).
usage: python3 v68_c2.py corpusname K restarts
"""
import json, math, os, random, sys
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v68_lib as V
import v53_lib as L53

MINC = 4


def vocab(train_lines):
    c = Counter(w for l in train_lines for w in l)
    keep = {w for w, n in c.items() if n >= MINC}
    return keep


def tok(w, keep):
    if w in keep:
        return w
    return ('<' + w[0], w[-1] + '>')   # rare words: entry from first glyph, exit from last glyph


def build(lines, keep):
    """bigram list of (exit-key, entry-key, word-key)."""
    big = []
    for l in lines:
        ts = [tok(w, keep) for w in l]
        for a, b in zip(ts, ts[1:]):
            ea = a if isinstance(a, str) else 'E' + a[1]
            sb = b if isinstance(b, str) else 'S' + b[0]
            big.append((ea, sb, b if isinstance(b, str) else b[0] + '|' + b[1]))
    return big


class Model:
    def __init__(self, K, toeplitz, rng):
        self.K, self.T, self.rng = K, toeplitz, rng

    def fit(self, big, sweeps=30, init=None):
        K = self.K
        E = sorted(set(e for e, _, _ in big)); S = sorted(set(s for _, s, _ in big))
        ei = {x: i for i, x in enumerate(E)}; si = {x: i for i, x in enumerate(S)}
        # pair counts exit-type x entry-type
        pc = Counter((ei[e], si[s]) for e, s, _ in big)
        self.E, self.S, self.ei, self.si = E, S, ei, si
        ne, ns = len(E), len(S)
        rows = np.array([k[0] for k in pc]); cols = np.array([k[1] for k in pc]); vals = np.array(list(pc.values()), float)
        self.out_by_e = defaultdict(list); self.in_by_s = defaultdict(list)
        for r, c, v in zip(rows, cols, vals):
            self.out_by_e[r].append((c, v)); self.in_by_s[c].append((r, v))
        ce = self.rng.integers(0, K, ne); cs = self.rng.integers(0, K, ns)
        if init is not None:
            ie, is_ = init
            for x, i in ei.items():
                if x in ie: ce[i] = ie[x]
            for x, i in si.items():
                if x in is_: cs[i] = is_[x]
        for sw in range(max(sweeps, 1)):
            if sweeps == 0: break
            C = np.zeros((K, K))
            np.add.at(C, (ce[rows], cs[cols]), vals)
            logP = self.cond(C)
            moved = 0
            # exit classes: word e contributes sum_c v * log P(cs[c] | class)
            for i in self.rng.permutation(ne):
                vec = np.zeros(K)
                for c, v in self.out_by_e[i]:
                    vec[cs[c]] += v
                best = int(np.argmax(logP @ vec))
                if best != ce[i]:
                    ce[i] = best; moved += 1
            C = np.zeros((K, K)); np.add.at(C, (ce[rows], cs[cols]), vals)
            logP = self.cond(C)
            # entry classes: term log P(s|e) + log P(v|s) -> P(v|s) = n(v)/n(s): use full LL
            nS = np.bincount(cs, weights=np.array([sum(v for _, v in self.in_by_s[c]) for c in range(ns)]), minlength=K)
            for c in self.rng.permutation(ns):
                vec = np.zeros(K)
                for r, v in self.in_by_s[c]:
                    vec[ce[r]] += v
                tot = vec.sum()
                cur = cs[c]
                sc = vec @ logP  # sum_e vec[e] log P(s|e) for each s
                ns_wo = nS.copy(); ns_wo[cur] -= tot
                sc = sc - tot * np.log(ns_wo + tot + 1e-9)
                best = int(np.argmax(sc))
                if best != cur:
                    nS[cur] -= tot; nS[best] += tot; cs[c] = best; moved += 1
            if moved == 0 and sw >= 2:
                break
            if sweeps == 0:
                break
        C = np.zeros((K, K)); np.add.at(C, (ce[rows], cs[cols]), vals)
        self.C = C; self.logP = self.cond(C); self.ce, self.cs = ce, cs
        return self

    def cond(self, C):
        K = self.K
        if not self.T:
            P = (C + 0.1) / (C + 0.1).sum(1, keepdims=True)
            return np.log(P)
        # log-linear a_s + b_{s-e}: fit by gradient ascent on sum C log P(s|e)
        a = np.log(C.sum(0) + 1) ; a -= a.mean(); b = np.zeros(2 * K - 1)
        D = np.arange(K)[None, :] - np.arange(K)[:, None] + K - 1
        rs = C.sum(1)
        for it in range(150):
            L = a[None, :] + b[D]
            L -= L.max(1, keepdims=True)
            P = np.exp(L); P /= P.sum(1, keepdims=True)
            G = C - rs[:, None] * P
            ga = G.sum(0); gb = np.bincount(D.ravel(), weights=G.ravel(), minlength=2 * K - 1)
            lr = 1.0 / max(1.0, C.sum()) * 5
            a += lr * ga; b += lr * gb
        L = a[None, :] + b[D]; L -= L.max(1, keepdims=True)
        P = np.exp(L); P /= P.sum(1, keepdims=True)
        self.kernel = b
        return np.log(P + 1e-12)

    def score(self, big_tr, big_te):
        """held-out bits/bigram gain over unigram, both with the same P(v) backoff."""
        wc = Counter(w for _, _, w in big_tr); tot = sum(wc.values()); nw = len(wc) + 1
        sc_ = Counter()
        for _, s, w in big_tr:
            if s in self.si:
                sc_[self.cs[self.si[s]]] += 1
        g = 0.0; n = 0
        for e, s, w in big_te:
            pw = (wc.get(w, 0) + 0.5) / (tot + 0.5 * nw)
            if e in self.ei and s in self.si:
                k = self.cs[self.si[s]]
                pk = sc_[k] / tot
                lp = self.logP[self.ce[self.ei[e]], k]
                g += (lp - math.log(pk + 1e-12)) / math.log(2)
            n += 1
        return g / n

    def train_ll(self):
        C = self.C
        return float((C * self.logP).sum())


def halves(lines_with_folio):
    fol = []
    for f, _ in lines_with_folio:
        if f not in fol:
            fol.append(f)
    A = set(fol[0::2])
    return [l for f, l in lines_with_folio if f in A], [l for f, l in lines_with_folio if f not in A]


def corpus(name, rng):
    """returns (train_lines, test_lines, truth) with truth = {word: (entry pitch, exit pitch)} for chant."""
    truth = None
    if name.split('_')[0] in ('ZL', 'IT'):
        vz = L53.load_voynich('ZL3b' if name.startswith('ZL') else 'IT2a')
        lf = [(l['folio'], l['words']) for l in vz]
    elif name.startswith('chant') or name.startswith('LA') or name.startswith('DE'):
        base = name.split('_')[0] if not name.startswith('chant') else 'chant'
        if base == 'chant':
            src = V.chant_lines(V.chant_units())[:9000]
            enc = {'rep': 'abs', 'fold': False, 'group': 'one', 'trunc': 0, 'rank': True, 'p2': 0.3,
                   'seed': 77, 'sympermute': True}
            coded = V.encode(src, enc)
            truth = {}
            for l, cl in zip(src, coded):
                for u, w in zip(l, cl):
                    truth[w] = (u[0], u[-1])
        else:
            src = V.lang_lines({'LA': 'LA_ency', 'DE': 'DE_herb'}[base])
            enc = {'rep': 'abs', 'fold': False, 'group': 'one', 'trunc': 0, 'rank': True, 'p2': 0.3,
                   'seed': 78, 'sympermute': True}
            coded = V.encode(src, enc, symmap=list(range(26)))
        lf = [(i // 25, l) for i, l in enumerate(coded)]   # pseudo-folios of 25 lines
    else:
        raise ValueError(name)
    if name.endswith('_wshuf'):
        lf = [(f, rng.sample(l, len(l))) for f, l in lf]
    tr, te = halves(lf)
    return tr, te, truth


def full_ll(m):
    C = m.C
    nS = C.sum(0)
    return float((C * m.logP).sum() - (nS[nS > 0] * np.log(nS[nS > 0])).sum())


def seriate(m):
    """order exit and entry classes of a free model jointly by the Fiedler vector of the bipartite PMI graph."""
    K = m.K; C = m.C + 0.5
    pmi = np.log(C * C.sum() / C.sum(1, keepdims=True) / C.sum(0, keepdims=True))
    A = np.zeros((2 * K, 2 * K)); A[:K, K:] = np.maximum(pmi, 0); A[K:, :K] = A[:K, K:].T
    Lp = np.diag(A.sum(1)) - A
    w, v = np.linalg.eigh(Lp)
    f = v[:, 1]
    re = np.argsort(np.argsort(f[:K])); rs = np.argsort(np.argsort(f[K:]))
    ie = {x: int(re[m.ce[i]]) for x, i in m.ei.items()}
    is_ = {x: int(rs[m.cs[i]]) for x, i in m.si.items()}
    return ie, is_


def ca_init(btr, K, dim=0, flip=False):
    """correspondence analysis of the exit x entry table; axis 'dim' cut into K weighted quantile classes."""
    E = sorted(set(e for e, _, _ in btr)); S = sorted(set(s for _, s, _ in btr))
    ei = {x: i for i, x in enumerate(E)}; si = {x: i for i, x in enumerate(S)}
    N = np.zeros((len(E), len(S)))
    for e, s_, _ in btr:
        N[ei[e], si[s_]] += 1
    P = N / N.sum(); r = P.sum(1); c = P.sum(0)
    Sm = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, d, Vt = np.linalg.svd(Sm, full_matrices=False)
    xr = U[:, dim] / np.sqrt(r); xc = Vt[dim] / np.sqrt(c)
    if flip:
        xc = -xc
    def q(x, w):
        o = np.argsort(x); cw = np.cumsum(w[o]) / w.sum()
        cl = np.empty(len(x), int); cl[o] = np.minimum(K - 1, (cw * K).astype(int))
        return cl
    ce = q(xr, r); cs = q(xc, c)
    return {x: int(ce[i]) for x, i in ei.items()}, {x: int(cs[i]) for x, i in si.items()}


def run(name, K, R, seed=0):
    rng = np.random.default_rng(seed); prng = random.Random(seed)
    tr, te, truth = corpus(name, prng)
    keep = vocab(tr)
    btr, bte = build(tr, keep), build(te, keep)
    res = {'name': name, 'K': K, 'ntr': len(btr), 'nte': len(bte), 'nkeep': len(keep)}
    frees = [Model(K, False, rng).fit(btr) for _ in range(R)]
    fb = max(frees, key=full_ll)
    res['free'] = {'ll': full_ll(fb) / len(btr), 'heldout_gain': fb.score(btr, bte)}
    cands = [('rand', Model(K, True, rng).fit(btr)) for _ in range(R)]
    for f in sorted(frees, key=full_ll)[-3:]:
        cands.append(('seriated', Model(K, True, rng).fit(btr, init=seriate(f))))
    for dim in (0, 1):
        for fl in (False, True):
            cands.append(('ca%d%s' % (dim, 'f' if fl else ''), Model(K, True, rng).fit(btr, init=ca_init(btr, K, dim, fl))))
    res['cand'] = {}
    for tag, m in cands:
        if tag not in res['cand'] or full_ll(m) / len(btr) > res['cand'][tag][0]:
            res['cand'][tag] = (full_ll(m) / len(btr), m.score(btr, bte))
    blind = max(cands, key=lambda c: full_ll(c[1]))
    # iterated local search from the best blind solution: perturb 15% of assignments, refit, keep if better
    bm = blind[1]
    for it in range(8):
        ie = {x: (int(rng.integers(0, K)) if rng.random() < 0.15 else int(bm.ce[i])) for x, i in bm.ei.items()}
        is_ = {x: (int(rng.integers(0, K)) if rng.random() < 0.15 else int(bm.cs[i])) for x, i in bm.si.items()}
        m2 = Model(K, True, rng).fit(btr, init=(ie, is_))
        if full_ll(m2) > full_ll(bm):
            bm = m2
    blind = ('ils', bm)
    res['blind'] = {'ll': full_ll(bm) / len(btr), 'heldout_gain': bm.score(btr, bte),
                    'kernel': [round(float(x), 2) for x in bm.kernel]}
    res['R_blind'] = res['blind']['heldout_gain'] / max(1e-9, res['free']['heldout_gain'])
    if truth:
        lo = min(p for e in truth.values() for p in e)
        ie = {w: min(K - 1, max(0, p[1] - 2)) for w, p in truth.items()}
        is_ = {w: min(K - 1, max(0, p[0] - 2)) for w, p in truth.items()}
        om = Model(K, True, rng).fit(btr, sweeps=0, init=(ie, is_))
        res['oracle_pitch'] = {'ll': full_ll(om) / len(btr), 'heldout_gain': om.score(btr, bte),
                               'kernel': [round(float(x), 2) for x in om.kernel]}
        cands.append(('oracle+sweep', Model(K, True, rng).fit(btr, init=(ie, is_))))
    best = max(cands, key=lambda c: full_ll(c[1]))
    m = best[1]
    res['pitch'] = {'ll': full_ll(m) / len(btr), 'heldout_gain': m.score(btr, bte), 'from': best[0],
                    'kernel': [round(float(x), 2) for x in m.kernel]}
    res['pitch_from_counts'] = dict(__import__('collections').Counter(c[0] for c in cands))
    if truth:
        from scipy.stats import spearmanr
        xs, ys, xe, ye = [], [], [], []
        mb = blind[1]
        xs2, xe2 = [], []
        for w in keep:
            if w in truth and w in mb.si and w in mb.ei:
                xs2.append(mb.cs[mb.si[w]]); xe2.append(mb.ce[mb.ei[w]])
        for w in keep:
            if w in truth and w in m.si and w in m.ei:
                xs.append(m.cs[m.si[w]]); ys.append(truth[w][0]); xe.append(m.ce[m.ei[w]]); ye.append(truth[w][1])
        res['recover_entry_rho'] = float(abs(spearmanr(xs, ys)[0]))
        res['recover_exit_rho'] = float(abs(spearmanr(xe, ye)[0]))
        res['blind_entry_rho'] = float(abs(spearmanr(xs2, ys)[0]))
        res['blind_exit_rho'] = float(abs(spearmanr(xe2, ye)[0]))
    res['R'] = res['pitch']['heldout_gain'] / max(1e-9, res['free']['heldout_gain'])
    return res


if __name__ == '__main__':
    name, K, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    r = run(name, K, R, seed)
    print(json.dumps(r))
    with open(os.path.join(V.CK, 'c2_results.jsonl'), 'a') as f:
        f.write(json.dumps(r) + '\n')
