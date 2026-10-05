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

    def fit(self, big, sweeps=12):
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
        for sw in range(sweeps):
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
            if moved == 0:
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
    if name in ('ZL', 'IT'):
        vz = L53.load_voynich('ZL3b' if name == 'ZL' else 'IT2a')
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


def run(name, K, R, seed=0):
    rng = np.random.default_rng(seed); prng = random.Random(seed)
    tr, te, truth = corpus(name, prng)
    keep = vocab(tr)
    btr, bte = build(tr, keep), build(te, keep)
    res = {'name': name, 'K': K, 'ntr': len(btr), 'nte': len(bte), 'nkeep': len(keep)}
    for tp in (False, True):
        best = None
        for r in range(R):
            m = Model(K, tp, rng).fit(btr)
            ll = m.train_ll()
            if best is None or ll > best[0]:
                best = (ll, m)
        m = best[1]
        key = 'pitch' if tp else 'free'
        res[key] = {'train_ll_per': best[0] / len(btr) / math.log(2), 'heldout_gain': m.score(btr, bte)}
        if tp:
            res['kernel'] = [round(float(x), 2) for x in m.kernel]
            if truth:
                # recovery: correlation of fitted entry class with true entry pitch over kept words
                xs, ys, xe, ye = [], [], [], []
                for w in keep:
                    if w in truth and w in m.si and w in m.ei:
                        xs.append(m.cs[m.si[w]]); ys.append(truth[w][0])
                        xe.append(m.ce[m.ei[w]]); ye.append(truth[w][1])
                from scipy.stats import spearmanr
                res['recover_entry_rho'] = float(abs(spearmanr(xs, ys)[0]))
                res['recover_exit_rho'] = float(abs(spearmanr(xe, ye)[0]))
    res['R'] = res['pitch']['heldout_gain'] / max(1e-9, res['free']['heldout_gain'])
    return res


if __name__ == '__main__':
    name, K, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    r = run(name, K, R, seed)
    print(json.dumps(r))
    with open(os.path.join(V.CK, 'c2_results.jsonl'), 'a') as f:
        f.write(json.dumps(r) + '\n')
