"""v63 cycle 1: WHICH TWIN IS THE SLIP?  Orientation of adjacent repeats and near-repeats.

For adjacent word pairs (A, B) (lag 1; lag 2 as a layout/position baseline) inside paragraph streams:
  rep    identical pairs per 1000 tokens
  pfx    share of proper-prefix pairs in which the FIRST word is the prefix (false start = first)
  near   pairs at edit distance 1-2 (no prefix): mean log f(A) - log f(B), mean len(A) - len(B)
  delo   deletability orientation: D(A) - D(B) over near pairs, D(x) = log P(next | prev) - log P(next | x)
         (interpolated word-bigram + glyph-junction model trained on the other half of the paragraphs):
         positive = the line reads better without the first twin (a slip written before the intended word)
  auc    for corpora with known slips: AUC of D for slips vs other tokens (detector sanity)
Corpora: Voynich ZL3b/IT2a; Plaoul Latin (opaque verbose code) clean, as really written (real struck words
kept), clean + planted real-mix slips at 1/3/10%; nulls: Voynich within-paragraph shuffle (x5), Markov-1
resynthesis (x3), copy-and-modify generator (x3).
"""
import json, os, sys, math, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v63_lib as L
import v63_plant as PL


class BiLM:
    def __init__(self, streams, lam=0.6):
        self.uni, self.bi, self.cnt = Counter(), Counter(), Counter()
        self.jn, self.jc, self.fg = Counter(), Counter(), Counter()
        for s in streams:
            ws = s['words']
            self.uni.update(ws)
            for a, b in zip(ws, ws[1:]):
                self.bi[(a, b)] += 1; self.cnt[a] += 1
                self.jn[(a[-1], b[0])] += 1; self.jc[a[-1]] += 1
            self.fg.update(w[0] for w in ws)
        self.N = sum(self.uni.values()); self.V = len(self.uni) + 1
        self.G = len(self.fg) + 1; self.lam = lam
        self.byfirst = Counter()
        for w, c in self.uni.items(): self.byfirst[w[0]] += c

    def lp(self, b, a):
        """log P(b | a): lam * bigram ML + (1-lam) * P(first glyph | last glyph) * P(word | first glyph)."""
        pj = (self.jn[(a[-1], b[0])] + 0.5) / (self.jc[a[-1]] + 0.5 * self.G)
        pw = (self.uni[b] + 0.1) / (self.byfirst[b[0]] + 0.1 * self.V)
        pb = self.bi[(a, b)] / self.cnt[a] if self.cnt[a] else 0.0
        return math.log(self.lam * pb + (1 - self.lam) * pj * pw)


def deletability(streams, seed=0):
    """D per token (None at paragraph edges), 2-fold over paragraphs."""
    idx = list(range(len(streams))); random.Random(seed).shuffle(idx)
    half = set(idx[:len(idx) // 2])
    D = [None] * len(streams)
    for fold in (0, 1):
        train = [s for k, s in enumerate(streams) if (k in half) != (fold == 0)]
        lm = BiLM(train)
        for k, s in enumerate(streams):
            if (k in half) != (fold == 1): continue
            ws = s['words']; d = [None] * len(ws)
            for i in range(1, len(ws) - 1):
                d[i] = lm.lp(ws[i + 1], ws[i - 1]) - lm.lp(ws[i + 1], ws[i])
            D[k] = d
    return D


def is_pfx(a, b):
    return len(a) < len(b) and b[:len(a)] == a


def metrics(streams, with_D=True):
    f = Counter(w for s in streams for w in s['words'])
    ntok = sum(len(s['words']) for s in streams)
    D = deletability(streams) if with_D else None
    out = {}
    for lag in (1, 2):
        rep = 0; pa = pb = 0; lf, ln, dor = [], [], []
        for k, s in enumerate(streams):
            ws = s['words']
            for i in range(len(ws) - lag):
                A, B = ws[i], ws[i + lag]
                if A == B: rep += 1; continue
                if is_pfx(A, B): pa += 1; continue
                if is_pfx(B, A): pb += 1; continue
                if abs(len(A) - len(B)) <= 2 and L.ed(A, B) <= 2:
                    lf.append(math.log(f[A]) - math.log(f[B])); ln.append(len(A) - len(B))
                    if lag == 1 and D is not None and D[k][i] is not None and D[k][i + 1] is not None:
                        dor.append(D[k][i] - D[k][i + 1])
        o = dict(rep=1000 * rep / ntok, pfx_first=pa / max(1, pa + pb), npfx=pa + pb,
                 near_rate=1000 * len(lf) / ntok, near_dlogf=float(np.mean(lf)) if lf else 0,
                 near_dlen=float(np.mean(ln)) if ln else 0)
        if lag == 1 and dor:
            o['delo'] = float(np.mean(dor)); o['delo_se'] = float(np.std(dor) / math.sqrt(len(dor))); o['ndelo'] = len(dor)
        out[lag] = o
    if D is not None and all('lab' in s for s in streams):
        pos, neg = [], []
        for k, s in enumerate(streams):
            for i, x in enumerate(D[k]):
                if x is None: continue
                (pos if s['lab'][i] else neg).append(x)
        if pos:
            neg = np.array(neg); pos = np.array(pos)
            r = np.argsort(np.argsort(np.concatenate([pos, neg])))
            auc = (r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
            out['auc'] = float(auc); out['nslip'] = len(pos)
    return out


def fmt(name, m):
    a, b = m[1], m[2]
    s = (f"{name:28s} rep {a['rep']:.2f}/{b['rep']:.2f}  pfx_first {a['pfx_first']:.3f} (n {a['npfx']}) / {b['pfx_first']:.3f}"
         f"  near {a['near_rate']:.1f}/{b['near_rate']:.1f}  dlogf {a['near_dlogf']:+.3f}/{b['near_dlogf']:+.3f}"
         f"  dlen {a['near_dlen']:+.3f}/{b['near_dlen']:+.3f}")
    if 'delo' in a: s += f"  delo {a['delo']:+.3f}+-{a['delo_se']:.3f} (n {a['ndelo']})"
    if 'auc' in m: s += f"  AUC(D,slips) {m['auc']:.3f} (n {m['nslip']})"
    return s


def main():
    res = {}
    C = {}
    C['V-ZL3b'] = L.voynich_paras('ZL3b')
    C['V-IT2a'] = L.voynich_paras('IT2a')
    lat = PL.latin_clean()
    C['LAT clean (opaque)'] = lat
    C['LAT as written (real slips)'] = PL.latin_as_written(wit='svict')
    for r in (0.01, 0.03, 0.10):
        C[f'LAT + planted slips {int(r*100)}%'] = PL.plant_slips(lat, r, seed=1)
    for name in list(C):
        m = metrics(C[name]); res[name] = m; print(fmt(name, m), flush=True)
    zl = C['V-ZL3b']
    for nm, fn, reps in (('V-ZL shuffle', PL.shuffle_streams, 5), ('V-ZL markov1', PL.markov1, 3), ('V-ZL copygen', PL.copygen, 3),
                         ('LAT clean shuffle', None, 3), ('LAT clean markov1', None, 3)):
        ms = []
        for sd in range(reps):
            if nm.startswith('LAT'):
                S = (PL.shuffle_streams if 'shuffle' in nm else PL.markov1)(lat, seed=sd)
            else:
                S = fn(zl, seed=sd)
            ms.append(metrics(S))
        res[nm] = ms
        for k, m in enumerate(ms):
            print(fmt(f'{nm} #{k}', m), flush=True)
    json.dump(res, open(os.path.join(L.CK, 'c1_metrics.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
