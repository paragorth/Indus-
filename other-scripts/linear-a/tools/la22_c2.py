#!/usr/bin/env python3
"""la22 cycle 2: restore masked NUMBERS and LOGOGRAMS; accounting vs lexical experts.

usage: python3 la22_c2.py CORPUS [seed] [nrand]   CORPUS = LA | LAshuf | LB
Numbers: candidate values 1..VMAX; experts
  GLOB  corpus value frequency (LOO doc) + 1/v tail
  ROUND multiples of 10 / 5 prior
  SCALE log-normal around the other numbers on the tablet           (accounting)
  REP   values written elsewhere on the tablet                      (accounting)
  TOT   arithmetic: entries since the last total must sum to the KU-RO / to-so total (accounting)
  WNUM  values after the same preceding word elsewhere (LOO)        (lexical)
  LNUM  log-magnitude after the same commodity logogram (LOO)       (lexical)
Logograms: candidate types; experts GLOB, DOCL (other logograms on the tablet), PREVW (preceding
word, LOO), FRAC (fraction letters of the following number, LOO), MAG (magnitude of the following
number, LOO), SITE.
"""
import sys, os, json, time, math, hashlib
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la22_data
from la22_c1 import load_corpus, metrics, combine, random_ensemble, doc_fold

CK = la22_data.CK
VMAX = 2000
EPS = 1e-7
TOTAL_WORDS = {('KU', 'RO'), ('PO', 'TO', 'KU', 'RO'), ('to', 'so'), ('to', 'sa'), ('to', 'so', 'de')}
NUM_EXPERTS = ['GLOB', 'ROUND', 'SCALE', 'REP', 'TOT', 'WNUM', 'LNUM']
NUM_GROUPS = {'accounting': ['SCALE', 'REP', 'TOT'], 'lexical': ['WNUM', 'LNUM'], 'base': ['GLOB', 'ROUND']}
LOG_EXPERTS = ['GLOB', 'DOCL', 'PREVW', 'FRAC', 'MAG', 'SITE']
LOG_GROUPS = {'accounting': ['DOCL', 'FRAC', 'MAG'], 'lexical': ['PREVW'], 'meta': ['SITE'], 'base': ['GLOB']}


def word_key(t):
    return tuple(t['tr']) if t['t'] == 'w' else None


def prev_content(T, i):
    for k in range(i - 1, -1, -1):
        if T[k]['t'] in ('div',): continue
        if T[k]['t'] in ('w', 'L'): return T[k]
        if T[k]['t'] in ('N', 'nl', 'gap', 'unk'): return None
    return None


def prev_word(T, i):
    for k in range(i - 1, -1, -1):
        t = T[k]['t']
        if t == 'w': return tuple(T[k]['tr'])
        if t in ('N', 'nl', 'gap'): return None
    return None


def prev_logo(T, i):
    for k in range(i - 1, -1, -1):
        t = T[k]['t']
        if t == 'L': return T[k]['c']
        if t in ('N', 'nl', 'gap', 'w'): return None
    return None


def sections(T):
    """yield (total_index, [entry indices]) for each total word followed by a number."""
    out = []; entries = []
    for i, t in enumerate(T):
        if t['t'] == 'w' and tuple(t['tr']) in TOTAL_WORDS:
            j = i + 1
            while j < len(T) and T[j]['t'] in ('div', 'L'): j += 1
            if j < len(T) and T[j]['t'] == 'N' and T[j]['v'] > 0:
                out.append((j, list(entries)))
                entries = []
            continue
        if t['t'] == 'N' and t['v'] > 0:
            if out and out[-1][0] == i: continue
            entries.append(i)
    return out


class NumModel:
    def __init__(self, docs):
        self.docs = docs
        self.G = Counter(); self.Gd = []
        self.WN = defaultdict(Counter); self.WNd = []
        self.LN = defaultdict(list); self.LNd = []
        for d in docs:
            T = d['toks']; g = Counter(); wn = defaultdict(Counter); ln = defaultdict(list)
            for i, t in enumerate(T):
                if t['t'] == 'N' and 0 < t['v'] <= VMAX and not t.get('dot'):
                    g[t['v']] += 1
                    w = prev_word(T, i)
                    if w: wn[w][t['v']] += 1
                    l = prev_logo(T, i)
                    if l: ln[l].append(math.log(t['v']))
            self.Gd.append(g); self.WNd.append(wn); self.LNd.append(ln)
            self.G.update(g)
            for w, c in wn.items(): self.WN[w].update(c)
            for l, v in ln.items(): self.LN[l].extend(v)
        self.vals = np.arange(1, VMAX + 1)
        self.logv = np.log(self.vals)

    def experts(self, di, i):
        d = self.docs[di]; T = d['toks']; n = VMAX
        g = np.zeros(n)
        for v, c in self.G.items(): g[v - 1] += c
        for v, c in self.Gd[di].items(): g[v - 1] -= c
        tail = 1.0 / self.vals; tail /= tail.sum()
        glob = 0.9 * g / max(g.sum(), 1) + 0.1 * tail
        E = {'GLOB': glob}
        r = np.ones(n); r[self.vals % 10 == 0] = 3; r[self.vals % 5 == 0] *= 1.5; r[self.vals < 10] = 6
        E['ROUND'] = r / r.sum()
        others = [t['v'] for k, t in enumerate(T) if k != i and t['t'] == 'N' and 0 < t['v'] <= VMAX]
        if others:
            lo = np.log(others); mu = lo.mean(); sd = max(lo.std(), 0.6)
            sc = np.exp(-0.5 * ((self.logv - mu) / sd) ** 2) / self.vals
            E['SCALE'] = 0.95 * sc / sc.sum() + 0.05 * glob
            rp = np.zeros(n)
            for v in others: rp[v - 1] += 1
            E['REP'] = 0.7 * rp / rp.sum() + 0.3 * glob
        else:
            E['SCALE'] = glob; E['REP'] = glob
        tot = None
        for ti, ents in sections(T):
            if ti == i and ents:
                tot = sum(T[k]['v'] for k in ents)
            elif i in ents:
                rest = T[ti]['v'] - sum(T[k]['v'] for k in ents if k != i)
                tot = rest
        if tot is not None and 1 <= tot <= VMAX:
            tv = np.zeros(n); tv[tot - 1] = 1.0
            # near misses (slips of 1 / 10 seen in la1) get a little mass
            for dlt, w in ((1, .15), (-1, .15), (10, .08), (-10, .08)):
                if 1 <= tot + dlt <= VMAX: tv[tot + dlt - 1] += w
            E['TOT'] = 0.6 * tv / tv.sum() + 0.4 * glob
        else:
            E['TOT'] = glob
        w = prev_word(T, i)
        if w and w in self.WN:
            c = self.WN[w].copy(); c.subtract(self.WNd[di].get(w, Counter()))
            wv = np.zeros(n)
            for v, k in c.items():
                if k > 0: wv[v - 1] += k
            E['WNUM'] = (wv + 0.5 * glob * 1) / (wv.sum() + 0.5) if wv.sum() > 0 else glob
        else:
            E['WNUM'] = glob
        l = prev_logo(T, i)
        if l and l in self.LN:
            own = Counter(self.LNd[di].get(l, []))
            lst = [x for x in self.LN[l]]
            for x, k in own.items():
                for _ in range(k): lst.remove(x)
            if len(lst) >= 2:
                mu = np.mean(lst); sd = max(np.std(lst), 0.6)
                sc = np.exp(-0.5 * ((self.logv - mu) / sd) ** 2) / self.vals
                E['LNUM'] = 0.9 * sc / sc.sum() + 0.1 * glob
            else:
                E['LNUM'] = glob
        else:
            E['LNUM'] = glob
        return E


class LogoModel:
    def __init__(self, docs):
        self.docs = docs
        types = Counter(t['c'] for d in docs for t in d['toks'] if t['t'] == 'L' and not t.get('dot'))
        self.V = [k for k, c in types.items()]
        self.ix = {k: i for i, k in enumerate(self.V)}
        nV = len(self.V)
        self.G = np.zeros(nV); self.Gd = []
        self.PW = defaultdict(lambda: np.zeros(nV)); self.PWd = []
        self.FR = defaultdict(lambda: np.zeros(nV)); self.FRd = []
        self.MG = defaultdict(lambda: np.zeros(nV)); self.MGd = []
        self.ST = defaultdict(lambda: np.zeros(nV))
        for d in docs:
            T = d['toks']; g = np.zeros(nV); pw = defaultdict(Counter); fr = defaultdict(Counter); mg = defaultdict(Counter)
            for i, t in enumerate(T):
                if t['t'] != 'L' or t.get('dot'): continue
                k = self.ix[t['c']]; g[k] += 1
                w = prev_word(T, i)
                if w: pw[w][k] += 1
                nx = T[i + 1] if i + 1 < len(T) else None
                if nx and nx['t'] == 'N':
                    for f in nx['frac']: fr[f][k] += 1
                    mg[self.mag(nx['v'])][k] += 1
            self.Gd.append(g); self.G += g; self.ST[d['site']] += g
            self.PWd.append(pw); self.FRd.append(fr); self.MGd.append(mg)
            for w, c in pw.items():
                for k, n in c.items(): self.PW[w][k] += n
            for f, c in fr.items():
                for k, n in c.items(): self.FR[f][k] += n
            for m, c in mg.items():
                for k, n in c.items(): self.MG[m][k] += n

    @staticmethod
    def mag(v):
        return 'f' if v == 0 else ('1' if v == 1 else ('s' if v < 10 else ('m' if v < 100 else 'l')))

    def experts(self, di, i):
        d = self.docs[di]; T = d['toks']; nV = len(self.V)
        g = self.G - self.Gd[di]
        glob = (g + 0.1) / (g + 0.1).sum()
        def nm(v, lam=1.0):
            v = np.maximum(v, 0)
            return (v + lam * glob) / (v.sum() + lam) if v.sum() > 0 else glob
        E = {'GLOB': glob}
        dl = np.zeros(nV)
        for k, t in enumerate(T):
            if k != i and t['t'] == 'L' and t['c'] in self.ix: dl[self.ix[t['c']]] += 1
        E['DOCL'] = nm(dl, 1.0)
        w = prev_word(T, i)
        if w and w in self.PW:
            v = self.PW[w].copy()
            for k, n in self.PWd[di].get(w, {}).items(): v[k] -= n
            E['PREVW'] = nm(v, 0.5)
        else:
            E['PREVW'] = glob
        nx = T[i + 1] if i + 1 < len(T) else None
        if nx and nx['t'] == 'N' and nx['frac']:
            acc = np.zeros(nV)
            for f in nx['frac']:
                v = self.FR[f].copy()
                for k, n in self.FRd[di].get(f, {}).items(): v[k] -= n
                acc += np.maximum(v, 0)
            E['FRAC'] = nm(acc, 1.0)
        else:
            E['FRAC'] = glob
        if nx and nx['t'] == 'N':
            m = self.mag(nx['v']); v = self.MG[m].copy()
            for k, n in self.MGd[di].get(m, {}).items(): v[k] -= n
            E['MAG'] = nm(v, 2.0)
        else:
            E['MAG'] = glob
        E['SITE'] = nm(self.ST[d['site']] - self.Gd[di], 10.0)
        return E


def run_task(kind, D, seed, nrand):
    if kind == 'num':
        M = NumModel(D); names = NUM_EXPERTS; groups = NUM_GROUPS
        insts = [(di, i) for di, d in enumerate(D) for i, t in enumerate(d['toks'])
                 if t['t'] == 'N' and 0 < t['v'] <= VMAX and not t.get('dot')]
        y = np.array([D[di]['toks'][i]['v'] - 1 for di, i in insts]); nV = VMAX
    else:
        M = LogoModel(D); names = LOG_EXPERTS; groups = LOG_GROUPS
        insts = [(di, i) for di, d in enumerate(D) for i, t in enumerate(d['toks'])
                 if t['t'] == 'L' and not t.get('dot')]
        y = np.array([M.ix[D[di]['toks'][i]['c']] for di, i in insts]); nV = len(M.V)
    if len(insts) > 6000:
        rs = np.random.default_rng(seed); sel = sorted(rs.choice(len(insts), 6000, replace=False))
        insts = [insts[k] for k in sel]; y = y[sel]
    LP = np.zeros((len(insts), len(names), nV), dtype=np.float32)
    for k, (di, i) in enumerate(insts):
        E = M.experts(di, i)
        for e, nm in enumerate(names): LP[k, e] = np.log(E[nm] + EPS)
    fold = np.array([doc_fold(D[di]['id']) for di, i in insts])
    rng = np.random.default_rng(seed + 11)
    P = np.zeros((len(insts), nV), dtype=np.float32); W = []
    for fa in (0, 1):
        A = fold == fa; B = ~A
        PB, wbar, lA, order = random_ensemble(LP, y, A, B, names, nrand, rng); P[B] = PB; W.append(wbar)
    base = np.exp(LP[:, names.index('GLOB')]); base /= base.sum(1, keepdims=True)
    out = {'n': len(insts), 'nV': nV, 'ensemble': metrics(P, y), 'glob_only': metrics(base, y),
           'weights': dict(zip(names, np.round(np.mean(W, 0), 3).tolist())), 'single': {}, 'ablation': {}}
    for e, nm in enumerate(names):
        Pe = np.exp(LP[:, e]); Pe /= Pe.sum(1, keepdims=True); out['single'][nm] = metrics(Pe, y)['top1']
    for g, mem in groups.items():
        if g == 'base': continue
        Pw = np.zeros_like(P); Po = np.zeros_like(P)
        for fa in (0, 1):
            A = fold == fa; B = ~A
            Pw[B] = random_ensemble(LP, y, A, B, names, nrand // 4, rng, allowed=[n for n in names if n not in mem])[0]
            Po[B] = random_ensemble(LP, y, A, B, names, nrand // 4, rng, allowed=mem + ['GLOB'])[0]
        out['ablation'][g] = {'without': metrics(Pw, y)['top1'], 'only_plus_glob': metrics(Po, y)['top1']}
    if kind == 'num':  # subset where a total constrains the value
        has = np.array([np.exp(LP[k, names.index('TOT')]).max() > 0.3 for k in range(len(insts))])
        out['with_total'] = {'n': int(has.sum()), 'ensemble': metrics(P[has], y[has]), 'glob_only': metrics(base[has], y[has])}
        out['without_total'] = {'n': int((~has).sum()), 'ensemble': metrics(P[~has], y[~has])}
    return out


def main():
    name = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    nrand = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    t0 = time.time()
    D = load_corpus(name, seed)
    if name == 'LB':   # numbers/logograms: use a larger LB sample matched on number tokens
        pass
    res = {'corpus': name}
    for kind in ('num', 'logo'):
        res[kind] = run_task(kind, D, seed, nrand)
        print(name, kind, json.dumps(res[kind]['ensemble']), 'glob', json.dumps(res[kind]['glob_only']), round(time.time() - t0), flush=True)
    json.dump(res, open(os.path.join(CK, 'c2_%s.json' % name), 'w'), indent=1)


if __name__ == '__main__':
    main()
