#!/usr/bin/env python3
"""la22 engine: expert predictors for a masked syllabic sign, leave-one-document-out.

An instance is (doc, token, sign position j, mode):
  int : sign j masked, rest of the word visible, word length known
  R   : right edge broken; signs[:j] survive, target = signs[j], length unknown
  L   : left edge broken; signs[j+1:] survive, target = signs[j], length unknown
Every expert returns a probability vector over the syllabic sign inventory V.
All corpus counts exclude the instance's own document (LOO-doc); the document's OTHER
tokens are visible only through the tablet experts (DOCW, DOCS).
"""
import numpy as np
from collections import Counter, defaultdict

BOW, EOW, UNKN = '<', '>', '?'

EXPERTS = ['FREQ', 'POS', 'BIL', 'BIR', 'TRI', 'LEX', 'LEX1', 'DOCW', 'DOCS',
           'SITE', 'SCRIBE', 'SUPP', 'SLOT', 'NNL', 'NND']
GROUPS = {'language': ['POS', 'BIL', 'BIR', 'TRI', 'LEX', 'LEX1', 'NNL'],
          'tablet': ['DOCW', 'DOCS', 'NND'],
          'meta': ['SITE', 'SCRIBE', 'SUPP'],
          'accounting_slot': ['SLOT'],
          'base': ['FREQ']}


def next_kind(T, i):
    for k in range(i + 1, len(T)):
        t = T[k]['t']
        if t == 'div': continue
        if t == 'N': return 'N'
        if t == 'L': return 'L'
        if t == 'w': return 'w'
        return 'x'
    return 'x'


class Corpus:
    def __init__(self, docs, vocab=None):
        self.docs = docs
        if vocab is None:
            vocab = sorted({c for d in docs for t in d['toks'] if t['t'] == 'w' for c in t['c']})
        self.V = vocab
        self.ix = {s: k for k, s in enumerate(vocab)}
        self.nV = len(vocab)
        self._count()

    def _doc_counts(self, d):
        u = Counter(); pos = Counter(); bi = Counter(); tri = Counter(); words = Counter()
        site = Counter(); slot = Counter()
        T = d['toks']
        for i, t in enumerate(T):
            if t['t'] != 'w' or t.get('dot'): continue
            s = t['c']
            words[tuple(s)] += 1
            nk = next_kind(T, i)
            ext = [BOW] + s + [EOW]
            n = len(s)
            for j, c in enumerate(s):
                u[c] += 1
                pc = 'S' if n == 1 else ('I' if j == 0 else ('F' if j == n - 1 else 'M'))
                pos[(pc, c)] += 1
                if j == n - 1: slot[(nk, c)] += 1
            for a, b in zip(ext, ext[1:]): bi[(a, b)] += 1
            for a, b, c in zip(ext, ext[1:], ext[2:]): tri[(a, b, c)] += 1
        return dict(u=u, pos=pos, bi=bi, tri=tri, words=words, slot=slot)

    def _count(self):
        self.dc = [self._doc_counts(d) for d in self.docs]
        G = defaultdict(Counter)
        for c in self.dc:
            for k, v in c.items(): G[k].update(v)
        self.G = G
        # index structures over vectors
        nV = self.nV
        self.U = np.zeros(nV)
        for s, n in G['u'].items(): self.U[self.ix[s]] = n
        self.siteU = defaultdict(lambda: np.zeros(nV)); self.scrU = defaultdict(lambda: np.zeros(nV))
        self.supU = defaultdict(lambda: np.zeros(nV))
        self.docU = []
        for d, c in zip(self.docs, self.dc):
            v = np.zeros(nV)
            for s, n in c['u'].items(): v[self.ix[s]] = n
            self.docU.append(v)
            self.siteU[d['site']] += v
            if d['scribe']: self.scrU[d['scribe']] += v
            self.supU[d['support']] += v
        # bigram vectors: left context a -> vec over b ; right context b -> vec over a
        self.biL = defaultdict(lambda: np.zeros(nV)); self.biR = defaultdict(lambda: np.zeros(nV))
        for (a, b), n in G['bi'].items():
            if b in self.ix: self.biL[a][self.ix[b]] += n
            if a in self.ix: self.biR[b][self.ix[a]] += n
        self.triB = defaultdict(lambda: np.zeros(nV)); self.triL = defaultdict(lambda: np.zeros(nV))
        self.triR = defaultdict(lambda: np.zeros(nV))
        for (a, b, c), n in G['tri'].items():
            if b in self.ix: self.triB[(a, c)][self.ix[b]] += n
            if c in self.ix: self.triL[(a, b)][self.ix[c]] += n
            if a in self.ix: self.triR[(b, c)][self.ix[a]] += n
        self.posV = defaultdict(lambda: np.zeros(nV))
        for (pc, s), n in G['pos'].items(): self.posV[pc][self.ix[s]] += n
        self.slotV = defaultdict(lambda: np.zeros(nV))
        for (k, s), n in G['slot'].items(): self.slotV[k][self.ix[s]] += n
        self.wordlist = list(G['words'].items())

    # ---- helpers -------------------------------------------------------
    def _loo(self, glob, di, kind, key, builder):
        v = glob[key].copy() if key in glob else np.zeros(self.nV)
        own = self.dc[di][kind]
        # subtract own-document contribution
        for k2, n in builder(own, key):
            v[self.ix[k2]] -= n
        return np.maximum(v, 0)

    def instances(self, modes=('int', 'R', 'L'), clean_only=True, docs=None):
        out = []
        for di, d in enumerate(self.docs):
            if docs is not None and di not in docs: continue
            for ti, t in enumerate(d['toks']):
                if t['t'] != 'w' or (clean_only and t.get('dot')): continue
                n = len(t['c'])
                for j in range(n):
                    if 'int' in modes: out.append((di, ti, j, 'int'))
                    if 'R' in modes and j >= 1: out.append((di, ti, j, 'R'))
                    if 'L' in modes and j <= n - 2: out.append((di, ti, j, 'L'))
        return out

    def context(self, inst):
        di, ti, j, mode = inst
        s = self.docs[di]['toks'][ti]['c']
        n = len(s)
        if mode == 'int':
            left = [BOW] + s[:j]; right = s[j + 1:] + [EOW]; L = n
        elif mode == 'R':
            left = [BOW] + s[:j]; right = [UNKN]; L = None
        else:
            left = [UNKN]; right = s[j + 1:] + [EOW]; L = None
        return s, left, right, L

    def experts(self, inst, sm):
        """sm: smoothing dict. Returns dict name -> prob vector (sums to 1)."""
        di, ti, j, mode = inst
        d = self.docs[di]; T = d['toks']
        s, left, right, L = self.context(inst)
        nV = self.nV; own = self.dc[di]
        a0 = sm.get('add', 0.1)
        U = np.maximum(self.U - self.docU[di], 0)
        freq = (U + a0) / (U + a0).sum()
        E = {'FREQ': freq}

        def norm(v, back, lam):
            tot = v.sum()
            if tot <= 0: return back
            return (v + lam * back * 1.0) / (tot + lam)

        # POS
        if mode == 'int':
            pc = 'S' if L == 1 else ('I' if j == 0 else ('F' if j == L - 1 else 'M'))
            pv = self.posV[pc].copy()
            for (p2, c2), n2 in own['pos'].items():
                if p2 == pc: pv[self.ix[c2]] -= n2
        elif mode == 'R':  # non-initial
            pv = self.posV['M'] + self.posV['F']
            pv = pv.copy()
            for (p2, c2), n2 in own['pos'].items():
                if p2 in ('M', 'F'): pv[self.ix[c2]] -= n2
        else:              # non-final
            pv = (self.posV['M'] + self.posV['I']).copy()
            for (p2, c2), n2 in own['pos'].items():
                if p2 in ('M', 'I'): pv[self.ix[c2]] -= n2
        E['POS'] = norm(np.maximum(pv, 0), freq, sm.get('lam', 2.0))

        # bigrams
        lam = sm.get('lam', 2.0)
        if left[-1] != UNKN:
            a = left[-1]
            v = self.biL[a].copy() if a in self.biL else np.zeros(nV)
            for (x, y), n2 in own['bi'].items():
                if x == a and y in self.ix: v[self.ix[y]] -= n2
            E['BIL'] = norm(np.maximum(v, 0), freq, lam)
        else:
            E['BIL'] = freq
        if right[0] != UNKN:
            b = right[0]
            v = self.biR[b].copy() if b in self.biR else np.zeros(nV)
            for (x, y), n2 in own['bi'].items():
                if y == b and x in self.ix: v[self.ix[x]] -= n2
            E['BIR'] = norm(np.maximum(v, 0), freq, lam)
        else:
            E['BIR'] = freq
        # trigram (both sides if possible, else one side)
        if left[-1] != UNKN and right[0] != UNKN:
            key = (left[-1], right[0]); v = self.triB[key].copy() if key in self.triB else np.zeros(nV)
            for (x, y, z), n2 in own['tri'].items():
                if (x, z) == key and y in self.ix: v[self.ix[y]] -= n2
            back = E['BIL'] * E['BIR']; back /= back.sum()
        elif left[-1] != UNKN and len(left) >= 2:
            key = (left[-2], left[-1]); v = self.triL[key].copy() if key in self.triL else np.zeros(nV)
            for (x, y, z), n2 in own['tri'].items():
                if (x, y) == key and z in self.ix: v[self.ix[z]] -= n2
            back = E['BIL']
        elif right[0] != UNKN and len(right) >= 2:
            key = (right[0], right[1]); v = self.triR[key].copy() if key in self.triR else np.zeros(nV)
            for (x, y, z), n2 in own['tri'].items():
                if (y, z) == key and x in self.ix: v[self.ix[x]] -= n2
            back = E['BIR']
        else:
            v = np.zeros(nV); back = E['BIL'] * E['BIR']; back /= back.sum()
        E['TRI'] = norm(np.maximum(v, 0), back, sm.get('lam3', 1.0))

        # lexicon matching
        def match_vec(wordcounts, exclude_own=None, allow_edit=False):
            v = np.zeros(nV); v1 = np.zeros(nV)
            for w, cnt in wordcounts:
                if exclude_own is not None:
                    cnt = cnt - exclude_own.get(w, 0)
                if cnt <= 0: continue
                lw = len(w)
                if mode == 'int':
                    if lw != L: continue
                    mm = sum(1 for k in range(lw) if k != j and w[k] != s[k])
                    tgt = w[j]
                elif mode == 'R':
                    if lw <= j: continue
                    mm = sum(1 for k in range(j) if w[k] != s[k])
                    tgt = w[j]
                else:
                    m = len(s) - j - 1  # surviving suffix length
                    if lw <= m: continue
                    mm = sum(1 for k in range(m) if w[lw - m + k] != s[j + 1 + k])
                    tgt = w[lw - m - 1]
                if tgt not in self.ix: continue
                if mm == 0: v[self.ix[tgt]] += cnt
                elif mm == 1 and allow_edit: v1[self.ix[tgt]] += cnt
            return v, v1

        v, v1 = match_vec(self.wordlist, exclude_own=own['words'], allow_edit=True)
        E['LEX'] = norm(v, freq, sm.get('lamlex', 0.5))
        E['LEX1'] = norm(v1, freq, sm.get('lamlex', 0.5) * 4)
        # tablet: other words on this document (exclude the target token itself)
        oth = Counter()
        for k, t in enumerate(T):
            if k != ti and t['t'] == 'w' and not t.get('dot'): oth[tuple(t['c'])] += 1
        v, _ = match_vec(list(oth.items()))
        E['DOCW'] = norm(v, freq, sm.get('lamdoc', 0.5))
        dv = np.zeros(nV)
        for w, cnt in oth.items():
            for c in w:
                if c in self.ix: dv[self.ix[c]] += cnt
        E['DOCS'] = norm(dv, freq, sm.get('lamdocs', 5.0))
        # meta priors (LOO doc)
        sv = np.maximum(self.siteU[d['site']] - self.docU[di], 0)
        E['SITE'] = norm(sv, freq, 20.0)
        if d['scribe'] and d['scribe'] in self.scrU:
            E['SCRIBE'] = norm(np.maximum(self.scrU[d['scribe']] - self.docU[di], 0), freq, 20.0)
        else:
            E['SCRIBE'] = freq
        E['SUPP'] = norm(np.maximum(self.supU[d['support']] - self.docU[di], 0), freq, 20.0)
        # accounting slot: last sign conditioned on what follows the word (number, logogram, word)
        if mode == 'int' and j == L - 1 or mode == 'R':
            nk = next_kind(T, ti)
            v = self.slotV[nk].copy()
            for (k2, c2), n2 in own['slot'].items():
                if k2 == nk: v[self.ix[c2]] -= n2
            sl = norm(np.maximum(v, 0), freq, 5.0)
            if mode == 'R':  # target final only sometimes: mix with POS
                sl = 0.5 * sl + 0.5 * E['POS']
            E['SLOT'] = sl
        else:
            E['SLOT'] = freq
        return E


def feats_for_nn(C, inst):
    """integer features for the neural experts."""
    s, left, right, L = C.context(inst)
    def code(x):
        if x in C.ix: return C.ix[x] + 3
        return {BOW: 0, EOW: 1, UNKN: 2}.get(x, 2)
    l1 = code(left[-1]); l2 = code(left[-2]) if len(left) >= 2 else 2
    r1 = code(right[0]); r2 = code(right[1]) if len(right) >= 2 else 2
    mode = {'int': 0, 'R': 1, 'L': 2}[inst[3]]
    lb = min(L, 6) if L else 0
    return [l2, l1, r1, r2, mode, lb]
