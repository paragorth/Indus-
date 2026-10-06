#!/usr/bin/env python3
"""LA-60 generative models of whole documents. Every model predicts the same events:
each token's kind and identity (word, logogram, number, line break, end) and every number's
integer value and fraction string. Bits are summed over held-out documents.

  B0   unigram over symbols
  B1   interpolated (Witten-Bell) trigram over symbols
  G(R) the reading's grammar: role trigram (roles from reading R) x identity within role
       x number given context, plus (a) arithmetic: the number after a TOT word is the running
       sum with probability theta, (b) commodity order: a commodity written after another is
       re-weighted by exp(beta * agreement with R's order)
  MIX  0.5 B1 + 0.5 G(R), per token
Shared by all: the spelling model for unseen types and the number model's backbone.
"""
import math
from collections import Counter, defaultdict
from la60_common import role_of, base_of, _num_after, close_test

LOG2 = math.log(2)


def nbin(v):
    return 0 if v <= 0 else int(math.floor(math.log2(v))) + 1


def bin_size(b):
    return 1 if b == 0 else 2 ** (b - 1)


class Spell:
    """P(new type) by its sign string: length geometric x sign unigram (add-0.5)."""
    def __init__(self, train):
        self.c = Counter(); self.L = Counter(); n = 0
        for d in train:
            for t in d['toks']:
                if t[0] in ('W', 'L'):
                    parts = t[1].split('-') if t[0] == 'W' else t[1].split('+')
                    self.L[len(parts)] += 1; n += 1
                    for p in parts:
                        self.c[p] += 1
        self.N = sum(self.c.values()); self.V = len(self.c) + 1; self.n = n

    def lp(self, w, kind):
        parts = w.split('-') if kind == 'W' else w.split('+')
        l = len(parts)
        lp = math.log((self.L[l] + 0.5) / (self.n + 0.5 * 12))
        for p in parts:
            lp += math.log((self.c[p] + 0.5) / (self.N + 0.5 * self.V))
        return lp + math.log(0.5)          # kind-specific half of the mass (W vs L)


class NumModel:
    """P(v, frac | ctx): bin by Witten-Bell backoff (prev identity -> prev role -> global),
    uniform within bin; fraction string by role ctx; arithmetic mixture handled by caller."""
    def __init__(self):
        self.bid = defaultdict(Counter); self.brole = defaultdict(Counter); self.ball = Counter()
        self.frole = defaultdict(Counter); self.fall = Counter()

    def add(self, v, fr, pid, prole):
        b = min(nbin(v), 14)
        self.bid[pid][b] += 1; self.brole[prole][b] += 1; self.ball[b] += 1
        self.frole[prole][fr] += 1; self.fall[fr] += 1

    @staticmethod
    def _wb(cnt, lower):
        n = sum(cnt.values()); t = len(cnt)
        if n == 0:
            return lower
        lam = n / (n + t)
        return lambda x: lam * cnt[x] / n + (1 - lam) * lower(x)

    def lp(self, v, fr, pid, prole):
        b = min(nbin(v), 14)
        p0 = lambda x: (self.ball[x] + 0.5) / (sum(self.ball.values()) + 0.5 * 15)
        p1 = self._wb(self.brole[prole], p0)
        p2 = self._wb(self.bid[pid], p1)
        pb = p2(b)
        pv = pb / (bin_size(b) if b < 14 else 2 ** 14)
        f0 = lambda x: (self.fall[x] + 0.5) / (sum(self.fall.values()) + 0.5 * (len(self.fall) + 1)) * (1.0 if x in self.fall else 2 ** -8)
        pf = self._wb(self.frole[prole], f0)(fr)
        return math.log(pv) + math.log(pf)


def ctx_of(toks, i, roles_fn):
    """Previous content token identity and role (skipping line breaks)."""
    j = i - 1
    while j >= 0 and toks[j][0] == 'NL':
        j -= 1
    if j < 0:
        return 'START', 'START'
    t = toks[j]
    if t[0] == 'N':
        return 'N', 'N'
    return t[0] + ':' + str(t[1]), roles_fn(t)


class SeqModel:
    """Witten-Bell interpolated n-gram over a symbol alphabet; unseen identities via Spell."""
    def __init__(self, order, spell):
        self.o = order; self.spell = spell
        self.c = [defaultdict(Counter) for _ in range(order)]

    def fit(self, seqs):
        for s in seqs:
            h = ['<s>'] * (self.o - 1)
            for x in s:
                for k in range(self.o):
                    self.c[k][tuple(h[len(h) - k:]) if k else ()][x] += 1
                h = (h + [x])[1:] if self.o > 1 else h
        self.vocab = set(self.c[0][()])
        self.N0 = sum(self.c[0][()].values()); self.T0 = len(self.vocab)

    def p(self, h, x):
        # level 0 with an unseen-symbol share
        n0, t0 = self.N0, self.T0
        p = (self.c[0][()][x]) / (n0 + t0) if x in self.vocab else t0 / (n0 + t0)
        for k in range(1, self.o):
            hk = tuple(h[len(h) - k:])
            cnt = self.c[k].get(hk)
            if not cnt:
                continue
            n = sum(cnt.values()); t = len(cnt)
            if x in self.vocab:
                p = (cnt[x] + t * p) / (n + t)
            else:
                p = (t * p) / (n + t)
        return p


def sym_of(t):
    if t[0] in ('W', 'L'):
        return t[0] + ':' + t[1]
    return t[0]


class Baseline:
    def __init__(self, train, order):
        self.spell = Spell(train)
        self.seq = SeqModel(order, self.spell)
        self.seq.fit([[sym_of(t) for t in d['toks']] + ['END'] for d in train])
        self.num = NumModel()
        rf = lambda t: t[0]
        for d in train:
            for i, t in enumerate(d['toks']):
                if t[0] == 'N':
                    pid, pr = ctx_of(d['toks'], i, rf)
                    self.num.add(t[1], t[2], pid, pr)

    def token_lps(self, d):
        rf = lambda t: t[0]
        h = ['<s>'] * (self.seq.o - 1); out = []
        toks = d['toks'] + [['END', None, None]]
        for i, t in enumerate(toks):
            x = sym_of(t)
            lp = math.log(self.seq.p(h, x))
            if x not in self.seq.vocab:
                lp += self.spell.lp(t[1], t[0])
            if t[0] == 'N':
                pid, pr = ctx_of(toks, i, rf)
                lp += self.num.lp(t[1], t[2], pid, pr)
            out.append(lp)
            h = (h + [x])[1:] if self.seq.o > 1 else h
        return out


class Grammar:
    """G(R): role trigram x identity-in-role x numbers, + arithmetic + commodity order."""
    def __init__(self, train, R, lb=False, arith=True, order=True):
        self.R = R; self.lb = lb; self.use_arith = arith; self.use_order = order
        self.spell = Spell(train)
        self.rf = lambda t: role_of(t, R)
        self.rseq = SeqModel(3, None)
        self.rseq.fit([[self.rf(t) for t in d['toks']] + ['END'] for d in train])
        self.ident = defaultdict(Counter)
        for d in train:
            for t in d['toks']:
                if t[0] in ('W', 'L'):
                    self.ident[self.rf(t)][t[0] + ':' + t[1]] += 1
        self.num = NumModel()
        hits = 0; trials = 0
        for d in train:
            run = self._running(d)
            for i, t in enumerate(d['toks']):
                if t[0] == 'N':
                    pid, pr = ctx_of(d['toks'], i, self.rf)
                    if run.get(i) is not None:
                        trials += 1
                        if t[1] in run[i]:
                            hits += 1
                            continue        # explained by arithmetic: not used to train the base
                    self.num.add(t[1], t[2], pid, pr)
        self.theta = (hits + 0.5) / (trials + 1.0) if trials else 0.0
        self.arith_hits = (hits, trials)
        self.beta = 0.0
        if order and R['order']:
            best = None
            for b in [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
                self.beta = b
                s = sum(self._id_lp(d, i, t) for d in train for i, t in enumerate(d['toks'])
                        if t[0] in ('W', 'L') and self.rf(t) == 'COM')
                if best is None or s > best[0]:
                    best = (s, b)
            self.beta = best[1]

    def _running(self, d):
        """For each number index right after a TOT word: candidate set of totals (>= 2 entries)."""
        toks = d['toks']; out = {}; ent = []; skip = set()
        for i, t in enumerate(toks):
            if t[0] == 'W' and self.rf(t) in ('TOT', 'RES'):
                j = _num_after(toks, i)
                if j is not None:
                    skip.add(j)
                    if self.rf(t) == 'TOT' and len(ent) >= 2 and self.use_arith:
                        if self.lb:
                            s = sum(e[1] + (float(e[2]) if e[2] else 0.0) for e in ent)
                            out[j] = {int(math.floor(s + 1e-6))}
                        else:
                            s = sum(e[1] for e in ent); m = sum(1 for e in ent if e[2])
                            out[j] = set(range(s, s + (m // 2 + 1 if m else 0) + 1))
                ent = []
                continue
            if t[0] == 'N' and i not in skip:
                ent.append(t)
        return out

    def _prev_com(self, d, i):
        for j in range(i - 1, -1, -1):
            t = d['toks'][j]
            if t[0] in ('W', 'L') and self.rf(t) == 'COM':
                return base_of(t[1])
        return None

    def _id_lp(self, d, i, t):
        r = self.rf(t); cnt = self.ident[r]; key = t[0] + ':' + t[1]
        n = sum(cnt.values()); T = len(cnt)
        if n == 0:
            return self.spell.lp(t[1], t[0])
        if r == 'COM' and self.beta and self.use_order:
            prev = self._prev_com(d, i)
            if prev is not None and prev in self.R['order']:
                po = self.R['order'][prev]
                def g(k):
                    b = base_of(k.split(':', 1)[1])
                    if b == prev or b not in self.R['order']:
                        return 0.0
                    return 1.0 if self.R['order'][b] >= po else -1.0
                Z = sum(c * math.exp(self.beta * g(k)) for k, c in cnt.items())
                if key in cnt:
                    return math.log(n / (n + T)) + math.log(cnt[key] * math.exp(self.beta * g(key)) / Z)
                return math.log(T / (n + T)) + self.spell.lp(t[1], t[0])
        if key in cnt:
            return math.log(cnt[key] / (n + T))
        return math.log(T / (n + T)) + self.spell.lp(t[1], t[0])

    def token_lps(self, d):
        toks = d['toks'] + [['END', None, None]]
        dd = dict(d, toks=toks)
        run = self._running(d)
        h = ['<s>', '<s>']; out = []
        for i, t in enumerate(toks):
            r = self.rf(t) if t[0] != 'END' else 'END'
            lp = math.log(max(self.rseq.p(h, r), 1e-12))
            if t[0] in ('W', 'L'):
                lp += self._id_lp(dd, i, t)
            elif t[0] == 'N':
                pid, pr = ctx_of(toks, i, self.rf)
                base = math.exp(self.num.lp(t[1], t[2], pid, pr))
                if run.get(i) is not None and self.theta > 0:
                    C = run[i]
                    pa = self.theta / len(C) if t[1] in C else 0.0
                    base = pa * self._frac_p(t[2], pr) + (1 - self.theta) * base
                lp += math.log(max(base, 1e-300))
            out.append(lp)
            h = (h + [r])[1:]
        return out

    def _frac_p(self, fr, prole):
        f = self.num.frole[prole]; n = sum(f.values()); T = len(f)
        allf = self.num.fall; na = sum(allf.values())
        p0 = (allf[fr] + 0.5) / (na + 0.5 * (len(allf) + 1)) * (1.0 if fr in allf else 2 ** -8)
        if n == 0:
            return p0
        lam = n / (n + T)
        return lam * f[fr] / n + (1 - lam) * p0


def doc_bits(model, d):
    return -sum(model.token_lps(d)) / LOG2


def mix_bits(m1, m2, d, w=0.5):
    a = m1.token_lps(d); b = m2.token_lps(d)
    return -sum(math.log(w * math.exp(x) + (1 - w) * math.exp(y)) for x, y in zip(a, b)) / LOG2
