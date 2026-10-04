"""v24 THE SCRIBE'S CORRECTIONS ARE THE RULEBOOK.  Shared library.

A correction site = (before word, after word, context).  Context = previous / next word in the line,
the word two back, line-initial flag, first glyph of the previous line's first word.
Words are tuples of glyph units (Voynich: vlib.glyphs units; Latin: letters).

Test: for each candidate rule R (real-valued, defined per word-in-context), the statistic is
D_R = mean over sites [R(after) - R(before)].  Null: before is replaced by a random single-glyph
edit of after AT THE SAME PLACE and of the same type (substitution at the same position; for a
deletion correction an inserted random glyph at the same position; for an insertion correction
a random glyph deleted), glyphs drawn by corpus frequency.  All corpus counts used by a rule are
computed with the site's own token removed (no self-support for 'after').
Conditional tests: identity|legality (null edits matched to the real before's bigram legality and
template class) and legality|identity (null edits matched to the real before's in-vocabulary flag).
"""
import os, sys, math, re, json, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

ROOT = vlib.ROOT
DATA = vlib.DATA
LOOPS = os.path.join(ROOT, 'loops')
CK = os.path.join(DATA, 'v24_ckpt'); os.makedirs(CK, exist_ok=True)
SCR = os.environ.get('V24_SCRATCH', '/tmp/v24')
B, E = '^', '$'


def U(w):
    return tuple(vlib.glyphs(w))


# ------------------------------------------------------------------ corpus model
class Model:
    """Counts over a corpus given as lines (list of list of glyph tuples)."""

    def __init__(self, lines, slot_order=None, slot_cuts=None):
        self.lines = lines
        self.vocab = Counter(); self.big = Counter(); self.tri = Counter(); self.uni = Counter()
        self.first = Counter(); self.last = Counter(); self.junc = Counter(); self.juncL = Counter()
        self.lchain = Counter(); self.lchainL = Counter()
        prevfirst = None
        for L in lines:
            for i, w in enumerate(L):
                self.add_word(w, +1)
                if i > 0:
                    self.junc[(L[i - 1][-1], w[0])] += 1; self.juncL[L[i - 1][-1]] += 1
            if L:
                if prevfirst is not None:
                    self.lchain[(prevfirst, L[0][0])] += 1; self.lchainL[prevfirst] += 1
                prevfirst = L[0][0]
        self.ctx2 = Counter()
        for (a_, b_, c_), n_ in self.tri.items():
            self.ctx2[(a_, b_)] += n_
        self.alpha = sorted(self.uni)
        tot = sum(self.uni.values())
        self.pw = np.array([self.uni[g] / tot for g in self.alpha])
        self.slot = None
        if slot_order:
            rank = {g: i for i, g in enumerate(slot_order)}
            self.slot = {}
            for g, r in rank.items():
                for s in range(len(slot_cuts) - 1):
                    if slot_cuts[s] <= r < slot_cuts[s + 1]:
                        self.slot[g] = s
        self.nfirst = sum(self.first.values()); self.nlast = sum(self.last.values())
        self.nbig = sum(self.big.values())

    def add_word(self, w, s):
        self.vocab[w] += s
        x = (B,) + w + (E,)
        for g in w:
            self.uni[g] += s
        for a, b in zip(x, x[1:]):
            self.big[(a, b)] += s
        for a, b, c in zip(x, x[1:], x[2:]):
            self.tri[(a, b, c)] += s
        self.first[w[0]] += s; self.last[w[-1]] += s


def bigrams(w):
    x = (B,) + w + (E,)
    return list(zip(x, x[1:]))


def trigrams(w):
    x = (B,) + w + (E,)
    return list(zip(x, x[1:], x[2:]))


# ------------------------------------------------------------------ rules
class Site:
    __slots__ = ('before', 'after', 'kind', 'pos', 'prev', 'next', 'prev2', 'li', 'pl', 'meta')

    def __init__(self, before, after, prev=None, nxt=None, prev2=None, li=False, pl=None, meta=None):
        self.before, self.after = tuple(before), tuple(after)
        self.prev, self.next, self.prev2, self.li, self.pl = prev, nxt, prev2, li, pl
        self.meta = meta or {}
        self.kind, self.pos = edit_kind(self.before, self.after)


def edit_kind(b, a):
    """single-unit edit type turning before b into after a: ('sub',i) / ('ins',i) a has an extra unit at i /
    ('del',i) b had an extra unit at i; else ('multi',None)."""
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        if len(d) == 1:
            return 'sub', d[0]
        return ('same', None) if not d else ('multi', None)
    if len(a) == len(b) + 1:
        for i in range(len(a)):
            if a[:i] + a[i + 1:] == b:
                return 'ins', i
    if len(b) == len(a) + 1:
        for i in range(len(b)):
            if b[:i] + b[i + 1:] == a:
                return 'del', i
    return 'multi', None


def null_pool(site, M):
    """all single edits of `after` at the same place, same type; weights by glyph frequency."""
    a = site.after; out, wts = [], []
    if site.kind == 'sub':
        i = site.pos
        for g, p in zip(M.alpha, M.pw):
            if g != a[i]:
                out.append(a[:i] + (g,) + a[i + 1:]); wts.append(p)
    elif site.kind == 'del':          # before had an extra glyph at pos: insert random glyph at pos
        i = site.pos
        for g, p in zip(M.alpha, M.pw):
            out.append(a[:i] + (g,) + a[i:]); wts.append(p)
    elif site.kind == 'ins':          # before lacked a glyph: delete a random glyph of after
        for i in range(len(a)):
            if len(a) > 1:
                out.append(a[:i] + a[i + 1:]); wts.append(1.0)
    wts = np.array(wts, float)
    return out, (wts / wts.sum() if len(wts) else wts)


class RuleSet:
    def __init__(self, M, n_random=1500, seed=0, slot=True):
        self.M = M
        rng = random.Random(seed)
        A = M.alpha
        self.names = ['vocab', 'logf', 'big1', 'big5', 'bilp', 'tri1', 'slot', 'junc_prev', 'junc_next',
                      'junc', 'lineinit', 'chsh2', 'firstlp', 'lastlp', 'len', 'q3lp']
        self.core = len(self.names)
        self.rand = []
        bigs = sorted({k for k in M.big if M.big[k] > 0}, key=lambda k: -M.big[k])
        for k in bigs[:400]:
            self.rand.append(('hasbig', k)); self.names.append('hasbig:%s%s' % k)
        for g in A:
            for t in ('first', 'last', 'has'):
                self.rand.append((t, g)); self.names.append('%s:%s' % (t, g))
        for _ in range(n_random // 3):
            k = rng.randint(2, max(2, min(6, len(A) - 1)))
            S = frozenset(rng.sample(A, k)); t = rng.choice(['firstS', 'lastS', 'hasS'])
            self.rand.append((t, S)); self.names.append('%s:%s' % (t, ''.join(sorted(S))))
        for _ in range(n_random - n_random // 3):
            S1 = frozenset(rng.sample(A, rng.randint(2, 5))); S2 = frozenset(rng.sample(A, rng.randint(2, 5)))
            t = rng.choice(['jprev', 'jnext'])
            self.rand.append((t, (S1, S2))); self.names.append('%s:%s|%s' % (t, ''.join(sorted(S1)), ''.join(sorted(S2))))
        self.use_slot = slot and M.slot is not None

    def score(self, w, site):
        """vector of rule values for word w in the site's context; site.after's own token is excluded."""
        M = self.M; a = site.after
        nan = float('nan'); v = []
        selfb = Counter(bigrams(a)); selft = Counter(trigrams(a))
        selfc2 = Counter((t[0], t[1]) for t in trigrams(a))
        cnt = max(M.vocab[w] - (1 if w == a else 0), 0)
        v.append(1.0 if cnt >= 1 else 0.0)
        v.append(math.log1p(max(cnt, 0)))
        bc = [max(M.big[b] - selfb[b], 0) for b in bigrams(w)]
        v.append(1.0 if min(bc) >= 1 else 0.0)
        v.append(1.0 if min(bc) >= 5 else 0.0)
        V = len(M.alpha) + 2
        lp = 0.0
        for b, c in zip(bigrams(w), bc):
            ctx = M.uni.get(b[0], 0) if b[0] != B else sum(M.first.values())
            lp += math.log((c + 0.1) / (ctx + 0.1 * V))
        v.append(lp / (len(w) + 1))
        tc = [M.tri[t] - selft[t] for t in trigrams(w)]
        v.append(1.0 if (not tc or min(tc) >= 1) else 0.0)
        if self.use_slot:
            sl = [M.slot.get(g, -1) for g in w]
            v.append(1.0 if all(x <= y for x, y in zip(sl, sl[1:])) and -1 not in sl else 0.0)
        else:
            v.append(nan)
        def jp(x, y, own_pair, own_den):
            c = max(M.junc[(x, y)] - own_pair, 0); n = max(M.juncL[x] - own_den, 0)
            return math.log((c + 0.1) / (n + 0.1 * V))
        jpv = jp(site.prev[-1], w[0], int(w[0] == a[0]), 1) if site.prev else nan
        jnv = jp(w[-1], site.next[0], int(w[-1] == a[-1]), int(w[-1] == a[-1])) if site.next else nan
        v += [jpv, jnv, (jpv if site.prev else 0) + (jnv if site.next else 0) if (site.prev or site.next) else nan]
        if site.li and site.pl is not None:
            own = 1 if w[0] == a[0] else 0
            c = max(M.lchain[(site.pl, w[0])] - own, 0); n = max(M.lchainL[site.pl] - 1, 0)
            v.append(math.log((c + 0.1) / (n + 0.1 * V)))
        else:
            v.append(nan)
        cs = lambda x: ('C' in x) - ('S' in x)
        if site.prev2 is not None and cs(w) != 0 and cs(site.prev2) != 0:
            v.append(1.0 if cs(w) == cs(site.prev2) else 0.0)
        else:
            v.append(nan)
        v.append(math.log((max(M.first[w[0]] - (1 if w[0] == a[0] else 0), 0) + 0.1) / M.nfirst))
        v.append(math.log((max(M.last[w[-1]] - (1 if w[-1] == a[-1] else 0), 0) + 0.1) / M.nlast))
        v.append(float(len(w)))
        q = 0.0; x = (B, B) + w + (E,)
        bcnt = M.big
        for i in range(2, len(x)):
            t = (x[i - 2], x[i - 1], x[i]) if x[i - 2] != B or x[i - 1] != B else None
            if t is None:
                c = max(M.first[x[i]] - (1 if x[i] == a[0] else 0), 0); n = M.nfirst - 1
            else:
                c = max(M.tri[t] - selft[t], 0)
                ctx = (x[i - 2], x[i - 1])
                n = max(M.ctx2.get(ctx, 0) - selfc2[ctx], 0)
            q += math.log((c + 0.05) / (n + 0.05 * V))
        v.append(q / (len(w) + 1))
        bs = set(bigrams(w))
        for t, k in self.rand:
            if t == 'hasbig':
                v.append(1.0 if k in bs else 0.0)
            elif t == 'first':
                v.append(1.0 if w[0] == k else 0.0)
            elif t == 'last':
                v.append(1.0 if w[-1] == k else 0.0)
            elif t == 'has':
                v.append(1.0 if k in w else 0.0)
            elif t == 'firstS':
                v.append(1.0 if w[0] in k else 0.0)
            elif t == 'lastS':
                v.append(1.0 if w[-1] in k else 0.0)
            elif t == 'hasS':
                v.append(1.0 if any(g in k for g in w) else 0.0)
            elif t == 'jprev':
                v.append((1.0 if (site.prev[-1] in k[0] and w[0] in k[1]) else 0.0) if site.prev else nan)
            elif t == 'jnext':
                v.append((1.0 if (w[-1] in k[0] and site.next[0] in k[1]) else 0.0) if site.next else nan)
        return np.array(v, dtype=np.float32)


# ------------------------------------------------------------------ test
def prepare(sites, RS):
    """per site: after vector, before vector, pool matrix, pool weights, pool words."""
    P = []
    for s in sites:
        if s.kind not in ('sub', 'ins', 'del') or len(s.after) == 0 or len(s.before) == 0:
            continue
        pool, w = null_pool(s, RS.M)
        pool = [(p, wt) for p, wt in zip(pool, w) if len(p) > 0]
        if not pool:
            continue
        pw = np.array([x[1] for x in pool]); pw /= pw.sum()
        pm = np.stack([RS.score(p, s) for p, _ in pool])
        P.append(dict(site=s, va=RS.score(s.after, s), vb=RS.score(s.before, s), pm=pm, pw=pw,
                      pwords=[p for p, _ in pool]))
    return P


def run_test(P, nsim=2000, seed=1, match=None, rule_idx=None):
    """D statistic vs random-edit null.  match: None, or a function(prep, pool_matrix)->bool mask
    of admissible null edits (conditional test).  Returns dict of arrays over rules."""
    rng = np.random.default_rng(seed)
    keep = []
    for p in P:
        pw = p['pw'].copy()
        if match is not None:
            m = match(p)
            if not m.any():
                continue
            pw = pw * m; pw /= pw.sum()
        keep.append((p, pw))
    n = len(keep)
    if n == 0:
        return None
    R = len(keep[0][0]['va'])
    va = np.stack([p['va'] for p, _ in keep]); vb = np.stack([p['vb'] for p, _ in keep])
    dobs = va - vb
    defined = ~np.isnan(dobs)
    nd = defined.sum(0)
    Dobs = np.where(nd > 0, np.nansum(dobs, 0) / np.maximum(nd, 1), np.nan)
    Dnull = np.zeros((nsim, R), np.float32)
    for k, (p, pw) in enumerate(keep):
        idx = rng.choice(len(pw), size=nsim, p=pw)
        d = p['va'][None, :] - p['pm'][idx]
        Dnull += np.nan_to_num(d)
    Dnull /= np.maximum(nd, 1)
    mu = Dnull.mean(0); sd = Dnull.std(0) + 1e-9
    z = (Dobs - mu) / sd
    z[nd < 3] = np.nan
    zn = (Dnull - mu) / sd
    zn[:, nd < 3] = 0
    mx = np.nanmax(np.abs(zn), 1)
    zmax = np.nanmax(np.abs(z))
    # one-sided-in-the-rule's-direction per-rule p (two-sided empirical)
    pr = ((np.abs(zn) >= np.abs(np.nan_to_num(z))[None, :]).sum(0) + 1) / (nsim + 1)
    pfw = np.array([((mx >= abs(zz)).sum() + 1) / (nsim + 1) if not np.isnan(zz) else np.nan for zz in z])
    return dict(n=n, Dobs=Dobs, mu=mu, sd=sd, z=z, p=pr, pfw=pfw, zmax=zmax, nd=nd)


def summarize(res, names, core_only=False, top=8):
    rows = []
    if res is None:
        return rows
    z = res['z']
    for i, nm in enumerate(names):
        if core_only and i >= 16:
            break
        rows.append((nm, int(res['nd'][i]), float(res['Dobs'][i]), float(res['mu'][i]), float(z[i]), float(res['p'][i]), float(res['pfw'][i])))
    if not core_only:
        rows = sorted([r for r in rows if not math.isnan(r[4])], key=lambda r: -abs(r[4]))[:top]
    return rows


def match_legal(p, RS=None):
    """null edits with the same bigram legality (big1) and slot class as the real before."""
    i1, i2 = 2, 6
    m = p['pm'][:, i1] == p['vb'][i1]
    if not np.isnan(p['vb'][i2]):
        m &= p['pm'][:, i2] == p['vb'][i2]
    return m


def match_lp(p, RS=None, tol=0.25):
    """null edits with the same bigram legality and a trigram log-probability within tol of the real before."""
    m = (p['pm'][:, 2] == p['vb'][2]) & (np.abs(p['pm'][:, 15] - p['vb'][15]) <= tol)
    return m


def match_vocab(p, RS=None):
    return p['pm'][:, 0] == p['vb'][0]


def fmt_core(rows):
    return '; '.join('%s D %.3f (null %.3f) z %+.1f p %.3g' % (r[0], r[2], r[3], r[4], r[5]) for r in rows if not math.isnan(r[4]))


def fmt_top(rows):
    return '; '.join('%s z %+.1f pFW %.3g' % (r[0], r[4], r[6]) for r in rows)
