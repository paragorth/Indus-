#!/usr/bin/env python3
"""LA-61 CRACK ATTEMPT STEP 2: READ THE WORDS by massive random typed-glossary search.

Every recurring word (not in the frozen la60 scaffold) gets a structured hypothesis
    (type, commodity affinity, site affinity)
with type in a fixed typed space
    PER person, PLA place, OFF office/occupation, TRX transaction/heading word,
    CSUB commodity sub-type, QUAL qualifier, MEAS measure/unit.
Types are given names only by weak a-priori slot templates (pseudo-counts, the same for every
corpus); everything else is learned from the training documents pooled over the words of a type.
A full assignment is scored by how much it improves the decoding of documents it was not fitted
on (per-occurrence log-likelihood of: slot (prev kind x next kind), attached-number size, first
line or not, commodity in scope, site) over the scaffold's single pooled 'entry word' class.
Search = thousands of fully random assignments + hill-climbing from random restarts, objective =
2-fold internal cross-validation inside the fitting documents; held-out documents are never seen.

Corpora, reduced to one form [kind, id, value, frac] (kind W word, L commodity/logogram, N number,
NL line break), with a scaffold of known roles:
    LA   Linear A administrative documents; scaffold = la60 PRIOR reading (commodity class -> L,
         KU-RO/PO-TO-KU-RO TOT, KI-RO RES, heading single signs HDR). TRX words of la60 are NOT given:
         they are typed by the search like every other word.
    LB   Linear B KN+PY (DAMOS via la57 cache), sign identities only; scaffold = logograms + to-so.
    UR3  Ur III administrative texts (CDLI via la57 cache), opaque token ids; scaffold = commodity
         words -> L, szu-nigin -> TOT.  Units are left to be typed (truth MEAS).
Truth for calibration only (la57 lists): LB PER/PLA/TRA, UR3 PER/PLA/TRA/HDR/UNI/COM.
No Linear B sound value enters any Linear A hypothesis.
"""
import os, sys, json, math, random, hashlib, re
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la61_ckpt')
os.makedirs(CK, exist_ok=True)
LOOPS = os.path.join(HERE, '..', 'loops')

from la60_common import load_la, admin_docs, prior_reading, base_of, seed, tab_of

TYPES = ['PER', 'PLA', 'OFF', 'TRX', 'CSUB', 'QUAL', 'MEAS']
NT = len(TYPES)
TRUTH_MAP = {'PER': 'PER', 'PLA': 'PLA', 'OFF': 'HDR', 'TRX': 'TRA', 'MEAS': 'UNI', 'CSUB': 'COM'}
PREV = ['DS', 'LS', 'W', 'L', 'N', 'H']
NEXT = ['N', 'L', 'W', 'H', 'LE', 'DE']
NSLOT = len(PREV) * len(NEXT)
NBIN = 8                      # 1, 2, 3-5, 6-10, 11-30, 31-100, >100, none
PI_C = 0.5                    # weight of a commodity-affinity hypothesis
PI_S = 0.7                    # weight of a site-affinity hypothesis
PRIOR_S = 3.0                 # strength of the a-priori type templates (pseudo-observations)
NDISP = 5                     # documents containing the word: 1, 2, 3-4, 5-9, 10+ (per fitting fold)
LAM_C = math.log(14)          # MDL cost (nats) of stating a commodity affinity
LAM_S = math.log(11)          # MDL cost of stating a site affinity
# dispersion templates (a priori): persons rare, offices / transactions / measures widespread
_DISP = dict(PER=[1, .6, .3, .1, .05], PLA=[.3, .6, 1, .8, .5], OFF=[.1, .3, .6, 1, 1], TRX=[.1, .3, .6, 1, 1],
             CSUB=[.3, .5, .8, .8, .6], QUAL=[.3, .5, .8, .8, .6], MEAS=[.1, .3, .6, 1, 1])

# a-priori templates (weights; normalised then scaled by PRIOR_S) -- fixed for every corpus
_T = {
    'PER': (dict(DS=.2, LS=1, W=.5, L=.3, N=1, H=.5), dict(N=1, L=.6, W=.2, H=.1, LE=.2, DE=.1),
            [1, 1, 1, .6, .3, .1, .1, .2], .3),
    'PLA': (dict(DS=1, LS=1, W=.3, L=.2, N=.5, H=.5), dict(N=.6, L=.6, W=.6, H=.3, LE=.6, DE=.2),
            [.1, .1, .2, .4, 1, 1, 1, .6], .6),
    'OFF': (dict(DS=.3, LS=.6, W=1, L=.2, N=.5, H=.3), dict(N=1, L=.4, W=1, H=.2, LE=.3, DE=.1),
            [1, 1, 1, .6, .3, .1, .1, .4], .4),
    'TRX': (dict(DS=1, LS=.5, W=.2, L=.2, N=.2, H=.4), dict(N=.1, L=.8, W=1, H=.6, LE=1, DE=.5),
            [.05, .05, .05, .05, .05, .05, .05, 1], .9),
    'CSUB': (dict(DS=.1, LS=.3, W=.3, L=1, N=.3, H=.2), dict(N=1, L=1, W=.2, H=.1, LE=.2, DE=.1),
             [.5, .5, .6, .8, 1, .8, .5, .3], .4),
    'QUAL': (dict(DS=.1, LS=.2, W=1, L=.5, N=.2, H=.3), dict(N=.4, L=.6, W=1, H=.3, LE=.5, DE=.3),
             [.2, .2, .2, .2, .2, .2, .2, 1], .4),
    'MEAS': (dict(DS=.05, LS=.2, W=.3, L=.6, N=1, H=.2), dict(N=1, L=.2, W=.1, H=.1, LE=.2, DE=.1),
             [1, 1, 1, .5, .2, .1, .05, .1], .3),
}


def templates():
    P = {}
    for k, t in enumerate(TYPES):
        pp, pn, nb, fl = _T[t]
        s = np.outer([pp[x] for x in PREV], [pn[x] for x in NEXT]).ravel()
        nb = np.array(nb, float)
        dp = np.array(_DISP[t], float)
        P[t] = dict(slot=PRIOR_S * s / s.sum(), nb=PRIOR_S * nb / nb.sum(), fl=PRIOR_S * np.array([1 - fl, fl]),
                    disp=PRIOR_S * dp / dp.sum())
    return P


TPL = templates()


# ===================================================================== corpora
def _la():
    A = admin_docs(load_la())
    R = prior_reading()
    roles = R['roles']
    out = []
    for d in A:
        toks = []
        for t in d['toks']:
            if t[0] == 'W' and roles.get(t[1]) in ('COM', 'CENT'):
                toks.append(['L', t[1], None, None])
            elif t[0] == 'W' and roles.get(t[1]) in ('TOT', 'RES', 'HDR'):
                toks.append(['H', t[1], roles[t[1]], None])
            else:
                toks.append([t[0], t[1], t[1] if t[0] == 'N' else None, t[2] if t[0] == 'N' else None])
        out.append(dict(id=d['id'], tab=d['tab'], site=d['site'], pub=d['pub'], toks=toks))
    return out


def _from57(name, tot_words, com_words):
    L = json.load(open(os.path.join(D, 'la57_ckpt', name + '_docs.json')))
    out = []
    for d in L:
        toks = []
        for x in d['toks']:
            if x[0] == 'L':
                if toks and toks[-1][0] != 'NL':
                    toks.append(['NL', None, None, None])
            elif x[0] == 'N':
                v = x[1]
                if v is None:
                    v = -1
                iv = int(math.floor(v + 1e-9)) if v >= 0 else -1
                toks.append(['N', 'N', iv, 'f' if x[2] else ''])
            else:
                w = x[1]
                if w.startswith('L:'):
                    toks.append(['L', w[2:], None, None])
                elif w in tot_words:
                    toks.append(['H', w, 'TOT', None])
                elif w in com_words:
                    toks.append(['L', w, None, None])
                else:
                    toks.append(['W', w, None, None])
        while toks and toks[-1][0] == 'NL':
            toks.pop()
        out.append(dict(id=d['id'], tab=d['id'], site=d['site'], pub='G', toks=toks))
    return out


def load(corpus):
    fn = os.path.join(CK, 'docs_%s.json' % corpus)
    if os.path.exists(fn):
        return json.load(open(fn))
    if corpus == 'LA':
        out = _la()
    elif corpus == 'LB':
        from la57_common import LB_TRUTH_W
        out = _from57('lb', set(LB_TRUTH_W['TOT'].split()), set())
    elif corpus == 'UR3':
        from la57_common import MESO_W
        out = _from57('ur3', set(MESO_W['TOT'].split()), set(MESO_W['COM'].split()))
    json.dump(out, open(fn, 'w'))
    return out


def truth(corpus):
    """Calibration truth (never used in fitting)."""
    fn = os.path.join(CK, 'truth_%s.json' % corpus)
    if os.path.exists(fn):
        return json.load(open(fn))
    if corpus == 'LB':
        from la57_common import lb_docs, lb_truth
        t = lb_truth(lb_docs())
    elif corpus == 'UR3':
        from la57_common import ur3_docs, meso_truth
        t = meso_truth(ur3_docs())
    else:
        t = {}
    t = {w: r for w, r in t.items() if not w.startswith('L:') and r != 'TOT'}
    json.dump(t, open(fn, 'w'))
    return t


def ntok(docs):
    return sum(1 for d in docs for t in d['toks'] if t[0] != 'NL')


def draw(docs, n, rng):
    idx = list(range(len(docs))); rng.shuffle(idx)
    out, c = [], 0
    for i in idx:
        if c >= n:
            break
        out.append(docs[i]); c += sum(1 for t in docs[i]['toks'] if t[0] != 'NL')
    return out


def shuffle_words(docs, rng):
    """Control: word tokens (kind W only) permuted over all W slots of the corpus."""
    ws = [t[1] for d in docs for t in d['toks'] if t[0] == 'W']
    rng.shuffle(ws); it = iter(ws)
    out = []
    for d in docs:
        out.append(dict(d, toks=[[t[0], next(it), t[2], t[3]] if t[0] == 'W' else t for t in d['toks']]))
    return out


def split_tabs(docs, k, name):
    """k folds by tablet, stratified by site."""
    rng = random.Random(seed(name))
    by = defaultdict(set)
    for d in docs:
        by[d['site']].add(d['tab'])
    fold = {}
    off = 0
    for s in sorted(by):
        u = sorted(by[s]); rng.shuffle(u)
        for i, t in enumerate(u):
            fold[t] = (i + off) % k
        off += len(u)
    return [[d for d in docs if fold[d['tab']] == f] for f in range(k)]


# ===================================================================== occurrences
def numbin(v):
    if v is None or v < 1:
        return 7
    return 0 if v == 1 else 1 if v == 2 else 2 if v <= 5 else 3 if v <= 10 else 4 if v <= 30 else 5 if v <= 100 else 6


def occurrences(docs):
    """One record per W token: (word, slot, numbin, firstline, commodity-in-scope, site, doc id)."""
    out = []
    for d in docs:
        toks = d['toks']
        lines = []; cur = []
        for i, t in enumerate(toks):
            if t[0] == 'NL':
                lines.append(cur); cur = []
            else:
                cur.append(i)
        lines.append(cur)
        line_of = {}
        for li, l in enumerate(lines):
            for i in l:
                line_of[i] = li
        line_com = [next((base_of(toks[i][1]) for i in l if toks[i][0] == 'L'), None) for l in lines]
        content = [i for i, t in enumerate(toks) if t[0] != 'NL']
        for p, i in enumerate(content):
            t = toks[i]
            if t[0] != 'W':
                continue
            li = line_of[i]
            if p == 0:
                pv = 'DS'
            elif lines[li][0] == i:
                pv = 'LS'
            else:
                pv = toks[content[p - 1]][0]
            if p == len(content) - 1:
                nx = 'DE'
            elif lines[li][-1] == i:
                nx = 'LE'
            else:
                nx = toks[content[p + 1]][0]
            slot = PREV.index(pv) * len(NEXT) + NEXT.index(nx)
            nb = 7
            if p + 1 < len(content):
                j = content[p + 1]
                if toks[j][0] == 'N':
                    nb = numbin(toks[j][2])
                elif toks[j][0] == 'L' and p + 2 < len(content) and toks[content[p + 2]][0] == 'N':
                    nb = numbin(toks[content[p + 2]][2])
            com = line_com[li]
            if com is None:
                for lj in range(li - 1, -1, -1):
                    if line_com[lj]:
                        com = line_com[lj]; break
            out.append((t[1], slot, nb, 1 if li == 0 else 0, com or 'NONE', d['site'], d['id']))
    return out


class Vocab:
    def __init__(self, fit_docs, min_occ=2, ncom=12):
        occ = occurrences(fit_docs)
        c = Counter(o[0] for o in occ)
        self.words = sorted(w for w, n in c.items() if n >= min_occ)
        self.wi = {w: i for i, w in enumerate(self.words)}
        cc = Counter(o[4] for o in occ if o[4] != 'NONE')
        self.coms = [k for k, _ in cc.most_common(ncom)] + ['OTH', 'NONE']
        self.ci = {k: i for i, k in enumerate(self.coms)}
        sc = Counter(o[5] for o in occ)
        self.sites = [k for k, _ in sc.most_common(10)] + ['OTH']
        self.si = {k: i for i, k in enumerate(self.sites)}
        self.W = len(self.words); self.K = len(self.coms); self.S = len(self.sites)

    def com_idx(self, c):
        return self.ci.get(c, self.ci['OTH']) if c != 'NONE' else self.ci['NONE']

    def site_idx(self, s):
        return self.si.get(s, self.si['OTH'])

    def counts(self, docs):
        """Count matrices per word for the vocab's words (occurrences of other words are dropped)."""
        W = self.W
        C = dict(slot=np.zeros((W, NSLOT)), nb=np.zeros((W, NBIN)), fl=np.zeros((W, 2)),
                 com=np.zeros((W, self.K)), site=np.zeros((W, self.S)), disp=np.zeros((W, NDISP)))
        nd = defaultdict(set)
        for o in occurrences(docs):
            if o[0] in self.wi:
                nd[o[0]].add(o[6])
        for w, s_ in nd.items():
            n = len(s_)
            C['disp'][self.wi[w], 0 if n == 1 else 1 if n == 2 else 2 if n <= 4 else 3 if n <= 9 else 4] += 1
        for o in occurrences(docs):
            i = self.wi.get(o[0])
            if i is None:
                continue
            C['slot'][i, o[1]] += 1; C['nb'][i, o[2]] += 1; C['fl'][i, o[3]] += 1
            C['com'][i, self.com_idx(o[4])] += 1; C['site'][i, self.site_idx(o[5])] += 1
        return C


# ===================================================================== scoring
TPL_ARR = None


def tpl_arrays():
    global TPL_ARR
    if TPL_ARR is None:
        TPL_ARR = dict(slot=np.stack([TPL[t]['slot'] for t in TYPES]), nb=np.stack([TPL[t]['nb'] for t in TYPES]),
                       fl=np.stack([TPL[t]['fl'] for t in TYPES]), disp=np.stack([TPL[t]['disp'] for t in TYPES]))
    return TPL_ARR


def _onehot(typ, T=NT):
    M = np.zeros((T, len(typ)))
    M[typ, np.arange(len(typ))] = 1.0
    return M


def ll_typed(asg, Ctr, Cte, prior=True):
    """Held-out log-likelihood (nats) of Cte's occurrences under the assignment fitted on Ctr.
    asg = (typ[W], cafx[W] (-1 none), safx[W] (-1 none)).  Returns per-word LL vector."""
    typ, caf, saf = asg
    M = _onehot(typ)
    P = tpl_arrays()
    tot = np.zeros(len(typ))
    for ch in ('slot', 'nb', 'fl', 'disp'):
        n = M @ Ctr[ch] + (P[ch] if prior else 0.5)
        lp = np.log(n / n.sum(1, keepdims=True))
        tot += (Cte[ch] * lp[typ]).sum(1)
    for ch, aff, pi in (('com', caf, PI_C), ('site', saf, PI_S)):
        n = M @ Ctr[ch] + 0.5
        q = (n / n.sum(1, keepdims=True))[typ]               # W x K
        has = aff >= 0
        p = q.copy()
        if has.any():
            p[has] *= (1 - pi)
            p[np.where(has)[0], aff[has]] += pi
        tot += (Cte[ch] * np.log(p)).sum(1)
    return tot - 2 * (LAM_C * (caf >= 0) + LAM_S * (saf >= 0)) / 2


def ll_pooled(Ctr, Cte):
    """The scaffold baseline: every unread word is one pooled 'entry word' class (flat prior)."""
    W = Ctr['slot'].shape[0]
    tot = np.zeros(W)
    for ch in ('slot', 'nb', 'fl', 'com', 'site', 'disp'):
        n = Ctr[ch].sum(0) + 0.5
        lp = np.log(n / n.sum())
        tot += (Cte[ch] * lp).sum(1)
    return tot


def cv_objective(asg, C1, C2):
    return ll_typed(asg, C1, C2).sum() + ll_typed(asg, C2, C1).sum()


def add(Ca, Cb):
    return {k: Ca[k] + Cb[k] for k in Ca}


def random_asg(V, rng, p_aff=0.5):
    typ = rng.integers(0, NT, V.W)
    caf = np.where(rng.random(V.W) < p_aff, rng.integers(0, V.K, V.W), -1)
    saf = np.where(rng.random(V.W) < p_aff, rng.integers(0, V.S, V.W), -1)
    return (typ, caf, saf)


class State:
    """Incremental 2-fold CV objective = sum over types of a multinomial part (slot, number size,
    first line) + per-word commodity/site terms that depend on the word's affinity and its type's q."""
    def __init__(self, V, C1, C2, asg):
        self.V = V; self.C = (C1, C2)
        self.typ, self.caf, self.saf = (a.copy() for a in asg)
        self.P = tpl_arrays()
        self.S = [{ch: np.zeros((NT, C1[ch].shape[1])) for ch in C1} for _ in range(2)]
        for f in range(2):
            for ch in C1:
                np.add.at(self.S[f][ch], self.typ, self.C[f][ch])
        self.mt = np.zeros(NT); self.wt = np.zeros(V.W)
        self.logq = {}
        for t in range(NT):
            self._retype(t)

    def _retype(self, t):
        sc = 0.0
        for a, b in ((0, 1), (1, 0)):
            for ch in ('slot', 'nb', 'fl', 'disp'):
                n = self.S[a][ch][t] + self.P[ch][t]
                sc += float(self.S[b][ch][t] @ np.log(n / n.sum()))
            for ch in ('com', 'site'):
                n = self.S[a][ch][t] + 0.5
                self.logq[(a, ch, t)] = n / n.sum()
        self.mt[t] = sc
        m = np.where(self.typ == t)[0]
        if len(m):
            self._words(m)

    def _words(self, m):
        """Vectorised affinity terms for word indices m (all of one type)."""
        t = self.typ[m[0]]; sc = np.zeros(len(m))
        for a, b in ((0, 1), (1, 0)):
            for ch, AF, pi in (('com', self.caf, PI_C), ('site', self.saf, PI_S)):
                q = self.logq[(a, ch, t)]
                c = self.C[b][ch][m]
                sc += c @ np.log(q)
                af = AF[m]; h = af >= 0
                if h.any():
                    ca = c[h]; k = af[h]
                    qa = q[k]
                    sc[h] += ca.sum(1) * math.log(1 - pi) + ca[np.arange(len(k)), k] * (
                        np.log((1 - pi) * qa + pi) - np.log((1 - pi) * qa))
        self.wt[m] = sc - 2 * (LAM_C * (self.caf[m] >= 0) + LAM_S * (self.saf[m] >= 0))

    def _word(self, w):
        self._words(np.array([w]))

    def total(self):
        return float(self.mt.sum() + self.wt.sum())

    def set_type(self, w, t):
        a = self.typ[w]
        for f in range(2):
            for ch in self.S[f]:
                self.S[f][ch][a] -= self.C[f][ch][w]; self.S[f][ch][t] += self.C[f][ch][w]
        self.typ[w] = t
        self._retype(a); self._retype(t)

    def set_aff(self, f, w, v):
        (self.caf if f == 1 else self.saf)[w] = v
        self._word(w)

    def get(self, f, w):
        return (self.typ, self.caf, self.saf)[f][w]

    def set(self, f, w, v):
        if f == 0:
            self.set_type(w, v)
        else:
            self.set_aff(f, w, v)

    def asg(self):
        return (self.typ.copy(), self.caf.copy(), self.saf.copy())


def hill_climb(V, C1, C2, rng, iters, start=None, temp0=2.0):
    """Random single-field proposals, simulated annealing to temp 0, then greedy polish.
    Returns (asg, score); the number of assignments scored is iters + polish evaluations."""
    st = State(V, C1, C2, start if start is not None else random_asg(V, rng))
    cur = st.total(); best = (cur, st.asg())
    for it in range(iters):
        temp = temp0 * max(0.0, 1 - it / (0.8 * iters))
        w = int(rng.integers(V.W)); f = int(rng.integers(3))
        old = st.get(f, w)
        new = int(rng.integers(NT)) if f == 0 else int(rng.integers(-1, V.K if f == 1 else V.S))
        if new == old:
            continue
        st.set(f, w, new)
        s = st.total()
        if s >= cur or (temp > 0 and rng.random() < math.exp((s - cur) / temp)):
            cur = s
            if s > best[0]:
                best = (s, st.asg())
        else:
            st.set(f, w, old)
    st = State(V, C1, C2, best[1]); cur = st.total()
    for sweep in range(2):
        for w in rng.permutation(V.W):
            for f, rv in ((0, range(NT)), (1, range(-1, V.K)), (2, range(-1, V.S))):
                old = st.get(f, w); bv, bs = old, cur
                for v in rv:
                    if v == old:
                        continue
                    st.set(f, w, v)
                    s = st.total()
                    if s > bs + 1e-9:
                        bv, bs = v, s
                st.set(f, w, bv); cur = bs
    return st.asg(), cur


def fit(V, docs, rng, restarts, iters, name):
    """Search an assignment on docs (internal 2-fold CV by tablet). Returns list of (asg, score)."""
    f1, f2 = split_tabs(docs, 2, name)
    C1, C2 = V.counts(f1), V.counts(f2)
    res = []
    for r in range(restarts):
        res.append(hill_climb(V, C1, C2, rng, iters))
    return res, C1, C2


def consensus(res, V):
    """Modal type / affinities across restarts and the modal share."""
    T = np.stack([r[0][0] for r in res]); Cf = np.stack([r[0][1] for r in res]); Sf = np.stack([r[0][2] for r in res])
    out = []
    for w in range(V.W):
        ct = Counter(T[:, w]); t, n = ct.most_common(1)[0]
        cc = Counter(Cf[:, w]); c, nc = cc.most_common(1)[0]
        sc = Counter(Sf[:, w]); s, ns = sc.most_common(1)[0]
        out.append((int(t), n / len(res), int(c), nc / len(res), int(s), ns / len(res)))
    return out


def best_asg(res):
    return max(res, key=lambda r: r[1])[0]


def wlog(path, row):
    with open(path, 'a') as f:
        f.write(row.rstrip() + '\n')


def hsh(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ===================================================================== held-out evaluation
CHANNELS = ('slot', 'nb', 'fl', 'com', 'site', 'disp')


def ll_channels(asg, Ctr, Cte, pooled=False):
    """Per-channel held-out LL (nats), summed over words."""
    out = {}
    if pooled:
        for ch in CHANNELS:
            n = Ctr[ch].sum(0) + 0.5
            out[ch] = float((Cte[ch] * np.log(n / n.sum())).sum())
        return out
    typ, caf, saf = asg
    M = _onehot(typ); P = tpl_arrays()
    for ch in ('slot', 'nb', 'fl', 'disp'):
        n = M @ Ctr[ch] + P[ch]
        lp = np.log(n / n.sum(1, keepdims=True))
        out[ch] = float((Cte[ch] * lp[typ]).sum())
    for ch, aff, pi in (('com', caf, PI_C), ('site', saf, PI_S)):
        n = M @ Ctr[ch] + 0.5
        p = (n / n.sum(1, keepdims=True))[typ].copy()
        h = aff >= 0
        if h.any():
            p[h] *= (1 - pi); p[np.where(h)[0], aff[h]] += pi
        out[ch] = float((Cte[ch] * np.log(p)).sum())
    return out


def gain(asg, Ctr, Cte):
    a = ll_channels(asg, Ctr, Cte); b = ll_channels(None, Ctr, Cte, pooled=True)
    g = {ch: (a[ch] - b[ch]) / math.log(2) for ch in CHANNELS}          # bits
    g['core'] = g['slot'] + g['nb'] + g['fl'] + g['com'] + g['disp']
    g['all'] = g['core'] + g['site']
    return g


def com_accuracy(asg, V, Ctr, te_docs, tr_docs):
    """Held-out commodity-in-scope accuracy (occurrences with a commodity): glossary vs site default."""
    typ, caf, _ = asg
    M = _onehot(typ)
    q = M @ Ctr['com'] + 0.5
    sd = defaultdict(Counter)
    for o in occurrences(tr_docs):
        if o[4] != 'NONE':
            sd[o[5]][o[4]] += 1
    allc = Counter(); [allc.update(c) for c in sd.values()]
    hit = hit_sd = n = 0
    none = V.ci['NONE']
    for o in occurrences(te_docs):
        i = V.wi.get(o[0])
        if i is None or o[4] == 'NONE':
            continue
        n += 1
        if caf[i] >= 0:
            pred = V.coms[caf[i]]
        else:
            qq = q[typ[i]].copy(); qq[none] = -1; qq[V.ci['OTH']] = -1
            pred = V.coms[int(np.argmax(qq))]
        hit += pred == o[4]
        dflt = (sd[o[5]] or allc).most_common(1)[0][0]
        hit_sd += dflt == o[4]
    return hit, hit_sd, n


def truth_acc(typ_by_word, tr, rng, nperm=2000):
    """Typing accuracy against calibration truth, with a permutation null (truth labels permuted
    among the labelled typed words; type sizes kept)."""
    ws = [w for w in typ_by_word if w in tr]
    if not ws:
        return dict(n=0)
    pred = [TRUTH_MAP.get(typ_by_word[w]) for w in ws]
    lab = [tr[w] for w in ws]
    acc = sum(p == l for p, l in zip(pred, lab))
    null = []
    lab2 = list(lab)
    for _ in range(nperm):
        rng.shuffle(lab2)
        null.append(sum(p == l for p, l in zip(pred, lab2)))
    null = np.array(null)
    per = {}
    for r in sorted(set(lab)):
        idx = [k for k, l in enumerate(lab) if l == r]
        per[r] = (sum(pred[k] == r for k in idx), len(idx), Counter(typ_by_word[ws[k]] for k in idx).most_common(3))
    return dict(n=len(ws), acc=int(acc), null_mean=float(null.mean()), null_sd=float(null.std()),
                P=float((1 + (null >= acc).sum()) / (1 + nperm)), per=per)
