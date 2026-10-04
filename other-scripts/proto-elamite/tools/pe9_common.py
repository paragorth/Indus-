"""pe9 ('the names were drawn from a bag'): shared code.

A corpus = list of tablets; a tablet = list of strings; a string = list of sign ids.
String lengths and the number of strings per tablet are taken as given by every
model (all models are conditioned on them identically).

Every model is written as a sequential predictive P(sign | history), where history =
the signs already drawn on this tablet (or session of S consecutive tablets) and
the previous sign in the current string.  base q(.|prev) = unigram (add-0.5) or a
bigram smoothed toward the unigram (q = (c(p,s) + a w_s) / (c(p) + a)).

 mode 0 LM        : P = q                                   (language-like names)
 mode 1 POLYA     : P = (n_s + th q_s) / (n + th)           (tablet topic / cache;
                                                             used signs get MORE likely)
 mode 2 BAG_WR    : bag of K sign types, drawn WITH replacement (inexhaustible);
                    seen type 1/K each, unseen types share (K-d)/K by q
 mode 3 BAG_WOR   : bag of K types x c copies, drawn WITHOUT replacement (depleting);
                    seen weight c-n_s, unseen weight (K-d)c; bag refilled when empty
 mode 4 STAMP     : K stamps; a stamp is not re-used inside one string but is
                    returned after the string (stamps, not counters)
 modes 2-4 mix in eps of draws straight from q (signs outside the bag).
The same predictive is used to SIMULATE corpora (ABC) and to SCORE held-out tablets.
"""
import json, os, sys, math
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CKPT = os.path.join(DATA, 'pe9_ckpt')
os.makedirs(CKPT, exist_ok=True)
LN2 = math.log(2)
MODES = ['LM', 'POLYA', 'BAG_WR', 'BAG_WOR', 'STAMP']


def load_corpora():
    return json.load(open(os.path.join(DATA, 'pe9_corpora.json')))


class Enc:
    """Flat encoding of a corpus."""
    def __init__(self, tablets, vocab=None):
        if vocab is None:
            vocab = sorted({s for t in tablets for m in t for s in m})
        self.vocab = vocab
        self.idx = {s: i for i, s in enumerate(vocab)}
        self.V = len(vocab)
        tok, ss, ts = [], [0], [0]
        for t in tablets:
            for m in t:
                tok.extend(self.idx[s] for s in m)
                ss.append(len(tok))
            ts.append(len(ss) - 1)
        self.tok = np.array(tok, np.int32)
        self.ss = np.array(ss, np.int32)
        self.ts = np.array(ts, np.int32)
        self.T = len(tablets)

    def subset(self, tabs_idx):
        """tablets as nested lists of ids (for refitting / resampling)."""
        out = []
        for i in tabs_idx:
            out.append([list(self.tok[self.ss[j]:self.ss[j + 1]]) for j in range(self.ts[i], self.ts[i + 1])])
        return out


def enc_ids(tablets_ids, V):
    """encode tablets already given as id lists"""
    tok, ss, ts = [], [0], [0]
    for t in tablets_ids:
        for m in t:
            tok.extend(m)
            ss.append(len(tok))
        ts.append(len(ss) - 1)
    return np.array(tok, np.int32), np.array(ss, np.int32), np.array(ts, np.int32)


def fit_base(tok, ss, ts, tabs, V, a=2.0):
    """unigram w (add-0.5) and bigram matrix B (V+1 rows; row V = string start)."""
    cnt = np.full(V, 0.5)
    big = np.zeros((V + 1, V))
    for i in tabs:
        for j in range(ts[i], ts[i + 1]):
            p = V
            for k in range(ss[j], ss[j + 1]):
                s = tok[k]
                cnt[s] += 1
                big[p, s] += 1
                p = s
    w = cnt / cnt.sum()
    B = (big + a * w[None, :]) / (big.sum(1, keepdims=True) + a)
    return w, B


@njit(cache=True)
def _q(base, w, B, prev, V, out):
    if base == 0:
        for s in range(V):
            out[s] = w[s]
    else:
        for s in range(V):
            out[s] = B[prev, s]


@njit(cache=True)
def score(tok, ss, ts, tabs, sess, V, w, B, base, mode, K, c, eps, th):
    """log-likelihood (nats) of each tablet in tabs (int array of tablet indices);
    sess[i] = 1 if history resets before tablet i (session start)."""
    out = np.zeros(len(tabs))
    n_s = np.zeros(V)
    seen_list = np.zeros(V, np.int32)
    used_str = np.zeros(V, np.int32)
    q = np.zeros(V)
    d = 0
    n = 0.0
    for ii in range(len(tabs)):
        i = tabs[ii]
        if sess[i] == 1 or ii == 0 or tabs[ii - 1] != i - 1:
            for k in range(d):
                n_s[seen_list[k]] = 0
            d = 0
            n = 0.0
        ll = 0.0
        for j in range(ts[i], ts[i + 1]):
            prev = V
            for k in range(ss[j], ss[j + 1]):
                s = tok[k]
                _q(base, w, B, prev, V, q)
                if mode == 0:
                    p = q[s]
                elif mode == 1:
                    p = (n_s[s] + th * q[s]) / (n + th)
                else:
                    # bag weights
                    if mode == 3:
                        tot = 0.0
                        for t in range(d):
                            x = c - n_s[seen_list[t]]
                            if x > 0:
                                tot += x
                        uw = (K - d) * c if K > d else 0.0
                        if tot + uw <= 0:      # bag empty -> refill
                            for t in range(d):
                                n_s[seen_list[t]] = 0
                            d = 0
                            n = 0.0
                            tot = 0.0
                            uw = K * c
                        seenw = c - n_s[s] if n_s[s] > 0 else 0.0
                        if seenw < 0:
                            seenw = 0.0
                    elif mode == 2:
                        tot = d if d < K else d
                        uw = (K - d) if K > d else 0.0
                        seenw = 1.0 if n_s[s] > 0 else 0.0
                    else:  # STAMP
                        tot = 0.0
                        for t in range(d):
                            if used_str[seen_list[t]] == 0:
                                tot += 1.0
                        uw = (K - d) if K > d else 0.0
                        seenw = 1.0 if (n_s[s] > 0 and used_str[s] == 0) else 0.0
                    Qs = 0.0
                    for t in range(d):
                        Qs += q[seen_list[t]]
                    Z = tot + uw
                    if n_s[s] > 0:
                        pb = seenw / Z if Z > 0 else 0.0
                    else:
                        pb = (uw / Z) * q[s] / max(1e-12, 1.0 - Qs) if Z > 0 else 0.0
                    p = eps * q[s] + (1 - eps) * pb
                if p < 1e-300:
                    p = 1e-300
                ll += math.log(p)
                if n_s[s] == 0:
                    seen_list[d] = s
                    d += 1
                n_s[s] += 1
                n += 1
                used_str[s] = 1
                prev = s
            for k in range(ss[j], ss[j + 1]):
                used_str[tok[k]] = 0
        out[ii] = ll
    return out


@njit(cache=True)
def _draw(q, V, u):
    acc = 0.0
    for s in range(V):
        acc += q[s]
        if u <= acc:
            return s
    return V - 1


@njit(cache=True)
def simulate(ss, ts, sess, V, w, B, base, mode, K, c, eps, th, seed):
    """sample a corpus with the given structure from the predictive. returns tok."""
    np.random.seed(seed)
    tok = np.zeros(ss[-1], np.int32)
    n_s = np.zeros(V)
    seen_list = np.zeros(V, np.int32)
    used_str = np.zeros(V, np.int32)
    q = np.zeros(V)
    d = 0
    n = 0.0
    T = len(ts) - 1
    for i in range(T):
        if sess[i] == 1:
            for k in range(d):
                n_s[seen_list[k]] = 0
            d = 0
            n = 0.0
        for j in range(ts[i], ts[i + 1]):
            prev = V
            for k in range(ss[j], ss[j + 1]):
                _q(base, w, B, prev, V, q)
                if mode == 0:
                    s = _draw(q, V, np.random.random())
                elif mode == 1:
                    if np.random.random() < n / (n + th):
                        s = tok[ss[ts[_sess_start(sess, i)]] + int(np.random.random() * n)]
                    else:
                        s = _draw(q, V, np.random.random())
                else:
                    if np.random.random() < eps:
                        s = _draw(q, V, np.random.random())
                    else:
                        if mode == 3:
                            tot = 0.0
                            for t in range(d):
                                x = c - n_s[seen_list[t]]
                                if x > 0:
                                    tot += x
                            uw = (K - d) * c if K > d else 0.0
                            if tot + uw <= 0:
                                for t in range(d):
                                    n_s[seen_list[t]] = 0
                                d = 0
                                n = 0.0
                                tot = 0.0
                                uw = K * c
                        elif mode == 2:
                            tot = float(d)
                            uw = (K - d) if K > d else 0.0
                        else:
                            tot = 0.0
                            for t in range(d):
                                if used_str[seen_list[t]] == 0:
                                    tot += 1.0
                            uw = (K - d) if K > d else 0.0
                        Z = tot + uw
                        r = np.random.random() * Z
                        if Z <= 0 or r >= tot:
                            # new type from q restricted to unseen (rejection)
                            s = -1
                            for _ in range(200):
                                x = _draw(q, V, np.random.random())
                                if n_s[x] == 0:
                                    s = x
                                    break
                            if s < 0:
                                s = _draw(q, V, np.random.random())
                        else:
                            acc = 0.0
                            s = seen_list[0]
                            for t in range(d):
                                x = seen_list[t]
                                if mode == 3:
                                    wt = c - n_s[x]
                                    if wt < 0:
                                        wt = 0.0
                                elif mode == 2:
                                    wt = 1.0
                                else:
                                    wt = 1.0 if used_str[x] == 0 else 0.0
                                acc += wt
                                if r < acc:
                                    s = x
                                    break
                tok[k] = s
                if n_s[s] == 0:
                    seen_list[d] = s
                    d += 1
                n_s[s] += 1
                n += 1
                used_str[s] = 1
                prev = s
            for k in range(ss[j], ss[j + 1]):
                used_str[tok[k]] = 0
    return tok


@njit(cache=True)
def _bdraw(cum, V, u):
    u = u * cum[V - 1]
    lo, hi = 0, V - 1
    while lo < hi:
        m = (lo + hi) // 2
        if cum[m] < u:
            lo = m + 1
        else:
            hi = m
    return lo


@njit(cache=True)
def simulate_fast(ss, ts, sess, V, CW, CB, base, mode, K, c, eps, th, seed):
    """simulate() with cumulative tables CW (V) and CB (V+1, V) and O(log V) draws."""
    np.random.seed(seed)
    tok = np.zeros(ss[-1], np.int32)
    n_s = np.zeros(V)
    seen_list = np.zeros(V, np.int32)
    used_str = np.zeros(V, np.int32)
    d = 0
    n = 0.0
    T = len(ts) - 1
    for i in range(T):
        if sess[i] == 1:
            for k in range(d):
                n_s[seen_list[k]] = 0
            d = 0
            n = 0.0
        for j in range(ts[i], ts[i + 1]):
            prev = V
            for k in range(ss[j], ss[j + 1]):
                row = CW if base == 0 else CB[prev]
                if mode == 0:
                    s = _bdraw(row, V, np.random.random())
                elif mode == 1:
                    if np.random.random() < n / (n + th):
                        s = tok[ss[ts[_sess_start(sess, i)]] + int(np.random.random() * n)]
                    else:
                        s = _bdraw(row, V, np.random.random())
                else:
                    if np.random.random() < eps:
                        s = _bdraw(row, V, np.random.random())
                    else:
                        if mode == 3:
                            tot = 0.0
                            for t in range(d):
                                x = c - n_s[seen_list[t]]
                                if x > 0:
                                    tot += x
                            uw = (K - d) * c if K > d else 0.0
                            if tot + uw <= 0:
                                for t in range(d):
                                    n_s[seen_list[t]] = 0
                                d = 0
                                n = 0.0
                                tot = 0.0
                                uw = K * c
                        elif mode == 2:
                            tot = float(d)
                            uw = (K - d) if K > d else 0.0
                        else:
                            tot = 0.0
                            for t in range(d):
                                if used_str[seen_list[t]] == 0:
                                    tot += 1.0
                            uw = (K - d) if K > d else 0.0
                        Z = tot + uw
                        r = np.random.random() * Z
                        if Z <= 0 or r >= tot:
                            # new type from q restricted to unseen (rejection)
                            s = -1
                            for _ in range(200):
                                x = _bdraw(row, V, np.random.random())
                                if n_s[x] == 0:
                                    s = x
                                    break
                            if s < 0:
                                s = _bdraw(row, V, np.random.random())
                        else:
                            acc = 0.0
                            s = seen_list[0]
                            for t in range(d):
                                x = seen_list[t]
                                if mode == 3:
                                    wt = c - n_s[x]
                                    if wt < 0:
                                        wt = 0.0
                                elif mode == 2:
                                    wt = 1.0
                                else:
                                    wt = 1.0 if used_str[x] == 0 else 0.0
                                acc += wt
                                if r < acc:
                                    s = x
                                    break
                tok[k] = s
                if n_s[s] == 0:
                    seen_list[d] = s
                    d += 1
                n_s[s] += 1
                n += 1
                used_str[s] = 1
                prev = s
            for k in range(ss[j], ss[j + 1]):
                used_str[tok[k]] = 0
    return tok


@njit(cache=True)
def _sess_start(sess, i):
    while i > 0 and sess[i] == 0:
        i -= 1
    return i


@njit(cache=True)
def stats(tok, ss, ts, V, ncommon, seed):
    """summary statistics:
    0 within-tablet string-pair rare-sign sharing
    1 cross-tablet string-pair rare sharing (random pairs)
    2 log ratio 0/1
    3 token repeat rate (type already used in an EARLIER string on the tablet)
    4 among (tablet,type) used in >= 2 strings: share used in >= 3 strings
    5 within-string repeat rate
    6 types / tokens
    7 hapax share of types
    8 top-10 token share
    9 adjacent-tablet rare sharing / random-tablet rare sharing (log)
    10 rare-only repeat rate (stat 3 restricted to rare signs)"""
    np.random.seed(seed)
    N = len(tok)
    cnt = np.zeros(V)
    for k in range(N):
        cnt[tok[k]] += 1
    order = np.argsort(-cnt)
    rare = np.ones(V, np.int32)
    for r in range(ncommon):
        rare[order[r]] = 0
    S = len(ss) - 1
    T = len(ts) - 1
    tab_of = np.zeros(S, np.int32)
    for i in range(T):
        for j in range(ts[i], ts[i + 1]):
            tab_of[j] = i
    mark = np.zeros(V, np.int32)

    def_share = 0.0
    # within-tablet pairs
    pw = 0.0
    sw = 0.0
    for i in range(T):
        for a in range(ts[i], ts[i + 1]):
            for b in range(a + 1, ts[i + 1]):
                pw += 1
                for k in range(ss[a], ss[a + 1]):
                    if rare[tok[k]] == 1:
                        mark[tok[k]] = 1
                hit = 0
                for k in range(ss[b], ss[b + 1]):
                    if mark[tok[k]] == 1:
                        hit = 1
                for k in range(ss[a], ss[a + 1]):
                    mark[tok[k]] = 0
                sw += hit
    o = np.zeros(11)
    o[0] = sw / max(pw, 1)
    pc = 0.0
    sc = 0.0
    for r in range(4000):
        a = int(np.random.random() * S)
        b = int(np.random.random() * S)
        if tab_of[a] == tab_of[b]:
            continue
        pc += 1
        for k in range(ss[a], ss[a + 1]):
            if rare[tok[k]] == 1:
                mark[tok[k]] = 1
        hit = 0
        for k in range(ss[b], ss[b + 1]):
            if mark[tok[k]] == 1:
                hit = 1
        for k in range(ss[a], ss[a + 1]):
            mark[tok[k]] = 0
        sc += hit
    o[1] = sc / max(pc, 1)
    o[2] = math.log((o[0] + 1e-3) / (o[1] + 1e-3))
    # repeats on tablet
    nstr = np.zeros(V)
    rep = 0.0
    rrep = 0.0
    rtot = 0.0
    ge2 = 0.0
    ge3 = 0.0
    wrep = 0.0
    seen_l = np.zeros(V, np.int32)
    instr = np.zeros(V, np.int32)
    for i in range(T):
        d = 0
        for j in range(ts[i], ts[i + 1]):
            for k in range(ss[j], ss[j + 1]):
                s = tok[k]
                if instr[s] == 1:
                    wrep += 1
                    continue
                instr[s] = 1
                if nstr[s] > 0:
                    rep += 1
                    if rare[s] == 1:
                        rrep += 1
                if rare[s] == 1:
                    rtot += 1
                if nstr[s] == 0:
                    seen_l[d] = s
                    d += 1
                nstr[s] += 1
            for k in range(ss[j], ss[j + 1]):
                instr[tok[k]] = 0
        for t in range(d):
            x = seen_l[t]
            if nstr[x] >= 2:
                ge2 += 1
            if nstr[x] >= 3:
                ge3 += 1
            nstr[x] = 0
    o[3] = rep / N
    o[4] = ge3 / max(ge2, 1)
    o[5] = wrep / N
    ntypes = 0.0
    hap = 0.0
    for s in range(V):
        if cnt[s] > 0:
            ntypes += 1
            if cnt[s] == 1:
                hap += 1
    o[6] = ntypes / N
    o[7] = hap / max(ntypes, 1)
    top = 0.0
    for r in range(10):
        top += cnt[order[r]]
    o[8] = top / N
    # adjacent tablets vs random tablets (tablet-level: any rare sign shared)
    adj = 0.0
    na = 0.0
    for i in range(T - 1):
        for k in range(ss[ts[i]], ss[ts[i + 1]]):
            if rare[tok[k]] == 1:
                mark[tok[k]] = 1
        hit = 0
        for k in range(ss[ts[i + 1]], ss[ts[i + 2]]):
            if mark[tok[k]] == 1:
                hit = 1
        for k in range(ss[ts[i]], ss[ts[i + 1]]):
            mark[tok[k]] = 0
        adj += hit
        na += 1
    rnd = 0.0
    nr = 0.0
    for r in range(3 * T):
        i = int(np.random.random() * T)
        j2 = int(np.random.random() * T)
        if abs(i - j2) <= 1:
            continue
        for k in range(ss[ts[i]], ss[ts[i + 1]]):
            if rare[tok[k]] == 1:
                mark[tok[k]] = 1
        hit = 0
        for k in range(ss[ts[j2]], ss[ts[j2 + 1]]):
            if mark[tok[k]] == 1:
                hit = 1
        for k in range(ss[ts[i]], ss[ts[i + 1]]):
            mark[tok[k]] = 0
        rnd += hit
        nr += 1
    o[9] = math.log((adj / max(na, 1) + 1e-3) / (rnd / max(nr, 1) + 1e-3))
    o[10] = rrep / max(rtot, 1)
    return o


STAT_NAMES = ['share_within', 'share_cross', 'log_share_ratio', 'repeat_rate', 'p3_given_2',
              'within_string_rep', 'types_per_token', 'hapax_share', 'top10_share',
              'log_adjacent_ratio', 'rare_repeat_rate']


def sess_flags(T, S, offset=0):
    f = np.zeros(T, np.int32)
    for i in range(T):
        if (i + offset) % S == 0:
            f[i] = 1
    f[0] = 1
    return f


@njit(cache=True)
def score_profile(tok, ss, ts, tabs, sess, V, w, B, base, f1, f2, f3, fw):
    """'use-profile' model: P(s) = q(s|prev) r(s) / Z with r = 1 for signs not yet
    used in this tablet/session, r = f1, f2, f3 after 1, 2, >= 3 earlier uses, and
    r = fw for a sign already used in the CURRENT string.
    Inexhaustible bag / tablet topic: f1 ~ f2 ~ f3 > 1 (or rising).
    Depleting bag of c copies: r collapses (< 1) once n_s reaches c."""
    out = np.zeros(len(tabs))
    n_s = np.zeros(V, np.int32)
    seen_list = np.zeros(V, np.int32)
    used = np.zeros(V, np.int32)
    q = np.zeros(V)
    fv = np.array([1.0, f1, f2, f3])
    d = 0
    for ii in range(len(tabs)):
        i = tabs[ii]
        if sess[i] == 1 or ii == 0 or tabs[ii - 1] != i - 1:
            for k in range(d):
                n_s[seen_list[k]] = 0
            d = 0
        ll = 0.0
        for j in range(ts[i], ts[i + 1]):
            prev = V
            for k in range(ss[j], ss[j + 1]):
                s = tok[k]
                _q(base, w, B, prev, V, q)
                Z = 1.0
                for t in range(d):
                    x = seen_list[t]
                    r = fw if used[x] == 1 else fv[min(n_s[x], 3)]
                    Z += q[x] * (r - 1.0)
                rs = fw if used[s] == 1 else fv[min(n_s[s], 3)]
                p = q[s] * rs / Z
                if p < 1e-300:
                    p = 1e-300
                ll += math.log(p)
                if n_s[s] == 0:
                    seen_list[d] = s
                    d += 1
                n_s[s] += 1
                used[s] = 1
                prev = s
            for k in range(ss[j], ss[j + 1]):
                used[tok[k]] = 0
        out[ii] = ll
    return out
