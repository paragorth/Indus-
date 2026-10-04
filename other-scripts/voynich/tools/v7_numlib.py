"""v7: 'every word is a number' -- shared machinery.

A corpus is a list of lines (dicts with 'words' as unit strings, one char per
glyph, and 'para' id).  Pipeline, identical for every corpus:
  1. learn a glyph order (test_c_slots.learn_order) on the corpus itself;
  2. cut that order into K contiguous glyph bins = digit slots, choosing the
     cuts that maximise the share of tokens whose filler in every slot is one
     of that slot's 10 commonest fillers (the empty filler counts);
  3. a reading = one injective map filler -> digit 0..9 per slot; the word's
     value is sum_k digit_k * 10^(K-1-k) (leftmost slot most significant, or
     reversed);
  4. numeric-regularity score of the reading, maximised by simulated annealing.
Tokens whose filler in some slot is outside the top 10 are 'not numbers' and
are skipped (adjacency is only counted between consecutive valid tokens).
"""
import sys, os, random, math, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from vlib import *
from test_c_slots import learn_order

BENF = np.log10(1 + 1 / np.arange(1, 10))

# ---------------------------------------------------------------- corpora
def add_para(lines):
    pid = -1; prev = None
    for L in lines:
        if L.get('para_start') or L.get('folio') != prev or pid < 0:
            pid += 1
        prev = L.get('folio')
        L['para'] = pid
    return lines

def voynich(name='ZL3b', illus=None, lang=None):
    from test_a_language import unitize
    L = unitize(load_voynich(name, ('P',), True))
    if illus: L = [l for l in L if l['illus'] in illus]
    if lang: L = [l for l in L if l['lang'] == lang]
    return add_para([dict(l) for l in L])

def latin_verbose(n_words=35000, seed=3):
    """Isidore (data/plain/la.txt) under a fixed verbose substitution: every
    letter -> 1-2 Voynich-like glyph units.  Lines of 8-10 words, paras of 10."""
    rng = random.Random(seed)
    txt = open(os.path.join(DATA, 'plain', 'la.txt'), encoding='utf-8').read().lower()
    words = [''.join(c for c in w if 'a' <= c <= 'z') for w in txt.split()]
    words = [w.replace('j', 'i').replace('v', 'u') for w in words if w]
    words = [w for w in words if w][2000:2000 + n_words]
    units = list('qoktpfCSKTPFedsainlrmgy')
    letters = sorted(set(''.join(words)))
    code = {}
    used = set()
    for c in letters:
        while True:
            k = 1 if c in 'aeiou' else rng.choice([1, 2])
            s = ''.join(rng.choice(units) for _ in range(k))
            if s not in used:
                used.add(s); code[c] = s; break
    enc = [''.join(code[c] for c in w) for w in words]
    return chop(enc, rng)

def chop(tokens, rng, lmin=8, lmax=10, para=10):
    out, i, ln = [], 0, 0
    while i < len(tokens):
        k = rng.randint(lmin, lmax)
        out.append({'words': tokens[i:i + k], 'para_start': ln % para == 0})
        i += k; ln += 1
    for j, L in enumerate(out):
        L['folio'] = 'x%d' % (j // 40)
    return add_para(out)

def shuffle_global(lines, seed):
    rng = random.Random(seed)
    ws = [w for L in lines for w in L['words']]; rng.shuffle(ws)
    out, i = [], 0
    for L in lines:
        k = len(L['words']); nl = dict(L); nl['words'] = ws[i:i + k]; i += k; out.append(nl)
    return out

def shuffle_inline(lines, seed):
    rng = random.Random(seed)
    out = []
    for L in lines:
        ws = list(L['words']); rng.shuffle(ws); nl = dict(L); nl['words'] = ws; out.append(nl)
    return out

# ---------------------------------------------------------------- slot parse
def slot_model(lines, K=4, order=None, cand=None):
    words = [w for L in lines for w in L['words']]
    if order is None:
        order = learn_order(words[: len(words)])
    rank = {c: i for i, c in enumerate(order)}
    U = len(order)
    tc = Counter(words)
    types = list(tc)
    rk = [[rank.get(c, U - 1) for c in w] for w in types]
    best = None
    for cuts in (cand if cand is not None else itertools.combinations(range(1, U), K - 1)):
        b = [0] + list(cuts) + [U]
        def binof(r):
            for k in range(K):
                if r < b[k + 1]: return k
        fill = []
        cnt = [Counter() for _ in range(K)]
        for w, r in zip(types, rk):
            f = [''] * K
            for c, x in zip(w, r):
                f[binof(x)] += c
            fill.append(f)
            for k in range(K): cnt[k][f[k]] += tc[w]
        top = [set(x for x, _ in cnt[k].most_common(10)) for k in range(K)]
        cov = sum(tc[w] for w, f in zip(types, fill) if all(f[k] in top[k] for k in range(K)))
        if best is None or cov > best[0]:
            best = (cov, b, [[x for x, _ in cnt[k].most_common(10)] for k in range(K)])
    cov, b, tops = best
    return {'order': order, 'cuts': b, 'fillers': tops, 'coverage': cov / len(words)}

def encode(lines, model):
    """-> dict of numpy arrays: F (n,K) filler index or -1; line, para, pos."""
    order, b, tops = model['order'], model['cuts'], model['fillers']
    K = len(tops); rank = {c: i for i, c in enumerate(order)}
    idx = [{f: i for i, f in enumerate(t)} for t in tops]
    F, LN, PA, PO = [], [], [], []
    for li, L in enumerate(lines):
        for p, w in enumerate(L['words']):
            f = [''] * K
            for c in w:
                r = rank.get(c, len(order) - 1)
                k = max(i for i in range(K) if b[i] <= r)
                f[k] += c
            F.append([idx[k].get(f[k], -1) for k in range(K)])
            LN.append(li); PA.append(L['para']); PO.append(p)
    F = np.array(F, dtype=np.int64); LN = np.array(LN); PA = np.array(PA); PO = np.array(PO)
    valid = (F >= 0).all(1)
    return build_index(F, LN, PA, PO, valid)

def build_index(F, LN, PA, PO, valid):
    n = len(F)
    vi = np.where(valid)[0]
    # in-line adjacency among consecutive valid tokens
    a, bb = vi[:-1], vi[1:]
    same = LN[a] == LN[bb]
    pa, pb = a[same], bb[same]
    t1 = vi[:-2]; t2 = vi[1:-1]; t3 = vi[2:]
    st = (LN[t1] == LN[t2]) & (LN[t2] == LN[t3])
    tri = (t1[st], t2[st], t3[st])
    # line-final sum: lines with >= 3 valid tokens
    lines_valid = {}
    for t in vi:
        lines_valid.setdefault(LN[t], []).append(t)
    lv = [v for v in lines_valid.values() if len(v) >= 3]
    last = np.array([v[-1] for v in lv]); body_tok = np.concatenate([v[:-1] for v in lv]) if lv else np.array([], int)
    body_grp = np.concatenate([[g] * (len(v) - 1) for g, v in enumerate(lv)]) if lv else np.array([], int)
    # columns: same position, consecutive lines, same paragraph
    key = {(LN[t], PO[t]): t for t in vi}
    c1, c2, c3 = [], [], []
    for t in vi:
        u = key.get((LN[t] + 1, PO[t]));
        if u is None or PA[u] != PA[t]: continue
        v = key.get((LN[t] + 2, PO[t]))
        if v is None or PA[v] != PA[t]: continue
        c1.append(t); c2.append(u); c3.append(v)
    return {'F': F, 'valid': valid, 'vi': vi, 'tri': tri,
            'last': last, 'body_tok': body_tok, 'body_grp': body_grp, 'nlv': len(lv),
            'col': (np.array(c1, int), np.array(c2, int), np.array(c3, int)), 'LN': LN, 'PA': PA}

def subset(E, keep_para):
    """Restrict an encoding to tokens whose paragraph is in keep_para (bool array over para ids)."""
    m = keep_para[E['PA']]
    idx = np.where(m)[0]
    F = E['F'][idx]; LN = E['LN'][idx]; PA = E['PA'][idx]
    # recover positions: count within line
    PO = np.zeros(len(idx), int)
    for i in range(1, len(idx)):
        PO[i] = PO[i - 1] + 1 if LN[i] == LN[i - 1] else 0
    return build_index(F, LN, PA, PO, (F >= 0).all(1))

# ---------------------------------------------------------------- scoring
def values(E, D, rev=False):
    F = E['F']; K = F.shape[1]
    w = 10 ** np.arange(K - 1, -1, -1) if not rev else 10 ** np.arange(K)
    v = np.zeros(len(F), np.int64)
    Fv = np.where(F >= 0, F, 0)
    for k in range(K):
        v += D[k][Fv[:, k]] * w[k]
    return v

def score(E, v, parts=False):
    t1, t2, t3 = E['tri']
    d1 = v[t2] - v[t1]; d2 = v[t3] - v[t2]
    mono = np.mean((np.sign(d1) == np.sign(d2)) & (d1 != 0)) if len(t1) else 0
    cd = np.mean((d1 == d2) & (d1 != 0)) if len(t1) else 0
    c1, c2, c3 = E['col']
    e1 = v[c2] - v[c1]; e2 = v[c3] - v[c2]
    cmono = np.mean((np.sign(e1) == np.sign(e2)) & (e1 != 0)) if len(c1) else 0
    ccd = np.mean((e1 == e2) & (e1 != 0)) if len(c1) else 0
    if E['nlv']:
        s = np.bincount(E['body_grp'], weights=v[E['body_tok']], minlength=E['nlv'])
        sm = np.mean((s == v[E['last']]) & (s > 0))
    else:
        sm = 0
    J = mono + cmono + 5 * (cd + ccd) + 5 * sm
    if parts:
        vv = v[E['vi']]; pos = vv[vv > 0]
        lead = pos // 10 ** np.floor(np.log10(pos)).astype(np.int64)
        ld = np.bincount(lead, minlength=10)[1:10] / max(1, len(pos))
        benf = 1 - 0.5 * np.abs(ld - BENF).sum()
        rnd = np.mean(pos % 10 == 0) if len(pos) else 0
        return {'J': float(J), 'mono': float(mono), 'cdiff': float(cd), 'colmono': float(cmono),
                'colcdiff': float(ccd), 'sum': float(sm), 'benford_fit': float(benf), 'round': float(rnd),
                'n_tri': int(len(t1)), 'n_col': int(len(c1)), 'n_sumlines': int(E['nlv'])}
    return J

def score_tok(E, v):
    vv = v[E['vi']]; pos = vv[vv > 0]
    if not len(pos): return 0
    lead = pos // 10 ** np.floor(np.log10(pos)).astype(np.int64)
    ld = np.bincount(lead, minlength=10)[1:10] / len(pos)
    return (1 - 0.5 * np.abs(ld - BENF).sum()) + np.mean(pos % 10 == 0)

def random_D(K, rng):
    return [np.array(rng.permutation(10)) for _ in range(K)]

def anneal(E, steps=12000, seed=0, rev=False, T0=0.05, T1=0.0005, obj=score, D0=None):
    rng = np.random.default_rng(seed)
    K = E['F'].shape[1]
    D = [d.copy() for d in D0] if D0 is not None else random_D(K, rng)
    cur = obj(E, values(E, D, rev)); best, bestD = cur, [d.copy() for d in D]
    for s in range(steps):
        T = T0 * (T1 / T0) ** (s / steps)
        k = rng.integers(K); i, j = rng.choice(10, 2, replace=False)
        D[k][i], D[k][j] = D[k][j], D[k][i]
        new = obj(E, values(E, D, rev))
        if new >= cur or rng.random() < math.exp((new - cur) / T):
            cur = new
            if cur > best: best, bestD = cur, [d.copy() for d in D]
        else:
            D[k][i], D[k][j] = D[k][j], D[k][i]
    return best, bestD
