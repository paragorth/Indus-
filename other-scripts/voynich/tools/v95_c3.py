"""v95 cycle 3: THE CHART WAS COMPUTED.  In a computed record (a geomantic chart: 4 random mothers, 12 figures derived by
bitwise XOR of fixed pairs) a token at one slot is a function of the tokens at two other slots, while each of them alone
says nothing about it: SYNERGY = held-out bits gained by predicting C3 from (C1, C2) jointly over the better single slot.
Samples = consecutive line pairs inside a paragraph; slot = (line 0/1, word index 0-7).  Hypotheses = random slot triples x
random word-class maps (top-k core words, top-k whole words, presence bits of random glyph sets, first/last glyph classes).
usage: python3 v95_c3.py CORPUS [NHYP]
"""
import sys, os, json, random, time, zlib
import numpy as np
import v95_lib as L, v72_lib as V

NW = 8
NTW = 4


def samples(pages, h):
    S = []
    for p in pages:
        if L.half(p['id']) != h: continue
        ls = p['lines']
        for i in range(len(ls) - 1):
            if ls[i + 1]['ps']: continue
            S.append((ls[i]['w'], ls[i + 1]['w']))
    return S


def hypotheses(n, seed=953):
    rng = random.Random(seed); H = []
    slots = [(a, b) for a in (0, 1) for b in range(NW)]
    for i in range(n):
        s = rng.sample(slots, 3)
        kind = rng.choice(['core', 'word', 'bits', 'edge'])
        if kind in ('core', 'word'): par = rng.choice([4, 8, 16, 32])
        elif kind == 'bits': par = [''.join(rng.sample(L.ALPHA[:-1], rng.randint(1, 3))) for _ in range(rng.randint(2, 5))]
        else: par = (rng.choice(['first', 'last', 'both']), rng.choice([4, 8, 16]), rng.randrange(10 ** 6))
        H.append(dict(i=i, slots=s, kind=kind, par=par))
    return H


def classer(hyp, train_words):
    k, par = hyp['kind'], hyp['par']
    if k in ('core', 'word'):
        f = (lambda w: V._core(w)) if k == 'core' else (lambda w: w)
        from collections import Counter
        top = [w for w, _ in Counter(f(w) for w in train_words).most_common(par - 1)]
        ix = {w: j for j, w in enumerate(top)}
        return lambda w: ix.get(f(w), par - 1), par
    if k == 'bits':
        return (lambda w: sum((1 << j) for j, gs in enumerate(par) if any(g in w for g in gs))), 2 ** len(par)
    side, m, salt = par
    def g(w):
        key = w[0] if side == 'first' else (w[-1] if side == 'last' else w[0] + w[-1])
        return zlib.crc32((key + str(salt)).encode()) % m
    return g, m


def get(sm, slot):
    li, wi = slot; ws = sm[li]
    return ws[wi] if wi < len(ws) else None


def ll(tr, te, nc, cols, a=0.5):
    """held-out bits per sample of C3 given the columns in cols (tuple of indices into (c1, c2))."""
    if cols:
        key_tr = sum(tr[:, c] * (nc ** j) for j, c in enumerate(cols)); key_te = sum(te[:, c] * (nc ** j) for j, c in enumerate(cols))
    else:
        key_tr = np.zeros(len(tr), int); key_te = np.zeros(len(te), int)
    nk = nc ** len(cols)
    T = np.zeros((nk, nc)); np.add.at(T, (key_tr, tr[:, 2]), 1)
    marg = (np.bincount(tr[:, 2], minlength=nc) + a) / (len(tr) + a * nc)
    P = (T + a * nc * marg) / (T.sum(1, keepdims=True) + a * nc)      # back off to the marginal
    return -np.mean(np.log2(P[key_te, te[:, 2]]))


def synergy(tr, te, nc):
    l1 = ll(tr, te, nc, (0,)); l2 = ll(tr, te, nc, (1,)); l12 = ll(tr, te, nc, (0, 1))
    return min(l1, l2) - l12


def arr(S, hyp, cl):
    rows = []
    for sm in S:
        ws = [get(sm, s) for s in hyp['slots']]
        if None in ws: continue
        rows.append([cl(w) for w in ws])
    return np.array(rows, int).reshape(-1, 3)


def evaluate(S0, S1, hyp, train_words):
    cl, nc = classer(hyp, train_words)
    tr = arr(S0, hyp, cl); te = arr(S1, hyp, cl)
    if len(tr) < 200 or len(te) < 200: return None
    # train score: 2-fold inside the train half; test score: fit on train, score on test
    h = len(tr) // 2
    s_tr = (synergy(tr[:h], tr[h:], nc) + synergy(tr[h:], tr[:h], nc)) / 2
    return float(s_tr), float(synergy(tr, te, nc)), len(te)


def run(name, nh):
    out = os.path.join(L.CK, 'c3_%s.json' % name)
    if os.path.exists(out): return
    t0 = time.time(); C = L.corpus(name)
    vs = [('real', C)] + [('tw%d' % s, L.twin(C, 9800 + s)) for s in range(NTW)]
    SS = {v: (samples(c, 0), samples(c, 1), [w for p in c if L.half(p['id']) == 0 for l in p['lines'] for w in l['w']])
          for v, c in vs}
    rows = []
    for hyp in hypotheses(nh):
        r = {'i': hyp['i']}
        for v, _ in vs:
            e = evaluate(*SS[v][:2], hyp, SS[v][2]); r[v] = e
        rows.append(r)
    json.dump(dict(name=name, nh=nh, rows=rows, sec=time.time() - t0), open(out, 'w'))
    print(name, 'done', round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
