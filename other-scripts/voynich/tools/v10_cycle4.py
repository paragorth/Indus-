"""v10 cycle 4: is ALL of the chain's structure one first-order margin rule?
If the chain hides a text (acrostic, key, counter) there must be structure beyond 'glyph above -> glyph below'.
(4a) First-order Markov surrogates (transition table of the corpus' own chain; same paragraph lengths and line-2
     glyphs; tables fitted globally, and separately for Currier A and B): do they reproduce lag-2/3 same-glyph rates,
     lag-2 MI, the period-2 spectral power and H(X2|X0X1)? 500 surrogates; z of observed vs surrogates.
(4b) Held-out order test: order-2 (backed off to order-1) vs order-1 transition model, trained on even folios,
     scored on odd folios (and reverse), bits/line gain; null = same gain on Markov-1 surrogates of the test half.
(4c) What alternates: the 2-step return rate per glyph (P(x at r+2 | x at r)) vs surrogate, to see which glyphs carry
     the A-B-A pattern.
Control: a planted order-2 chain (each body glyph copies the glyph two lines up with p=0.25, else follows the real
     first-order table) must be flagged by (4a)/(4b); the planted acrostic of cycle 3 (Latin through a lossy map)
     must be flagged by (4b)."""
import sys, os, json, random, math
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v10_lib import *
from v10_cycle1 import spectrum

REPS = int(os.environ.get('REPS', 500))

def body_seqs(paras): return [p['chain'][1:] for p in paras if len(p['chain']) > 2]

def fit(seqs):
    T = defaultdict(Counter)
    for s in seqs:
        for a, b in zip(s, s[1:]): T[a][b] += 1
    return {a: (list(c), list(c.values())) for a, c in T.items()}

def surrogate(paras, rng, by_lang=True):
    groups = defaultdict(list)
    for p in paras: groups[p['lang'] if by_lang else 0].append(p)
    tabs = {g: fit(body_seqs(ps)) for g, ps in groups.items()}
    out = []
    for p in paras:
        tab = tabs[p['lang'] if by_lang else 0]; s = p['chain'][1:]
        if not s: out.append(dict(p)); continue
        q = [s[0]]
        for _ in range(len(s) - 1):
            ks, ws = tab.get(q[-1]) or tab['d']; q.append(rng.choices(ks, ws)[0])
        d = dict(p); d['chain'] = [p['chain'][0]] + q; out.append(d)
    return out

def stats(paras):
    S = body_seqs(paras); R = {}
    for lag in (2, 3):
        P = [(s[i], s[i + lag]) for s in S for i in range(len(s) - lag)]
        R['lag%d same' % lag] = sum(a == b for a, b in P) / len(P)
        R['lag%d mi' % lag] = mi(P)
    sp = spectrum(paras, periods=(2, 3)); R['power T2'] = sp[2]; R['power T3'] = sp[3]
    ctx = Counter(); jt = Counter()
    for s in S:
        for i in range(2, len(s)): ctx[(s[i - 2], s[i - 1])] += 1; jt[(s[i - 2], s[i - 1], s[i])] += 1
    n = sum(jt.values()); R['H2|1'] = -sum(v / n * math.log2(v / ctx[k[:2]]) for k, v in jt.items())
    c2 = Counter(); r2 = Counter()
    for s in S:
        for i in range(len(s) - 2): c2[s[i]] += 1; r2[s[i]] += s[i + 2] == s[i]
    for g in 'dyoqstSC': R['ret2 ' + g] = r2[g] / max(1, c2[g])
    return R

def heldout_gain(train, test, alpha=0.5):
    """bits/glyph gain of order-2 (backoff) over order-1 (backoff to unigram) on test."""
    u = Counter(); T1 = defaultdict(Counter); T2 = defaultdict(Counter)
    for s in train:
        for i, x in enumerate(s):
            u[x] += 1
            if i >= 1: T1[s[i - 1]][x] += 1
            if i >= 2: T2[(s[i - 2], s[i - 1])][x] += 1
    V = sorted(set(u) | {x for s in test for x in s}); N = sum(u.values())
    pu = {v: (u[v] + alpha) / (N + alpha * len(V)) for v in V}
    def p1(a, x):
        r = T1[a]; return (r[x] + 5 * pu[x]) / (sum(r.values()) + 5)
    g, n = 0.0, 0
    for s in test:
        for i in range(2, len(s)):
            a, b, x = s[i - 2], s[i - 1], s[i]; r = T2[(a, b)]
            q1 = p1(b, x); q2 = (r[x] + 5 * q1) / (sum(r.values()) + 5)
            g += math.log2(q2 / q1); n += 1
    return g / n

def halves(paras):
    ev = [p for p in paras if fnum(p['folio'])[0] % 2 == 0]; od = [p for p in paras if fnum(p['folio'])[0] % 2 == 1]
    return ev, od

def ho(paras):
    ev, od = halves(paras)
    return (heldout_gain(body_seqs(ev), body_seqs(od)) + heldout_gain(body_seqs(od), body_seqs(ev))) / 2

_P = {}
def load(name, plant):
    if (name, plant) in _P: return _P[(name, plant)]
    paras = chain_paras(name)
    if plant == 'order2':
        rng = random.Random(4); tab = fit(body_seqs(paras))
        for p in paras:
            s = p['chain'][1:]
            for i in range(1, len(s)):
                if i >= 2 and rng.random() < 0.25: s[i] = s[i - 2]
                else: ks, ws = tab.get(s[i - 1]) or tab['d']; s[i] = rng.choices(ks, ws)[0]
            p['chain'] = [p['chain'][0]] + s
    elif plant == 'acro_la':
        txt = letters_of('Latin-Caesar'); txt = txt[len(txt) // 3:]
        glyphs = ['d', 'y', 'o', 'q', 's', 't', 'S', 'C', 'l', 'p']
        order = sorted(set(txt), key=lambda c: -txt.count(c)); random.Random(9).shuffle(order)
        lmap = {c: glyphs[i % len(glyphs)] for i, c in enumerate(order)}; pos = 0
        for p in paras:
            nb = len(p['chain']); p['chain'] = [lmap[c] for c in txt[pos:pos + nb]]; pos += nb
    _P[(name, plant)] = paras
    return paras

def job(a):
    name, plant, seed = a
    paras = load(name, plant); sur = surrogate(paras, random.Random(seed))
    return stats(sur), ho(sur)

if __name__ == '__main__':
    out = {}
    with Pool(2) as pool:
        for name, plant in (('ZL3b', None), ('IT2a', None), ('ZL3b', 'order2'), ('ZL3b', 'acro_la')):
            ck = os.path.join(CK, 'c4_%s_%s.json' % (name, plant))
            if os.path.exists(ck): out[(name, plant)] = json.load(open(ck)); continue
            paras = load(name, plant); o = stats(paras); oh = ho(paras)
            reps = REPS if plant is None else 200
            res = pool.map(job, [(name, plant, s) for s in range(reps)])
            S = {k: zstat(o[k], [r[0][k] for r in res]) for k in o}
            S['heldout order-2 gain bits/glyph'] = zstat(oh, [r[1] for r in res])
            json.dump(S, open(ck, 'w'), indent=1); out[(name, plant)] = S
            print(name, plant, json.dumps(S), flush=True)
