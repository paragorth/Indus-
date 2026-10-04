"""v12 cycle 3: what are the tandem swaps, and do they mark stream boundaries?
A. Hamming-1 neighbours (same length, one glyph differs) by glyph pair x/y at lag 1 and lag 2, vs 20
   within-line shuffles (obs/exp, z). Weave prediction: swaps between streams -> no lag-1 excess but lag-2 excess.
B. Phase: excess (over shuffle) of same-gallows concordance, Hamming-1 rate and junction agreement per
   junction index j (word j -> j+1) from line start; odd-minus-even contrast. Gloss-pair (text+gloss pairs)
   is the positive control for a pair phase.
C. Second-order alternation: I(x_t ; x_{t-2} | x_{t-1}) for word features (first glyph, gallows, a/o vowel,
   last glyph), excess over shuffle. A two-stream weave puts information at lag 2 that lag 1 does not
   carry (A B A); a single first-order process (mode persistence) does not.
D. Filler: word-type entropy and type/token of odd vs even line positions (vs shuffle); and a W2 weave model
   fitted on the whole corpus: per-stream type entropy, top words, P(switch) at Hamming-1 / gallows-change
   junctions vs others. For Weave-LF-hmm the decoded streams are scored against the planted truth."""
import sys, os, json, math, random, time
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v12_lib as V, v12_hmm as H

GALL = set('ktpfKTPF'); NS = 20


def shuf(lines, seed):
    rng = random.Random(seed); out = []
    for l in lines:
        ws = list(l['words']); rng.shuffle(ws); out.append(dict(l, words=ws))
    return out


def gal(w):
    for c in w:
        if c in GALL: return c
    return '-'


def vow(w):
    for c in w:
        if c in 'ao': return c
    return '-'


FEATS = {'first': lambda w: w[0], 'gallows': gal, 'vowel': vow, 'last': lambda w: w[-1]}


def ham_pairs(lines, lag):
    c = Counter()
    for l in lines:
        ws = l['words']
        for i in range(len(ws) - lag):
            a, b = ws[i], ws[i + lag]
            if V.hamming1(a, b):
                k = [i for i in range(len(a)) if a[i] != b[i]][0]
                c[tuple(sorted((a[k], b[k])))] += 1
    return c


def phase(lines, J=8):
    r = defaultdict(lambda: [0, 0])
    for l in lines:
        ws = l['words']
        for j in range(min(len(ws) - 1, J)):
            a, b = ws[j], ws[j + 1]
            ga, gb = gal(a), gal(b)
            if ga != '-' and gb != '-':
                r[('gconc', j)][0] += ga == gb; r[('gconc', j)][1] += 1
            r[('ham', j)][0] += V.hamming1(a, b); r[('ham', j)][1] += 1
            r[('vconc', j)][0] += (vow(a) == vow(b) and vow(a) != '-'); r[('vconc', j)][1] += 1
    return {k: v[0] / max(v[1], 1) for k, v in r.items()}


def H_(c):
    n = sum(c.values()); return -sum(v / n * math.log2(v / n) for v in c.values() if v)


def cmi2(lines, f):
    """I(x_t ; x_{t-2} | x_{t-1}) plug-in, within lines."""
    c3 = Counter(); 
    for l in lines:
        x = [f(w) for w in l['words']]
        for t in range(2, len(x)):
            c3[(x[t - 2], x[t - 1], x[t])] += 1
    ab = Counter(); bc = Counter(); b = Counter()
    for (p, q, r), v in c3.items():
        ab[(p, q)] += v; bc[(q, r)] += v; b[q] += v
    n = sum(c3.values())
    return sum(v / n * math.log2(v * b[q] / (ab[(p, q)] * bc[(q, r)])) for (p, q, r), v in c3.items())


def mi_lag(lines, f, lag):
    c = Counter(); a = Counter(); bb = Counter()
    for l in lines:
        x = [f(w) for w in l['words']]
        for t in range(lag, len(x)):
            c[(x[t - lag], x[t])] += 1
    for (p, q), v in c.items(): a[p] += v; bb[q] += v
    n = sum(c.values())
    return sum(v / n * math.log2(v * n / (a[p] * bb[q])) for (p, q), v in c.items())


def parity_entropy(lines):
    o = Counter(); e = Counter()
    for l in lines:
        for i, w in enumerate(l['words']):
            (o if i % 2 == 0 else e)[w] += 1
    return H_(o), H_(e), len(o) / sum(o.values()), len(e) / sum(e.values())


def stats(lines):
    r = {}
    for lag in (1, 2):
        r[f'ham{lag}'] = dict((f'{a}/{b}', v) for (a, b), v in ham_pairs(lines, lag).items())
    r['phase'] = {f'{k[0]}{k[1]}': v for k, v in phase(lines).items()}
    for fn, f in FEATS.items():
        r[f'cmi2_{fn}'] = cmi2(lines, f); r[f'mi1_{fn}'] = mi_lag(lines, f, 1); r[f'mi2_{fn}'] = mi_lag(lines, f, 2)
    r['par_ent'] = parity_entropy(lines)
    return r


def part_abc(name):
    ck = os.path.join(V.CK, f'c3abc_{name}.json')
    if os.path.exists(ck): return json.load(open(ck))
    C = V.corpus(name)
    obs = stats(C); nulls = [stats(shuf(C, s)) for s in range(1, NS + 1)]
    out = {'name': name, 'obs': obs, 'nulls': nulls}
    json.dump(out, open(ck, 'w'))
    print('abc', name, flush=True)
    return out


def part_d(name):
    ck = os.path.join(V.CK, f'c3d_{name}.json')
    if os.path.exists(ck): return json.load(open(ck))
    t0 = time.time()
    C = V.corpus(name)
    truth = list(V.LAST_LABELS) if name.startswith('Weave') else None
    best = None
    for sd in (1, 2):
        m = H.StreamModel(2, 'W', seed=sd); m.fit(C, iters=10)
        if best is None or m.train_bits > best.train_bits: best = m
    m = best
    wc = [Counter(), Counter()]; sw_h = [0, 0]; sw_o = [0, 0]; sw_g = [0, 0]; agree = [0, 0]; k = 0
    for l in C:
        ws = l['words']
        if not ws: continue
        P, sw = m.decode(ws)
        for t, w in enumerate(ws):
            for s in range(2): wc[s][w] += P[t][s]
            if truth is not None:
                agree[0] += (P[t][1] > 0.5) == (truth[k] == 1); agree[1] += 1
            k += 1
            if t >= 1:
                a, b = ws[t - 1], ws[t]
                if V.hamming1(a, b): sw_h[0] += sw[t]; sw_h[1] += 1
                else: sw_o[0] += sw[t]; sw_o[1] += 1
                if gal(a) != '-' and gal(b) != '-' and gal(a) != gal(b): sw_g[0] += sw[t]; sw_g[1] += 1
    ent = [H_(c) for c in wc]
    out = {'name': name, 'A': m.A, 'pi': m.pi, 'train_bits': m.train_bits,
           'share': [sum(c.values()) for c in wc], 'type_entropy': ent,
           'top': [[(w, round(v, 1)) for w, v in c.most_common(15)] for c in wc],
           'P_switch_hamming1': sw_h[0] / max(sw_h[1], 1), 'n_hamming1': sw_h[1],
           'P_switch_other': sw_o[0] / max(sw_o[1], 1),
           'P_switch_gallows_change': sw_g[0] / max(sw_g[1], 1), 'n_gchange': sw_g[1],
           'truth_agree': (max(agree[0], agree[1] - agree[0]) / agree[1]) if truth is not None and agree[1] else None,
           'sec': time.time() - t0}
    json.dump(out, open(ck, 'w'), indent=1)
    print('d', name, round(out['sec']), flush=True)
    return out


def job(a):
    kind, name = a
    return part_abc(name) if kind == 'abc' else part_d(name)


if __name__ == '__main__':
    names = V.VOY + V.POS + V.NEG + ['Gloss-pair']
    jobs = [('abc', n) for n in names] + [('d', n) for n in ['Voynich-ZL', 'Voynich-IT', 'Weave-LF-hmm', 'Weave-VV-alt', 'vLatin', 'Shuf-line']]
    with Pool(2) as p:
        p.map(job, jobs, chunksize=1)
    print('done all')
