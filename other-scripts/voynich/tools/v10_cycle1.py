"""v10 cycle 1: characterise the line-initial chain as its own text.
Chain = first glyph of each line, down each paragraph (paragraph-first line kept separately: it is mostly a gallows).
(1a) alphabet, unigram entropy, conditional entropies H1|0, H2|1 (plug-in) vs within-paragraph line permutation.
(1b) lag profile: excess MI and same-glyph rate at lags 1..8 (within paragraph body).
(1c) spectrum: page-wise concatenated body chain, one indicator series per common glyph, summed periodogram at
     periods 2..16 lines; excess over permutation; best period reported with max-statistic null.
(1d) fixed cyclic order (rotating key / counter mod k): over all cyclic orders of the top-k glyphs (k=4..8),
     maximise the share of transitions that step +1 (and +2) along the cycle; null = same max on permuted chains.
(1e) counter tied to line index: MI(glyph, line index in paragraph) and MI(glyph, index mod b) b=2..6.
(1f) restarts: does the chain continue across paragraph, page, bifolio(approx.) and quire boundaries?
     MI of (last body-line glyph before the boundary, second-line glyph after it) vs re-pairing null;
     also paragraph-first line -> line 2 inside a paragraph.
Controls: (i) the same battery on the permuted chain must give nothing (z ~ 0 everywhere, by construction for the
     permutation, so we also show a 'planted counter' corpus: Voynich lines whose first glyph is replaced by a mod-6
     counter through d,y,o,q,s,t restarting at each paragraph, and a 'planted acrostic': first glyphs replaced by a
     Latin text through a lossy letter->glyph map; both must light up (1d)/(1e) or (1a)/(1b) respectively).
Both transcriptions. Checkpoint: data/results/v10/c1_<name>.json"""
import sys, os, json, random, math, itertools
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v10_lib import *

REPS = int(os.environ.get('REPS', 300))
TOP = ['d', 'y', 'o', 'q', 's', 't', 'S', 'C']

def body(paras): return [p['chain'][1:] for p in paras]

def condH(seqs, order):
    ctx = Counter(); joint = Counter()
    for s in seqs:
        for i in range(order, len(s)):
            c = tuple(s[i - order:i]); ctx[c] += 1; joint[c + (s[i],)] += 1
    n = sum(joint.values())
    return -sum(v / n * math.log2(v / ctx[k[:-1]]) for k, v in joint.items())

def lagstats(seqs, lag):
    P = [(s[i], s[i + lag]) for s in seqs for i in range(len(s) - lag)]
    return mi(P), (sum(a == b for a, b in P) / len(P) if P else 0)

def spectrum(paras, glyphs=TOP[:6], periods=range(2, 17)):
    # page-wise concatenation of body chains
    pages = defaultdict(list)
    for p in paras: pages[p['page_idx']].extend(p['chain'][1:])
    out = {}
    for T in periods:
        tot = 0.0
        for g in glyphs:
            for s in pages.values():
                if len(s) < 2 * T: continue
                x = [1.0 if c == g else 0.0 for c in s]; m = sum(x) / len(x); x = [v - m for v in x]
                re_ = sum(v * math.cos(2 * math.pi * i / T) for i, v in enumerate(x))
                im_ = sum(v * math.sin(2 * math.pi * i / T) for i, v in enumerate(x))
                tot += (re_ * re_ + im_ * im_) / len(x)
        out[T] = tot
    return out

def cycle_scores(seqs, k):
    syms = TOP[:k]; P = Counter((s[i], s[i + 1]) for s in seqs for i in range(len(s) - 1) if s[i] in syms and s[i + 1] in syms)
    n = sum(P.values()); best1 = (0, None); best2 = (0, None)
    for perm in itertools.permutations(syms[1:]):
        cyc = (syms[0],) + perm; nxt = {cyc[i]: cyc[(i + 1) % k] for i in range(k)}; nx2 = {cyc[i]: cyc[(i + 2) % k] for i in range(k)}
        s1 = sum(P[(a, nxt[a])] for a in syms) / n; s2 = sum(P[(a, nx2[a])] for a in syms) / n
        if s1 > best1[0]: best1 = (s1, ''.join(cyc))
        if s2 > best2[0]: best2 = (s2, ''.join(cyc))
    return best1, best2

def idx_mi(paras, mod=None, cap=8):
    P = []
    for p in paras:
        for i, g in enumerate(p['chain'][1:], start=1):
            P.append(((i % mod) if mod else min(i, cap), g))
    return mi(P)

def boundary_pairs(paras, kind):
    """(glyph of last line before boundary, glyph of 2nd line after boundary) for consecutive paragraphs."""
    P = []
    for a, b in zip(paras, paras[1:]):
        if len(b['chain']) < 2: continue
        same_page = a['page_idx'] == b['page_idx']
        if kind == 'paragraph' and not same_page: continue
        if kind == 'page' and (same_page or b['page_idx'] != a['page_idx'] + 1): continue
        if kind == 'bifolio' and (a['bifolio'] == b['bifolio'] or same_page): continue
        if kind == 'quire' and (a['quire'] == b['quire']): continue
        if kind == 'page' and a['bifolio'] != b['bifolio']: pass
        P.append((a['chain'][-1], b['chain'][1]))
    return P

def repair_null(P, reps, seed):
    xs = []; rng = random.Random(seed); A = [a for a, _ in P]; B = [b for _, b in P]
    for _ in range(reps):
        rng.shuffle(B); xs.append(mi(list(zip(A, B))))
    return xs

def battery(paras):
    seqs = body(paras); R = {}
    allg = Counter(g for s in seqs for g in s)
    R['alphabet'] = {'types': len(allg), 'types>=1%': sum(v >= 0.01 * sum(allg.values()) for v in allg.values()),
                     'H0': round(math.log2(len(allg)), 3), 'H1': round(H(allg), 3)}
    R['H1|0'] = condH(seqs, 1); R['H2|1'] = condH(seqs, 2)
    for lag in range(1, 9):
        m, s = lagstats(seqs, lag); R['lag%d mi' % lag] = m; R['lag%d same' % lag] = s
    R['spec'] = spectrum(paras)
    for k in range(4, 9):
        (s1, c1), (s2, c2) = cycle_scores(seqs, k); R['cyc%d +1' % k] = s1; R['cyc%d +2' % k] = s2
        R['cyc%d order' % k] = c1
    R['idx mi'] = idx_mi(paras)
    for b in range(2, 7): R['idx mod%d mi' % b] = idx_mi(paras, b)
    R['first->2nd mi'] = mi([(p['chain'][0], p['chain'][1]) for p in paras if len(p['chain']) > 1])
    return R

def null_job(args):
    name, seed, plant = args
    paras = load(name, plant)
    return battery(perm_body(paras, random.Random(seed)))

_cache = {}
def load(name, plant=None):
    key = (name, plant)
    if key in _cache: return _cache[key]
    paras = chain_paras(name)
    if plant == 'counter':
        cyc = ['d', 'y', 'o', 'q', 's', 't']; rng = random.Random(5)
        for p in paras:
            st = rng.randrange(6)
            p['chain'] = [p['chain'][0]] + [cyc[(st + i) % 6] for i in range(len(p['chain']) - 1)]
    elif plant == 'acrostic':
        txt = letters_of('Latin-Caesar'); rng = random.Random(9)
        glyphs = ['d', 'y', 'o', 'q', 's', 't', 'S', 'C', 'l', 'p']
        lmap = {c: glyphs[i % len(glyphs)] for i, c in enumerate(sorted(set(txt), key=lambda c: -txt.count(c)))}
        pos = 0
        for p in paras:
            p['chain'] = [p['chain'][0]] + [lmap[txt[pos + i]] for i in range(len(p['chain']) - 1)]; pos += len(p['chain'])
    _cache[key] = paras
    return paras

def summarise(obs, nulls):
    S = {}
    for k, v in obs.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k != 'alphabet':
            xs = [n[k] for n in nulls]; S[k] = zstat(v, xs)
    # spectrum: excess per period and max-statistic
    ex = {T: obs['spec'][T] - sum(n['spec'][T] for n in nulls) / len(nulls) for T in obs['spec']}
    sd = {T: (sum((n['spec'][T] - (obs['spec'][T] - ex[T])) ** 2 for n in nulls) / (len(nulls) - 1)) ** .5 for T in obs['spec']}
    zT = {T: ex[T] / sd[T] for T in ex}; bestT = max(zT, key=zT.get)
    nullmax = [max((n['spec'][T] - (obs['spec'][T] - ex[T])) / sd[T] for T in ex) for n in nulls]
    S['spectrum best period'] = {'T': bestT, 'z': round(zT[bestT], 2),
                                 'p_max': round((1 + sum(x >= zT[bestT] for x in nullmax)) / (1 + len(nullmax)), 4),
                                 'z_by_T': {T: round(z, 1) for T, z in zT.items()}}
    S['alphabet'] = obs['alphabet']
    for k in range(4, 9): S['cyc%d order' % k] = obs['cyc%d order' % k]
    return S

if __name__ == '__main__':
    res = {}
    jobs = [('ZL3b', None), ('IT2a', None), ('ZL3b', 'counter'), ('ZL3b', 'acrostic')]
    with Pool(2) as pool:
        for name, plant in jobs:
            ck = os.path.join(CK, 'c1_%s_%s.json' % (name, plant))
            if os.path.exists(ck): res[(name, plant)] = json.load(open(ck)); continue
            paras = load(name, plant); obs = battery(paras)
            reps = REPS if plant is None else 100
            nulls = pool.map(null_job, [(name, s, plant) for s in range(reps)])
            S = summarise(obs, nulls)
            if plant is None:
                for kind in ('paragraph', 'page', 'bifolio', 'quire'):
                    P = boundary_pairs(paras, kind); S['boundary ' + kind] = dict(zstat(mi(P), repair_null(P, 1000, 1)), n=len(P))
                # inside-paragraph reference with same n as paragraph boundary
            json.dump(S, open(ck, 'w'), indent=1); res[(name, plant)] = S
            print(name, plant, json.dumps(S)[:3000], flush=True)
