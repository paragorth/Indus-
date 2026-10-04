"""v10 cycle 2: is the line-initial chain a KEY STREAM for the line it heads (or the line below)?
Key k_r = first glyph of line r (source 'own') or of line r-1 ('above'). Line body = words 2..n of line r (the
word that carries the key glyph is dropped). Only body lines of paragraphs (line 2..n), so every key is a chain member.
(2a) omnibus: MI(key, body glyph) and MI(key, body word-initial glyph), MI(key, 2nd word) vs keys permuted among the
     body lines of the same paragraph. Any key-dependent substitution of a skewed glyph distribution must show here.
(2b) brute force of simple key-application rules; each rule transforms every body line given its key, then we score
     the regularity of the whole transformed text: word-type entropy (bits/word), within-word glyph conditional
     entropy H(g_i|g_i-1), and number of word types. Rules:
       shift   : g -> O[(pos(g) + s*v(key)) mod |O|], O in {frequency order, EVA alphabetical, chain cycle order},
                 s in {+1,-1,+2,-2}, v(key) = rank of key in the key-frequency order; scope in {all glyphs,
                 word-initial, word-final, first body word only}
       select  : keep glyph j of the body iff (j - v(key)) mod n == 0, n in {2,3}; read the kept glyphs as text
                 (scored by glyph H(g|g-1) only)
     Statistic per rule: regularity with the true keys minus mean regularity with keys permuted within paragraph
     (the 'wrong key' null); max over all rules compared with the null max (keys permuted, same search).
Controls: planted key (Voynich ZL3b body glyphs shifted on the frequency order by +v(own key), all glyphs; and a
     weaker one shifting only word-initial glyphs) must be found by the search; a line-shuffled (whole lines moved
     between paragraphs at random) Voynich must give nothing."""
import sys, os, json, random, math
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v10_lib import *

REPS = int(os.environ.get('REPS', 100))

def items(paras):
    """list of (key_own, key_above, body word glyph lists, paragraph id)"""
    out = []
    for pi, p in enumerate(paras):
        for i in range(1, len(p['lines'])):
            l = p['lines'][i]; g = [U(w) for w in l[1:]]   # body = words 2..n (the key's own word is dropped,
            g = [w for w in g if w]                         # so in-word dependence on the key glyph cannot leak in)
            if not g: continue
            out.append([p['chain'][i], p['chain'][i - 1], g, pi])
    return out

def orders(its):
    fc = Counter(x for it in its for w in it[2] for x in w)
    freq = [x for x, _ in fc.most_common()]
    alpha = sorted(fc)
    cyc = list('dqCtsSoy'); cyc += [x for x in freq if x not in cyc]
    return {'freq': freq, 'alpha': alpha, 'cycle': cyc}

def keyrank(its):
    kc = Counter(it[0] for it in its); return {k: i for i, (k, _) in enumerate(kc.most_common())}

def transform(its, keys, O, s, scope, vk):
    pos = {x: i for i, x in enumerate(O)}; n = len(O); out = []
    for (it, k) in zip(its, keys):
        sh = s * vk.get(k, 0); ws = []
        for wi, w in enumerate(it[2]):
            if scope == 'all': w2 = [O[(pos[x] + sh) % n] for x in w]
            elif scope == 'init': w2 = [O[(pos[w[0]] + sh) % n]] + w[1:]
            elif scope == 'final': w2 = w[:-1] + [O[(pos[w[-1]] + sh) % n]]
            elif scope == 'firstword': w2 = [O[(pos[x] + sh) % n] for x in w] if wi == 0 else w
            ws.append(w2)
        out.append(ws)
    return out

def regularity(lines):
    wc = Counter(''.join(w) for l in lines for w in l)
    big = Counter(); ctx = Counter()
    for l in lines:
        for w in l:
            w = ['^'] + w
            for a, b in zip(w, w[1:]): big[(a, b)] += 1; ctx[a] += 1
    n = sum(big.values())
    Hc = -sum(v / n * math.log2(v / ctx[a]) for (a, b), v in big.items())
    return {'Hword': H(wc), 'Hglyph': Hc, 'types': len(wc)}

def select_reg(its, keys, nstep, vk):
    big = Counter(); ctx = Counter()
    for it, k in zip(its, keys):
        g = [x for w in it[2] for x in w + [' ']]
        kept = [x for j, x in enumerate(g) if (j - vk.get(k, 0)) % nstep == 0]
        for a, b in zip(kept, kept[1:]): big[(a, b)] += 1; ctx[a] += 1
    n = sum(big.values())
    return -sum(v / n * math.log2(v / ctx[a]) for (a, b), v in big.items())

RULES = [('shift', o, s, sc, src) for o in ('freq', 'alpha', 'cycle') for s in (1, -1, 2, -2)
         for sc in ('all', 'init', 'final', 'firstword') for src in ('own', 'above')] + \
        [('select', None, n, None, src) for n in (2, 3) for src in ('own', 'above')]

def rule_scores(its, keysets, O, vk):
    """keysets: {'own': keys, 'above': keys}. Returns {rule: (Hword, Hglyph, types)}"""
    R = {}
    for r in RULES:
        kind, o, s, sc, src = r
        if kind == 'shift':
            g = regularity(transform(its, keysets[src], O[o], s, sc, vk)); R[r] = (g['Hword'], g['Hglyph'], g['types'])
        else:
            R[r] = (0.0, select_reg(its, keysets[src], s, vk), 0)
    return R

def permkeys(its, rng):
    byp = defaultdict(list)
    for j, it in enumerate(its): byp[it[3]].append(j)
    own = [None] * len(its); above = [None] * len(its)
    for js in byp.values():
        perm = rng.sample(js, len(js))
        for a, b in zip(js, perm): own[a] = its[b][0]; above[a] = its[b][1]
    return {'own': own, 'above': above}

def omnibus(its, keys):
    P1 = [(k, x) for it, k in zip(its, keys) for w in it[2] for x in w]
    P2 = [(k, w[0]) for it, k in zip(its, keys) for w in it[2]]
    P3 = [(k, ''.join(it[2][0])) for it, k in zip(its, keys) if it[2]]
    c = Counter(w for _, w in P3); P3 = [(k, w if c[w] >= 5 else '*') for k, w in P3]
    return mi(P1), mi(P2), mi(P3)

_C = {}
def setup(name, plant):
    if (name, plant) in _C: return _C[(name, plant)]
    paras = chain_paras(name)
    if plant == 'shuffled':
        rng = random.Random(3); allb = [l for p in paras for l in p['lines'][1:]]; rng.shuffle(allb); k = 0
        for p in paras:
            nb = len(p['lines']) - 1; p['lines'] = [p['lines'][0]] + allb[k:k + nb]; k += nb
            p['chain'] = [U(l[0])[0] for l in p['lines']]
    its = items(paras); O = orders(its); vk = keyrank(its)
    if plant in ('key_all', 'key_init'):
        pos = {x: i for i, x in enumerate(O['freq'])}; n = len(O['freq'])
        for it in its:
            sh = vk.get(it[0], 0)
            it[2] = [[O['freq'][(pos[x] + sh) % n] for x in w] if plant == 'key_all' else [O['freq'][(pos[w[0]] + sh) % n]] + w[1:]
                     for w in it[2]]
        # decipher with the true key = shift by -v: our rules include s=-1 on freq order
    _C[(name, plant)] = (its, O, vk)
    return _C[(name, plant)]

def job(args):
    name, plant, seed = args
    its, O, vk = setup(name, plant)
    ks = {'own': [it[0] for it in its], 'above': [it[1] for it in its]} if seed < 0 else permkeys(its, random.Random(seed))
    return seed, rule_scores(its, ks, O, vk), omnibus(its, ks['own']), omnibus(its, ks['above'])

if __name__ == '__main__':
    out = {}
    with Pool(2) as pool:
        for name, plant in (('ZL3b', None), ('IT2a', None), ('ZL3b', 'key_all'), ('ZL3b', 'key_init'), ('ZL3b', 'shuffled')):
            ck = os.path.join(CK, 'c2_%s_%s.json' % (name, plant))
            if os.path.exists(ck): out[(name, plant)] = json.load(open(ck)); print(name, plant, 'cached'); continue
            reps = REPS if plant is None else 30
            res = pool.map(job, [(name, plant, s) for s in [-1] + list(range(reps))])
            obs = res[0]; nulls = res[1:]
            S = {}
            for j, lab in ((2, 'own'), (3, 'above')):
                for q, nm in enumerate(('glyph', 'word-initial glyph', 'first body word (word 2)')):
                    S['omnibus MI(key %s, %s)' % (lab, nm)] = zstat(obs[j][q], [n[j][q] for n in nulls])
            # rule search: improvement = null mean - observed (lower entropy = more regular), z by null sd
            best = []; nullmax = []
            for mi_, nm in ((0, 'Hword'), (1, 'Hglyph')):
                zs = {}
                for r in RULES:
                    if r[0] == 'select' and mi_ == 0: continue
                    xs = [n[1][r][mi_] for n in nulls]; m = sum(xs) / len(xs)
                    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** .5 or 1e-9
                    zs[r] = ((m - obs[1][r][mi_]) / sd, m - obs[1][r][mi_], m, sd)
                rb = max(zs, key=lambda r: zs[r][0])
                # null max: each null replicate scored against the other replicates' mean/sd (leave-in approx)
                nm_ = [max((zs[r][2] - n[1][r][mi_]) / zs[r][3] for r in zs) for n in nulls]
                S['best rule by %s' % nm] = {'rule': str(rb), 'gain_bits': round(zs[rb][1], 4), 'z': round(zs[rb][0], 2),
                                             'p_max': round((1 + sum(x >= zs[rb][0] for x in nm_)) / (1 + len(nm_)), 4),
                                             'null_max_z_95': round(sorted(nm_)[int(.95 * len(nm_))], 2)}
                worst = min(zs, key=lambda r: zs[r][0])
                S['least regular rule by %s' % nm] = {'rule': str(worst), 'z': round(zs[worst][0], 2)}
            out[(name, plant)] = S; json.dump(S, open(ck, 'w'), indent=1)
            print(name, plant, json.dumps(S), flush=True)
