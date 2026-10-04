#!/usr/bin/env python3
"""LA-13 cycle 2: dialect alignment by annealed sign maps (no shared words needed).
For each target site group B, a bigram model (types, ^/$ boundaries, add-0.1) is trained on all OTHER groups (the reference).
A sign map s: B-signs -> reference signs is annealed to maximise  sum_w log2 P_ref(s(w)) - LAMBDA * #(non-identity entries).
A sound law / spelling convention X(ref) -> Y(B) shows up as s(Y) = X with a large gain.
Null: the same anneal on pseudo-sites drawn by permuting site labels within support type (same docs per label, same support mix).
Positive controls: rules planted in Khania (LA) and in Pylos (LB, LA-sized subsample); negative: unplanted LB.
Two workers; checkpoints in data/la13/c2_*.json."""
import sys, os, json, random, math, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la13_common import *
from la13_c1 import plant

LAMBDA = 10.0
ALPHA = 0.1

class Model:
    def __init__(self, words, alphabet):
        self.idx = {s: i for i, s in enumerate(alphabet)}
        V = len(alphabet)
        C = collections.defaultdict(collections.Counter)
        for w in words:
            s = ['^'] + list(w) + ['$']
            for a, b in zip(s, s[1:]): C[a][b] += 1
        self.C = C; self.V = V
        self.tot = {a: sum(c.values()) for a, c in C.items()}
        self.cache = {}
    def lp(self, a, b):
        k = (a, b)
        v = self.cache.get(k)
        if v is None:
            v = math.log2((self.C[a][b] + ALPHA) / (self.tot.get(a, 0) + ALPHA * (self.V + 1)))
            self.cache[k] = v
        return v
    def word(self, w):
        s = ['^'] + list(w) + ['$']
        return sum(self.lp(a, b) for a, b in zip(s, s[1:]))

def anneal(ref_words, tgt_words, seed, steps=20000, lam=LAMBDA):
    rnd = random.Random(seed)
    alphabet = sorted({s for w in ref_words for s in w} | {s for w in tgt_words for s in w})
    M = Model(ref_words, alphabet)
    tgt = sorted(set(tgt_words))
    bsc = collections.Counter(s for w in tgt for s in w)
    ysigns = [s for s, c in bsc.items() if c >= 2]
    rsc = collections.Counter(s for w in ref_words for s in w)
    xsigns = [s for s, c in rsc.most_common(90)]
    occ = collections.defaultdict(list)
    for i, w in enumerate(tgt):
        for s in set(w): occ[s].append(i)
    sig = {}
    def mapw(w): return tuple(sig.get(s, s) for s in w)
    cur = [M.word(w) for w in tgt]
    base = sum(cur)
    score = base
    T0, T1 = 8.0, 0.05
    for t in range(steps):
        T = T0 * (T1 / T0) ** (t / steps)
        y = rnd.choice(ysigns)
        old = sig.get(y, y)
        new = y if (old != y and rnd.random() < 0.3) else rnd.choice(xsigns)
        if new == old: continue
        if new == y: sig.pop(y)
        else: sig[y] = new
        d = 0.0; nv = {}
        for i in occ[y]:
            v = M.word(mapw(tgt[i])); nv[i] = v; d += v - cur[i]
        d -= lam * ((new != y) - (old != y))
        if d >= 0 or rnd.random() < math.exp(d / T):
            score += d
            for i, v in nv.items(): cur[i] = v
        else:
            if old == y: sig.pop(y, None)
            else: sig[y] = old
    # marginal gain of each entry
    marg = {}
    for y, x in list(sig.items()):
        sig.pop(y)
        g = sum(cur[i] - M.word(mapw(tgt[i])) for i in occ[y]) - lam
        sig[y] = x
        marg[y] = (x, round(g, 2), bsc[y])
    return dict(gain=round(score - base, 2), n=len(sig), entries=sorted(marg.items(), key=lambda kv: -kv[1][1]))

def split(units, labels, target):
    ref = [w for u, l in zip(units, labels) for w in set(u[3]) if l != target]
    tg = [w for u, l in zip(units, labels) for w in set(u[3]) if l == target]
    return sorted(set(ref)), sorted(set(tg))

def job(args):
    kind, target, seed = args
    rnd = random.Random(seed)
    U = UNITS if kind != 'LB' else LBS
    labs = [u[1] for u in U] if seed < 0 else shuffle_labels(U, rnd, True)
    ref, tg = split(U, labs, target)
    r = anneal(ref, tg, abs(seed) + 7)
    return (kind, target, seed, r)

def ckpt(name, obj): json.dump(obj, open(os.path.join(OUT, f'c2_{name}.json'), 'w'), indent=1)
def load(name):
    p = os.path.join(OUT, f'c2_{name}.json'); return json.load(open(p)) if os.path.exists(p) else None

def lb_sub(seed):
    LBU = lb_units(); r = random.Random(seed); outu = []
    for site, n in {'KN': 439, 'PY': 87, 'TH': 112}.items():
        docs = [u for u in LBU if u[1] == site]; r.shuffle(docs); seen = set()
        for u in docs:
            if len(seen) >= n: break
            outu.append(u); seen.update(u[3])
    return outu

UNITS = la_units()
LBS = lb_sub(100)

def main():
    out = open(os.path.join(OUT, 'c2_report.txt'), 'w')
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()
    targets = ['HT', 'KH', 'ZA', 'PH', 'KN', 'OTH']
    NN = 100
    res = load('main')
    if res is None:
        jobs = [('LA', t, -1) for t in targets] + [('LA', t, s) for t in targets for s in range(NN)]
        jobs += [('LB', 'PY', -1)] + [('LB', 'PY', s) for s in range(NN)]
        with Pool(2) as pool: res = pool.map(job, jobs)
        ckpt('main', res)
    by = collections.defaultdict(list); real = {}
    for kind, t, s, r in res:
        if s < 0: real[(kind, t)] = r
        else: by[(kind, t)].append(r)
    for key, r in real.items():
        ng = [x['gain'] for x in by[key]]; nn = [x['n'] for x in by[key]]
        topnull = sorted(e[1][1] for x in by[key] for e in x['entries'][:1])
        best = r['entries'][0][1][1] if r['entries'] else 0
        nbest = [x['entries'][0][1][1] if x['entries'] else 0 for x in by[key]]
        say(f'{key}: real gain {r["gain"]} entries {r["n"]} | null gain mean {sum(ng)/len(ng):.1f} max {max(ng)} '
            f'P(>=) {sum(1 for g in ng if g >= r["gain"])/len(ng):.3f} | entries null mean {sum(nn)/len(nn):.1f} | '
            f'best-entry gain {best} vs null best mean {sum(nbest)/len(nbest):.1f} P(>=) {sum(1 for g in nbest if g >= best)/len(nbest):.3f}')
        say('    real entries (B sign -> ref sign, marginal bits, B count):', r['entries'][:12])
    # ---- planted controls
    pc = load('planted')
    if pc is None:
        pc = []
        rnd = random.Random(21)
        for kind in ('LA', 'LB'):
            U = UNITS if kind == 'LA' else LBS
            site = 'KH' if kind == 'LA' else 'PY'
            sc = collections.Counter(s for u in U if u[1] == site for w in set(u[3]) for s in w)
            allsc = collections.Counter(s for u in U for w in set(u[3]) for s in w)
            cand = [s for s, c in sc.items() if 4 <= c <= 15 and not s.startswith('*')]
            for rate in (1.0, 0.5):
                for rep in range(3):
                    xs = rnd.sample(cand, 3)
                    ys = rnd.sample([s for s in allsc if s not in xs and not s.startswith('*') and sc.get(s, 0) <= 3], 3)
                    rules = list(zip(xs, ys))
                    U2 = plant(U, site, rules, rate, rnd)
                    labs = [u[1] for u in U2]
                    ref, tg = split(U2, labs, site)
                    r = anneal(ref, tg, rep)
                    got = {y: e for y, e in r['entries']}
                    hit = [(x, y, got.get(y)) for x, y in rules]
                    nh = sum(1 for x, y, e in hit if e and e[0] == x)
                    pc.append(dict(kind=kind, rate=rate, rep=rep, rules=rules, hits=nh, detail=hit, gain=r['gain'], n=r['n'],
                                   top=r['entries'][:5]))
                    say('planted', kind, rate, rep, 'recovered', nh, '/3', hit, 'top', r['entries'][:5])
        ckpt('planted', pc)
    else:
        for p in pc: say('planted', p['kind'], p['rate'], p['rep'], 'recovered', p['hits'], '/3', p['detail'], 'top', p['top'])
    out.close()

if __name__ == '__main__':
    main()
