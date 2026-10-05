"""v61 cycle 2b: random and greedy rule SYSTEMS, scored by whether undoing them makes edge words and
medial words of the same stem agree (the sandhi criterion), with coupling and vocabulary as side scores.
Selection on train pages; survivors re-scored on held-out pages. Same pipeline on controls and nulls."""
import sys, json, math, random
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import v61_lib as L
from v61_c1 import build


def prep(lines):
    T = L.tokens(lines)
    words = [t['w'] for t in T]
    nxt = [i + 1 if t['nxt'] is not None else -1 for i, t in enumerate(T)]
    edge = [t['fin'] is not None for t in T]
    return T, words, nxt, edge


def rule_hits(words, nxt, lex, S, B, ctx):
    h = {}
    for i, w in enumerate(words):
        j = nxt[i]
        if j < 0:
            continue
        if (w.endswith(S) if S else True) and len(w) > len(S) and words[j][0] in ctx:
            c = w[:len(w) - len(S)] + B
            if c in lex:
                h[i] = c
    return h


def edge_agreement(words, edge, idx, k=2, min_n=5):
    """Pooled JSD between ending distributions (last k glyphs) of edge vs medial tokens of the same stem."""
    E = defaultdict(Counter); M = defaultdict(Counter)
    for i in idx:
        w = words[i]
        if len(w) <= k:
            st, en = '', w
        else:
            st, en = w[:-k], w[-k:]
        (E if edge[i] else M)[st][en] += 1
    tot, wsum = 0.0, 0
    for st in E:
        ne, nm = sum(E[st].values()), sum(M[st].values())
        if ne < min_n or nm < min_n:
            continue
        keys = set(E[st]) | set(M[st]); js = 0.0
        for x in keys:
            p, q = E[st][x] / ne, M[st][x] / nm; m = (p + q) / 2
            if p: js += 0.5 * p * math.log2(p / m)
            if q: js += 0.5 * q * math.log2(q / m)
        w = ne; tot += w * js; wsum += w
    return tot / max(1, wsum)


def coupling(words, nxt, idx):
    c = Counter()
    for i in idx:
        j = nxt[i]
        if j >= 0:
            c[(words[i][-1], words[j][0])] += 1
    n = sum(c.values()); a = Counter(); b = Counter()
    for (x, y), v in c.items(): a[x] += v; b[y] += v
    return sum(v / n * math.log2(v * n / (a[x] * b[y])) for (x, y), v in c.items())


def metrics(words, nxt, edge, idx):
    return {'A': edge_agreement(words, edge, idx), 'MI': coupling(words, nxt, idx),
            'types': len({words[i] for i in idx})}


def apply_sys(words, hits_list):
    w2 = list(words)
    done = set()
    for h in hits_list:
        for i, c in h.items():
            if i not in done:
                w2[i] = c; done.add(i)
    return w2, len(done)


class Inc:
    """Incremental metrics over a token subset; only medial tokens are rewritten."""
    def __init__(self, words, nxt, edge, idx, k=2):
        self.words, self.nxt, self.edge, self.k = words, nxt, edge, k
        self.idx = set(idx)
        self.E = defaultdict(Counter); self.M = defaultdict(Counter)
        self.P = Counter(); self.W = Counter(words[i] for i in idx)
        for i in idx:
            st, en = self.se(words[i])
            (self.E if edge[i] else self.M)[st][en] += 1
            if nxt[i] >= 0:
                self.P[(words[i][-1], words[nxt[i]][0])] += 1
        self.stems = [st for st in self.E if sum(self.E[st].values()) >= 5]

    def se(self, w):
        k = self.k
        return ('', w) if len(w) <= k else (w[:-k], w[-k:])

    def score(self, changes):
        dM = defaultdict(Counter); dP = Counter(); dW = Counter()
        for i, c in changes.items():
            if i not in self.idx or self.nxt[i] < 0:
                continue
            o = self.words[i]
            a, b = self.se(o); dM[a][b] -= 1
            a, b = self.se(c); dM[a][b] += 1
            nx = self.words[self.nxt[i]][0]
            dP[(o[-1], nx)] -= 1; dP[(c[-1], nx)] += 1
            dW[o] -= 1; dW[c] += 1
        tot = wsum = 0.0
        for st in self.stems:
            Ec = self.E[st]; Mc = self.M[st]
            if st in dM:
                Mc = {x: Mc.get(x, 0) + dM[st].get(x, 0) for x in set(Mc) | set(dM[st])}
            ne = sum(Ec.values()); nm = sum(v for v in Mc.values() if v > 0)
            if nm < 5:
                continue
            js = 0.0
            for x in set(Ec) | set(Mc):
                p = Ec.get(x, 0) / ne; q = max(Mc.get(x, 0), 0) / nm; m = (p + q) / 2
                if p: js += 0.5 * p * math.log2(p / m)
                if q: js += 0.5 * q * math.log2(q / m)
            tot += ne * js; wsum += ne
        A = tot / max(1, wsum)
        P = Counter(self.P); P.update(dP)
        P = {k: v for k, v in P.items() if v > 0}
        n = sum(P.values()); a = Counter(); b = Counter()
        for (x, y), v in P.items(): a[x] += v; b[y] += v
        MI = sum(v / n * math.log2(v * n / (a[x] * b[y])) for (x, y), v in P.items())
        Wc = Counter(self.W); Wc.update(dW)
        return {'A': A, 'MI': MI, 'types': sum(1 for v in Wc.values() if v > 0)}


def run(arg, n_random=3000, seed=11):
    name, d = arg
    lines = build(name)
    if d == 'P':
        lines = L.reverse_text(lines)
    recs = L.jload('c1_%s_%s.json' % (name.replace(':', '_'), d))['recs']
    train = L.page_split(lines, 1)
    T, words, nxt, edge = prep(lines)
    tr = [i for i, t in enumerate(T) if t['page'] in train]; te = [i for i, t in enumerate(T) if t['page'] not in train]
    lex = set(words)
    pool = [r for r in recs if r['z_train'] > 4][:300]
    hits = [rule_hits(words, nxt, lex, r['S'], r['B'], set(r['cls'])) for r in pool]
    Itr = Inc(words, nxt, edge, tr); Ite = Inc(words, nxt, edge, te)
    base_tr = Itr.score({}); base_te = Ite.score({})
    rng = random.Random(seed)
    res = []
    def merged(hl):
        ch = {}
        for h in hl:
            for i, c in h.items():
                if i not in ch:
                    ch[i] = c
        return ch
    def score(sel):
        ch = merged([hits[k] for k in sel])
        return Itr.score(ch), len(ch)
    for it in range(n_random):
        k = rng.randint(2, 25)
        sel = rng.sample(range(len(pool)), min(k, len(pool)))
        m, n = score(sel)
        res.append((m['A'] - base_tr['A'], sel, m, n))
    res.sort(key=lambda x: x[0])
    cur, curA = [], base_tr['A']
    for step in range(30):
        best = None
        for k in range(len(pool)):
            if k in cur:
                continue
            m, n = score(cur + [k])
            if best is None or m['A'] < best[0]:
                best = (m['A'], k)
        if best is None or best[0] > curA - 1e-4:
            break
        cur.append(best[1]); curA = best[0]
    def held(sel):
        ch = merged([hits[k] for k in sel]); n = len(ch)
        m = Ite.score(ch)
        return {'dA_rel': (m['A'] - base_te['A']) / max(1e-9, base_te['A']), 'dMI_rel': (m['MI'] - base_te['MI']) / base_te['MI'],
                'dTypes_rel': (m['types'] - base_te['types']) / base_te['types'], 'rewrites': n}
    top = [held(x[1]) for x in res[:20]]
    # rule-shuffled control: same top systems but each rule's context class replaced by a random class of equal size
    glyphs = sorted({w[0] for w in words})
    shuf = []
    for x in res[:20]:
        hl = []
        for k in x[1]:
            r = pool[k]; ctx = set(rng.sample(glyphs, min(len(glyphs), len(set(r['cls'])))))
            hl.append(rule_hits(words, nxt, lex, r['S'], r['B'], ctx))
        m = Ite.score(merged(hl))
        shuf.append((m['A'] - base_te['A']) / max(1e-9, base_te['A']))
    out = {'name': name, 'dir': d, 'base_train': base_tr, 'base_test': base_te, 'pool': len(pool), 'n_random': n_random,
           'best_train_dA_rel': [x[0] / max(1e-9, base_tr['A']) for x in res[:5]],
           'random_median_train_dA_rel': float(np.median([x[0] for x in res])) / max(1e-9, base_tr['A']),
           'top20_test_dA_rel_median': float(np.median([t['dA_rel'] for t in top])),
           'top20_test_dMI_rel_median': float(np.median([t['dMI_rel'] for t in top])),
           'top20_test_dTypes_rel_median': float(np.median([t['dTypes_rel'] for t in top])),
           'top20_ctxshuffled_test_dA_rel_median': float(np.median(shuf)),
           'greedy_rules': [(pool[k]['S'], pool[k]['B'], pool[k]['cls']) for k in cur], 'greedy_test': held(cur) if cur else None}
    if lines[0].get('base'):
        bw = [t['base'] for t in T]
        mg = Inc(bw, nxt, edge, te).score({})
        out['gold_test'] = {'dA_rel': (mg['A'] - base_te['A']) / max(1e-9, base_te['A']), 'dMI_rel': (mg['MI'] - base_te['MI']) / base_te['MI'],
                            'dTypes_rel': (mg['types'] - base_te['types']) / base_te['types']}
    L.jsave('c2b_%s_%s.json' % (name.replace(':', '_'), d), out)
    return out


if __name__ == '__main__':
    jobs = [('Sanskrit', 'R'), ('Italian', 'R'), ('Welsh', 'P'), ('VMS-planted', 'R'), ('VMS-ZL', 'R'), ('VMS-ZL', 'P'),
            ('VMS-IT', 'R'), ('VMS-IT', 'P'), ('null-pairblind:VMS-ZL', 'R'), ('null-pairblind:VMS-ZL', 'P'),
            ('null-pair:VMS-ZL', 'P'), ('null-pair:Welsh', 'P')]
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    with Pool(2) as p:
        for r in p.imap_unordered(run, jobs):
            print(json.dumps(r), flush=True)
