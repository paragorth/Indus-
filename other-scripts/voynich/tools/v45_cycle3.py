"""v45 cycle 3: extract the residual stream as a new text and run the language battery on it.

Streams (reading order, per corpus):
  SURF  every word (reference)
  SURP  only the words the rule model did not expect: (surprisal - model entropy) in the corpus's top 30%
  PIT   the probability-integral transform of each choice among the allowed alternatives, 8 bins (+ 1 for
        words outside the allowed set): the 'which alternative' stream, as an arithmetic decoder would read it
Battery: Zipf slope, TTR and hapax share at fixed length; burstiness (repeat of a type seen in the previous
100 stream tokens, / the same after shuffling the stream within section); lag-1/2/4 mutual information
excess over a within-page shuffle; arrow of time (held-out bigram cross-entropy forward minus backward);
for PIT: held-out order-2 prediction gain over unigram.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v45_lib import *

FN = 'v45_cycle3.txt'
NTOK = 6000


def streams(name):
    C = get_corpus(name); R = residual(name)
    seq = []
    for p in C:
        if p['id'] not in R: continue
        for r in R[p['id']]['recs']:
            seq.append((p['id'], p['sec'], r[0], -r[5] - r[6], r[8]))
    ex = np.array([x[3] for x in seq]); thr = np.quantile(ex, 0.7)
    surf = [(a, b, w) for a, b, w, e, u in seq]
    surp = [(a, b, w) for a, b, w, e, u in seq if e >= thr]
    pit = [(a, b, (8 if u < 0 else min(7, int(u * 8)))) for a, b, w, e, u in seq]
    return dict(SURF=surf, SURP=surp, PIT=pit)


def zipf(tok):
    c = np.array(sorted(Counter(tok).values(), reverse=True), float)[:300]
    r = np.arange(1, len(c) + 1)
    return float(np.polyfit(np.log(r), np.log(c), 1)[0])


def mi(a, b):
    n = len(a); ca = Counter(a); cb = Counter(b); cab = Counter(zip(a, b))
    return sum(v / n * math.log2(v * n / (ca[x] * cb[y])) for (x, y), v in cab.items())


def shuffle_within(st, key, rng):
    out = list(st); groups = defaultdict(list)
    for i, x in enumerate(st): groups[x[key]].append(i)
    for ix in groups.values():
        vals = [st[i] for i in ix]; rng.shuffle(vals)
        for i, v in zip(ix, vals): out[i] = v
    return out


def burst(tok, W=100):
    hit = 0; last = {}
    for i, w in enumerate(tok):
        if w in last and i - last[w] <= W: hit += 1
        last[w] = i
    return hit / len(tok)


def lagmi(tok, pages, L):
    a = [tok[i] for i in range(len(tok) - L) if pages[i] == pages[i + L]]
    b = [tok[i + L] for i in range(len(tok) - L) if pages[i] == pages[i + L]]
    return mi(a, b)


def wb_xent(train, test):
    u = Counter(); bg = defaultdict(Counter)
    for s in train:
        for x, y in zip(['<s>'] + s, s): u[y] += 1; bg[x][y] += 1
    N = sum(u.values()); V = len(u) + 1; lp = 0; n = 0
    for s in test:
        for x, y in zip(['<s>'] + s, s):
            pu = (u[y] + 1) / (N + V)
            c = bg.get(x)
            if c:
                t = len(c); m = sum(c.values()); p = (c[y] + t * pu) / (m + t)
            else: p = pu
            lp -= math.log2(p); n += 1
    return lp / n


def arrow(st, rng):
    pages = defaultdict(list)
    for a, b, w in st: pages[a].append(w)
    cnt = Counter(w for a, b, w in st)
    P = [[w if cnt[w] >= 2 else '<unk>' for w in v] for v in pages.values()]
    ix = list(range(len(P))); rng.shuffle(ix)
    tr = [P[i] for i in ix[:len(ix) // 2]]; te = [P[i] for i in ix[len(ix) // 2:]]
    f = wb_xent(tr, te); b = wb_xent([s[::-1] for s in tr], [s[::-1] for s in te])
    u = Counter(w for s in tr for w in s); N = sum(u.values())
    uni = -np.mean([math.log2((u[w] + 1) / (N + len(u) + 1)) for s in te for w in s])
    return f, b, float(uni)


def battery(st, rng, sym=False):
    st = st[:NTOK] if not sym else st
    tok = [x[2] for x in st]; pages = [x[0] for x in st]
    out = {}
    if not sym:
        out['zipf'] = zipf(tok); out['ttr'] = len(set(tok)) / len(tok)
        c = Counter(tok); out['hapax'] = sum(1 for v in c.values() if v == 1) / len(c)
        b0 = burst(tok); bs = np.mean([burst([x[2] for x in shuffle_within(st, 1, rng)]) for _ in range(5)])
        out['burst'] = b0 / bs
    for L in (1, 2, 4):
        m0 = lagmi(tok, pages, L)
        ms = np.mean([lagmi([x[2] for x in shuffle_within(st, 0, rng)], pages, L) for _ in range(5)])
        out[f'mi{L}'] = m0 - ms
    f, b, uni = arrow(st, rng)
    out['arrow'] = f - b; out['gain2'] = uni - f
    return out


if __name__ == '__main__':
    rng = random.Random(453)
    names = sys.argv[1:] or ['V', 'VI', 'GEN0', 'GEN1', 'PL', 'LAw', 'LAl', 'GEw', 'GEl']
    res = jload('c3_' + '_'.join(sys.argv[1:]) + '.json') or {}
    for nm in names:
        if not os.path.exists(os.path.join(CK, f'resid_{nm}.json')): print('missing', nm); continue
        S = streams(nm)
        res[nm] = {k: battery(v, rng, sym=(k == 'PIT')) for k, v in S.items()}
        jsave('c3_' + '_'.join(sys.argv[1:]) + '.json', res)
        for k, v in res[nm].items(): print(nm, k, ' '.join('%s %+.4f' % kv for kv in v.items()), flush=True)
