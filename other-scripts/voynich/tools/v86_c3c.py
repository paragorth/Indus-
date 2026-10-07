"""v86 cycle 3c: (1) bootstrap CI for the context test (does a line-end stem+m behave like its page preimage?);
(2) how much of the line-end vocabulary does 'm = final form of r/l' explain: Jensen-Shannon separation of line-end vs
mid-line word distributions before/after restoring m -> r/l (page majority), vs restoring to a random final glyph;
(3) information lost: entropy of the r/l/n choice among preimages."""
import random, math
from collections import Counter, defaultdict
import v86_lib as L
import v86_c3b as B


def ctx_items(pages):
    prev_of = defaultdict(Counter); allprev = Counter()
    for f, ls in pages.items():
        for ln in ls:
            u = ln['units']
            for j in range(1, len(u)):
                if u[j] and u[j - 1]:
                    prev_of[u[j]][u[j - 1][:2]] += 1; allprev[u[j - 1][:2]] += 1
    tot = sum(allprev.values())
    def lp(word, ctx):
        c = prev_of[word]; n = sum(c.values())
        return math.log((c[ctx] + 5 * allprev[ctx] / tot) / (n + 5)) - math.log(allprev[ctx] / tot)
    rng = random.Random(1); items = []
    for f, ls in pages.items():
        pc = Counter(w for ln in ls for w in ln['units'] if w)
        for ln in ls:
            u = ln['units']
            if len(u) < 2 or not u[-1] or not u[-2] or not u[-1].endswith('m') or len(u[-1]) < 2:
                continue
            s = u[-1]; ctx = u[-2][:2]
            pre = [v for v in pc if len(v) == len(s) and v[:-1] == s[:-1] and v[-1] != 'm']
            cand = [v for v in pc if len(v) == len(s) and v[:-1] != s[:-1] and not v.endswith('m') and v[:2] == s[:2]]
            if not pre or not cand:
                continue
            items.append(sum(lp(v, ctx) for v in pre) / len(pre) - sum(lp(rng.choice(cand), ctx) for _ in range(20)) / 20)
    m = sum(items) / len(items)
    bs = sorted(sum(rng.choice(items) for _ in items) / len(items) for _ in range(1000))
    return m, bs[25], bs[975], len(items)


def js(p, q):
    keys = set(p) | set(q); sp = sum(p.values()); sq = sum(q.values()); d = 0.0
    for k in keys:
        a = p[k] / sp; b = q[k] / sq; mm = (a + b) / 2
        if a: d += 0.5 * a * math.log2(a / mm)
        if b: d += 0.5 * b * math.log2(b / mm)
    return d


def restore(pages, mode, rng):
    out = {}; changed = 0
    for f, ls in pages.items():
        pc = Counter(w for ln in ls for w in ln['units'] if w)
        nl = []
        for ln in ls:
            u = list(ln['units'])
            j = len(u) - 1
            if ln['cls'][j] == 'E' and u[j] and u[j].endswith('m') and len(u[j]) >= 2:
                s = u[j]
                pre = Counter({v: pc[v] for v in pc if len(v) == len(s) and v[:-1] == s[:-1] and v[-1] in 'rln'})
                if pre:
                    if mode == 'major':
                        u[j] = pre.most_common(1)[0][0]
                    else:
                        u[j] = s[:-1] + rng.choice('rlnyos')
                    changed += 1
            nl.append(dict(ln, units=u))
        out[f] = nl
    return out, changed


def sep(pages):
    E = Counter(); M = Counter()
    for ls in pages.values():
        for ln in ls:
            for w, c in zip(ln['units'], ln['cls']):
                if not w: continue
                if c == 'E': E[w] += 1
                elif c == 'M': M[w] += 1
    return js(E, M), sum(1 for ls in pages.values() for ln in ls for w, c in zip(ln['units'], ln['cls']) if c == 'E' and w and w.endswith('m')), sum(E.values())


if __name__ == '__main__':
    rng = random.Random(5)
    for tr in ('ZL3b', 'IT2a'):
        VP, _ = L.voynich_pages(tr)
        m, lo, hi, n = ctx_items(VP)
        print(tr, 'context LLR preimage - same-onset random: %.3f [%.3f, %.3f] n=%d' % (m, lo, hi, n), flush=True)
        j0, nm, nE = sep(VP)
        R1, c1 = restore(VP, 'major', rng); R2, c2 = restore(VP, 'rand', rng)
        print(tr, 'line-end m-final tokens %d of %d (%.1f%%); JS(E,M) raw %.4f, after m->page r/l/n %.4f (changed %d), after m->random glyph %.4f'
              % (nm, nE, 100 * nm / nE, j0, sep(R1)[0], c1, sep(R2)[0]), flush=True)
        # entropy of preimage final among recoverable stems
        H = []; 
        for f, ls in VP.items():
            pc = Counter(w for ln in ls for w in ln['units'] if w)
            for ln in ls:
                s = ln['units'][-1]
                if ln['cls'][-1] != 'E' or not s or not s.endswith('m') or len(s) < 2: continue
                pre = Counter({v[-1]: pc[v] for v in pc if len(v) == len(s) and v[:-1] == s[:-1] and v[-1] in 'rln'})
                if len(pre) >= 1:
                    t = sum(pre.values()); H.append(-sum(c / t * math.log2(c / t) for c in pre.values()))
        print(tr, 'r/l/n choice entropy among recoverable line-end m tokens: mean %.2f bits, ambiguous share %.2f (n=%d)'
              % (sum(H) / len(H), sum(1 for h in H if h > 0) / len(H), len(H)), flush=True)
    lat = L.layout_like(VP, L.ref_words('Latin-Caesar', sum(len(ln['units']) for ls in VP.values() for ln in ls)))
    latp = L.plant(lat, lambda w, r: (w[:-1] + 'm') if len(w) > 2 and w[-1] in 'rls' else w, classes=('E',), p=0.8, seed=4)
    m, lo, hi, n = ctx_items(latp)
    print('Latin plant (final r/l/s -> m at line end) context: %.3f [%.3f, %.3f] n=%d' % (m, lo, hi, n), flush=True)
    vp = L.plant(L.shuffle_positions(VP, random.Random(9)), lambda w, r: (w[:-1] + 'm') if len(w) > 2 and w[-1] in 'rl' else w, classes=('E',), p=0.8, seed=4)
    m, lo, hi, n = ctx_items(vp)
    print('Voynich pos-shuffled + context-free r/l->m plant: %.3f [%.3f, %.3f] n=%d' % (m, lo, hi, n), flush=True)
