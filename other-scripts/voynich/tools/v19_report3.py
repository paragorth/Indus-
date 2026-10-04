"""v19 cycle 3 report: location scan (windows anywhere, page breaks ignored), family-wise null from scans on shuffled corpora,
planted glossary rank and order recovery; GA search-adequacy check on the top Voynich windows."""
import json, os, glob, random, sys
import numpy as np
from v19_lib import *
from v19_cycle3 import windows, planted_glossary, shuffled_within_page

D = os.path.join(RES, 'c3')


def load(c, kind, key):
    f = os.path.join(D, '%s_%s_%s.json' % (c, kind, key))
    return json.load(open(f)) if os.path.exists(f) else None


def ga(ent, syms, depth, rev, pop=200, gens=300, seed=0):
    """genetic search over glyph orders: order crossover + swap mutation, tau as fitness (python, numpy)."""
    rng = random.Random(seed)
    k = len(syms); idx = {g: i for i, g in enumerate(syms)}
    # pairwise decision matrix W[a][b]
    W = np.zeros((k, k))
    keys = []
    for _, _, w in ent:
        w2 = (w[::-1] if rev else w)
        if depth:
            w2 = w2[:depth]
        keys.append([idx[g] for g in w2] + [idx['#']])
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            for t in range(min(len(a), len(b))):
                if a[t] != b[t]:
                    W[a[t], b[t]] += 1; break
    T = W.sum()

    def fit(o):
        pos = np.empty(k, int); pos[o] = np.arange(k)
        return (2 * W[pos[:, None] < pos[None, :]].sum() - T) / T

    P = [rng.sample(range(k), k) for _ in range(pop)]
    F = [fit(np.array(o)) for o in P]
    for g in range(gens):
        new = []
        order = np.argsort(F)[::-1]
        elite = [P[i] for i in order[:pop // 5]]
        new.extend(elite)
        while len(new) < pop:
            p1, p2 = rng.sample(elite, 2)
            a, b = sorted(rng.sample(range(k), 2))
            mid = p1[a:b]; rest = [x for x in p2 if x not in mid]
            child = rest[:a] + mid + rest[a:]
            for _ in range(rng.randint(0, 2)):
                i, j = rng.randrange(k), rng.randrange(k); child[i], child[j] = child[j], child[i]
            new.append(child)
        P = new; F = [fit(np.array(o)) for o in P]
    return max(F)


if __name__ == '__main__':
    out = {}
    for kind in ['LI12', 'LI24', 'AW4']:
        for key in ['F', 'L']:
            print('== %s / %s ==' % (kind, key))
            for c in ['ZL', 'IT', 'NULLH1', 'NULLH2', 'NULLW1', 'NULLW2', 'PLANTG1', 'PLANTG2']:
                d = load(c, kind, key)
                if not d:
                    continue
                rows = d['rows']; zs = np.array([r['z'] for r in rows])
                top = sorted(rows, key=lambda r: -r['z'])[:3]
                s = '%-8s n=%4d  z>4: %3d  z>5: %2d  max %.1f  top: %s' % (c, len(rows), (zs > 4).sum(), (zs > 5).sum(), zs.max(),
                                                                         '; '.join('%s z%.1f tau%.2f' % (r['id'], r['z'], r['tau']) for r in top))
                if c.startswith('PLANTG'):
                    s0 = d['meta']['start']; perm = d['meta']['perm']; rank = {g: i for i, g in enumerate(perm)}
                    # windows overlapping the plant (window ids hold the start index in their own line list)
                    zl = voynich_lines('ZL3b')
                    lines, s1, _ = planted_glossary(zl, int(c[6:]))
                    st = windows(lines, kind)
                    # locate windows containing the planted lines by content match
                    planted_heads = [tuple(lines[s1 + k]['words'][0]) for k in range(15)]
                    best = None
                    for r, (sid, ent) in zip(rows, st):
                        hs = [tuple(w) for _, _, w in ent]
                        ov = sum(1 for h in planted_heads if h in hs)
                        if ov >= 8 and (best is None or r['z'] > best[0]['z']):
                            best = (r, ov)
                    if best:
                        r, ov = best
                        rk = 1 + int((zs > r['z']).sum())
                        rec = order_agreement(r['order_s'].split(), rank)
                        s += '\n           planted window (overlap %d/15): z %.1f, rank %d of %d, order recovery %.2f' % (ov, r['z'], rk, len(rows), rec)
                print(s)
                out['%s_%s_%s' % (c, kind, key)] = dict(n=len(rows), z4=int((zs > 4).sum()), max=float(zs.max()), top=top)
    # GA adequacy on top ZL windows
    print('== GA vs ILS on the top ZL windows ==')
    zl = voynich_lines('ZL3b')
    for kind, key in [('LI12', 'F'), ('LI24', 'L'), ('AW4', 'F')]:
        d = load('ZL', kind, key)
        if not d:
            continue
        st = dict(windows(zl, kind))
        for r in sorted(d['rows'], key=lambda r: -r['z'])[:2]:
            ent = st[r['id']]
            syms = sorted({g for _, _, w in ent for g in w} | {'#'})
            depth, rev = KEYS[key]
            g = ga(ent, syms, depth, rev)
            print('%s/%s %s: ILS tau %.4f  GA tau %.4f' % (kind, key, r['id'], r['tau'], g))
            out.setdefault('ga', []).append((kind, key, r['id'], r['tau'], g))
    json.dump(out, open(os.path.join(RES, 'c3_summary.json'), 'w'), indent=1)
