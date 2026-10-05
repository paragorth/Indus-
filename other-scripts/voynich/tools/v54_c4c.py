"""v54 cycle 4c: the third word as a witness.  When a word pair (a, b) recurs in a section, is the next word c the
SAME word as last time (exact), a one-edit VARIANT of it, or unrelated?  Observed counts vs 200 successor shuffles
(successors of b permuted within section).  Copying with spelling variation predicts variants >> chance;
a pure word-pair generator predicts both ~1."""
import sys, os, json, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v54_lib as V
from v54_c4b import triples, shuffle_succ, plain_pages

def lev1(x, y):
    if x == y: return False
    lx, ly = len(x), len(y)
    if abs(lx - ly) > 1: return False
    if lx == ly: return sum(a != b for a, b in zip(x, y)) == 1
    if lx > ly: x, y = y, x
    i = 0
    while i < len(x) and x[i] == y[i]: i += 1
    return x[i:] == y[i + 1:]

def counts(T):
    g = collections.defaultdict(list)
    for s, pi, a, b, c in T: g[(s, a, b)].append(c)
    ex = var = 0
    for v in g.values():
        if len(v) < 2: continue
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if v[i] == v[j]: ex += 1
                elif lev1(v[i], v[j]): var += 1
    return ex, var

def run(pages, nperm=200, seed=2):
    T = triples(pages); rng = random.Random(seed)
    ex, var = counts(T)
    N = [counts(shuffle_succ(T, rng)) for _ in range(nperm)]
    def zz(o, k):
        xs = [n[k] for n in N]; m = sum(xs) / len(xs); sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** .5
        return round(o / m, 3) if m else None, round((o - m) / (sd or 1), 1), round(m, 1)
    r1, z1, m1 = zz(ex, 0); r2, z2, m2 = zz(var, 1)
    return dict(exact=ex, exact_null=m1, exact_ratio=r1, exact_z=z1, variant=var, variant_null=m2, variant_ratio=r2, variant_z=z2)

if __name__ == '__main__':
    from v54_c4 import null_ngram
    C = json.load(open(os.path.join(V.ROOT, 'data', 'v30_ckpt', 'corpora.json')))['corpora']
    def cat_pages(key):
        P = C[key]
        return [dict(id=str(i), vars=dict(sec='s%d' % (6 * i // len(P))), lines=[ws[j:j + 9] for j in range(0, len(ws), 9)]) for i, ws in enumerate(P)]
    zl = V.voynich('ZL3b')
    corp = [('ZL', zl), ('IT', V.voynich('IT2a')), ('ZL_selfcit', V.null_selfcit(zl, 3)), ('ZL_markov', V.null_markov(zl, 3)),
            ('BRU_clean', V.brumati(plant=False)[0]), ('BRU_planted', V.brumati()[0]), ('GER_real', V.german()),
            ('LAT_catmus', cat_pages('I_Lat')), ('ITA_catmus', cat_pages('I_Ita'))]
    for k in ['la', 'it', 'de']:
        f = os.path.join(V.ROOT, 'data', 'plain', k + '.txt')
        if os.path.exists(f): corp.append(('plain_' + k, plain_pages(f)))
    out = {}
    for nm, P in corp:
        r = run(P); out[nm] = r; print(nm, json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(V.CK, 'c4c.json'), 'w'))
