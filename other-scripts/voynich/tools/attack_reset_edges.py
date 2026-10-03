"""Is the line-break reset of junction coupling (V3) just edge vocabulary? Compare junction MI for:
(a) cross-line pairs (last word of line -> first word of next line, same paragraph);
(b) within-line pairs that touch an edge: (first word -> second word) and (second-last -> last word);
(c) inner within-line pairs (no edge word);
(d) cross-line pairs restricted to 'ordinary' words: last and next-first word both NOT among the 60 most
    edge-enriched words (line-final and line-initial vocab).
All at equal n, excess over 200 pair-shuffles. Both transcriptions."""
import sys, os, math, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

def mi(P):
    n = len(P); a = Counter(x for x, _ in P); b = Counter(y for _, y in P); ab = Counter(P)
    return sum(v/n*math.log2(v*n/(a[x]*b[y])) for (x, y), v in ab.items())

def ex(P, rng, n):
    P = rng.sample(P, min(n, len(P))); o = mi(P); s = []
    for _ in range(200):
        bb = [y for _, y in P]; rng.shuffle(bb); s.append(mi(list(zip([x for x, _ in P], bb))))
    m = sum(s)/len(s); return round(o-m, 4), sum(x >= o for x in s)/200, len(P)

for name in ('ZL3b', 'IT2a'):
    vl = vlib.load_voynich(name, drop_uncertain=True)
    L = [[''.join(vlib.glyphs(w)) for w in r['words']] for r in vl]
    allw = Counter(w for l in L for w in l); first = Counter(l[0] for l in L if l); last = Counter(l[-1] for l in L if l)
    N = sum(allw.values()); nl = len(L)
    enr = lambda c: sorted(((c[w]/nl)/(allw[w]/N), w) for w in c if allw[w] >= 10)
    edge = {w for _, w in enr(first)[-60:]} | {w for _, w in enr(last)[-60:]}
    J = lambda a, b: (a[-1], b[0])
    cross = [J(L[i][-1], L[i+1][0]) for i in range(len(L)-1) if L[i] and L[i+1] and not vl[i+1].get('para_start')]
    cross_ord = [J(L[i][-1], L[i+1][0]) for i in range(len(L)-1) if L[i] and L[i+1] and not vl[i+1].get('para_start')
                 and L[i][-1] not in edge and L[i+1][0] not in edge]
    edgein = [J(l[0], l[1]) for l in L if len(l) >= 3] + [J(l[-2], l[-1]) for l in L if len(l) >= 3]
    inner = [J(l[i], l[i+1]) for l in L for i in range(1, len(l)-2)]
    n = min(len(cross_ord), 1500); rng = random.Random(1)
    print(f'{name} (n={n} each): cross-line {ex(cross, rng, n)} | cross-line ordinary words {ex(cross_ord, rng, n)} | '
          f'within, edge-touching {ex(edgein, rng, n)} | within, inner {ex(inner, rng, n)}')
