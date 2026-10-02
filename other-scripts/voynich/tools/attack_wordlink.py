"""Attack on chink 6 (word-to-word dependency). Where does the Voynich word-to-word information come from?
For adjacent word pairs inside lines: (1) MI between the LAST glyph of word i and the FIRST glyph of word i+1;
(2) MI between whole words; both as excess over a within-line word shuffle (20x). Run with doubled words removed and
with line edges removed. Controls: Latin, Italian, German prose (same tokenisation), Dante verse, and the
Markov-3 / self-citation generators. In languages the junction (last->first letter) carries little information;
a mechanical generator that writes words 'from' their neighbours carries a lot."""
import sys, os, math, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, gen

def mi(pairs):
    n = len(pairs); a = Counter(x for x, _ in pairs); b = Counter(y for _, y in pairs); ab = Counter(pairs)
    return sum(v/n*math.log2(v*n/(a[x]*b[y])) for (x, y), v in ab.items())

def pairs_of(lines, f, drop_dup=False, inner=False):
    P = []
    for L in lines:
        w = [x for x in L['words'] if x]
        idx = range(1, len(w)-2) if inner else range(len(w)-1)
        for i in idx:
            if drop_dup and w[i] == w[i+1]: continue
            P.append((f(w[i], 'last'), f(w[i+1], 'first')))
    return P

def junction(w, side): return w[-1] if side == 'last' else w[0]
def whole(w, side): return w

def excess(lines, f, **kw):
    obs = mi(pairs_of(lines, f, **kw)); sims = []
    for s in range(20):
        rng = random.Random(s)
        sh = [{'words': rng.sample(L['words'], len(L['words']))} for L in lines]
        sims.append(mi(pairs_of(sh, f, **kw)))
    return round(obs - sum(sims)/len(sims), 4)

def cap(lines, n=20000):
    out, k = [], 0
    for L in lines:
        out.append(L); k += len(L['words'])
        if k >= n: break
    return out

vl = vlib.load_voynich('ZL3b', drop_uncertain=True)
V = cap([{'words': [''.join(vlib.glyphs(w)) for w in L['words']]} for L in vl])
texts = {'Voynich': V}
for key in ('Latin-Caesar', 'Italian-Manzoni', 'German-Kafka', 'Italian-Dante'):
    texts[key] = cap([{'words': [w for w in L['words'] if w]} for L in vlib.load_ref(key)])
texts['Markov-3 from Voynich'] = cap(gen.char_markov(V, order=3, seed=3))
texts['self-citation generator'] = cap(gen.self_citation(V, seed=3))
print(f"{'text':28s} {'junction MI':>12s} {'no doubles':>11s} {'no line edges':>14s} {'word MI':>9s} {'word MI no doubles':>19s}")
res = {}
for k, L in texts.items():
    r = [excess(L, junction), excess(L, junction, drop_dup=True), excess(L, junction, inner=True), excess(L, whole), excess(L, whole, drop_dup=True)]
    res[k] = r
    print(f"{k:28s} {r[0]:12.4f} {r[1]:11.4f} {r[2]:14.4f} {r[3]:9.4f} {r[4]:19.4f}")
# top junction pairs in Voynich vs shuffle
P = pairs_of(V, junction); c = Counter(P); a = Counter(x for x, _ in P); b = Counter(y for _, y in P); n = len(P)
top = sorted(((v/(a[x]*b[y]/n), x, y, v) for (x, y), v in c.items() if v >= 40), reverse=True)
print('strongest Voynich junctions (obs/expected, last->first, count):', [(round(r,2), x+'->'+y, v) for r, x, y, v in top[:8]])
print('weakest:', [(round(r,2), x+'->'+y, v) for r, x, y, v in top[-6:]])
vlib.save('attack_wordlink', res)
