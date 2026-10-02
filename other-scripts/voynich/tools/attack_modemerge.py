"""N5: are the two line modes (N3) two settings of one device? Split lines into q-type and a-type (top and bottom
third of the N4 score, so the split is not circular on most glyphs). Greedy search: apply glyph rewrites
x->y (single glyph or glyph unit, both directions allowed) to a-type lines, keep the rewrite that most lowers the
Jensen-Shannon divergence between the two word distributions; up to 6 rewrites, fitted on half the pages and
scored on the other half. Baselines: JSD between two random halves of the same mode; and the same greedy search
applied to an Italian (Manzoni) split built the same way (lines rich vs poor in words starting with 'c' minus
words containing 'ar'), to show how much a greedy search can 'merge' by overfitting."""
import sys, os, math, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

def jsd(A, B):
    a = Counter(A); b = Counter(B); na = sum(a.values()); nb = sum(b.values()); s = 0
    for w in set(a) | set(b):
        p = a[w]/na; q = b[w]/nb; m = (p+q)/2
        if p: s += .5*p*math.log2(p/m)
        if q: s += .5*q*math.log2(q/m)
    return s

def rw(words, rules):
    out = []
    for w in words:
        for x, y in rules: w = w.replace(x, y)
        out.append(w)
    return out

def split_modes(lines, score):
    sc = sorted(lines, key=score); k = len(sc)//3
    return sc[:k], sc[-k:]

def run(lines, score, alphabet, label, seed=1):
    rng = random.Random(seed)
    lo, hi = split_modes(lines, score)
    def halves(L):
        L = L[:]; rng.shuffle(L); return L[:len(L)//2], L[len(L)//2:]
    lo1, lo2 = halves(lo); hi1, hi2 = halves(hi)
    W = lambda L: [w for l in L for w in l]
    base_same = (jsd(W(lo1), W(lo2)) + jsd(W(hi1), W(hi2))) / 2
    start_fit = jsd(W(lo1), W(hi1)); start_test = jsd(W(lo2), W(hi2))
    rules = []; cur = W(lo1)
    cands = [(x, y) for x in alphabet for y in alphabet + [''] if x != y]
    for step in range(6):
        best = None
        for x, y in cands:
            v = jsd(rw(cur, [(x, y)]), W(hi1))
            if best is None or v < best[0]: best = (v, x, y)
        if best[0] >= jsd(cur, W(hi1)) - 1e-4: break
        rules.append((best[1], best[2])); cur = rw(cur, [(best[1], best[2])])
    test = jsd(rw(W(lo2), rules), W(hi2))
    print(f'{label}: modes JSD fit {start_fit:.3f} -> {jsd(cur, W(hi1)):.3f}; HELD-OUT {start_test:.3f} -> {test:.3f}; '
          f'same-mode baseline {base_same:.3f}; rules {rules}; '
          f'held-out gap closed {(start_test-test)/(start_test-base_same)*100:.0f}%')

vl = vlib.load_voynich('ZL3b', drop_uncertain=True)
V = [[''.join(vlib.glyphs(w)) for w in L['words']] for L in vl if len(L['words']) >= 4]
vs = lambda ws: (sum(w.startswith('q') for w in ws) - sum(('ai' in w) or ('ar' in w) for w in ws))/len(ws)
alph = sorted({c for l in V for w in l for c in w}) + ['ai', 'ar', 'ed', 'ee', 'ol', 'ok', 'qo', 'dy', 'Ce', 'eo']
run(V, vs, alph, 'Voynich (q-type vs a-type lines)')
for key in ('Italian-Manzoni', 'Latin-Caesar'):
    I = [[w for w in L['words'] if w] for L in vlib.load_ref(key) if len([w for w in L['words'] if w]) >= 4][:4000]
    isc = lambda ws: (sum(w.startswith('c') for w in ws) - sum('ar' in w for w in ws))/len(ws)
    ialph = sorted({c for l in I for w in l for c in w})[:26] + ['ar', 'er', 'on', 'ch', 'qu', 'in', 'an', 'en', 're', 'co']
    run(I, isc, ialph, key + ' (same split recipe)')
