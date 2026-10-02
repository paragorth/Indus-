"""Novel test N1: line quotas. If each line was filled to a budget (a tally, a checksum, a metre), the count of a
glyph class per line should vary LESS than if the same words were dealt to lines at random.
Null: within each paragraph, deal the paragraph's words back into its lines at random, keeping every line's word
count (200x). Statistic: variance of per-line count / mean null variance (<1 = quota-like, >1 = clumped).
Glyph classes: gallows (k t p f + benched), q, ch/sh, e, i, d, y, l, r, n, m/g, a, o, total glyphs.
Controls: the same on Latin, Italian, German prose and Dante verse (letters: vowels, s, t, n, r, total letters)."""
import sys, os, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

def paras(lines):
    P, cur = [], []
    for L in lines:
        if L.get('para_start') and cur: P.append(cur); cur = []
        cur.append([w for w in L['words'] if w])
    if cur: P.append(cur)
    return [p for p in P if len(p) >= 3]

def count(words, cls): return sum(1 for w in words for c in w if c in cls) if cls != 'ALL' else sum(len(w) for w in words)

def var(xs):
    m = sum(xs)/len(xs); return sum((x-m)**2 for x in xs)/len(xs)

def ratio(P, cls, reps=200, seed=1):
    rng = random.Random(seed)
    obs = [count(l, cls) for p in P for l in p]
    # residual variance after removing the paragraph mean, to compare like with like
    def within(PP):
        vals = []
        for p in PP:
            c = [count(l, cls) for l in p]; m = sum(c)/len(c); vals += [x - m for x in c]
        return sum(v*v for v in vals)/len(vals)
    o = within(P); sims = []
    for _ in range(reps):
        NP = []
        for p in P:
            ws = [w for l in p for w in l]; rng.shuffle(ws); k = 0; np_ = []
            for l in p: np_.append(ws[k:k+len(l)]); k += len(l)
            NP.append(np_)
        sims.append(within(NP))
    sims.sort(); m = sum(sims)/len(sims)
    p_low = sum(s <= o for s in sims)/reps; p_high = sum(s >= o for s in sims)/reps
    return round(o/m, 3), p_low, p_high

vl = vlib.load_voynich('ZL3b', drop_uncertain=True)
VL = [{'words': [''.join(vlib.glyphs(w)) for w in L['words']], 'para_start': L.get('para_start')} for L in vl]
VP = paras(VL)
VC = {'gallows': set('ktpfTKPF'), 'q': {'q'}, 'ch/sh': {'C', 'S'}, 'e': {'e'}, 'i': {'i'}, 'd': {'d'}, 'y': {'y'},
      'l': {'l'}, 'r': {'r'}, 'n': {'n'}, 'm/g': {'m', 'g'}, 'a': {'a'}, 'o': {'o'}, 'all glyphs': 'ALL'}
print('Voynich ZL: paragraphs', len(VP), 'lines', sum(len(p) for p in VP))
for k, c in VC.items(): print(f'  {k:10s} variance ratio {ratio(VP, c)}')
LC = {'vowels': set('aeiouy'), 's': {'s'}, 't': {'t'}, 'n': {'n'}, 'r': {'r'}, 'all letters': 'ALL'}
for key in ('Latin-Caesar', 'Italian-Manzoni', 'German-Kafka', 'Italian-Dante'):
    P = paras(vlib.load_ref(key))[:400]
    print(key, 'paragraphs', len(P))
    for k, c in LC.items(): print(f'  {k:10s} variance ratio {ratio(P, c, reps=100)}')
