"""Novel test N2: line bracketing. Does the start of a line predict its end (first glyph of first word vs last glyph
of last word), like an opener/closer pair or a checksum? Also: line length (words) vs first glyph.
Null: re-pair line starts and line ends at random within the same page (500x). Paragraph-first lines excluded.
Controls: prose and verse lines."""
import sys, os, math, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

def mi(P):
    n = len(P); a = Counter(x for x, _ in P); b = Counter(y for _, y in P); ab = Counter(P)
    return sum(v/n*math.log2(v*n/(a[x]*b[y])) for (x, y), v in ab.items())

def test(groups, reps=500, seed=1):
    rng = random.Random(seed)
    P = [p for g in groups for p in g]; o = mi(P); s = []
    for _ in range(reps):
        Q = []
        for g in groups:
            ends = [y for _, y in g]; rng.shuffle(ends); Q += list(zip([x for x, _ in g], ends))
        s.append(mi(Q))
    m = sum(s)/len(s); return round(o - m, 4), sum(x >= o for x in s)/reps, len(P)

vl = vlib.load_voynich('ZL3b', drop_uncertain=True)
byp = defaultdict(list); byp_len = defaultdict(list)
for L in vl:
    if L.get('para_start') or len(L['words']) < 3: continue
    g1 = vlib.glyphs(L['words'][0]); g2 = vlib.glyphs(L['words'][-1])
    byp[L.get('page') or L.get('folio')].append((g1[0], g2[-1]))
    byp_len[L.get('page') or L.get('folio')].append((g1[0], min(len(L['words']), 12)))
print('Voynich first-glyph vs last-glyph of line:', test(list(byp.values())))
print('Voynich first-glyph vs line length:', test(list(byp_len.values())))
for key in ('Latin-Caesar', 'Italian-Manzoni', 'German-Kafka', 'Italian-Dante'):
    lines = [L for L in vlib.load_ref(key) if not L.get('para_start') and len([w for w in L['words'] if w]) >= 3]
    groups = [[(l['words'][0][0], [w for w in l['words'] if w][-1][-1]) for l in lines[i:i+20] if l['words'][0]] for i in range(0, len(lines), 20)]
    print(key, 'first vs last:', test([g for g in groups if g], reps=200))
