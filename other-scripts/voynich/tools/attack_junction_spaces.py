"""Control for V2/V3: is the junction coupling an artefact of uncertain word spaces (',' in IVTFF), i.e. of where
transcribers cut words? Split junctions into certain ('.') and uncertain (',') spaces; junction MI (last glyph ->
next first glyph) excess over a within-class pairing shuffle (200x), at equal sample size."""
import sys, os, re, math, random
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

def mi(P):
    n = len(P); a = Counter(x for x, _ in P); b = Counter(y for _, y in P); ab = Counter(P)
    return sum(v/n*math.log2(v*n/(a[x]*b[y])) for (x, y), v in ab.items())

def clean(t):
    t = re.sub(r'<[^>]*>', '.', t); t = re.sub(r'\[([^:\]]*):[^\]]*\]', r'\1', t)
    t = re.sub(r'@\d+;', '?', t); t = t.replace('{', '').replace('}', '').replace("'", '')
    return t

def junctions(path):
    J = {'.': [], ',': []}
    for line in open(path, encoding='utf-8', errors='replace'):
        if line.startswith('#') or not line.startswith('<f'): continue
        m = re.match(r'<([^>]*)>\s*(.*)', line.rstrip('\n'))
        if not m or ',P' not in m.group(1) and '+P' not in m.group(1) and '@P' not in m.group(1) and '*P' not in m.group(1) and '=P' not in m.group(1): continue
        t = clean(m.group(2))
        toks = re.split(r'([.,])', t)
        words = toks[0::2]; seps = toks[1::2]
        for i, s in enumerate(seps):
            a, b = words[i], words[i+1]
            if not a or not b or '?' in a or '?' in b or '%' in a+b or '!' in a+b: continue
            ga, gb = vlib.glyphs(a), vlib.glyphs(b)
            J[s].append((ga[-1], gb[0]))
    return J

def excess(P, rng, reps=200):
    o = mi(P); s = []
    for _ in range(reps):
        bb = [y for _, y in P]; rng.shuffle(bb); s.append(mi(list(zip([x for x, _ in P], bb))))
    return round(o - sum(s)/len(s), 4), sum(x >= o for x in s)/reps

for name in ('ZL3b-n.txt', 'IT2a-n.txt'):
    J = junctions(os.path.join(vlib.DATA, name)); rng = random.Random(1)
    n = min(len(J['.']), len(J[',']))
    print(name, 'certain spaces', len(J['.']), 'uncertain', len(J[',']))
    if n < 200: print('  too few uncertain spaces'); 
    print('  certain (all):', excess(J['.'], rng))
    if n >= 200:
        print('  certain (n=%d):' % n, excess(rng.sample(J['.'], n), rng), ' uncertain (n=%d):' % n, excess(rng.sample(J[','], n), rng))
        c = Counter(J[',']); cc = Counter(J['.'])
        print('  top uncertain-space junctions:', [(a+'|'+b, v) for (a, b), v in c.most_common(6)])
