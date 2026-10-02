"""N3: line 'flavours'. N1b found glyph classes clump by line (lines differ in make-up beyond chance). If each line
used a different key or table row (a per-line substitution), clumping should come in SWAP pairs: a line rich in one
glyph is poor in its substitute (negative residual correlation between classes across lines). If lines are just
'about' different words (self-citation from a seed word), classes that co-occur inside common words rise together
(positive correlations). Statistic: correlation of per-line residuals (count - expected from line length and
paragraph rate), observed minus null mean (words re-dealt within paragraphs, 100x)."""
import sys, os, random, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

CL = {'gal': set('ktpfTKPF'), 'q': {'q'}, 'ch': {'C', 'S'}, 'e': {'e'}, 'i': {'i'}, 'd': {'d'}, 'y': {'y'}, 'l': {'l'},
      'r': {'r'}, 'n': {'n'}, 'a': {'a'}, 'o': {'o'}}
K = list(CL)

def paras(lines):
    P, cur = [], []
    for L in lines:
        if L.get('para_start') and cur: P.append(cur); cur = []
        cur.append([w for w in L['words'] if w])
    if cur: P.append(cur)
    return [p for p in P if len(p) >= 3]

def residuals(P):
    R = {k: [] for k in K}
    for p in P:
        tot = [sum(len(w) for w in l) for l in p]; T = sum(tot)
        for k, c in CL.items():
            cnt = [sum(1 for w in l for ch in w if ch in c) for l in p]; rate = sum(cnt)/T if T else 0
            R[k] += [(x - t*rate) / ((t*rate)**.5 if t*rate > 0 else 1) for x, t in zip(cnt, tot)]
    return R

def corr(a, b):
    n = len(a); ma = sum(a)/n; mb = sum(b)/n
    va = sum((x-ma)**2 for x in a); vb = sum((y-mb)**2 for y in b)
    return sum((x-ma)*(y-mb) for x, y in zip(a, b)) / (va*vb)**.5 if va and vb else 0

vl = vlib.load_voynich('ZL3b', drop_uncertain=True)
P = paras([{'words': [''.join(vlib.glyphs(w)) for w in L['words']], 'para_start': L.get('para_start')} for L in vl])
R = residuals(P); obs = {(a, b): corr(R[a], R[b]) for a, b in itertools.combinations(K, 2)}
rng = random.Random(5); null = {k: [] for k in obs}
for _ in range(100):
    NP = []
    for p in P:
        ws = [w for l in p for w in l]; rng.shuffle(ws); j = 0; q = []
        for l in p: q.append(ws[j:j+len(l)]); j += len(l)
        NP.append(q)
    RN = residuals(NP)
    for a, b in obs: null[(a, b)].append(corr(RN[a], RN[b]))
rows = []
for k, o in obs.items():
    s = sorted(null[k]); m = sum(s)/len(s); sd = (sum((x-m)**2 for x in s)/len(s))**.5 or 1e-9
    rows.append(((o-m)/sd, k, round(o, 3), round(m, 3)))
rows.sort()
print('most NEGATIVE (swap-like) line co-variation, z:'); [print('  ', f'{z:6.1f}', a+'~'+b, 'obs', o, 'null', m) for z, (a, b), o, m in rows[:8]]
print('most POSITIVE (travel together), z:'); [print('  ', f'{z:6.1f}', a+'~'+b, 'obs', o, 'null', m) for z, (a, b), o, m in rows[-8:]]
