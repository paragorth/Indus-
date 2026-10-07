"""v88 cycle 1b: power at the Voynich's own junction strength. The ZL3b word stream (binding
order) re-poured across spreads (row by row across each opening) or in normal flow, no encoding;
Latin/Cury through a plain opaque letter substitution (no padding)."""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v88_lib as G, v65_lib as V, v88_c1 as C

def subst(words, seed):
    rng = random.Random(seed); al = sorted({c for w in words for c in w}); sh = al[:]; rng.shuffle(sh)
    m = dict(zip(al, sh)); return [''.join(m[c] for c in w) for w in words]

quires, _ = V.structure('ZL3b'); tmpl, meta = G.vpages('ZL3b')
stream = [w for k in G.phys_seq(quires) if k != G.GAP and k in tmpl for L in tmpl[k] for w in L]
corp = [('Voy-SPREAD', G.pour_spread(tmpl, stream, 0, quires, encode=False)),
        ('Voy-FLOW', G.pour_flow(tmpl, stream, 0, quires, encode=False)),
        ('IsidoreSub-SPREAD', G.pour_spread(tmpl, subst(V.isidore_words(), 3), 0, quires, encode=False)),
        ('CurySub-SPREAD', G.pour_spread(tmpl, subst(G.cury_words(), 4), 0, quires, encode=False)),
        ('CurySub-FLOW', G.pour_flow(tmpl, subst(G.cury_words(), 4), 0, quires, encode=False))]
out = []
for name, p in corp:
    r = C.analyse(name, p, meta, quires); out.append(r); c = r['classes']
    print(name, 'WL %.3f WRAP %.3f' % (r['within_line'], r['wrap']),
          ' '.join('%s J0 %.3f zT %.1f zD %.1f | S zD %.1f' % (cl, c[cl]['J']['T0'], c[cl]['J']['zT0'], c[cl]['J']['zD'], c[cl]['S']['zD']) for cl in ('GUT', 'GUTR', 'LEAF', 'FLAT')),
          'blind', r['blind'], flush=True)
json.dump(out, open(os.path.join(G.CK, 'c1b.json'), 'w'), indent=1, default=float)
