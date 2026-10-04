"""v29 cycle 3: WHICH SIDE OF THE JUNCTION ADAPTS?  (sandhi direction)
If the Voynich junction agreement is phonological sandhi, one side should adapt to the other: in regressive
sandhi the final glyph of word 1 varies with the next word's initial, the rest of word 1 held fixed;
in progressive sandhi the initial of word 2 varies with the previous word's final, the rest of word 2 held fixed.
  REG  = I(final1 ; initial2 | stem1 = word1 minus its final)
  PROG = I(initial2 ; final1 | rest2 = word2 minus its initial)
Plug-in conditional MI, excess over a within-line word shuffle (20 replicates).  Stroke version: the glyphs are
replaced by their v25 stroke class (DESC/CROSS/LOOP/MINIM/CURVE ... signature) before computing MI.
Positive controls: Latin with planted REGRESSIVE sandhi (final s/m/t -> depends on the next initial) and planted
PROGRESSIVE sandhi (initial c/p/t -> depends on the previous final); the same planted into Voynich text.
Kill control: the shape-neutral copy-and-edit generator (v29_copygen).
usage: python3 v29_cycle3.py -> data/v29_ckpt/c3_dir.json"""
import os, sys, json, random, math
import numpy as np
from collections import Counter, defaultdict
import v29_lib as L, v25_shapes as S, v29_copygen
from v29_cycle1 import corpus


def cmi(triples):
    """I(X;Y|Z) plug-in, triples = list of (z, x, y)."""
    g = defaultdict(Counter)
    for z, x, y in triples:
        g[z][(x, y)] += 1
    N = len(triples); tot = 0.0
    for z, c in g.items():
        n = sum(c.values())
        if n < 2:
            continue
        cx = Counter(); cy = Counter()
        for (x, y), k in c.items():
            cx[x] += k; cy[y] += k
        mi = 0.0
        for (x, y), k in c.items():
            mi += k / n * math.log2(k * n / (cx[x] * cy[y]))
        tot += n / N * mi
    return tot


def chent(triples):
    """H(X|Z) plug-in for triples (z, x, y)."""
    g = defaultdict(Counter)
    for z, x, y in triples:
        g[z][x] += 1
    N = len(triples); h = 0.0
    for z, c in g.items():
        n = sum(c.values())
        h += n / N * -sum(k / n * math.log2(k / n) for k in c.values())
    return h


def pairs(lines, cls=None):
    reg, prog = [], []
    f = (lambda g: g) if cls is None else cls
    for L_ in lines:
        for a, b in zip(L_, L_[1:]):
            if len(a) < 2 or len(b) < 2:
                continue
            reg.append((a[:-1], f(a[-1]), f(b[0])))
            prog.append((b[1:], f(b[0]), f(a[-1])))
    return reg, prog


def score(lines, cls=None, R=20, seed=0):
    r0, p0 = pairs(lines, cls)
    o = np.array([cmi(r0), cmi(p0)])
    n = []
    for q in range(R):
        r1, p1 = pairs(L.wshuffle(lines, seed + q), cls)
        n.append([cmi(r1), cmi(p1)])
    n = np.array(n)
    ex = o - n.mean(0); sd = np.maximum(n.std(0, ddof=1), 1e-4)
    hr, hp = chent(r0), chent(p0)   # how free the final (given stem) and the initial (given rest) are
    return dict(reg=float(ex[0]), prog=float(ex[1]), Hreg=hr, Hprog=hp,
                nreg=float(ex[0] / max(hr, 1e-9)), nprog=float(ex[1] / max(hp, 1e-9)), zreg=float(ex[0] / sd[0]), zprog=float(ex[1] / sd[1]),
                asym=float((ex[0] - ex[1]) / max(abs(ex[0]) + abs(ex[1]), 1e-9)), n=len(r0))


def plant_sandhi(lines, mode, finals, initials, rule, p=0.5, seed=0):
    """mode 'reg': a final glyph in `finals` is replaced by rule[class(next initial)];
       mode 'prog': an initial glyph in `initials` is replaced by rule[class(previous final)].
       class(g) = g in the first class set of `rule['_cls']`."""
    rng = random.Random(seed); C = rule['_cls']
    out = []
    for L_ in lines:
        nl = [list(w) for w in L_]
        for i in range(len(nl) - 1):
            a, b = nl[i], nl[i + 1]
            if rng.random() > p:
                continue
            if mode == 'reg' and a and a[-1] in finals and b:
                a[-1] = rule[b[0] in C]
            if mode == 'prog' and b and b[0] in initials and a:
                b[0] = rule[a[-1] in C]
        out.append([tuple(w) for w in nl])
    return out


def stroke_class(g):
    sh = S.VOYNICH.get(g)
    return g if sh is None else '+'.join(sorted(k for k in sh if k in ('DESC', 'CROSS', 'LEGS', 'BAR', 'MINIM', 'CURVE', 'LOOP', 'RTAIL', 'PLUME', 'ASC')))


if __name__ == '__main__':
    out = {}
    la = L.plain_lines('la')
    V = set('aeiouy')
    lat_rule = {'_cls': V, True: 'd', False: 's'}
    lat_prog = {'_cls': V, True: 'g', False: 'c'}
    voy = L.voynich_lines('ZL3b')
    voy_reg = {'_cls': {'q', 'o'}, True: 'y', False: 'n'}     # planted: final y/n/l chosen by next initial
    voy_prog = {'_cls': {'y'}, True: 'q', False: 'o'}         # planted: initial q/o chosen by previous final
    runs = {
        'la': la,
        'la_plant_reg': plant_sandhi(la, 'reg', {'s', 'm', 't'}, None, lat_rule),
        'la_plant_prog': plant_sandhi(la, 'prog', None, {'c', 'p', 't'}, lat_prog),
        'tr': corpus('tr')[0], 'hu': corpus('hu')[0], 'cs': corpus('cs')[0],
        'ZL': voy, 'IT': L.voynich_lines('IT2a'),
        'ZL_A': L.voynich_lines('ZL3b', lang='A'), 'ZL_B': L.voynich_lines('ZL3b', lang='B'),
        'ZL_plant_reg': plant_sandhi(voy, 'reg', {'y', 'n', 'l'}, None, voy_reg, p=0.3),
        'ZL_plant_prog': plant_sandhi(voy, 'prog', None, {'q', 'o'}, voy_prog, p=0.3),
        'copy5': v29_copygen.gen(voy, 0.5, 1.0, seed=3), 'copy8': v29_copygen.gen(voy, 0.8, 1.5, seed=3),
    }
    for k, lines in runs.items():
        out[k] = dict(glyph=score(lines))
        if k.startswith(('ZL', 'IT', 'copy')):
            out[k]['stroke'] = score(lines, cls=stroke_class)
        print(k, json.dumps(out[k]), flush=True)
    json.dump(out, open(os.path.join(L.CK, 'c3_dir.json'), 'w'), indent=1)
