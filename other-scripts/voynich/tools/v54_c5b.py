"""v54 cycle 5b: collate only the context-anchored variants (target words that differ by one edit when a 2-word
context recurs), minus the same tally under target shuffles (20x).  These are the textual critic's true variant
classes."""
import sys, os, json, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v54_lib as V
from v54_c4c import lev1
from v54_c5 import trip_dirs

def tally(T, tgt):
    g = collections.defaultdict(list)
    for (t, x1, x2, _, _), y in zip(T, tgt): g[(t, x1, x2)].append(y)
    ops = collections.Counter(); pos = collections.Counter()
    for v in g.values():
        if len(v) < 2: continue
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if v[i] != v[j] and lev1(v[i], v[j]):
                    for o in V.align_ops(v[i], v[j]): ops[o[:-1]] += 1; pos[o[-1]] += 1
    return ops, pos

def run(pages, inv=None, nperm=20, seed=7):
    T = trip_dirs(pages); rng = random.Random(seed); tgt = [t[3] for t in T]
    O, Op = tally(T, tgt)
    grp = collections.defaultdict(list)
    for k, t in enumerate(T): grp[(t[0], t[4])].append(k)
    N = collections.Counter(); Np = collections.Counter()
    for _ in range(nperm):
        tt = list(tgt)
        for idx in grp.values():
            vals = [tgt[k] for k in idx]; rng.shuffle(vals)
            for k, v in zip(idx, vals): tt[k] = v
        a, b = tally(T, tt); N.update(a); Np.update(b)
    show = lambda k: tuple((inv.get(c, c) if inv else c) for c in k)
    ex = sorted(((O[k] - N[k] / nperm, O[k], round(N[k] / nperm, 1), show(k)) for k in set(O) | set(N)), reverse=True)
    return dict(top=[(round(a, 1), b, c, d) for a, b, c, d in ex[:15]], bottom=[(round(a, 1), b, c, d) for a, b, c, d in ex[-6:]],
                pos_obs=dict(Op), pos_null={k: round(v / nperm, 1) for k, v in Np.items()})

if __name__ == '__main__':
    bru, M = V.brumati(); inv = {v: k for k, v in M.items()}
    out = {}
    for nm, P, iv in [('BRU_planted', bru, inv), ('ZL', V.voynich('ZL3b'), None), ('IT', V.voynich('IT2a'), None), ('GER_real', V.german(), None)]:
        r = run(P, iv); out[nm] = r; print(nm, json.dumps(r, ensure_ascii=False), flush=True)
    json.dump(out, open(os.path.join(V.CK, 'c5b.json'), 'w'), ensure_ascii=False)
