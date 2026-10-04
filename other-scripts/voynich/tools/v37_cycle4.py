"""v37 cycle 4: rotation null for line schemas. Each line is rotated cyclically by a random offset (words keep
their neighbours except at one seam; the line's bag is kept), so any schema anchored to the line start is
destroyed while local word-to-word structure survives. Alignment = held-out gain lost by rotation, for the
left-to-right field automaton (k 4, 6) and for fixed position bins. Schema beyond edge rules = LR loss - posbin
loss. Usage: python3 v37_cycle4.py NAME [NAME...]"""
import sys, time
from v37_lib import *
from v37_cycle1 import get

def rotate(C, rng):
    out = []
    for p in C:
        q = dict(p); q['paras'] = []
        for pa in p['paras']:
            npa = []
            for l in pa:
                if len(l) > 2:
                    r = rng.randint(1, len(l) - 1); l = l[r:] + l[:r]
                npa.append(list(l))
            q['paras'].append(npa)
        out.append(q)
    return out

if __name__ == '__main__':
  for name in sys.argv[1:]:
    t = time.time(); C = get(name, 0); body = name.startswith('V')
    out = {'real': field_scores(C, seed=0, ks=(4, 6), restarts=6, body_only=body),
           'rot': [field_scores(rotate(C, random.Random(s)), seed=s, ks=(4, 6), restarts=6, body_only=body) for s in (1, 2)]}
    save(f'c4_{name}.json', out)
    r = out['real']; ro = {k: np.mean([x[k] for x in out['rot']]) for k in r}
    lr = max(r['LR4'], r['LR6']); lro = max(ro['LR4'], ro['LR6']); er = max(r['ER4'], r['ER6']); ero = max(ro['ER4'], ro['ER6'])
    print(name, f'{time.time()-t:.0f}s', 'LR real %.1f rot %.1f loss %.1f | ER loss %.1f | posbin real %.1f rot %.1f loss %.1f | schema beyond edges %.1f' %
          (lr, lro, lr - lro, er - ero, r['posbin'], ro['posbin'], r['posbin'] - ro['posbin'], (lr - lro) - (r['posbin'] - ro['posbin'])), flush=True)


def bin_profile(C, body_only, seed=0):
    """Held-out gain (mb/word) of position-bin emissions over a unigram, per bin (first, 2nd, 3rd, middle, 2nd-last, last)."""
    tr, te = split_half(C, seed=37 + seed); g = np.zeros(6); n = np.zeros(6)
    for A_, B_ in ((tr, te), (te, tr)):
        cl = Classer(A_)
        X, M = encode([l for _, _, l in line_units(A_, body_only)], cl); Xt, Mt = encode([l for _, _, l in line_units(B_, body_only)], cl)
        V = len(cl.idx) + 1
        def bins(Mm):
            nn = Mm.sum(1); T = Mm.shape[1]; t = np.arange(T)[None, :]
            b = np.full(Mm.shape, 3); b[:, 0] = 0
            b = np.where(t == 1, 1, b); b = np.where(t == 2, 2, b)
            b = np.where(t == nn[:, None] - 2, 4, b); b = np.where(t == nn[:, None] - 1, 5, b)
            return b
        b = bins(M); Bc = np.zeros((6, V)) + 0.5; np.add.at(Bc, (b[M], X[M]), 1); Bc /= Bc.sum(1, keepdims=True)
        uni = np.bincount(X[M], minlength=V) + 0.5; uni = uni / uni.sum()
        bt = bins(Mt)
        d = np.log2(Bc[bt[Mt], Xt[Mt]]) - np.log2(uni[Xt[Mt]])
        np.add.at(g, bt[Mt], d); np.add.at(n, bt[Mt], 1)
    return (1000 * g / np.maximum(n, 1)).round(0).tolist()
