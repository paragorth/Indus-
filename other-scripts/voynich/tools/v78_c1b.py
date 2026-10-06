"""v78 cycle 1b: are lag-1 doubles the same population as lag-2/3 repeats?

A dedicated doubling mechanism (reduplication, tally, ditto, dittography) writes lag-1 copies of its own kinds of
token; lag-2/3 repeats come from the text's ordinary re-use. A process with no special adjacency writes the same
kinds at every lag. Per corpus (E1c tokens, within lines):
  JSD   Jensen-Shannon divergence between the type distributions of lag-1 doubles and lag-2/3 repeats, against a
        label-permutation null (500 permutations): z and p.
  top1  share of lag-1 doubles taken by the 5 types most over-represented at lag 1 relative to lag 2/3.
  respell  share of lag-1 doubles whose raw forms differ / same share at lag 2/3 (copy re-spelled as often?)
  pos   mean relative line position of lag-1 doubles vs lag-2/3 repeats.
"""
import os, sys, random, pickle, json, math
import numpy as np
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v78_lib as L


def reps(pages):
    out = []   # (lag, type, rawdiff, relpos)
    for p in pages:
        for l in p['lines']:
            W = l['w']; T = [L.e1c(w) for w in W]; n = len(T)
            for k in (1, 2, 3):
                for i in range(n - k):
                    if T[i] == T[i + k]: out.append((1 if k == 1 else 2, T[i], W[i] != W[i + k], i / max(1, n - 1)))
    return out


def jsd(a, b):
    ks = set(a) | set(b); na = sum(a.values()); nb = sum(b.values()); s = 0
    for k in ks:
        p = a[k] / na; q = b[k] / nb; m = (p + q) / 2
        if p: s += 0.5 * p * math.log2(p / m)
        if q: s += 0.5 * q * math.log2(q / m)
    return s


def analyse(pages, nperm=500, seed=0):
    R = reps(pages)
    A = [r for r in R if r[0] == 1]; B = [r for r in R if r[0] == 2]
    if len(A) < 5 or len(B) < 5: return dict(n1=len(A), n2=len(B))
    ca = Counter(r[1] for r in A); cb = Counter(r[1] for r in B)
    j0 = jsd(ca, cb)
    rng = random.Random(seed); lab = [r[1] for r in R]; nA = len(A); null = []
    for _ in range(nperm):
        rng.shuffle(lab); null.append(jsd(Counter(lab[:nA]), Counter(lab[nA:])))
    null = np.array(null)
    over = sorted(ca, key=lambda t: -(ca[t] / len(A) - cb[t] / len(B)))[:5]
    return dict(n1=len(A), n2=len(B), jsd=j0, jsd_null=float(null.mean()), z=float((j0 - null.mean()) / (null.std() + 1e-12)),
                p=float((null >= j0).mean()), over=[(t, ca[t], cb[t]) for t in over],
                top1=sum(ca[t] for t in over) / len(A),
                respell1=float(np.mean([r[2] for r in A])), respell2=float(np.mean([r[2] for r in B])),
                pos1=float(np.mean([r[3] for r in A])), pos2=float(np.mean([r[3] for r in B])))


def main():
    C = pickle.load(open(os.path.join(L.CK, 'corpora.pkl'), 'rb'))
    res = {}
    for name, c in C.items():
        r = analyse(c['pages']); res[name] = dict(kind=c['kind'], **r)
        print(name, c['kind'], json.dumps(r, default=float), flush=True)
    for nm in ('VOY_ZL', 'VOY_IT'):
        for h in (0, 1):
            r = analyse(L.half(C[nm]['pages'], h)); res['%s_h%d' % (nm, h)] = r
            print(nm, h, json.dumps(r, default=float), flush=True)
    for mech, P in (('REDUP', dict(REDUP=dict(k=0.02, a=0.5, r=0.12, t=0.05))), ('DITTOG', dict(DITTOG=dict(r=0.01, x=0.2))),
                    ('TALLY', dict(TALLY=dict(m=2, r=0.06, g=0.4)))):
        S, lab = L.apply(C['VOY_ZL']['pages'], P, 79)
        sh, nd = L.shares(S, lab)
        r = analyse(S); res['PLANT_' + mech] = dict(share=sh[mech], **r)
        print('PLANT', mech, 'share %.2f' % sh[mech], json.dumps(r, default=float), flush=True)
    json.dump(res, open(os.path.join(L.CK, 'c1b.json'), 'w'), default=float, indent=1)


if __name__ == '__main__':
    main()
