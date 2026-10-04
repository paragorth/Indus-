#!/usr/bin/env python3
"""LA-45: is a stable Linear A meaning carried by word ORDER, or only by which documents a word sits in?
Compares per-type modal names of real LA populations with the order-shuffled (S2) populations of the same tag.
A type's meaning is ORDER-BORNE if it is stable in LA (agree >= 0.8) and S2's modal name differs (or S2 agree < 0.5).
Usage: la45_compare.py TAG REAL SHUF"""
import sys, os, json, glob, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la45_common as C

tag, real, shuf = sys.argv[1:4]


def load(n):
    fs = [f for f in glob.glob(os.path.join(C.CK, '%s_%s_*.json' % (tag, n)))
          if re.fullmatch(r'%s_%s_\d+\.json' % (tag, n), os.path.basename(f))]
    return [json.load(open(f)) for f in sorted(fs)]


def modal(runs, minc=3):
    out = {}
    allw = set().union(*[set(r['tnames']) for r in runs])
    for w in allw:
        ns = [r['tnames'][w] for r in runs if w in r['tnames']]
        cs = [r['tcnt'][w] for r in runs if w in r['tcnt']]
        if len(ns) < 0.8 * len(runs) or sum(cs) / len(cs) < minc:
            continue
        c = collections.Counter(ns).most_common(1)[0]
        out[w] = (c[0], c[1] / len(ns), sum(cs) / len(cs))
    return out


A, S = modal(load(real)), modal(load(shuf))
stable = {w: v for w, v in A.items() if v[1] >= 0.8}
order = {w: v for w, v in stable.items() if w in S and (S[w][0] != v[0] or S[w][1] < 0.5)}
same = {w: v for w, v in stable.items() if w in S and w not in order}
print('stable in %s: %d; also stable-same in %s: %d; ORDER-BORNE: %d' % (real, len(stable), shuf, len(same), len(order)))
by = collections.defaultdict(list)
for w, v in sorted(order.items(), key=lambda x: -x[1][2]):
    by[v[0]].append('%s(%s->%s %.0f)' % (w, v[0], S[w][0], v[2]))
for k, v in by.items():
    print('  ', k, len(v), ' '.join(v[:20]))
json.dump({'stable': stable, 'order_borne': order, 'shuf_modal': {w: S.get(w) for w in stable}},
          open(os.path.join(C.CK, '%s_compare_%s_%s.json' % (tag, real, shuf)), 'w'), ensure_ascii=False, indent=0)
