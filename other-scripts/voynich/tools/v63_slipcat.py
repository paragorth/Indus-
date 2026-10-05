"""v63: catalogue of REAL slip-and-rewrite events in the Plaoul diplomatic witnesses.
A = whole-word deletion (scribe struck it), B = the next kept word (what was meant instead).
Relation of A to its context: dittography, false start (A proper prefix of B), near-miss (ed<=2 of B),
anticipation (A ~ a later word, offset 1..12 after B, i.e. eye jumped ahead), perseveration (A ~ an earlier
word within 12), same ending as B (homeoteleuton-like), other.  Writes data/v63_ckpt/slipcat.json."""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v63_lib as L


def near(a, b):
    return a == b or (len(a) >= 3 and (b.startswith(a) or L.ed(a, b) <= 1))


def classify(A, B, after_next, before_prev):
    if A == B: return 'ditto', 0
    if B.startswith(A): return 'falsestart', 0
    for k, w in enumerate(after_next[:12]):
        if near(A, w): return 'anticip', k + 1
    if L.ed(A, B) <= 2: return 'nearmiss', 0
    for k, w in enumerate(before_prev[:12]):
        if near(A, w): return 'persev', k + 1
    if len(A) >= 3 and A[-2:] == B[-2:]: return 'sameend', 0
    return 'other', 0


def main():
    P = L.plaoul_witnesses()
    cat, offs, ex, lbA = Counter(), Counter(), {}, Counter()
    events = []
    for p in P:
        ts = p['toks']
        kept = [(i, t['after']) for i, t in enumerate(ts) if t['st'] != 'd' and t['after']]
        for i, t in enumerate(ts):
            if t['st'] != 'd' or not t['before']:
                continue
            nxt = [w for j, w in kept if j > i]
            prv = [w for j, w in reversed(kept) if j < i]
            if not nxt:
                continue
            A, B = t['before'], nxt[0]
            c, k = classify(A, B, nxt[1:], prv)
            cat[c] += 1
            if c == 'anticip': offs[k] += 1
            ex.setdefault(c, []).append((A, B))
            lbA[(c, t['lb'])] += 1
            events.append(dict(A=A, B=B, c=c, k=k, lb=t['lb'], wit=p['wit']))
    n = sum(cat.values())
    print('n', n, {c: (v, round(v / n, 3)) for c, v in cat.most_common()})
    print('anticipation offsets', sorted(offs.items()))
    for c in cat:
        print(c, ex[c][:10])
    print('line break right after A:', {c: round(lbA[(c, True)] / max(1, lbA[(c, True)] + lbA[(c, False)]), 3) for c in cat})
    json.dump(dict(cat=cat, offs=offs, events=events), open(os.path.join(L.CK, 'slipcat.json'), 'w'))


if __name__ == '__main__':
    main()
