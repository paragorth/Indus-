#!/usr/bin/env python3
"""la71 cycle 2: the la46 SA-RA2 basket (B) in three corpus versions, as a direct count.
Basket = commodity items (logograms by base sign, and the commodity word NI) after SA-RA2 up to the next
SA-RA2 / KU-RO / KI-RO / PO-TO-KU-RO or the end of the document.  Ordered = first item GRA or CYP, and NI
before VIN when both are present, and NI or VIN present.  A basket with a '#R' placeholder (removed token)
before its last item is incomplete and is not scored.  Null: item order shuffled inside each basket
(10,000), and decoy: the same rule after every other word that heads >= 3 item runs on HT tablets."""
import os, sys, json, random, re
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la71_parse as P
STOP = {'SA-RA₂', 'KU-RO', 'KI-RO', 'PO-TO-KU-RO'}


def baskets(C, head='SA-RA₂', site='Haghia Triada'):
    out = []
    for d in C:
        if site and d['site'] != site: continue
        T = [t for t in d['tokens'] if t['t'] not in ('nl', 'div')]
        for i, t in enumerate(T):
            if t['t'] == 'word' and '-'.join(t['s']) == head:
                items = []; inc = False
                for u in T[i + 1:]:
                    if u['t'] == 'word' and '-'.join(u['s']) in STOP | {head}: break
                    if u['t'] == 'unk' and u.get('v') == '#R': inc = True; items.append('#'); continue
                    if u['t'] == 'logo': items.append(re.sub(r'\+.*', '', u['v']))
                    elif u['t'] == 'word' and u['s'] == ['NI']: items.append('NI')
                while items and items[-1] == '#': items.pop()
                inc = '#' in items
                out.append((d['id'], [x for x in items if x != '#'], inc))
    return out


def ordered(items):
    if not items or items[0] not in ('GRA', 'CYP'): return False
    if 'NI' not in items and 'VIN' not in items: return False
    if 'NI' in items and 'VIN' in items and items.index('NI') > items.index('VIN'): return False
    return True


def score(B, rng=None):
    n = 0
    for _, it, inc in B:
        if inc: continue
        x = it[:]
        if rng: rng.shuffle(x)
        n += ordered(x)
    return n


def run(ver):
    C = P.load(ver)
    B = baskets(C)
    obs = score(B); tested = sum(1 for b in B if not b[2] and b[1])
    rng = random.Random(71)
    null = [score(B, rng) for _ in range(10000)]
    p = (sum(x >= obs for x in null) + 1) / 10001
    # decoy heads: other words heading >= 3 item runs at HT
    heads = Counter()
    for d in C:
        if d['site'] != 'Haghia Triada': continue
        T = [t for t in d['tokens'] if t['t'] not in ('nl', 'div')]
        for i, t in enumerate(T[:-1]):
            if t['t'] == 'word' and T[i + 1]['t'] == 'logo': heads['-'.join(t['s'])] += 1
    dec = []
    for h, c in heads.items():
        if c < 3 or h in STOP: continue
        Bh = baskets(C, h); o = score(Bh)
        nl = [score(Bh, rng) for _ in range(500)]
        dec.append((h, o, (sum(x >= o for x in nl) + 1) / 501))
    dfalse = sum(1 for _, o, pp in dec if pp <= 0.05 and o >= 3) / max(1, len(dec))
    return dict(ver=ver, baskets=len(B), tested=tested, ordered=obs, null_mean=sum(null) / len(null), P=p,
                ids=[b[0] for b in B if not b[2] and ordered(b[1])], decoys=len(dec), decoy_false=dfalse)


if __name__ == '__main__':
    res = {}
    for v in ('all', 'rd', 'read'):
        res[v] = run(v); print(json.dumps(res[v]))
    json.dump(res, open(os.path.join(P.CK, 'c2_basket.json'), 'w'), indent=1)
