#!/usr/bin/env python3
"""Test d: header (tablet-opening) signs.
Header = first line of the tablet when it carries signs and no numeral.
Enrichment = share of headers starting with sign X / share of entry tokens that are X.
Control: binomial-normal z, and a permutation that draws 'fake headers' as the
first sign of random entries (1000 draws of the same number of headers)."""
import collections, json, math, os, random
from common import load, entries, header, DATA

random.seed(4)
T = load()
H = [(t, header(t)) for t in T]
H = [(t, h) for t, h in H if h and h[0] != 'x']
ntab = len(T)
print('tablets', ntab, 'with a numeral-free first line (header)', len(H))
hc = collections.Counter(h[0] for _, h in H)
E = entries(T)
tok = collections.Counter(s for e in E for s in e['signs'] if s != 'x')
ntok = sum(tok.values())
first_entry = [e['signs'][0] for e in E if e['signs'][0] != 'x']
fc = collections.Counter(first_entry)
# permutation: draw len(H) random entry-initial signs
perm = collections.defaultdict(list)
for _ in range(1000):
    smp = collections.Counter(random.choices(first_entry, k=len(H)))
    for x in hc:
        perm[x].append(smp[x])
rows = []
for x, n in hc.most_common(25):
    p = tok[x] / ntok
    exp = p * len(H)
    z = (n - exp) / math.sqrt(max(exp * (1 - p), 1e-9))
    pv = sum(1 for v in perm[x] if v >= n) / 1000
    rows.append({'sign': x, 'headers': n, 'share': round(n / len(H), 3),
                 'entry_tokens': tok[x], 'enrich': round(n / len(H) / max(p, 1e-9), 1),
                 'z': round(z, 1), 'perm_p_vs_entry_initial': pv})
    print(rows[-1])
top = hc.most_common(1)[0]
print('most common header sign %s: %d/%d = %.2f of headers' % (top[0], top[1], len(H), top[1] / len(H)))
print('top-3 header signs cover %.2f of headers' % (sum(n for _, n in hc.most_common(3)) / len(H)))
# full header strings
hs = collections.Counter(' '.join(h) for _, h in H)
print('most common full header strings', hs.most_common(10))
print('distinct header strings', len(hs), 'singletons', sum(1 for v in hs.values() if v == 1))
# header sign -> which systems the entries use
by = collections.defaultdict(collections.Counter)
for t, h in H:
    for e in entries([t]):
        by[h[0]][e['system']] += 1
for x, _ in hc.most_common(8):
    print('header', x, 'entry systems', dict(by[x]))
json.dump({'n_headers': len(H), 'rows': rows, 'strings': hs.most_common(40),
           'systems_by_header': {k: dict(v) for k, v in by.items()}},
          open(os.path.join(DATA, 'res_d_headers.json'), 'w'), indent=1)
