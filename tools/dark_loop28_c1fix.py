"""S-DARK-28 cycle 1 post-fix.  (1) The 32x32 matcher cannot always separate a one-stroke
difference, so a derived sign's remainder sometimes matches a hapax near-duplicate of the
real base (W741 -> W750 instead of W740).  For every edge whose base has < 3 tokens, the
best base with >= 3 tokens scoring within 0.10 of the top (and above threshold) is added as
an 'alt' edge.  (2) Reciprocal strokes edges (a -> b and b -> a) keep only the direction in
which the derived glyph has more ink.  (3) Union with the attached-modification edges of
cycle 1b when present.  Rewrites loop28_edges.json (original kept as loop28_edges_raw.json)
and loop28_edges_all.json.
"""
import sys, json, collections, os
sys.argv = ['x']; sys.path.insert(0, 'tools')
import numpy as np
src = open('tools/dark_loop28_c1.py').read().split('# ---------------- planted controls')[0]
exec(src)
OUT = 'data/derived/dark/'
M = json.load(open('data/derived/merged-corpus-canonical.json'))
FREQ = collections.Counter(x for r in M for x in r['seq_raw'] if x not in (0, 999))
A = json.load(open(OUT + 'loop28_edges.json'))
if not os.path.exists(OUT + 'loop28_edges_raw.json'): json.dump(A, open(OUT + 'loop28_edges_raw.json', 'w'), indent=0)
else: A = json.load(open(OUT + 'loop28_edges_raw.json'))
THR = 0.863
INK = {w: int(G[w].sum()) for w in SIGNS}
# (2) reciprocal edges
pairs = {(e['derived'], e['base'], e['type']) for e in A}
A2 = []
for e in A:
    if (e['base'], e['derived'], e['type']) in pairs and INK[e['derived']] < INK[e['base']]: continue
    A2.append(e)
# (1) alt frequent base
added = []
for e in list(A2):
    if FREQ[e['base']] >= 3 or e['type'] not in ('strokes', 'enclosure', 'roof'): continue
    D = G[e['derived']]
    best = None
    for c in analyse(D):
        if c['type'] != e['type']: continue
        s = match(c['part'])
        if s is None: continue
        top = float(s.max())
        for bi in np.argsort(-s):
            b = SIGNS[bi]
            if s[bi] < THR or s[bi] < top - 0.10: break
            if b == e['derived'] or FREQ[b] < 3: continue
            # among passing frequent bases prefer the most frequent (bases are frequent, derivatives rare)
            if best is None or FREQ[b] > FREQ[best[0]]:
                best = (b, float(s[bi]), c)
    if best and not any(x['derived'] == e['derived'] and x['base'] == best[0] for x in A2):
        b, sc, c = best
        ne = dict(e, base=b, score=sc, fine=float(SIM_FINE[IDX[e['derived']], IDX[b]]), conf=('mid' if sc >= THR + 0.04 else 'low'), alt_of=e['base'])
        if 'k' in c: ne['k'] = c['k']; ne['pos'] = c['pos']; ne['sub'] = c.get('sub', e.get('sub'))
        A2.append(ne); added.append(ne)
print('reciprocal edges dropped', len(A) - len(A2) + len(added), 'alt edges added', len(added))
for e in added: print(f"  W{e['derived']} <- W{e['base']} ({e['type']}, was W{e['alt_of']}) score {e['score']:.3f}")
json.dump(A2, open(OUT + 'loop28_edges.json', 'w'), indent=0)
if os.path.exists(OUT + 'loop28_edges_b.json'):
    EB = json.load(open(OUT + 'loop28_edges_b.json'))
    have = {(e['derived'], e['base']) for e in A2}
    new = [dict(e, type=e['type'].rstrip('+'), method='attached') for e in EB if (e['derived'], e['base']) not in have]
    U = A2 + new
    json.dump(U, open(OUT + 'loop28_edges_all.json', 'w'), indent=0)
    print('union', len(U), 'edges (component', len(A2), '+ attached-new', len(new), ')')
    print('by type', collections.Counter(e['type'] for e in U))
    print('edges with both signs >= 5 tokens:', sum(1 for e in U if FREQ[e['derived']] >= 5 and FREQ[e['base']] >= 5), '; >= 20:', sum(1 for e in U if FREQ[e['derived']] >= 20 and FREQ[e['base']] >= 20))
