#!/usr/bin/env python3
"""LA-8: evidence for each class assignment in the best Linear A machine. For every lexicon type,
move it to the other word class (open word <-> commodity) and record the fitness change (bits).
Positive = the evolved assignment is better by that many bits (held-out + DL)."""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la8_gp as G
C = sys.argv[1] if len(sys.argv) > 1 else 'LA'
D = G.Data(json.load(open(os.path.join(G.OUT, 'corpus_%s.json' % C))))
g = G.from_json(json.load(open(os.path.join(G.OUT, 'best_%s_final.json' % C)))['genome'])
cnt = Counter(t[0] for d in D.docs for t in d['toks'])
f0 = G.fitness(D, g)[0]
sizes = Counter(g['tc'].tolist())
word = [c for c in sizes if any(D.lex[i].startswith('W:') for i in range(D.V) if g['tc'][i] == c)]
word.sort(key=lambda c: -sizes[c])
a, b = word[0], word[1]
res = []
for i, w in enumerate(D.lex):
    if g['tc'][i] not in (a, b) or cnt[w] < 3: continue
    h = {'K': g['K'], 'tc': g['tc'].copy(), 'kc': g['kc'], 'nst': g['nst']}
    h['tc'][i] = b if g['tc'][i] == a else a
    res.append((G.fitness(D, h)[0] - f0, w, cnt[w], 'word' if g['tc'][i] == a else 'commodity'))
res.sort(key=lambda x: -x[0])
json.dump(res, open(os.path.join(G.OUT, 'margins_%s.json' % C), 'w'), ensure_ascii=False)
for cls in ('commodity', 'word'):
    r = [x for x in res if x[3] == cls]
    print(cls, 'n=%d' % len(r), 'margin>=2 bits: %d' % sum(x[0] >= 2 for x in r), 'margin<0 (misplaced): %d' % sum(x[0] < 0 for x in r))
    print('  strongest:', ['%s(%d) %.1f' % (x[1], x[2], x[0]) for x in r[:20]])
    print('  weakest:', ['%s(%d) %.1f' % (x[1], x[2], x[0]) for x in r[-8:]])
