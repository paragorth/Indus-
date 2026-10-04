"""v12 cycle 3c: position-preserving null for the ch/sh lag-2 swap and per-position word entropy (zig-zag or trend?).
Null: words are permuted among lines of the same Currier language AT THE SAME LINE POSITION (keeps every position's
vocabulary, destroys pairing within a line). 50 permutations."""
import sys, os, json, math, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v12_lib as V
from v12_cycle3b import cs_pairs, H

def colshuf(lines, seed):
    rng = random.Random(seed); buckets = defaultdict(list)
    for li, l in enumerate(lines):
        n = len(l['words'])
        for i, w in enumerate(l['words']):
            key = (l['lang'], i if i < n - 1 else 'last')
            buckets[key].append((li, i))
    new = [list(l['words']) for l in lines]
    for key, slots in buckets.items():
        ws = [lines[li]['words'][i] for li, i in slots]; rng.shuffle(ws)
        for (li, i), w in zip(slots, ws): new[li][i] = w
    return [dict(l, words=new[k]) for k, l in enumerate(lines)]

out = {}
for name in ['Voynich-ZL', 'Voynich-IT']:
    C = V.corpus(name); r = {}
    for lag in (1, 2):
        n = cs_pairs(C, lag)[0]; nulls = [cs_pairs(colshuf(C, s), lag) for s in range(1, 51)]
        m = sum(x[0] for x in nulls) / 50; sd = math.sqrt(sum((x[0] - m) ** 2 for x in nulls) / 49)
        pos1 = cs_pairs(C, lag)[1][1]; pm = sum(x[1][1] for x in nulls) / 50
        print(name, f'lag{lag} ch/sh pairs {n} vs position-preserving null {m:.1f} (z {(n-m)/sd:.1f}); start pos 2: {pos1} vs {pm:.1f}')
        r[f'lag{lag}'] = (n, m, (n - m) / sd, pos1, pm)
    # per-position entropy, equal samples
    pos = defaultdict(list)
    for l in C:
        ws = l['words']
        for i, w in enumerate(ws[:-1]):
            if i < 9: pos[i].append(w)
    m = min(len(pos[i]) for i in range(9)); rng = random.Random(0)
    ent = [H(Counter(rng.sample(pos[i], m))) for i in range(9)]
    print(name, 'entropy by position 1..9 (last word excluded, n =', m, '):', ' '.join(f'{e:.3f}' for e in ent))
    r['pos_entropy'] = ent
    out[name] = r
json.dump(out, open(os.path.join(V.CK, 'c3c.json'), 'w'), indent=1)
