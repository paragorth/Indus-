"""S-DARK-8.2b: does the Wells '/' convention agree with IM77 line numbering?
Map Wells two-register texts to M numbers with the bridge, find IM77 two-line sides with the
same sign multiset, and record whether Wells segment order (canonical 12 = after-'/' first)
matches IM77 line 1 -> line 2 or the reverse. Also: per IM77 two-line side, nll of order 12
minus order 21 (paired), to quantify the line-order margin."""
import json, csv, collections, sys, math, random
sys.path.insert(0, '/home/user/Indus-/tools')
from dark8_permute import Tri
from dark8_lines import wells, im77, nll_text
OUT = '/home/user/Indus-/data/derived/dark/'
bridge = {int(k): v for k, v in json.load(open('/home/user/Indus-/data/derived/bridge_extended.json')).items()}
one_w, two_w = wells(); one_m, two_m = im77()
lines = []
mi = collections.defaultdict(list)
for st, l1, l2, d1, d2 in two_m:
    mi[tuple(sorted(l1 + l2))].append((l1, l2))
agree12 = agree21 = nomatch = ambiguous = 0; examples = []
for st, a, b in two_w:
    # a = canonical first line (after '/'), b = second
    opts_a = [bridge.get(x, []) for x in a]; opts_b = [bridge.get(x, []) for x in b]
    if any(not o for o in opts_a + opts_b): nomatch += 1; continue
    # try all bridge combinations (small)
    import itertools
    hit = set()
    for ca in itertools.product(*opts_a):
        for cb in itertools.product(*opts_b):
            key = tuple(sorted(ca + cb))
            for l1, l2 in mi.get(key, []):
                if tuple(ca) == l1 and tuple(cb) == l2: hit.add('12')
                elif tuple(ca) == l2 and tuple(cb) == l1: hit.add('21')
                else: hit.add('other')
    if not hit: nomatch += 1
    elif hit == {'12'}: agree12 += 1
    elif hit == {'21'}: agree21 += 1; examples.append((st, a, b))
    else: ambiguous += 1
lines.append(f'Wells two-register texts {len(two_w)}: matched IM77 two-line side with canonical order (after-"/" first) = IM77 line1->line2: {agree12}; reversed (Wells canonical first line = IM77 line 2): {agree21}; mixed/other split: {ambiguous}; no match: {nomatch}')
lines.append('Wells texts whose canonical first line is IM77 line 2: ' + '; '.join(f'{st} {a}|{b}' for st, a, b in examples[:12]))
# IM77 margin 12 vs 21
V = len({x for s in one_m for x in s}) + 1; m = Tri([s for _, s in one_m], V)
d = [nll_text(m, l1 + l2) - nll_text(m, l2 + l1) for _, l1, l2, _, _ in two_m if len(l1) >= 2 and len(l2) >= 2]
rng = random.Random(3); N = len(d)
bs = sorted(sum(d[rng.randrange(N)] for _ in range(N)) / N for _ in range(2000))
wins = sum(1 for x in d if x < 0)
lines.append(f'IM77 sides with both lines >= 2 signs: {N}; nll(12) - nll(21) mean {sum(d)/N:+.3f} nats/token (95% CI {bs[50]:+.3f}..{bs[1950]:+.3f}); order 12 wins in {wins}/{N} texts (binomial two-sided P = {2*sum(math.comb(N,k) for k in range(wins, N+1))/2**N if wins > N/2 else 2*sum(math.comb(N,k) for k in range(0, wins+1))/2**N:.2e})')
# same for Wells
V = len({x for s in one_w for x in s}) + 1; mw = Tri([s for _, s in one_w], V)
d = [nll_text(mw, l1 + l2) - nll_text(mw, l2 + l1) for _, l1, l2 in two_w if len(l1) >= 2 and len(l2) >= 2]
N = len(d); wins = sum(1 for x in d if x < 0)
bs = sorted(sum(d[rng.randrange(N)] for _ in range(N)) / N for _ in range(2000))
lines.append(f'Wells "/" texts with both segments >= 2 signs: {N}; nll(canonical 12) - nll(21) mean {sum(d)/N:+.3f} (95% CI {bs[50]:+.3f}..{bs[1950]:+.3f}); canonical wins in {wins}/{N}')
open(OUT + 'loop8_cycle2b_log.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
