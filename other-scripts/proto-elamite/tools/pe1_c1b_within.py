#!/usr/bin/env python3
"""PE-1.1b: what are the within-tablet repeats? Same designation twice on one
tablet: same final class sign (a repeated item) or different final (one holder
receiving several goods)? Null N3: permute designations across slots (keeps the
exact string multiset and each tablet's slot count). Null N4: bigram generator."""
import json, random
from pe1_lib import *
T = load()
DE = designation_entries(T, minlen=2)
D = [(e['tablet'], e['des']) for e in DE]
by = collections.defaultdict(list)
for e in DE:
    by[(e['tablet'], e['des'])].append(e)
same_fin = diff_fin = 0; ex = []
for (t, d), es in by.items():
    if len(es) < 2: continue
    fins = [e['final'] for e in es]
    for i in range(len(es)):
        for j in range(i + 1, len(es)):
            if fins[i] == fins[j]: same_fin += 1
            else: diff_fin += 1
    ex.append((t, ' '.join(d), fins, [e['val'] for e in es]))
print('within-tablet repeat pairs: same final', same_fin, 'different final', diff_fin)
for x in ex[:25]: print('  ', x)
# baseline: random pairs of entries on the same tablet - same final rate
pairs_same = pairs = 0
bt = collections.defaultdict(list)
for e in DE: bt[e['tablet']].append(e['final'])
for t, f in bt.items():
    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            pairs += 1; pairs_same += f[i] == f[j]
print('baseline same-final rate for any two designations on one tablet: %.3f' % (pairs_same / pairs))
rng = random.Random(5)
obs = recurrence_stats(D)
n3 = []
for _ in range(1000):
    ds = [d for _, d in D]; rng.shuffle(ds)
    n3.append(recurrence_stats([(t, d) for (t, _), d in zip(D, ds)]))
out = {'within_same_final_pairs': same_fin, 'within_diff_final_pairs': diff_fin,
       'baseline_same_final': pairs_same / pairs, 'examples': ex[:40]}
for k in ['within_tablet_dups', 'types_on_2plus_tablets', 'cross_tablet_type_pairs']:
    out[k + '|placement'] = zp(obs[k], [x[k] for x in n3], greater=(k == 'within_tablet_dups'))
    print(k, out[k + '|placement'])
# bigram generator
class Markov1(Markov2):
    def gen(self, L, rng):
        out = ['<s>']
        for _ in range(L):
            c1 = self.m1.get(out[-1]) if out[-1] != '<s>' else self.m2[('<s>', '<s>')]
            out.append(self._draw(c1 if c1 else self.u, rng))
        return tuple(out[1:])
mk = Markov1([d for _, d in D]); n4 = []
for _ in range(200):
    n4.append(recurrence_stats(markov_null(D, mk, rng)))
for k in ['types_on_2plus_tablets', 'tokens_in_recurring', 'within_tablet_dups', 'cross_tablet_type_pairs']:
    out[k + '|bigram'] = zp(obs[k], [x[k] for x in n4]); print(k, 'bigram', out[k + '|bigram'])
# by length
for L in [2, 3]:
    DL = [(t, d) for t, d in D if (len(d) == L if L == 2 else len(d) >= 3)]
    mk = Markov1([d for _, d in DL]); nl = [recurrence_stats(markov_null(DL, mk, rng)) for _ in range(200)]
    o = recurrence_stats(DL)
    out[f'len{L}_types2tab|bigram'] = zp(o['types_on_2plus_tablets'], [x['types_on_2plus_tablets'] for x in nl])
    print('len', L, out[f'len{L}_types2tab|bigram'])
json.dump(out, open(os.path.join(DATA, 'pe1_c1b_within.json'), 'w'), indent=1, default=str)
