#!/usr/bin/env python3
"""LA-5 cycle 4b: is the consonant-avoidance (cecho) signal phonological, or a trivial artefact?
(i) identical-sign doubling removed (pairs x == y dropped): does the avoidance remain?
(ii) partition specificity: the same obs / P-shuffle ratio for 500 RANDOM partitions of the valued signs into classes
     of the real consonant-series sizes; percentile of the real (conventional-value) partition. A code that merely avoids
     repeating marks would make every partition show avoidance; phonology makes the real partition special.
Full type sets; 50 cached P-shuffles per population. Output ../data/la5_c4b.txt
"""
import sys, os, random, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la5_common as C
CONS = re.compile(r'^(D|J|K|M|N|P|Q|R|S|T|W|Z|DW|NW|TW|PH)?([AEIOU])[23]?$')
def cons(s):
    m = CONS.match(s); return (m.group(1) or '') if m and m.group(1) else None

la = C.la_docs(); lb = C.lb_docs(); wa = C.words_of(la); wb = C.words_of(lb)
T = lambda ws, f=lambda s: True: sorted(set(w for _, s, w in ws if f(s)))
pops = {'LA': T(wa), 'LA_nonHT': T(wa, lambda s: s != 'Haghia Triada'), 'LB_KN': T(wb, lambda s: s == 'KN'), 'LBpers': C.comparators()['LBpers']}
r0 = random.Random(7)
pops['LA_randID'] = sorted(set(C.random_id_code(pops['LA'], r0).values()))
pops['LA_slot'] = sorted(set(C.slot_code(pops['LA'], r0).values()))
out = ['LA-5 cycle 4b: consonant avoidance without doubling, and specificity of the conventional consonant partition']
for name, types in pops.items():
    r = random.Random(41)
    def pairs(tt, nodup):
        return [(t[i], t[i + 1]) for t in tt for i in range(len(t) - 1) if cons(t[i]) and cons(t[i + 1]) and (not nodup or t[i] != t[i + 1])]
    for nodup in (False, True):
        obs = pairs(types, nodup); nulls = [pairs(C.shuffle_pos(types, r), nodup) for _ in range(50)]
        def ratio(lab):
            o = sum(1 for x, y in obs if lab[x] == lab[y]) / len(obs)
            n = sum(sum(1 for x, y in pp if lab[x] == lab[y]) / len(pp) for pp in nulls) / len(nulls)
            return o / n
        signs = sorted(set(a for t in types for a in t if cons(a)))
        real = {a: cons(a) for a in signs}; rr = ratio(real)
        ks = list(real); vs = [real[k] for k in ks]; rnd = []
        for _ in range(500):
            r.shuffle(vs); rnd.append(ratio(dict(zip(ks, vs))))
        pct = sum(1 for x in rnd if x <= rr) / len(rnd)
        rnd.sort()
        out.append(f'{name:10s} {"no doubling" if nodup else "all pairs  "}: pairs {len(obs)}  real-partition ratio {rr:.3f}  random partitions median {rnd[250]:.3f} [{rnd[12]:.3f}, {rnd[487]:.3f}]  share of random partitions <= real {pct:.3f}')
        print(out[-1], flush=True)
open(os.path.join(C.LAD, 'la5_c4b.txt'), 'w').write('\n'.join(out) + '\n')
