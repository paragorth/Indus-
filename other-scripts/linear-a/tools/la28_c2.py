#!/usr/bin/env python3
"""la28 cycle 2: numbers. Do receipt quantities, or the COUNTS of identical receipts,
reappear as tablet amounts or totals at the same site?

(a) Quantity form: every receipt quantity (integer + fraction letters) with a term:
    is the same term/commodity written with the same quantity form on a same-site tablet?
    Also: share of fraction-only quantities on receipts vs tablets (Fisher).
(b) Counts as numbers (arrow in the dark): number of receipts carrying each term at a
    site (KH roundel classes, HT nodule signs), single counts, pair sums and triple sums,
    against (i) KU-RO / PO-TO-KU-RO totals and (ii) all distinct tablet amounts >= 10 at
    the same site. Null J: every count jittered uniformly within +-20 % (same search size,
    same magnitudes; 10,000 draws). Null S: counts of the OTHER site. Planted: a KU-RO
    equal to one pair sum inserted into a random same-site tablet (power).
"""
import random, sys, os, json, collections, itertools, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la28_common as C

R = random.Random(282)
out = []


def say(s):
    print(s, flush=True)
    out.append(s)


la = C.load_la()
REC = [d for d in la if d['cls'] == 'R']
TAB = [d for d in la if d['cls'] == 'T']

# (a) quantity forms
say('(a) receipt quantities')
rq = []
for d in REC:
    terms = [t for it in d['items'] for t in it['terms']]
    for it in d['items']:
        for q in it.get('q', []):
            rq.append((d['site'], d['id'], tuple(terms), q))
for x in rq:
    say(f'  {x[1]} {x[0]}: terms {list(x[2])} quantity {x[3][0]}{"+" + "".join(x[3][1]) if x[3][1] else ""}')
tq = collections.defaultdict(set)      # (site, term-base) -> quantity forms on tablets
tqs = collections.defaultdict(set)     # site -> quantity forms on tablets
for d in TAB:
    for it in d['items']:
        for q in it.get('q', []):
            tqs[d['site']].add(q)
            for t in it['terms']:
                tq[(d['site'], C.base(t))].add(q)
hit_term = [x for x in rq if any(x[3] in tq[(x[0], C.base(t))] for t in x[2])]
hit_form = [x for x in rq if x[3] in tqs[x[0]]]
say(f'  {len(rq)} receipt quantities; same term+same quantity on a same-site tablet: {len(hit_term)} '
    f'{[h[1] for h in hit_term]}; same quantity form anywhere on same-site tablets: {len(hit_form)}')
# site-swap null for the form match
vals = []
sites = [x[0] for x in rq]
for _ in range(5000):
    s2 = sites[:]
    R.shuffle(s2)
    vals.append(sum(1 for s, x in zip(s2, rq) if x[3] in tqs[s]))
m = sum(vals) / len(vals)
say(f'  quantity-form site-swap null {m:.2f}, P(>= {len(hit_form)}) {(1+sum(v>=len(hit_form) for v in vals))/5001:.3f}')
# term+quantity null: quantities shuffled among receipt quantities (terms and sites kept), and sites swapped
qs = [x[3] for x in rq]
v1, v2 = [], []
for _ in range(5000):
    q2 = qs[:]
    R.shuffle(q2)
    v1.append(sum(1 for x, q in zip(rq, q2) if any(q in tq[(x[0], C.base(t))] for t in x[2])))
    s2 = sites[:]
    R.shuffle(s2)
    v2.append(sum(1 for x, s in zip(rq, s2) if any(x[3] in tq[(s, C.base(t))] for t in x[2])))
say(f'  term+quantity: obs {len(hit_term)}; quantity-shuffle null {sum(v1)/5000:.2f} (P {(1+sum(v>=len(hit_term) for v in v1))/5001:.3f}); '
    f'site-swap null {sum(v2)/5000:.2f} (P {(1+sum(v>=len(hit_term) for v in v2))/5001:.3f})')
fr_only_r = sum(1 for x in rq if x[3][0] == 0 and x[3][1])
tall = [(d['site'], q) for d in TAB for it in d['items'] for q in it.get('q', [])]
fr_only_t = sum(1 for s, q in tall if q[0] == 0 and q[1])
a, b, c, dd = fr_only_r, len(rq) - fr_only_r, fr_only_t, len(tall) - fr_only_t
# Fisher exact (one-sided, receipts enriched)
def lch(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
N, K, n = a + b + c + dd, a + c, a + b
pf = sum(math.exp(lch(K, i) + lch(N - K, n - i) - lch(N, n)) for i in range(a, min(K, n) + 1))
say(f'  fraction-only quantities: receipts {a}/{a+b}, tablets {c}/{c+dd}; Fisher one-sided P {pf:.2g}')

# (b) counts as numbers
say('(b) receipt counts as tablet numbers')
counts = {}
MINC = {'Haghia Triada': 10, 'Khania': 5}
for site in ('Haghia Triada', 'Khania'):
    cc = collections.Counter()
    for d in REC:
        if d['site'] == site:
            ts = C.doc_terms(d)
            if len(ts) == 1:
                cc[next(iter(ts))] += 1
    counts[site] = [v for k, v in cc.most_common() if v >= MINC[site]]
    say(f'  {site}: receipt class counts {cc.most_common(14)}')

def targets(site):
    kuro, allnum = set(), set()
    for d in TAB:
        if d['site'] != site:
            continue
        for it in d['items']:
            if any(t in ('W:KU-RO', 'W:PO-TO-KU-RO') for t in it['terms']):
                kuro.update(n for n in it['nums'] if n > 0)
            allnum.update(n for n in it['nums'] if n >= 10)
    return kuro, allnum

def gens(cs):
    g = set(cs)
    g |= {a + b for a, b in itertools.combinations(cs, 2)}
    return g

for site in ('Haghia Triada', 'Khania'):
    cs = counts[site]
    kuro, allnum = targets(site)
    for tname, T in (('KU-RO totals', kuro), ('all amounts >= 10', allnum)):
        obs = len(gens(cs) & T)
        js = []
        for _ in range(10000):
            cj = [max(MINC[site] if False else 2, int(round(c * R.uniform(0.8, 1.2)))) for c in cs]
            js.append(len(gens(cj) & T))
        mj = sum(js) / len(js)
        pj = (1 + sum(v >= obs for v in js)) / 10001
        other = counts['Khania' if site == 'Haghia Triada' else 'Haghia Triada']
        so = len(gens(other) & T)
        say(f'  {site} counts {cs[:12]} -> {tname} ({len(T)} targets): {obs} hits '
            f'{sorted(gens(cs) & T)[:12]} vs jitter null {mj:.2f} (P {pj:.3f}); other-site counts {so}')

# planted: one KU-RO equal to a pair sum of HT counts, plus 2 more
say('  planted power (HT): insert m KU-RO totals equal to random pair/triple sums of HT counts into random HT tablets')
cs = counts['Haghia Triada']
kuro, _ = targets('Haghia Triada')
g = sorted(gens(cs))
for mplant in (1, 2, 4):
    det = 0
    for rep in range(200):
        T = set(kuro) | set(R.sample(g, mplant))
        obs = len(gens(cs) & T)
        js = []
        for _ in range(1000):
            cj = [max(MINC[site] if False else 2, int(round(c * R.uniform(0.8, 1.2)))) for c in cs]
            js.append(len(gens(cj) & T))
        det += ((1 + sum(v >= obs for v in js)) / 1001) < 0.05
    say(f'    m={mplant}: detected at P<0.05 in {det/200:.2f} of 200')

open(os.path.join(C.CK, 'c2.out'), 'w').write('\n'.join(out) + '\n')
