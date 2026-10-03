"""Loop 15 cycle 2: classify the gaps of cycle 1 (loop15_c1_<seqkey>.json) into rule kinds and test each against a
second, model-free null: sign tokens permuted across texts within site x object-class strata, keeping text lengths
(NPERM corpora).  A gap is 'robust' if it is also a gap under this null (expected >= EMIN, observed 0), so that it
does not depend on the order-2 slot model's misfit.
Kinds: REPEAT (X..X), PARADIGM (X and Y never in one text; co gap), ORDER (X..Y never, Y..X attested >= 3),
BOUND (X,Y attested adjacent >= 3 but never separated), TRIPLE (all three pairs attested, triple never), OTHER.
Usage: python3 loop15_classify.py --seqkey seq_raw [--nperm 200]
"""
import json, sys, os, random, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.argv += []
ARGS = sys.argv[1:]
def arg(n, d): return ARGS[ARGS.index(n) + 1] if n in ARGS else d
SEQKEY = arg('--seqkey', 'seq_raw'); NPERM = int(arg('--nperm', 200)); EMIN = float(arg('--emin', 5))
import importlib.util
spec = importlib.util.spec_from_file_location('eng', os.path.join(HERE, 'loop15_engine.py'))
sys.argv = ['x', '--seqkey', SEQKEY]; eng = importlib.util.module_from_spec(spec); spec.loader.exec_module(eng)
HOME, HELD, ALL = eng.HOME, eng.HELD, eng.ALL
J = json.load(open(os.path.join(HERE, f'loop15_c1_{SEQKEY}.json')))
gaps = [(tuple(k), e, eh, oh) for k, e, eh, oh in J['gaps']]
voc = eng.vocab_of(HOME, 15)
rng = random.Random(152)
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.append(s)
P(f'LOOP15 cycle2 classify seqkey={SEQKEY} gaps={len(gaps)} nperm={NPERM}')

CLASS = {}
for x in (817, 861, 820, 920, 692): CLASS[x] = 'opener'
for x in (2, 60): CLASS[x] = 'marker'
for x in (400, 90): CLASS[x] = 'suffix'
for x in (740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700): CLASS[x] = 'closer'
for x in (741, 742, 745): CLASS[x] = 'marked-jar'
for x in (235, 240, 233, 231, 220): CLASS[x] = 'fish'
for x in (1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56): CLASS[x] = 'numeral'
for x in (390, 405, 407): CLASS[x] = 'tree'
for x in (90, 91, 176, 140, 142, 100): CLASS[x] = 'person'
for x in (760, 100, 904, 636, 923, 61): CLASS.setdefault(x, 'pre-jar')
def cl(x): return CLASS.get(x, '-')

# observed counts (home) for pairs in both orders, adjacency, co-occurrence
obs = eng.count(HOME, voc)
adj = collections.Counter()
for _, _, s in HOME:
    for i in range(len(s) - 1):
        if s[i] in voc and s[i + 1] in voc: adj[(s[i], s[i + 1])] += 1

# model-free stratified permutation null
strata = collections.defaultdict(list)
for i, (site, c, s) in enumerate(HOME): strata[(site, c)].append(i)
keys = set(k for k, *_ in gaps)
tot = collections.Counter(); per_zero = collections.Counter()
for r in range(NPERM):
    new = [None] * len(HOME)
    for idx in strata.values():
        pool = [x for i in idx for x in HOME[i][2]]; rng.shuffle(pool); p = 0
        for i in idx:
            L = len(HOME[i][2]); new[i] = (HOME[i][0], HOME[i][1], tuple(pool[p:p + L])); p += L
    c = eng.count(new, voc)
    for k in keys:
        tot[k] += c.get(k, 0)
        if c.get(k, 0) == 0: per_zero[k] += 1
expP = {k: tot[k] / NPERM for k in keys}

def kind(k):
    t = k[0]
    if t == 'co':
        x, y = k[1], k[2]
        if x == y: return 'REPEAT'
        return 'PARADIGM'
    if t in ('ord', 'skip'):
        x, y = k[1], k[2]
        if x == y: return 'REPEAT'
        if obs.get(('co', min(x, y), max(x, y)), 0) == 0: return 'PARADIGM'
        if adj.get((x, y), 0) >= 3 and t == 'ord': return 'BOUND'
        if obs.get(('ord', y, x), 0) + adj.get((y, x), 0) >= 3: return 'ORDER'
        return 'OTHER'
    if t == 'tri':
        x, y, z = k[1:]
        if len({x, y, z}) < 3: return 'REPEAT'
        pairs = [(x, y), (y, z), (x, z)]
        if all(obs.get(('co', min(a, b), max(a, b)), 0) >= 3 for a, b in pairs): return 'TRIPLE'
        return 'REDUCIBLE'   # reduces to a pair gap / near-gap
    return 'OTHER'

rows = []
for k, e, eh, oh in gaps:
    kd = kind(k); ep = expP.get(k, 0)
    rows.append((kd, k, e, eh, oh, ep))
P('\nkind counts (all home gaps):', dict(collections.Counter(r[0] for r in rows)))
P('kind counts, robust under stratified permutation null (E_perm >= %.0f):' % EMIN,
  dict(collections.Counter(r[0] for r in rows if r[5] >= EMIN)))
P('kind counts, persistent on held-out (E_held >= 2, O_held = 0):',
  dict(collections.Counter(r[0] for r in rows if r[3] >= 2 and r[4] == 0)))
P('\nkind      pattern                      classes                  E_model  E_perm  E_held  O_held  reverse/adj (home)')
for kd, k, e, eh, oh, ep in sorted(rows, key=lambda r: (r[0], -r[2])):
    sig = k[1:]; cls = '/'.join(cl(x) for x in sig)
    extra = ''
    if k[0] in ('ord', 'skip') and len(sig) == 2:
        x, y = sig
        extra = f'rev ord={obs.get(("ord", y, x), 0)} rev adj={adj.get((y, x), 0)} adj={adj.get((x, y), 0)}'
    if k[0] == 'tri':
        x, y, z = sig
        extra = 'pairs co=' + ','.join(str(obs.get(('co', min(a, b), max(a, b)), 0)) for a, b in ((x, y), (y, z), (x, z)))
    P(f'{kd:<9} {str(k):<28} {cls:<24} {e:6.1f}  {ep:6.1f}  {eh:6.1f}  {oh:5d}   {extra}')
json.dump([dict(kind=r[0], pattern=list(r[1]), E_model=r[2], E_held=r[3], O_held=r[4], E_perm=r[5]) for r in rows],
          open(os.path.join(HERE, f'loop15_c2_{SEQKEY}.json'), 'w'))
open(os.path.join(HERE, f'loop15_c2_{SEQKEY}.txt'), 'w').write('\n'.join(out) + '\n')
