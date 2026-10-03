"""S-DARK-34 cycle 4: EXCEPTIONLESS-RULE CENSUS. Instead of a learner, enumerate every candidate rule
'antecedent -> consequent' on seals where the antecedent is one fact about the rest of the object (a sign present,
a sign at the first / last / before-closer position, a closer unit, the emblem class, the sub-type, length bucket, a
numeral value, sum-of-indices mod k = r) and the consequent is the value of one target (last sign, closer, first
sign, emblem, material, boss, shape, numeral present, numeral value); keep rules with >= 10 training supports and
precision 1.0 on Mohenjo-daro + Harappa seals; test each on (a) other-site seals and (b) the IM77-only texts;
report survivors with 0 exceptions (bound < 3/n), and compare the count with the same census on (i) label-shuffled
seals (consequents permuted across objects within sub-type) and (ii) a planted corpus where emblem := f(closer).
Usage: python3 tools/dark_loop34_c4.py <seq_raw|seq_strong|seq_all>"""
import sys, json
sys.argv = [sys.argv[0], '4'] + sys.argv[1:]
from dark_loop34 import *

seals = [o for o in OBJ if o['ot'] == 'SEAL']
train = [o for o in seals if o['big']]; sites = [o for o in seals if not o['big']]; im77 = [o for o in NEW]
log = open(SP + f'loop34_c4_{LV}.txt', 'w')
def P(s): print(s); log.write(s + '\n'); log.flush()
EXCLN = {1, 2, 31}
def numerals(seq): return [VAL[t] for t in seq if t in NUMER and t not in EXCLN]

def facts(o, hide):
    """antecedent facts about the object with target field `hide` removed."""
    s = o['seq']; f = set()
    cl, cp = closer_of(s)
    for t in set(s): f.add(('has', t))
    f.add(('first', s[0])); f.add(('last', s[-1])); f.add(('len', min(len(s), 8)))
    if cp > 0: f.add(('before_closer', s[cp - 1]))
    f.add(('closer', cl)); f.add(('type', o['type']))
    for k in (3, 5, 7, 10): f.add((f'summod{k}', sum(abs(t) for t in s) % k))
    nv = numerals(s); f.add(('numeral', 'yes' if nv else 'no'))
    for v in nv: f.add(('numval', v))
    for fk in ('emblem', 'material', 'boss', 'shape'):
        if o[fk] is not None: f.add((fk, o[fk]))
    # remove facts that reveal the hidden target
    if hide == 'last_sign': f = {x for x in f if x[0] not in ('last', 'closer', 'before_closer') and not (x[0] == 'has' and x[1] == s[-1] and s.count(s[-1]) == 1) and not x[0].startswith('summod')}
    if hide == 'closer': f = {x for x in f if x[0] not in ('last', 'closer') and not (x[0] == 'has' and x[1] == cl) and not x[0].startswith('summod')}
    if hide == 'first_sign': f = {x for x in f if x[0] != 'first' and not (x[0] == 'has' and x[1] == s[0] and s.count(s[0]) == 1) and not x[0].startswith('summod')}
    if hide in ('emblem', 'material', 'boss', 'shape'): f = {x for x in f if x[0] != hide}
    if hide == 'numeral': f = {x for x in f if x[0] not in ('numeral', 'numval') and not (x[0] == 'has' and x[1] in NUMER) and not x[0].startswith('summod')}
    if hide == 'numval': f = {x for x in f if x[0] not in ('numval',) and not (x[0] == 'has' and x[1] in NUMER) and not x[0].startswith('summod')}
    return f

def target(o, name):
    s = o['seq']
    if name == 'last_sign': return s[-1]
    if name == 'closer': return closer_of(s)[0]
    if name == 'first_sign': return s[0]
    if name in ('emblem', 'material', 'boss', 'shape'): return o[name]
    if name == 'numeral': return 'yes' if numerals(s) else 'no'
    if name == 'numval': nv = numerals(s); return nv[0] if len(nv) == 1 else None

def census(train, tests, name, minsup=10, label_perm=None):
    tab = collections.defaultdict(collections.Counter)
    ys = [target(o, name) for o in train]
    if label_perm is not None: ys = label_perm(train, ys)
    for o, y in zip(train, ys):
        if y is None: continue
        for a in facts(o, name): tab[a][y] += 1
    rules = []
    for a, c in tab.items():
        n = sum(c.values()); y, k = c.most_common(1)[0]
        if n >= minsup and k == n: rules.append((a, y, n))
    out = []
    for a, y, n in sorted(rules, key=lambda r: -r[2]):
        res = {}
        for tn, objs in tests.items():
            hit = exc = 0
            for o in objs:
                t = target(o, name)
                if t is None or a not in facts(o, name): continue
                if t == y: hit += 1
                else: exc += 1
            res[tn] = (hit, exc)
        out.append((a, y, n, res))
    return out

TN = ['last_sign', 'closer', 'first_sign', 'emblem', 'material', 'boss', 'shape', 'numeral', 'numval']
tests = {'sites': sites, 'im77': im77}
def perm_within_type(objs, ys):
    g = collections.defaultdict(list)
    for i, o in enumerate(objs): g[o['type']].append(i)
    ys = list(ys)
    for idx in g.values():
        vals = [ys[i] for i in idx]; rnd.shuffle(vals)
        for i, v in zip(idx, vals): ys[i] = v
    return ys
P(f'# S-DARK-34 cycle 4, level {LV}: rule census on {len(train)} train seals; tests: sites {len(sites)} seals, IM77-only {len(im77)} texts')
summary = {}
for name in TN:
    R = census(train, tests, name)
    def is_trivial(a, y):
        # rules that restate the sub-type code or the frame definition
        return (name == 'closer' and a[0] == 'before_closer') or (name == 'shape' and a[0] == 'type') or (name == 'boss' and a[0] == 'type')
    surv = [(a, y, n, r) for a, y, n, r in R if all(r[t][1] == 0 for t in r) and sum(r[t][0] for t in r) >= 3]
    killed = [(a, y, n, r) for a, y, n, r in R if any(r[t][1] > 0 for t in r)]
    untested = [(a, y, n, r) for a, y, n, r in R if sum(r[t][0] + r[t][1] for t in r) < 3 and not any(r[t][1] > 0 for t in r)]
    P(f'--- target {name}: {len(R)} exceptionless training rules (support >= 10); survive both hold-outs with >= 3 cases: {len(surv)}; killed: {len(killed)}; untestable (< 3 held-out cases): {len(untested)}')
    for a, y, n, r in surv[:25]:
        P(f'    SURVIVES  {a} -> {name}={y}: train {n}/{n}; sites {r["sites"][0]}/{sum(r["sites"])}, IM77 {r["im77"][0]}/{sum(r["im77"])} (bound < 3/{sum(r["sites"])+sum(r["im77"])} = {3/max(1,sum(r["sites"])+sum(r["im77"])):.2f}){"  [restates type/frame]" if is_trivial(a, y) else ""}')
    for a, y, n, r in killed[:12]:
        P(f'    killed    {a} -> {name}={y}: train {n}/{n}; sites {r["sites"][0]}/{sum(r["sites"])}, IM77 {r["im77"][0]}/{sum(r["im77"])}')
    # shuffled-label census (3 reps) and its survivors
    sh = []
    for rep in range(3):
        Rs = census(train, tests, name, label_perm=perm_within_type)
        sh.append((len(Rs), sum(1 for a, y, n, r in Rs if all(r[t][1] == 0 for t in r) and sum(r[t][0] for t in r) >= 3)))
    P(f'    shuffled-label census (3 reps): training rules {[x[0] for x in sh]}, survivors {[x[1] for x in sh]}')
    summary[name] = dict(rules=len(R), survivors=len(surv), killed=len(killed), untested=len(untested), shuffled=sh,
                         surv_list=[(str(a), str(y), n, r) for a, y, n, r in surv])
# planted: emblem := f(closer) on every object (train and tests), then census
P('--- PLANTED: emblem := lookup[closer] on all objects')
LOOK = {c: e for c, e in zip(['none'] + CLOSERS, ['unicorn', 'zebu', 'elephant', 'tiger', 'rhino', 'buffalo', 'goat', 'gharial', 'shorthorn', 'other', 'none', 'unicorn', 'zebu'])}
import copy
def plant(objs):
    out = []
    for o in objs:
        o2 = dict(o); o2['emblem'] = LOOK[closer_of(o['seq'])[0]]; out.append(o2)
    return out
Rp = census(plant(train), {'sites': plant(sites), 'im77': plant(im77)}, 'emblem')
survp = [(a, y, n, r) for a, y, n, r in Rp if all(r[t][1] == 0 for t in r) and sum(r[t][0] for t in r) >= 3]
P(f'    planted emblem rules: {len(Rp)} training, {len(survp)} survive; closer-based survivors: {[(a, y, n) for a, y, n, r in survp if a[0] in ("closer", "last")][:12]}')
summary['planted_emblem'] = dict(rules=len(Rp), survivors=len(survp))
json.dump(summary, open(SP + f'loop34_c4_{LV}.json', 'w'), indent=1, default=str)
