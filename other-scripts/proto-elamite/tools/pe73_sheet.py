#!/usr/bin/env python3
"""pe73: build blind check sheets with PLANTED errors.
Each numeral reading (validation tablets and key tablets) is shown either as transliterated (true) or with one
planted change (count +-1, unit swap, unit dropped/added); the reader marks AGREE / DISAGREE / UNREADABLE from the
CDLI photo (line art used only to locate the line).  Sign-presence questions in the validation set are true or
decoy (sign taken from another line of the same tablet / a common sign).  Keys are written to *_key.json and are
opened only by pe73_score.py.  Plant rate: validation 50%, key 25%."""
import json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data'); CK = os.path.join(DATA, 'pe73_ckpt')
C = {t['id']: t for t in json.load(open(os.path.join(DATA, 'pe_corpus.json')))}
SWAP = {'N01': 'N14', 'N14': 'N01', 'N39B': 'N24', 'N24': 'N39B', 'N30C': 'N30D', 'N30D': 'N30C', 'N34': 'N14', 'N39C': 'N30D'}

def fmt(nums):
    return ' '.join('%d(%s)' % (n, s) for n, s in nums)

def corrupt(nums, rng):
    nums = [list(x) for x in nums]
    k = rng.randrange(len(nums))
    m = rng.choice(['inc', 'dec', 'swap', 'add'])
    if m == 'dec' and nums[k][0] <= 1: m = 'inc'
    if m == 'inc': nums[k][0] += 1
    elif m == 'dec': nums[k][0] -= 1
    elif m == 'swap':
        nums[k][1] = SWAP.get(nums[k][1], 'N14')
    else:
        u = 'N14' if all(u != 'N14' for _, u in nums) else 'N01'
        if any(x[1] == u for x in nums): nums[[x[1] for x in nums].index(u)][0] += 1
        else: nums.insert(0 if u == 'N14' else len(nums), [1, u])
    out = []
    for n, u in nums:
        for o in out:
            if o[1] == u: o[0] += n; break
        else:
            out.append([n, u])
    return m, fmt([o for o in out if o[0] > 0])

def build(items, rate, seed, name):
    rng = random.Random(seed)
    Q, K = [], []
    for it in items:
        L = C[it['pid']]['lines'][it['idx']]
        true = fmt(L['numerals'])
        plant = rng.random() < rate
        shown, m = true, None
        if plant:
            m, shown = corrupt(L['numerals'], rng)
            if shown == true: plant, m = False, None
        Q.append(dict(pid=it['pid'], surface=L['surface'], col=L['column'], label=L['label'], shown=shown, n_signs=len(L['signs'])))
        K.append(dict(true=true, plant=plant, how=m, result=it.get('result', 'val')))
    json.dump(Q, open(os.path.join(CK, name + '_q.json'), 'w'), indent=0)
    json.dump(K, open(os.path.join(CK, name + '_key.json'), 'w'))
    return Q

# validation (second draw; the first draw's sheet was exposed with giveaway plants and is discarded):
# 22 random MDP tablets with >= 3 numeral lines, not key tablets, not in the first draw
first = {q['pid'] for q in json.load(open(os.path.join(CK, 'val_q.json')))}
keyt = {c['pid'] for c in json.load(open(os.path.join(CK, 'claims.json')))}
pool = sorted(p for p, t in C.items() if p not in first | keyt and t['designation'].startswith('MDP')
              and sum(bool(l['numerals']) for l in t['lines']) >= 3 and len(t['lines']) <= 12)
vt = random.Random('pe73-val2').sample(pool, 22)
json.dump(vt, open(os.path.join(CK, 'val2_tablets.json'), 'w'))
items = [dict(pid=p, idx=i) for p in vt for i, l in enumerate(C[p]['lines']) if l['numerals']]
val = [dict(pid=p) for p in vt]
build(items, 0.5, 'pe73-val2', 'vnum')
# key: every NUM claim
cl = [c for c in json.load(open(os.path.join(CK, 'claims.json'))) if c['kind'] == 'NUM']
build(cl, 0.25, 'pe73-key', 'knum')
# validation sign questions: 2 per tablet, one true, one decoy, order random
rng = random.Random('pe73-vsign'); Q, K = [], []
COMMON = ['M288', 'M157', 'M388', 'M218', 'M124', 'M009', 'M003', 'M054']
for p in sorted({q['pid'] for q in val}):
    L = C[p]['lines']
    cand = [(j, s) for j, l in enumerate(L) for s in l['signs'] if s in COMMON]
    if not cand: continue
    j, s = rng.choice(cand)
    others = [jj for jj, l in enumerate(L) if l['signs'] and s not in l['signs'] and jj != j]
    pairs = [(j, s, True)]
    if others: pairs.append((rng.choice(others), s, False))
    rng.shuffle(pairs)
    for jj, ss, tr in pairs:
        Q.append(dict(pid=p, surface=L[jj]['surface'], col=L[jj]['column'], label=L[jj]['label'], sign=ss)); K.append(tr)
json.dump(Q, open(os.path.join(CK, 'vsign_q.json'), 'w'), indent=0); json.dump(K, open(os.path.join(CK, 'vsign_key.json'), 'w'))
print('vnum', len(items), 'knum', len(cl), 'vsign', len(Q))
