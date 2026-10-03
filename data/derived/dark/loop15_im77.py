"""Loop 15: transcription-robust check of the candidate gap rules in IM77 (Mahadevan 1977 lines, site+text dedup,
damaged lines dropped).  For each rule: observed presence count, expectation under a model-free null in which sign tokens
are permuted across texts within site x object-type strata keeping line lengths (NPERM corpora), and the chance of 0.
Rules are given in Mahadevan numbers via data/derived/bridge_extended.json.
"""
import csv, collections, random, json, os, sys
ROOT = '/home/user/Indus-'
rows = list(csv.DictReader(open(os.path.join(ROOT, 'data/im77/im77_corpus_lines.csv'))))
sides = collections.defaultdict(list)
for r in rows: sides[(r['text_no'], r['side'])].append(r)
seen = set(); T = []
for k, ls in sides.items():
    ls = sorted(ls, key=lambda r: int(r['line'])); s = []
    for l in ls: s += l['signs_clean'].split()
    if len(s) < 2 or any(x in ('0', '000') or not x.isdigit() for x in s): continue
    key = (ls[0]['site'], tuple(s))
    if key in seen: continue
    seen.add(key); T.append((ls[0]['site'], ls[0]['object_type'], tuple(int(x) for x in s)))
print('IM77 texts (dedup, >=2 signs, undamaged):', len(T))

def has(seq, rule):
    kind = rule[0]; a = rule[1:]
    if kind == 'co':
        return a[0] in seq and (a[1] in seq if a[0] != a[1] else seq.count(a[0]) >= 2)
    if kind == 'ord':   # a0 before a1 with gap >= 1
        return any(seq[i] == a[0] and seq[j] == a[1] for i in range(len(seq)) for j in range(i + 2, len(seq)))
    if kind == 'any':   # a0 anywhere before a1 (adjacent allowed)
        return any(seq[i] == a[0] and seq[j] == a[1] for i in range(len(seq)) for j in range(i + 1, len(seq)))
    if kind == 'tri':
        return any(seq[i] == a[0] and seq[j] == a[1] and seq[k] == a[2]
                   for i in range(len(seq)) for j in range(i + 1, len(seq)) for k in range(j + 1, len(seq)))

RULES = [
    ('W2 twice in one text (M99 x2)', ('co', 99, 99)),
    ('W2 ... W741 marked jar, gap>=1 (M99 ... M343)', ('ord', 99, 343)),
    ('W2 before W741 at any distance (M99 .. M343)', ('any', 99, 343)),
    ('W2 ... W745 (M99 ... M345)', ('any', 99, 345)),
    ('W240 ... W2 with gap (M67 ... M99)', ('ord', 67, 99)),
    ('W240 twice (M67 x2)', ('co', 67, 67)),
    ('W240 ... W900 with gap (M67 ... M287)', ('ord', 67, 287)),
    ('W235 with W700 (M65 & M328)', ('co', 65, 328)),
    ('W60 with W861/817 opener (M123 & M267)', ('co', 123, 267)),
    ('W705 ... W700 (M336 ... M328)', ('any', 336, 328)),
    ('W741 ... W740 ... W400 (M343, M342, M176)', ('tri', 343, 342, 176)),
    ('W220 ... W176 ... W400 (M59, M48, M176)', ('tri', 59, 48, 176)),
    ('W2 ... W240 ... W400 (M99, M67, M176)', ('tri', 99, 67, 176)),
    ('W32 ... W220 ... W400 (M87, M59, M176)', ('tri', 87, 59, 176)),
    ('W240 ... W407 ... W740 (M67, M169, M342)', ('tri', 67, 169, 342)),
    ('W240 ... W61 ... W740 (M67, M124/125, M342)', ('tri', 67, 124, 342)),
    ('W501 with W740 (M210 & M342)', ('co', 210, 342)),
    ('W405/390 tree with W482 (M162 & M135)', ('co', 162, 135)),
    ('W741 ... W176 ... W400 (M343, M48, M176)', ('tri', 343, 48, 176)),
]
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = random.Random(7)
strata = collections.defaultdict(list)
for i, (site, ot, s) in enumerate(T): strata[(site, ot)].append(i)
obs = {name: sum(has(s, r) for _, _, s in T) for name, r in RULES}
tot = collections.Counter(); zero = collections.Counter()
for _ in range(NPERM):
    new = [None] * len(T)
    for idx in strata.values():
        pool = [x for i in idx for x in T[i][2]]; rng.shuffle(pool); p = 0
        for i in idx:
            L = len(T[i][2]); new[i] = pool[p:p + L]; p += L
    for name, r in RULES:
        c = sum(has(s, r) for s in new); tot[name] += c
        if c == 0: zero[name] += 1
print(f'{"rule":<55} obs  E_perm  P(0|null)')
for name, r in RULES:
    print(f'{name:<55} {obs[name]:3d}  {tot[name]/NPERM:6.1f}  {zero[name]/NPERM:.3f}')
