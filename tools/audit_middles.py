"""Audit: is middle uniqueness evidence for personal names, or just combinatorics?
Seal texts (SEAL*), strip frame: leading opener(+W2/W60), trailing jar/paradigm closer (+W400/W90).
Measure: share of middle tokens (seal instances, len>=2) whose middle string occurs on only one seal.
Null: bigram (Markov) model trained on the same middles, same length distribution, 200 corpora.
Names in real populations recur (common names), so real names should be LESS unique than a bigram generator."""
import json, random, collections
d = json.load(open('/home/user/Indus-/data/derived/merged-corpus-canonical.json'))
OP = {817, 861, 820}; MARK = {2, 60}
CLOSE = {740, 520, 151, 156, 527, 226, 617, 154, 158, 700, 236}; SUF = {400, 90}
mids = []
for x in d:
    if not x['type'].startswith('SEAL'): continue
    s = list(x['seq_raw'])
    if s and s[-1] in SUF: s = s[:-1]
    if s and s[-1] in CLOSE: s = s[:-1]
    if s and s[0] in OP: s = s[1:]
    if s and s[0] in MARK: s = s[1:]
    if len(s) >= 2: mids.append(tuple(s))
def uniq(ms):
    c = collections.Counter(ms); return sum(1 for m in ms if c[m] == 1) / len(ms)
def top_rep(ms):
    c = collections.Counter(ms); return c.most_common(1)[0][1]
o = uniq(ms := mids)
big = collections.defaultdict(collections.Counter)
for m in mids:
    p = 'S'
    for c in m: big[p][c] += 1; p = c
def gen(n):
    out = []; p = 'S'
    for _ in range(n):
        src = big[p] if big[p] else big['S']
        ks, ws = zip(*src.items()); c = random.choices(ks, ws)[0]; out.append(c); p = c
    return tuple(out)
random.seed(0); nu = []; nt = []
L = [len(m) for m in mids]
for _ in range(200):
    g = [gen(n) for n in L]; nu.append(uniq(g)); nt.append(top_rep(g))
nu.sort()
print('seal middles n=%d; observed unique share %.3f; bigram null median %.3f [2.5%% %.3f, 97.5%% %.3f]; P(null<=obs)=%.3f'
      % (len(mids), o, nu[100], nu[5], nu[194], sum(v <= o for v in nu)/200))
print('most repeated middle: obs %d copies %s; null median max %d' % (top_rep(mids), collections.Counter(mids).most_common(3), sorted(nt)[100]))
for k in (2, 3, 4):
    sub = [m for m in mids if len(m) == k]; print('len %d: n=%d unique %.3f' % (k, len(sub), uniq(sub)))
