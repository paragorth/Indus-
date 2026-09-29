"""Audit: is W2 (two short strokes) a connective or the numeral 2?
Compare the next-sign distribution of W2 (split: right after an opener vs elsewhere)
with short numerals W3/W4/W5 and with connective W60. Cosine similarity; null = cosine of
W3 vs frequency-matched random signs (500 draws). Site+text dedup."""
import json, random, collections, math
d = json.load(open('/home/user/Indus-/data/derived/merged-corpus-canonical.json'))
OP = {817, 861, 820}
seen = set(); texts = []
for x in d:
    k = (x['site'], tuple(x['seq_raw']))
    if k in seen: continue
    seen.add(k); texts.append(x['seq_raw'])
def nextdist(pred):
    c = collections.Counter()
    for s in texts:
        for i in range(len(s)):
            if pred(s, i): c[s[i+1] if i+1 < len(s) else 'END'] += 1
    return c
def cos(a, b):
    ks = set(a) | set(b); num = sum(a[k]*b[k] for k in ks)
    return num / math.sqrt(sum(v*v for v in a.values()) * sum(v*v for v in b.values()))
W2_op = nextdist(lambda s, i: s[i] == 2 and i > 0 and s[i-1] in OP)
W2_else = nextdist(lambda s, i: s[i] == 2 and not (i > 0 and s[i-1] in OP))
sig = {n: nextdist(lambda s, i, n=n: s[i] == n) for n in (3, 4, 5, 60, 1, 32)}
print('W2 after opener n=%d, elsewhere n=%d' % (sum(W2_op.values()), sum(W2_else.values())))
print('W2 elsewhere top next:', W2_else.most_common(12))
print('W3 top next:', sig[3].most_common(10))
for name, v in [('W2_else', W2_else), ('W2_op', W2_op)]:
    print(name, ' '.join('vs W%s %.2f' % (n, cos(v, sig[n])) for n in sig))
print('W3 vs W4 %.2f, W3 vs W5 %.2f, W4 vs W5 %.2f' % (cos(sig[3], sig[4]), cos(sig[3], sig[5]), cos(sig[4], sig[5])))
# Null: cosine between W2_else and random signs with 50-400 tokens
freq = collections.Counter(c for s in texts for c in s)
pool = [k for k, v in freq.items() if 50 <= v <= 400 and k not in (2, 3, 4, 5)]
random.seed(0)
null = sorted(cos(W2_else, nextdist(lambda s, i, n=n: s[i] == n)) for n in pool)
o = cos(W2_else, sig[3])
print('null (W2_else vs %d peer signs): median %.2f, 95%% %.2f, max %.2f; rank of W3: %d/%d above' % (
    len(pool), null[len(null)//2], null[int(.95*len(null))], null[-1], sum(v >= o for v in null), len(null)))
# counted-item share after each numeral
COUNTED = {740, 220, 390, 405, 900, 700, 407, 904, 384, 645}
for name, v in [('W2_else', W2_else), ('W2_op', W2_op), ('W3', sig[3]), ('W4', sig[4]), ('W5', sig[5]), ('W60', sig[60])]:
    t = sum(v.values()); print('%s -> counted item %.2f' % (name, sum(v[k] for k in COUNTED)/t))
