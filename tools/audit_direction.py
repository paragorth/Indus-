"""Audit: do texts marked L/R, BUS, NR, '-' look normalised to reading order?
Frame markers: openers W817/861/820 should be initial, jar W740 final.
Also: direction mix of the 25 opener-final texts (S260/S286). Control: permutation of dir labels."""
import json, random, collections
d = json.load(open('/home/user/Indus-/data/derived/merged-corpus-canonical.json'))
OP = {817, 861, 820}; JAR = 740
def norm(x): return x.strip().upper()
rows = [x for x in d if len(x['seq_raw']) >= 3]
def stats(sub):
    n = len(sub)
    of = sum(s['seq_raw'][0] in OP for s in sub); ol = sum(s['seq_raw'][-1] in OP for s in sub)
    jf = sum(s['seq_raw'][0] == JAR for s in sub); jl = sum(s['seq_raw'][-1] == JAR for s in sub)
    return n, of, ol, jf, jl
grp = collections.defaultdict(list)
for x in rows: grp[norm(x['dir.'])].append(x)
print('dir n opener_first opener_last jar_first jar_last')
for k, v in sorted(grp.items(), key=lambda t: -len(t[1])):
    print(k, *stats(v))
# Opener-final texts: direction mix vs opener-initial
ofin = [x for x in rows if x['seq_raw'][-1] in OP]
oini = [x for x in rows if x['seq_raw'][0] in OP]
print('opener-final dirs', collections.Counter(norm(x['dir.']) for x in ofin))
print('opener-initial dirs', collections.Counter(norm(x['dir.']) for x in oini).most_common(6))
# 'reversal index' per text: position-based statistic. Use all texts containing both an opener and jar
both = [x for x in rows if (set(x['seq_raw']) & OP) and JAR in x['seq_raw']]
def rev(x):
    s = x['seq_raw']; io = min(i for i, c in enumerate(s) if c in OP); ij = max(i for i, c in enumerate(s) if c == JAR)
    return io > ij
obs = collections.Counter((norm(x['dir.']), rev(x)) for x in both)
print('texts with opener+jar: (dir, jar-before-opener) counts', dict(obs))
# permutation: is 'reversed' rate in non-R/L higher than R/L?
lab = [norm(x['dir.']) != 'R/L' for x in both]; r = [rev(x) for x in both]
def stat(l): 
    a = [ri for li, ri in zip(l, r) if li]; b = [ri for li, ri in zip(l, r) if not li]
    return (sum(a)/max(1,len(a))) - (sum(b)/max(1,len(b)))
o = stat(lab); random.seed(1); ge = 0
for _ in range(10000):
    random.shuffle(lab); ge += stat(lab) >= o
print('reversed-rate diff nonRL-RL = %.3f, perm P = %.4f' % (o, (ge+1)/10001))
