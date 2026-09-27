"""S201: Replicate S197-S198 in the Mahadevan transcription (IM77). Numeral signs M87-M122 (M86 and M99 excluded: frame markers).
Deduplicated lines. For signs preceded by a numeral 8+ times: share of the top numeral. Control: permute numerals across bigrams 2000x."""
import csv,random,collections as C,json
NUM=set(range(87,123))-{99}
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
seen=set(); pairs=[]
for r in rows:
    s=tuple(int(x) for x in r['signs_clean'].split() if x.isdigit() and x!='0')
    if s in seen: continue
    seen.add(s)
    for i in range(1,len(s)):
        if s[i-1] in NUM and s[i] not in NUM: pairs.append((s[i-1],s[i]))
by=C.defaultdict(list)
for n,w in pairs: by[w].append(n)
def stats(by): return {w:(C.Counter(v).most_common(1)[0][1]/len(v),len(v)) for w,v in by.items() if len(v)>=8}
real=stats(by); fixed=[w for w,(r,n) in real.items() if r>=0.85]
print('tested',len(real),'fixed',len(fixed))
b=json.load(open('data/derived/bridge_extended.json')); inv={}
for w,ms in b.items():
    for m in ms: inv.setdefault(m,[]).append(int(w))
for w,(r,n) in sorted(real.items(),key=lambda z:-z[1][0]):
    print('M%d W%s n=%d top %.2f'%(w,inv.get(w),n,r),C.Counter(by[w]).most_common(3))
random.seed(5); ns=[n for n,w in pairs]; null=[]
for _ in range(2000):
    random.shuffle(ns); bb=C.defaultdict(list)
    for (n0,w),n in zip(pairs,ns): bb[w].append(n)
    null.append(sum(1 for r,n in stats(bb).values() if r>=0.85))
print('null mean %.2f P=%.4f'%(sum(null)/2000,sum(v>=len(fixed) for v in null)/2000))
