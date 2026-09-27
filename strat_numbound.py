"""S198: For every sign with >=8 tokens preceded by a numeral, is the numeral fixed (a lexical compound like '3-comb')
or does it vary (a real count)? Measure: share of the most common preceding numeral. Control: permute the numerals
among all numeral->sign bigrams 2000x and count signs that are this fixed by chance."""
import json,random,collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
b=json.load(open('data/derived/bridge_extended.json'))
NUM=set(range(1,8))|set(range(12,21))|set(range(25,30))|set(range(31,38))|{39,42,43,55,56}
pairs=[]
import sys
DEDUP='dedup' in sys.argv; seen=set()
for x in d:
    s=x['seq']
    if DEDUP:
        if tuple(s) in seen: continue
        seen.add(tuple(s))
    for i in range(1,len(s)):
        if s[i-1] in NUM and s[i] not in NUM: pairs.append((s[i-1],s[i],x['type'].split(':')[0]))
by=C.defaultdict(list)
for n,w,t in pairs: by[w].append(n)
def stats(by):
    out={}
    for w,ns in by.items():
        if len(ns)>=8:
            c=C.Counter(ns); out[w]=(c.most_common(1)[0][1]/len(ns),len(c),len(ns))
    return out
real=stats(by)
fixed=[w for w,(r,k,n) in real.items() if r>=0.85]
varied=[w for w,(r,k,n) in real.items() if r<0.5]
print('signs tested',len(real),'fixed(>=85%)',len(fixed),'varied(<50%)',len(varied))
for w,(r,k,n) in sorted(real.items(),key=lambda z:-z[1][0]):
    top=C.Counter(by[w]).most_common(3)
    print(w,b.get(str(w)),'n=%d top-share %.2f'%(n,r),top)
random.seed(1); ns=[n for n,w,t in pairs]; null=[]
for _ in range(2000):
    random.shuffle(ns); bb=C.defaultdict(list)
    for (n0,w,t),n in zip(pairs,ns): bb[w].append(n)
    null.append(sum(1 for r,k,n in stats(bb).values() if r>=0.85))
print('null mean fixed %.2f  P(>=real)=%.4f'%(sum(null)/2000,sum(v>=len(fixed) for v in null)/2000))
