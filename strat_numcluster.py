"""S233: Adjacent numeral clusters (two numeral signs in a row). Is the order fixed (e.g. tall-stroke group before short-stroke group, like
tens before units), and does the first element take a limited set of values? W2 (opener marker M99) and W31 (M86 marker) excluded;
W55/56 kept as short (12, 24). Control: binomial for tall-first vs short-first among mixed pairs; by object type."""
import json,collections as C,math
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
SHORT=(set(range(1,8))|set(range(12,21))|set(range(25,30))|{55,56})-{2}; TALL=set(range(32,38))|{39}
VAL={**{i:i for i in range(1,8)},**{i:i-10 for i in range(12,21)},**{i:i-20 for i in range(25,30)},**{i:i-30 for i in range(32,38)},39:9,55:12,56:24}
seen=set(); pairs=C.Counter(); byobj=C.defaultdict(C.Counter); ex=C.defaultdict(list)
for x in d:
    s=tuple(x['seq'])
    if (s,x['site']) in seen: continue
    seen.add((s,x['site']))
    t=x['type'].split(':')[0]
    for i in range(1,len(s)):
        a,b=s[i-1],s[i]
        if a in SHORT|TALL and b in SHORT|TALL:
            k=('T' if a in TALL else 'S')+('T' if b in TALL else 'S')
            pairs[k]+=1; byobj[t][k]+=1; ex[k].append((VAL[a],VAL[b],t,s))
print(pairs); print({t:dict(c) for t,c in byobj.items()})
n=pairs['TS']+pairs['ST']; k=pairs['TS']
p=2*sum(math.comb(n,i) for i in range(max(k,n-k),n+1))/2**n
print('tall-first %d vs short-first %d, two-sided P=%.2g'%(k,n-k,min(1,p)))
for kk in ('TS','ST','SS','TT'):
    print(kk,C.Counter((a,b) for a,b,t,s in ex[kk]).most_common(8))
    for a,b,t,s in ex[kk][:4]: print('   ',t,s)
