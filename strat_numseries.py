"""S204: Two stroke-numeral series (glyph stroke counts, docs/numerals.png): SHORT (W1-7 one row = 1-7; W12-20 two tiers = 2-10; W25-29 = 5-9 clusters)
and TALL (W32-39 = 2-9 long strokes; W31 = M86 excluded as a frame marker). Does each following sign take one series only
(like proto-cuneiform commodity-specific number systems)? Deduplicated text+site. Control: permute series labels among
numeral->sign tokens 5000x; statistic = sum over signs (8+ tokens) of the majority-series share excess."""
import json,random,collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
SHORT=set(range(1,8))|set(range(12,21))|set(range(25,30))
SHORT-={2}   # W2 = M99, opener marker
TALL=set(range(32,38))|{39}
VAL={**{i:i for i in range(1,8)},**{i:i-10 for i in range(12,21)},**{i:i-20 for i in range(25,30)},**{i:i-30 for i in range(32,38)},39:9}
seen=set(); tok=[]
for x in d:
    s=tuple(x['seq']); key=(s,x['site'])
    if key in seen: continue
    seen.add(key)
    for i in range(1,len(s)):
        n=s[i-1]; w='comb' if 422<=s[i]<=426 else s[i]
        if n in SHORT|TALL and s[i] not in SHORT|TALL|{2,31,55,56}:
            tok.append((w,'T' if n in TALL else 'S',VAL[n]))
def stat(tok):
    g=C.defaultdict(C.Counter)
    for w,sr,v in tok: g[w][sr]+=1
    tot=0; out={}
    for w,c in g.items():
        n=sum(c.values())
        if n>=8: m=max(c.values())/n; out[w]=(c['S'],c['T']); tot+=m
    return tot,out
real,out=stat(tok)
base=C.Counter(sr for w,sr,v in tok); print('tokens',len(tok),base)
for w,(s,t) in sorted(out.items(),key=lambda z:-(z[1][0]+z[1][1])):
    vs=C.Counter(v for ww,sr,v in tok if ww==w)
    print(w,'short',s,'tall',t,'values',dict(sorted(vs.items())))
random.seed(9); labs=[(sr,v) for w,sr,v in tok]; null=[]
for _ in range(5000):
    random.shuffle(labs); null.append(stat([(w,sr,v) for (w,_,_),(sr,v) in zip(tok,labs)])[0])
print('sum majority share %.1f null %.1f P=%.4f'%(real,sum(null)/5000,sum(v>=real for v in null)/5000))
mixed=[w for w,(s,t) in out.items() if min(s,t)/(s+t)>=0.2]
print('signs taking both series (>=20% minority):',mixed)

# S208: control for value. Permute series labels only among tokens of the same numeral value.
byv=C.defaultdict(list)
for i,(w,sr,v) in enumerate(tok): byv[v].append(i)
null2=[]
for _ in range(3000):
    lab=[sr for w,sr,v in tok]
    for v,ix in byv.items():
        p=[lab[i] for i in ix]; random.shuffle(p)
        for i,x in zip(ix,p): lab[i]=x
    null2.append(stat([(w,l,v) for (w,_,v),l in zip(tok,lab)])[0])
print('value-stratified: stat %.1f null %.1f P=%.4f'%(real,sum(null2)/3000,sum(x>=real for x in null2)/3000))
for v in sorted(byv): print('value',v,C.Counter(tok[i][1] for i in byv[v]))

# S208b: value 3 only (the one value drawn equally often in both forms)
t3=[(w,sr,v) for w,sr,v in tok if v==3]
g=C.defaultdict(C.Counter)
for w,sr,v in t3: g[w][sr]+=1
def s3(t):
    g=C.defaultdict(C.Counter)
    for w,sr,v in t: g[w][sr]+=1
    return sum(max(c.values())/sum(c.values()) for c in g.values() if sum(c.values())>=5)
r3=s3(t3); lab=[sr for w,sr,v in t3]; n3=[]
for _ in range(5000):
    random.shuffle(lab); n3.append(s3([(w,l,v) for (w,_,v),l in zip(t3,lab)]))
print('value 3 only: stat %.2f null %.2f P=%.4f'%(r3,sum(n3)/5000,sum(x>=r3 for x in n3)/5000))
for w,c in sorted(g.items(),key=lambda kv:-sum(kv[1].values())):
    if sum(c.values())>=5: print(' W%s'%w,dict(c))
