"""S229: Is the pre-jar title slot filled by fixed two-sign compounds? For each sign X occurring as '... A X 740' 15+ times, share of its
commonest A. Control: permute A among all (A,X) pairs 5000x; count X with top-share >= 0.5."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
seen=set(); pairs=[]
for x in d:
    s=tuple(x['seq'])
    if (s,x['site']) in seen: continue
    seen.add((s,x['site']))
    for i in range(2,len(s)):
        if s[i]==740: pairs.append((s[i-2],s[i-1]))
def stat(pairs):
    g=C.defaultdict(C.Counter)
    for a,x in pairs: g[x][a]+=1
    return {x:c.most_common(1)[0][1]/sum(c.values()) for x,c in g.items() if sum(c.values())>=15},g
r,g=stat(pairs); k=sum(v>=0.5 for v in r.values())
random.seed(97); A=[a for a,x in pairs]; nl=[]
for _ in range(5000):
    random.shuffle(A); nl.append(sum(v>=0.5 for v in stat(list(zip(A,[x for a,x in pairs])))[0].values()))
print('X tested',len(r),'with fixed partner (>=50%)',k,'null %.2f P=%.4f'%(sum(nl)/5000,sum(v>=k for v in nl)/5000))
for x,v in sorted(r.items(),key=lambda kv:-kv[1]): print(x,round(v,2),g[x].most_common(2),sum(g[x].values()))
