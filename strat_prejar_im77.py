"""S230: IM77 replication of S229 (pre-jar fixed compounds). '... A X M342'; X with 15+ cases; share of commonest A. Control: permute A 5000x."""
import csv,collections as C,random
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv'))); seen=set(); pairs=[]
for r in rows:
    s=tuple(int(t) for t in r['signs_clean'].split() if t.isdigit() and t!='0')
    if s in seen: continue
    seen.add(s)
    for i in range(2,len(s)):
        if s[i]==342: pairs.append((s[i-2],s[i-1]))
def stat(p):
    g=C.defaultdict(C.Counter)
    for a,x in p: g[x][a]+=1
    return {x:c.most_common(1)[0][1]/sum(c.values()) for x,c in g.items() if sum(c.values())>=15},g
r,g=stat(pairs); k=sum(v>=0.5 for v in r.values()); random.seed(101); A=[a for a,x in pairs]; nl=[]
for _ in range(5000):
    random.shuffle(A); nl.append(sum(v>=0.5 for v in stat(list(zip(A,[x for a,x in pairs])))[0].values()))
print('X tested',len(r),'fixed',k,'null %.2f P=%.4f'%(sum(nl)/5000,sum(v>=k for v in nl)/5000))
for x,v in sorted(r.items(),key=lambda kv:-kv[1])[:10]: print('M%d'%x,round(v,2),g[x].most_common(2),sum(g[x].values()))
