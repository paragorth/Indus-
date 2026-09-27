"""S210b: IM77 replication of S210. Tall-3 = M89, short-3 = M102/103; opener frame = M267, M391, M99. Control as S210."""
import csv,collections as C,random
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
seen=set(); t=[]
for r in rows:
    s=tuple(int(x) for x in r['signs_clean'].split() if x.isdigit() and x!='0')
    if s in seen: continue
    seen.add(s)
    for i in range(1,len(s)):
        if s[i-1] in (89,102,103) and not 86<=s[i]<=122:
            start='initial' if i==1 else 'after-opener' if s[i-2] in (267,391,99) else 'medial'
            t.append(('T' if s[i-1]==89 else 'S',start))
def tvd(t):
    a=C.Counter(k[1] for k in t if k[0]=='T'); b=C.Counter(k[1] for k in t if k[0]=='S')
    return 0.5*sum(abs(a[x]/sum(a.values())-b[x]/sum(b.values())) for x in set(a)|set(b)),a,b
r,a,b=tvd(t); lab=[k[0] for k in t]; random.seed(5); nl=[]
for _ in range(5000):
    random.shuffle(lab); nl.append(tvd([(l,k[1]) for k,l in zip(t,lab)])[0])
print('TVD %.3f null %.3f P=%.4f'%(r,sum(nl)/5000,sum(v>=r for v in nl)/5000),'tall',dict(a),'short',dict(b))
