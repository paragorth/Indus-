"""S210: Do tall-3 terms and short-3 terms (S208/S209) fill different slots? For each value-3 pair (3|33 + sign), record what follows:
text end, jar W740, a suffix (W400/W90), or anything else. Control: permute the tall/short label among pairs 5000x (chi-square-like TVD)."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
seen=set(); t=[]
for x in d:
    s=tuple(x['seq']); key=(s,x['site'])
    if key in seen: continue
    seen.add(key)
    for i in range(1,len(s)-0):
        if s[i-1] in (3,33) and i<len(s) and not (s[i]<60):
            nx=s[i+1] if i+1<len(s) else None
            f='END' if nx is None else 'jar' if nx==740 else 'suffix' if nx in (400,90) else 'other'
            start='initial' if i==1 else 'after-opener' if s[i-2] in (817,861,820,2) else 'medial'
            t.append(('T' if s[i-1]==33 else 'S',f,start,x['type'].split(':')[0],s[i]))
def tvd(t,j):
    a=C.Counter(k[j] for k in t if k[0]=='T'); b=C.Counter(k[j] for k in t if k[0]=='S')
    A=sum(a.values()); B=sum(b.values())
    return 0.5*sum(abs(a[x]/A-b[x]/B) for x in set(a)|set(b)),a,b
random.seed(23)
for j,name in [(1,'follows'),(2,'precedes'),(3,'object')]:
    r,a,b=tvd(t,j); lab=[k[0] for k in t]; nl=[]
    for _ in range(5000):
        random.shuffle(lab); nl.append(tvd([(l,)+k[1:] for k,l in zip(t,lab)],j)[0])
    print(name,'TVD %.3f null %.3f P=%.4f'%(r,sum(nl)/5000,sum(v>=r for v in nl)/5000),'tall',dict(a),'short',dict(b))
# excluding W520 (dominant tall-3 sign) for robustness
t2=[k for k in t if k[4]!=520]
r,a,b=tvd(t2,1); lab=[k[0] for k in t2]; nl=[]
for _ in range(5000):
    random.shuffle(lab); nl.append(tvd([(l,)+k[1:] for k,l in zip(t2,lab)],1)[0])
print('follows, W520 excluded: TVD %.3f P=%.4f'%(r,sum(v>=r for v in nl)/5000),'tall',dict(a),'short',dict(b))
