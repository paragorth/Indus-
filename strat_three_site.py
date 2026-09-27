"""S209: Is the tall-3 / short-3 split (S208) a sign-class rule or a local scribal habit? Re-run the value-3 test inside
Mohenjo-daro and inside Harappa separately, and check that the same signs take the same form in both cities.
Control: permute form labels among value-3 tokens within each city 5000x."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
seen=set(); t=[]
for x in d:
    s=tuple(x['seq']); key=(s,x['site'])
    if key in seen: continue
    seen.add(key)
    for i in range(1,len(s)):
        if s[i-1] in (3,13,33):
            w='comb' if 422<=s[i]<=426 else s[i]
            if w in (3,13,33) or w in range(1,57) and w not in (55,56) and w<60: continue
            t.append((x['site'],w,'T' if s[i-1]==33 else 'S'))
def st(z):
    g=C.defaultdict(C.Counter)
    for c,w,l in z: g[w][l]+=1
    return sum(max(v.values())/sum(v.values()) for v in g.values() if sum(v.values())>=4),g
random.seed(17)
maj={}
for city in ('Mohenjo-daro','Harappa'):
    z=[k for k in t if k[0]==city]; r,g=st(z); lab=[k[2] for k in z]; nl=[]
    for _ in range(5000):
        random.shuffle(lab); nl.append(st([(c,w,l) for (c,w,_),l in zip(z,lab)])[0])
    print(city,'n',len(z),C.Counter(k[2] for k in z),'stat %.2f null %.2f P=%.4f'%(r,sum(nl)/5000,sum(v>=r for v in nl)/5000))
    for w,c in sorted(g.items(),key=lambda kv:-sum(kv[1].values())):
        if sum(c.values())>=4: print('  W%s'%w,dict(c)); maj.setdefault(w,{})[city]=c.most_common(1)[0][0]
both=[(w,v) for w,v in maj.items() if len(v)==2]
print('signs in both cities:',both,'agree',sum(v['Mohenjo-daro']==v['Harappa'] for w,v in both),'of',len(both))
