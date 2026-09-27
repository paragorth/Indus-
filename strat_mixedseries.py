"""S208: For the signs that take both numeral series (jar W740, fish W220; S204), what decides the series?
Candidates: object type, site, the value, what comes before the numeral. Control: permute series labels within sign 5000x."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
SHORT=(set(range(1,8))|set(range(12,21))|set(range(25,30)))-{2}; TALL=set(range(32,38))|{39}
VAL={**{i:i for i in range(1,8)},**{i:i-10 for i in range(12,21)},**{i:i-20 for i in range(25,30)},**{i:i-30 for i in range(32,38)},39:9}
seen=set(); tok=[]
for x in d:
    s=tuple(x['seq']); key=(s,x['site'])
    if key in seen: continue
    seen.add(key)
    for i in range(1,len(s)):
        if s[i] in (740,220) and s[i-1] in SHORT|TALL:
            tok.append(dict(w=s[i],ser='T' if s[i-1] in TALL else 'S',val=VAL[s[i-1]],obj=x['type'].split(':')[0],
              site=x['site'] if x['site'] in ('Harappa','Mohenjo-daro') else 'other',initial=(i==1),nxt=s[i+1] if i+1<len(s) else 'END'))
random.seed(21)
def assoc(z,f):
    tab=C.defaultdict(C.Counter)
    for t in z: tab[t[f]][t['ser']]+=1
    n=len(z); pT=sum(t['ser']=='T' for t in z)/n
    return sum(abs(c['T']-pT*sum(c.values())) for c in tab.values()),tab
for w in (740,220):
    z=[t for t in tok if t['w']==w]
    print('== W%d n=%d'%(w,len(z)))
    for f in ('obj','site','val','initial','nxt'):
        obs,tab=assoc(z,f); sers=[t['ser'] for t in z]; null=[]
        for _ in range(3000):
            random.shuffle(sers); null.append(assoc([dict(t,ser=s) for t,s in zip(z,sers)],f)[0])
        print(f,'P=%.4f'%(sum(v>=obs for v in null)/3000),{k:dict(v) for k,v in sorted(tab.items(),key=lambda kv:-sum(kv[1].values()))[:6]})
