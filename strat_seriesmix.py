"""S205: Do the two numeral series (S204) avoid each other within one text? Control: permute series labels (short/tall, keeping values)
among all numeral tokens within each object class 3000x; count texts containing both series."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
SHORT=(set(range(1,8))|set(range(12,21))|set(range(25,30)))-{2}; TALL=set(range(32,38))|{39}
seen=set(); texts=[]
for x in d:
    s=tuple(x['seq']); key=(s,x['site'])
    if key in seen: continue
    seen.add(key)
    lab=['S' if w in SHORT else 'T' for w in s if w in SHORT|TALL]
    if len(lab)>=2: texts.append((x['type'].split(':')[0],lab))
real=sum(1 for t,l in texts if len(set(l))==2)
by=C.defaultdict(list)
for i,(t,l) in enumerate(texts): by[t].append(i)
random.seed(11); null=[]
for _ in range(3000):
    c=0
    for t,ix in by.items():
        pool=[z for i in ix for z in texts[i][1]]; random.shuffle(pool); k=0
        for i in ix:
            n=len(texts[i][1]); seg=pool[k:k+n]; k+=n; c+=len(set(seg))==2
    null.append(c)
print('texts with 2+ numerals',len(texts),'mixed',real,'null mean %.1f P(<=)=%.4f'%(sum(null)/3000,sum(v<=real for v in null)/3000))
