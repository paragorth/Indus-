"""S215: Two-faced objects (ids n.1, n.2...). When one face carries a count (numeral + counted sign or numeral alone) and another face carries
a text, does the count VALUE depend on the other face's text? Measure: for text faces that recur on 3+ objects, variance of the count value
within text vs between. Control: permute count values among objects 5000x (statistic: number of recurring texts whose count is constant)."""
import csv,collections as C,random,re
r=list(csv.DictReader(open('data/raw/inscriptions.csv')))
NUM={**{i:i for i in range(1,8)},**{i:i-10 for i in range(12,21)},**{i:i-20 for i in range(25,30)},**{i:i-30 for i in range(32,38)},39:9}
objs=C.defaultdict(list)
for x in r:
    oid,_,face=x['id'].partition('.')
    seq=[int(t) for t in re.findall(r'\d+',x['text'])]
    objs[oid].append((x['site'],x['type'],seq))
pairs=[]
for oid,fs in objs.items():
    if len(fs)<2: continue
    cnt=[f for f in fs if f[2] and all(w in NUM or w in (740,220,700,390,900,405,176,100,70,74) for w in f[2]) and any(w in NUM for w in f[2])]
    txt=[f for f in fs if f not in cnt and len(f[2])>=1 and 0 not in f[2]]
    if len(cnt)==1 and len(txt)==1:
        v=sum(NUM[w] for w in cnt[0][2] if w in NUM)
        pairs.append((fs[0][0],fs[0][1],tuple(txt[0][2]),v,tuple(cnt[0][2])))
print('objects with one count face + one text face',len(pairs),C.Counter((p[0][:6],p[1]) for p in pairs).most_common(6))
g=C.defaultdict(list)
for p in pairs: g[p[2]].append(p[3])
rec={t:v for t,v in g.items() if len(v)>=3}
def const(g2): return sum(1 for v in g2.values() if len(set(v))==1)
real=const(rec); print('recurring texts',len(rec),'with constant count',real)
for t,v in sorted(rec.items(),key=lambda kv:-len(kv[1]))[:12]: print(' ',t,sorted(v))
vals=[p[3] for p in pairs]; random.seed(43); nl=[]
keys=[p[2] for p in pairs]
for _ in range(5000):
    random.shuffle(vals); g2=C.defaultdict(list)
    for k,v in zip(keys,vals):
        if k in rec: g2[k].append(v)
    nl.append(const(g2))
print('null mean %.2f P(>=)=%.4f'%(sum(nl)/5000,sum(v>=real for v in nl)/5000))
