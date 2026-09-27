"""S222: Which seal texts are attested as impressions (TAG* objects with the identical text)? Compare 'used' seal texts with the rest:
opener present (W817/861/820), counted title (S198), jar-final, length, animal. Control: permutation of the used/unused label among seal texts
of the same length 5000x (length-matched), per feature."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
NUM=(set(range(1,8))|set(range(12,21))|set(range(25,30))|set(range(32,38))|{39,55,56})-{2}
CT={220,700,900,405,390,740}
tags=set(tuple(x['seq']) for x in d if x['type'].startswith('TAG') and len(x['seq'])>=2)
seals={}
for x in d:
    if x['type'].startswith('SEAL') and len(x['seq'])>=2:
        seals.setdefault(tuple(x['seq']),x)
feat=lambda s,x:{'opener':bool(set(s)&{817,861,820}),'count':any(s[i] in CT and s[i-1] in NUM for i in range(1,len(s))),
  'jarfinal':s[-1]==740,'unicorn':x['symbol'].startswith('Bull1'),'noanimal':x['symbol'] in ('-','')}
rows=[(s,s in tags,feat(s,x)) for s,x in seals.items()]
print('seal texts',len(rows),'used (impression known)',sum(u for s,u,f in rows))
bylen=C.defaultdict(list)
for i,(s,u,f) in enumerate(rows): bylen[min(len(s),7)].append(i)
random.seed(71)
for k in ['opener','count','jarfinal','unicorn','noanimal']:
    def diff(lab):
        a=[rows[i][2][k] for i in range(len(rows)) if lab[i]]; b=[rows[i][2][k] for i in range(len(rows)) if not lab[i]]
        return sum(a)/len(a)-sum(b)/len(b),sum(a),len(a),sum(b),len(b)
    lab=[u for s,u,f in rows]; o=diff(lab); nl=[]
    for _ in range(3000):
        l2=list(lab)
        for L,ix in bylen.items():
            p=[lab[i] for i in ix]; random.shuffle(p)
            for i,v in zip(ix,p): l2[i]=v
        nl.append(diff(l2)[0])
    print(k,'used %d/%d vs rest %d/%d diff %+.3f P=%.4f'%(o[1],o[2],o[3],o[4],o[0],sum(abs(v)>=abs(o[0]) for v in nl)/3000))
