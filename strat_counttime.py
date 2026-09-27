"""S203: At Harappa, does the use of counts (numeral + counted sign, S198) or fixed-numeral terms change from Period 3B to 3C?
Separately for seals and for tablets. Control: Fisher exact / permutation of period labels 5000x within object class."""
import json,random,collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
NUM=(set(range(1,8))|set(range(12,21))|set(range(25,30))|set(range(31,38))|{39,42,43,55,56})-{2,31}
CT={740,220,700,390,900,405}
FX={(32,877),(32,632),(17,585),(17,575),(33,520),(33,923),(32,226)}
def per(t):
    if t in ('Period 3B',): return '3B'
    if t.startswith('Period 3C'): return '3C'
    return None
def feats(s):
    c=any(s[i] in CT and s[i-1] in NUM for i in range(1,len(s)))
    f=any((s[i-1],s[i]) in FX or (s[i-1]==3 and 422<=s[i]<=426) for i in range(1,len(s)))
    n=any(w in NUM for w in s)
    return c,f,n
random.seed(4)
for cls,test in [('seal',lambda t:t.startswith('SEAL')),('tablet',lambda t:t.startswith('TAB')),('all',lambda t:True)]:
    rows=[(per(x['time']),feats(x['seq'])) for x in d if x['site']=='Harappa' and test(x['type']) and per(x['time'])]
    for j,name in enumerate(['count','fixed-term','any numeral']):
        g=C.defaultdict(lambda:[0,0])
        for p,f in rows: g[p][0]+=f[j]; g[p][1]+=1
        diff=g['3C'][0]/max(1,g['3C'][1])-g['3B'][0]/max(1,g['3B'][1])
        labs=[p for p,f in rows]; vals=[f[j] for p,f in rows]; null=[]
        for _ in range(3000):
            random.shuffle(labs); a=[v for l,v in zip(labs,vals) if l=='3C']; b=[v for l,v in zip(labs,vals) if l=='3B']
            null.append(sum(a)/max(1,len(a))-sum(b)/max(1,len(b)))
        p=sum(abs(v)>=abs(diff) for v in null)/3000
        print(cls,name,'3B %d/%d 3C %d/%d diff %+.3f P=%.3f'%(g['3B'][0],g['3B'][1],g['3C'][0],g['3C'][1],diff,p))
