"""S189: Seals sharing an identical text (3+ signs): same owner (same site, animal, size) or an office (different)?
Control: random pairs of seals with matched text length."""
import csv,json,random,collections as C,itertools,statistics as st
raw={x['cisi']:x for x in csv.DictReader(open('data/raw/inscriptions.csv')) if x['cisi'] not in ('-','')}
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
def size(x):
    r=raw.get(x['cisi'])
    try: return max(float(r['horizontal(mm)']),float(r['vertical(mm)'])) if r else None
    except: return None
S=[x for x in d if x['type']=='SEAL:S' and len(x['seq'])>=3]
g=C.defaultdict(list)
for x in S: g[tuple(x['seq'])].append(x)
dup=[v for v in g.values() if len(v)>=2]
pairs=[(a,b) for v in dup for a,b in itertools.combinations(v,2)]
def stats(P):
    site=sum(a['site']==b['site'] for a,b in P)/len(P)
    ani=sum(a['symbol'].split(':')[0]==b['symbol'].split(':')[0] for a,b in P)/len(P)
    ds=[abs(size(a)-size(b)) for a,b in P if size(a) and size(b)]
    return site,ani,(st.median(ds) if ds else None),len(ds)
print('duplicate texts',len(dup),'pairs',len(pairs)); print('dup pairs: same site %.2f same animal %.2f median |size diff| %s (n=%d)'%stats(pairs))
byL=C.defaultdict(list)
for x in S: byL[len(x['seq'])].append(x)
random.seed(0); R=[]
for a,b in pairs:
    L=len(a['seq']); c=random.sample(byL[L],2); R.append(tuple(c))
    c=random.sample(byL[L],2); R.append(tuple(c))
print('random pairs (same length): same site %.2f same animal %.2f median |size diff| %s (n=%d)'%stats(R))
for v in sorted(dup,key=len,reverse=True)[:8]: print(v[0]['seq'],[(x['site'],x['symbol'][:6],size(x)) for x in v])
