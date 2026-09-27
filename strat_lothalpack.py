"""S226: Lothal sealings (Frenez & Tosi 2005 Table 1, transcribed to data/derived/lothal-sealings-frenez-tosi2005.json) joined to
corpus texts by CISI number. Does the seal (text) predict the fastening type on the back? Diagnostic types only; single-seal tags only.
Statistic: number of same-text tag pairs that also share the fastening type. Control: permute types among tags 10000x.
Second test excludes the elephant seal 817-2-48-740."""
import json,collections as C,random,itertools
L=json.load(open('data/derived/lothal-sealings-frenez-tosi2005.json'))['rows']
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
byc=C.defaultdict(set)
for x in d:
    if x['site']=='Lothal' and x['seq']: byc[x['cisi']].add(tuple(x['seq']))
rows=[(r[0],r[2],list(byc[r[0]])[0]) for r in L if r[0] in byc and len(byc[r[0]])==1 and r[2] not in ('undiagnostic','other types')]
print('tags',len(rows)); tx=C.Counter(t for _,_,t in rows)
for t,n in tx.most_common():
    if n>=2: print(n,t,C.Counter(ty for _,ty,tt in rows if tt==t))
def stat(rows):
    return sum(1 for a,b in itertools.combinations(rows,2) if a[2]==b[2] and a[1]==b[1])
random.seed(79)
for name,rr in [('all',rows),('without elephant seal',[r for r in rows if r[2]!=(817,2,48,740)])]:
    o=stat(rr); ty=[r[1] for r in rr]; nl=[]
    for _ in range(10000):
        random.shuffle(ty); nl.append(stat([(a,t,c) for (a,_,c),t in zip(rr,ty)]))
    print(name,'n',len(rr),'same-text same-type pairs',o,'null %.1f P=%.4f'%(sum(nl)/10000,sum(v>=o for v in nl)/10000))
print('types:',C.Counter(r[1] for r in rows))
