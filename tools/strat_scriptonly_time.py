"""S259: do script-only seals (no motif) become commoner in later levels? (Binjor pattern, S258)
Merged corpus seals with a stratigraphic 'time' at Harappa, Mohenjo-daro, Dholavira, Kalibangan.
symbol '' = script only; '-' (unknown/damaged) excluded. Control: permute time labels within site."""
import json,collections,random
m=json.load(open('data/derived/merged-corpus-reading-order.json'))
m=m if isinstance(m,list) else list(m.values())[0]
ORD={'Period 3A':0,'Period 3B':1,'Period 3B-1':1,'Period 3B-2':1,'Period 3B/C':2,'Period 3C':3,'Period 3C-1':3,'Period 3C-2':4,'Period 3C-3':5,'Period 3C-4':6,'Period 4':7}
rows=[]
for r in m:
    if not str(r.get('type','')).startswith('SEAL'): continue
    if r['symbol']=='-' : continue
    t=r.get('time'); 
    if t not in ORD or r['site'] not in ('Harappa','Mohenjo-daro','Dholavira','Kalibangan'): continue
    rows.append((r['site'],ORD[t],r['symbol']=='' ,r['type']))
for site in ('Harappa','Mohenjo-daro','Dholavira','Kalibangan'):
    c=collections.defaultdict(lambda:[0,0])
    for s,t,so,ty in rows:
        if s==site: c[t][0]+=so; c[t][1]+=1
    print(site,' '.join(f"t{t}:{a}/{n}" for t,(a,n) in sorted(c.items())))
def stat(rs):
    # mean time rank of script-only minus mean of motif seals, z-scored within site
    tot=0
    for site in set(r[0] for r in rs):
        sub=[r for r in rs if r[0]==site]; ts=[r[1] for r in sub]
        mu=sum(ts)/len(ts)
        tot+=sum((r[1]-mu) for r in sub if r[2])
    return tot
obs=stat(rows); random.seed(0); N=5000; ge=0
bysite=collections.defaultdict(list)
for r in rows: bysite[r[0]].append(r)
for _ in range(N):
    perm=[]
    for site,sub in bysite.items():
        so=[r[2] for r in sub]; random.shuffle(so)
        perm+=[(r[0],r[1],x,r[3]) for r,x in zip(sub,so)]
    ge+= stat(perm)>=obs
print('n',len(rows),'script-only',sum(r[2] for r in rows),'obs',round(obs,1),'P(later)',ge/N)
# same test restricted to square seals (shape confound: script-only are mostly rectangular)
sq=[r for r in rows if r[3]=='SEAL:S']; print('square seals only: n',len(sq),'script-only',sum(r[2] for r in sq))
rect=[r for r in rows if r[3]=='SEAL:R']
for lab,sub in (('rect',rect),('all',rows)):
    c=collections.Counter((r[1]>=5, r[3]=='SEAL:R') for r in sub if r[0]=='Mohenjo-daro' or r[0]=='Harappa')

# do script-only seal texts differ? (length, opener first, jar present), late levels only to hold time fixed
OPEN={817,861,820}
def desc(sel):
    L=[len(r['seq']) for r in sel]; 
    return f"n={len(sel)} meanlen={sum(L)/len(L):.2f} opener1st={sum(r['seq'][0] in OPEN for r in sel)/len(sel):.2f} jar={sum(740 in r['seq'] for r in sel)/len(sel):.2f} endsW90={sum(r['seq'][-1]==90 for r in sel)/len(sel):.2f}"
late=[r for r in m if str(r.get('type','')).startswith('SEAL') and r.get('time') in ORD and ORD[r['time']]>=4 and r['symbol']!='-' and r.get('seq')]
print('late script-only ',desc([r for r in late if r['symbol']=='']))
print('late motif seals ',desc([r for r in late if r['symbol']!='']))
allm=[r for r in m if str(r.get('type','')).startswith('SEAL') and r['symbol']!='-' and r.get('seq')]
print('all script-only  ',desc([r for r in allm if r['symbol']=='']))
print('all motif seals  ',desc([r for r in allm if r['symbol']!='']))
