"""S179: Do larger unicorn seals carry the opener more often (authority seals)? Size groups from CISI vol. 1 page headers
(unicorn seals are arranged by size group I >=43.5mm ... VI <=17mm); CISI numbers mapped to groups by nearest header, monotone.
Control: permutation of size labels within text-length strata."""
import json,re,collections as C,random
S='/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
pts=json.load(open('data/derived/cisi1-unicorn-size-groups.json'))
R={'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6}
by=C.defaultdict(list)
for pg,site,a,b,g in pts:
    gs=[R[x] for x in g.split(',') if x in R]
    if len(gs)==1 and b-a<10 and a<2000: by[site].append(((a+b)/2,gs[0]))
def clean(p):
    out=[]
    for x,g in sorted(p):
        if out and g<out[-1][1]: continue
        out.append((x,g))
    return out
B={s:clean(v) for s,v in by.items()}
def grp(site,n):
    p=B.get(site)
    if not p or n>p[-1][0]+15: return None
    return min(p,key=lambda t:abs(t[0]-n))[1]
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
OP={817,861,820}
rows=[]
for x in d:
    m=re.match(r'([MHL])-(\d+)$',x['cisi'])
    if not m or x['type']!='SEAL:S' or not x['symbol'].startswith('Bull1') or len(x['seq'])<2: continue
    g=grp(m.group(1),int(m.group(2)))
    if g: rows.append(dict(big=g<=2,small=g>=4,L=min(len(x['seq']),7),op=x['seq'][0] in OP and x['seq'][1]==2))
rows=[r for r in rows if r['big'] or r['small']]
strata=C.defaultdict(list)
for r in rows: strata[r['L']].append(r)
obs=sum(r['op'] for r in rows if r['big']); random.seed(0); ge=0; tot=0
for _ in range(10000):
    v=0
    for rs in strata.values():
        k=sum(r['big'] for r in rs); v+=sum(r['op'] for r in random.sample(rs,k))
    tot+=v; ge+=v>=obs
nb=sum(r['big'] for r in rows)
print(f"large seals (I-II) n={nb}, opener {obs}; expected within length strata {tot/10000:.1f}; P={ge/10000:.3f}")
for L in sorted(strata):
    rs=strata[L]; b=[r for r in rs if r['big']]; s=[r for r in rs if r['small']]
    print(L, 'big',sum(r['op'] for r in b),'/',len(b),' small',sum(r['op'] for r in s),'/',len(s))
