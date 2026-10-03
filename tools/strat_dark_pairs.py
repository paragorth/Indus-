"""S363: ARROW-IN-THE-DARK, pair version. Random (text-pair relation) x (object-pair fact) hypotheses over random pairs of
objects within the same site. Statistic: MI; null: object-level permutation of the fact within site (so pair dependence is
respected). Train: MD + Harappa pairs; survivors p<0.001 re-tested on held-out-site pairs (p<0.01) and on seq_all."""
import json,csv,random,sys,collections
import numpy as np
NH=int(sys.argv[1]) if len(sys.argv)>1 else 1500; CONTROL=len(sys.argv)>2
C=json.load(open('data/derived/merged-corpus-canonical.json'))
raw={r['cisi']:r for r in csv.DictReader(open('data/raw/inscriptions.csv')) if r['cisi']}
NUM={1:1,3:3,4:4,5:5,16:6,17:7,18:8,31:1,32:2,33:3,34:4}
def num(x):
    try: v=float(x); return v if v>0 else None
    except: return None
OBJ=[]
for r in C:
    if not r.get('seq_raw') or len(r['seq_raw'])<2: continue
    x=raw.get(r['cisi'],{})
    f={'site':r['site'],'type':r['type'].split(':')[0],'emblem':(r.get('symbol') or '').split(':')[0] or None,'material':x.get('material'),
       'shape':x.get('shape'),'boss':x.get('boss'),'area':r.get('area-section'),'time':x.get('time') or r.get('period'),'h':num(x.get('horizontal(mm)')),'cult':x.get('cult'),'color':x.get('color')}
    for k,v in list(f.items()):
        if v in ('','-','--','None','?'): f[k]=None
    OBJ.append({'raw':tuple(r['seq_raw']),'all':tuple(r['seq_all']),'f':f})
rng=random.Random(11)
FACTS=['type','emblem','material','shape','boss','area','time','h','cult','color']
if CONTROL:
    for key in FACTS:
        bysite=collections.defaultdict(list)
        for o in OBJ: bysite[o['f']['site']].append(o)
        for os_ in bysite.values():
            vals=[o['f'][key] for o in os_]; rng.shuffle(vals)
            for o,v in zip(os_,vals): o['f'][key]=v
def pairs_for(objs,n):
    bysite=collections.defaultdict(list)
    for i,o in enumerate(objs): bysite[o['f']['site']].append(i)
    P=[]
    sites=[s for s in bysite if len(bysite[s])>=20]; w=[len(bysite[s]) for s in sites]
    for _ in range(n):
        s=rng.choices(sites,w)[0]; a,b=rng.sample(bysite[s],2); P.append((a,b))
    return P
def relation(kind,p,s,t):
    A,B=set(s),set(t)
    if kind=='jacc': return min(3,int(4*len(A&B)/len(A|B)))
    if kind=='first': return s[0]==t[0]
    if kind=='last': return s[-1]==t[-1]
    if kind=='edit1': return (len(s)==len(t) and sum(a!=b for a,b in zip(s,t))==1)
    if kind=='nest': return (len(s)!=len(t)) and (any(s==t[i:i+len(s)] for i in range(len(t)-len(s)+1)) or any(t==s[i:i+len(t)] for i in range(len(s)-len(t)+1)))
    if kind=='disjoint': return not (A&B)
    if kind=='samelen': return len(s)==len(t)
    if kind=='lensum': return min(12,len(s)+len(t))
    if kind=='sharesign': return p['s'] in A and p['s'] in B
    if kind=='xorsign': return (p['s'] in A)!=(p['s'] in B)
    if kind=='samenum': return bool({NUM[x] for x in s if x in NUM}&{NUM[x] for x in t if x in NUM})
    if kind=='sharepair': return bool(set(zip(s,s[1:]))&set(zip(t,t[1:])))
    if kind=='posk': k=p['k']; return (len(s)>k and len(t)>k and s[k]==t[k])
    if kind=='mirror': return s[::-1]==t or s[0]==t[-1]
    if kind=='sumw': return min(5,abs(sum(s)-sum(t))//300)
def pairfact(kind,fa,fb):
    if fa is None or fb is None: return None
    if kind=='h': return min(3,int(abs(fa-fb)//5))
    return fa==fb
def make(rng):
    kind=rng.choice(['jacc','first','last','edit1','nest','disjoint','samelen','lensum','sharesign','xorsign','samenum','sharepair','posk','mirror','sumw'])
    p={}
    if kind in('sharesign','xorsign'): p['s']=rng.choice([740,2,390,220,520,817,861,820,700,400,90,60,176,100,415,590,235,240,233,231,226,900,405,407,3,4,32,33,34,1,575,125,416,413,920,495,840,460,70,35,440,435,690,717])
    if kind=='posk': p['k']=rng.randint(1,3)
    return kind,p,rng.choice(FACTS)
def mi(a,b):
    _,xi=np.unique(a,return_inverse=True); _,yi=np.unique(b,return_inverse=True); nx=xi.max()+1; ny=yi.max()+1
    if nx<2 or ny<2: return 0.0
    t=np.bincount(xi*ny+yi,minlength=nx*ny).reshape(nx,ny)/len(a); px=t.sum(1,keepdims=True); py=t.sum(0,keepdims=True); nz=t>0
    return float((t[nz]*np.log(t[nz]/(px@py)[nz])).sum())
def evaluate(kind,p,fact,objs,P,level,nperm):
    rel=np.array([str(relation(kind,p,objs[a][level],objs[b][level])) for a,b in P])
    facts=[o['f'][fact] for o in objs]
    def pf(fv):
        return [pairfact(fact,fv[a],fv[b]) for a,b in P]
    y=pf(facts); keep=[i for i,v in enumerate(y) if v is not None]
    if len(keep)<200: return None
    ya=np.array([str(y[i]) for i in keep]); ra=rel[keep]
    if len(set(ra))<2 or len(set(ya))<2: return None
    o=mi(ra,ya); ge=0
    bysite=collections.defaultdict(list)
    for i,ob in enumerate(objs): bysite[ob['f']['site']].append(i)
    for k in range(nperm):
        fv=facts[:]
        for idx in bysite.values():
            vals=[fv[i] for i in idx]; rng.shuffle(vals)
            for i,v in zip(idx,vals): fv[i]=v
        y2=pf(fv); ya2=np.array([str(y2[i]) for i in keep]); ge+=mi(ra,ya2)>=o
        if ge>=5 and k>=20: return o,(ge+1)/(k+2),len(keep)
    return o,(ge+1)/(nperm+1),len(keep)
TR=[o for o in OBJ if o['f']['site'] in('Mohenjo-daro','Harappa')]; TE=[o for o in OBJ if o['f']['site'] not in('Mohenjo-daro','Harappa')]
PTR=pairs_for(TR,40000); PTE=pairs_for(TE,20000)
print('train objects',len(TR),'pairs',len(PTR),'| held-out objects',len(TE),'pairs',len(PTE))
surv=[]
for h in range(NH):
    kind,p,fact=make(rng)
    r=evaluate(kind,p,fact,TR,PTR,'raw',500)
    if not r or r[1]>0.0021: continue
    r2=evaluate(kind,p,fact,TE,PTE,'raw',500); r3=evaluate(kind,p,fact,TR,PTR,'all',200)
    ok=(r2 is not None and r2[1]<0.01) and (r3 is not None and r3[1]<0.01)
    surv.append(dict(kind=kind,p=p,fact=fact,train=r,held=r2,all=r3,ok=ok))
print('tried',NH,'train survivors',len(surv),'replicated',sum(s['ok'] for s in surv))
surv.sort(key=lambda s:-s['train'][0])
for s in surv:
    print(('SURVIVOR ' if s['ok'] else '   train-only '),s['kind'],s['p'],'x same/diff',s['fact'],'train MI %.4f p %.4f n %d'%s['train'],'held',tuple(round(x,4) for x in s['held'][:2]) if s['held'] else None)
json.dump(surv,open('data/derived/strat_dark_pairs%s.json'%('_control' if CONTROL else ''),'w'),default=str,indent=0)
