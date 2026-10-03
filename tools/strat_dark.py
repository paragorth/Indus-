"""ARROW-IN-THE-DARK machine (S362 family; loop-1 extension).
Random hypotheses = (random READING of a text) x (random OUTSIDE FACT) x (relation = mutual information).
SITE CONFOUND: texts and facts both differ by city, so the statistic is the CONDITIONAL mutual information given
site, I(X;Y|site), and every permutation null shuffles facts WITHIN site (train: within Mohenjo-daro and within
Harappa; held-out: within each other site). 'site' and 'region' are therefore not facts any more.
Train: Mohenjo-daro + Harappa. Screen: G-test on 2n*CMI; candidates (screen p<0.01) get a within-site permutation
null with enough permutations to resolve the Bonferroni level (up to --maxperm). Multiple testing: Bonferroni and
Benjamini-Hochberg over every arrow fired in the cycle. Survivors must replicate on HELD-OUT sites (perm p<0.01)
and on seq_raw / seq_strong / seq_all (train and held-out). A permutation within site x type x length-bin strata
flags frame-rule rediscoveries (opener->seal, suffix->tablet, length->type).
Whole-machine control: --control shuffles facts within site before firing (expected survivors ~0).
Usage: python3 tools/strat_dark.py --n 3000 --seed 7 --cycle 1 [--families legacy|all|mid,parity,...] [--control]
"""
import json,csv,random,math,sys,collections,argparse,time,os
import numpy as np
from scipy.stats import chi2
ap=argparse.ArgumentParser()
ap.add_argument('--n',type=int,default=3000); ap.add_argument('--seed',type=int,default=7); ap.add_argument('--cycle',type=int,default=0)
ap.add_argument('--families',default='all'); ap.add_argument('--control',action='store_true'); ap.add_argument('--out',default=None)
ap.add_argument('--maxperm',type=int,default=10000); ap.add_argument('--tag',default='loop1')
A=ap.parse_args(); NH=A.n; CONTROL=A.control
C=json.load(open('data/derived/merged-corpus-canonical.json'))
raw={r['cisi']:r for r in csv.DictReader(open('data/raw/inscriptions.csv')) if r['cisi']}
NUM={1:1,3:3,4:4,5:5,16:6,17:7,18:8,31:1,32:2,33:3,34:4}
def fam(s): return s//100 if s>=100 else 0
GS=json.load(open('data/derived/glyph_sim_signs.json')); GSI={s:i for i,s in enumerate(GS)}; GSIM=np.load('data/derived/glyph_sim.npy')
GTHR=float(np.quantile(GSIM[np.triu_indices(len(GS),1)],0.9))   # top-decile glyph similarity = 'same shape family'
def gsim(a,b):
    if a in GSI and b in GSI: return float(GSIM[GSI[a],GSI[b]])
    return None
# ---------- objects with facts ----------
def fnum(x):
    try: v=float(x)
    except: return None
    return None if v<=0 else v
def depth_ft(x):
    x=(x or '').strip()
    if not x or x=='- -': return None
    try: v=float(x.split()[0])
    except: return None
    if x.endswith('cm'): v=v*0.0328
    elif x.endswith('m'): v=v*3.281
    v=abs(v)
    return 0 if v<2 else 1 if v<4 else 2 if v<6 else 3 if v<9 else 4 if v<13 else 5
def clean(v):
    v=(v or '').strip()
    return None if v in ('','-','--','None','?','- -') else v
OBJ=[]
for r in C:
    if not r.get('seq_raw') or len(r['seq_raw'])<2: continue
    x=raw.get(r['cisi'],{})
    f={'site':r['site'],'type':r['type'].split(':')[0],'type2':r['type'],
       'emblem':((r.get('symbol') or '').split(':')[0] or None),'cult':clean(x.get('cult')),'material':clean(x.get('material')),
       'color':clean(x.get('color')),'shape':clean(x.get('shape')),'xsec':clean(x.get('cross-section')),'boss':clean(x.get('boss')),'dir':clean(x.get('dir.')),
       'sides':clean(x.get('sides')),'area':clean(r.get('area-section')),'time':clean(x.get('time')) or clean(r.get('period')),
       'h':fnum(x.get('horizontal(mm)')),'v':fnum(x.get('vertical(mm)')),'th':fnum(x.get('thickness(mm)')),
       'depth':depth_ft(x.get('depth')),'condition':(clean(x.get('condition')) or '').capitalize() or None,
       'preservation':clean(x.get('preservation'))}
    if f['emblem'] in ('','None'): f['emblem']=None
    if f['cult']=='None': f['cult']=None
    if f['preservation']=='complsete': f['preservation']='complete'
    OBJ.append({'cisi':r['cisi'],'seq_raw':r['seq_raw'],'seq_strong':r['seq_strong'],'seq_all':r['seq_all'],'f':f})
for k in ('h','v','th'):   # size bins: quartiles over objects with a value
    vals=np.array([o['f'][k] for o in OBJ if o['f'][k]]); qs=np.quantile(vals,[0.25,0.5,0.75])
    for o in OBJ:
        v=o['f'][k]; o['f'][k]=None if v is None else int(np.searchsorted(qs,v))
TRAIN=[o for o in OBJ if o['f']['site'] in ('Mohenjo-daro','Harappa')]; TEST=[o for o in OBJ if o['f']['site'] not in ('Mohenjo-daro','Harappa')]
FACTS=['type','type2','emblem','cult','material','color','shape','xsec','boss','dir','sides','area','time','h','v','th','depth','condition','preservation']
rng=random.Random(A.seed)
if CONTROL:  # scrambled-fact control: permute facts within site (keeps the site confound, destroys any text link)
    for key in FACTS:
        bysite=collections.defaultdict(list)
        for o in OBJ: bysite[o['f']['site']].append(o)
        for os_ in bysite.values():
            vals=[o['f'][key] for o in os_]; rng.shuffle(vals)
            for o,v in zip(os_,vals): o['f'][key]=v
FREQ={lv:collections.Counter(s for o in OBJ for s in o[lv]) for lv in ('seq_raw','seq_strong','seq_all')}
ALLSIGNS=sorted(FREQ['seq_raw']); COMMON=[s for s,c in FREQ['seq_raw'].most_common(60)]
# ---------- reading families ----------
LEGACY=['pos','pair','mod','even','odd','famfl','nfam','rep','nnum','sumnum','maxw','lenpar','run','fampos','dist','first2','last2','contains','posrel']
NEW={'mid':['midpos','midfam','midpar'],'parity':['parpos','xorpar','npar','parfl'],'set3':['set3','set3fam','nset3'],
     'w500':['nabove','fracabove','nabove_bin','allabove'],'shapefl':['flsame','flsim','adjsim','minadjsim'],
     'rare':['rarest','rarepos','rarestfam','nrare'],'numpos':['firstnum','lastnum','numval_first','numgap'],
     'ascrun':['ascrun','descrun','nasc','monot'],'base':['modw','modfam','modrev','digitsum'],
     'len':['len','lenbin','lenfam'],'gap':['sumgap','maxgap','mingap','gapsign'],'rank':['rankfirst','ranklast','medrank']}
def families():
    if A.families=='all': return LEGACY+[k for v in NEW.values() for k in v]
    if A.families=='legacy': return LEGACY
    out=[]
    for f in A.families.split(','): out+=NEW.get(f,[f] if f in LEGACY else [])
    return out
KINDS=families()
def make_reading(rng):
    kind=rng.choice(KINDS); p={}
    if kind=='pos': p['i']=rng.choice([0,1,2,-1,-2,-3])
    if kind=='pair': p['d']=rng.randint(1,4); p['i']=rng.choice([0,1,-2,'any'])
    if kind=='mod': p['base']=rng.randint(2,13); p['m']=rng.randint(2,12); p['digit']=rng.choice(['w','fam','num'])
    if kind=='fampos': p['i']=rng.choice([0,1,-1,-2])
    if kind=='dist': p['a']=rng.choice(COMMON[:24]); p['b']=rng.choice(COMMON[:24])
    if kind=='contains': p['s']=rng.choice(ALLSIGNS)
    if kind=='posrel': p['s']=rng.choice(COMMON[:14])
    if kind in ('midpos','midfam','midpar'): p['i']=rng.choice([-2,-1,0,1,2])
    if kind=='parpos': p['i']=rng.choice([0,1,2,-1,-2,-3])
    if kind in ('set3','set3fam','nset3'): p['S']=rng.sample(ALLSIGNS if kind!='set3fam' else list(range(0,10)),3)
    if kind in ('nabove','fracabove','nabove_bin','allabove'): p['t']=rng.choice([100,200,300,400,500,600,700,800,900])
    if kind=='rarepos': p['which']=rng.choice(['rarest','commonest'])
    if kind in ('modw','modfam','modrev','digitsum'): p['base']=rng.randint(2,16); p['m']=rng.randint(2,13)
    if kind in ('rankfirst','ranklast','medrank'): p['bins']=rng.choice([3,4,6])
    if kind=='lenfam': p['f']=rng.randint(0,9)
    if kind=='lenbin': p['cut']=rng.choice([3,4,5,6,7])
    return kind,p
def mid_index(L,i):
    c=(L-1)/2; j=int(math.floor(c))+i if i<=0 else int(math.ceil(c))+i
    return j if 0<=j<L else None
def rankbin(s,lv,bins):
    r=math.log2(max(FREQ[lv][s],1)); return min(int(r*bins/11),bins-1)
def read(kind,p,s,lv='seq_raw'):
    L=len(s)
    if kind=='pos': i=p['i']; return s[i] if -L<=i<L else None
    if kind=='pair':
        d=p['d']
        if p['i']=='any': return min((s[j],s[j+d]) for j in range(L-d)) if L>d else None
        i=p['i']; j=i+d if i>=0 else i-d
        return (s[i],s[j]) if -L<=i<L and -L<=j<L else None
    if kind=='mod':
        dig={'w':lambda x:x,'fam':fam,'num':lambda x:NUM.get(x,0)}[p['digit']]; v=0
        for x in s: v=(v*p['base']+dig(x))%p['m']
        return v
    if kind=='even': return tuple(s[0::2])
    if kind=='odd': return tuple(s[1::2]) or None
    if kind=='famfl': return (fam(s[0]),fam(s[-1]))
    if kind=='nfam': return len({fam(x) for x in s})
    if kind=='rep': return L-len(set(s))
    if kind=='nnum': return sum(x in NUM for x in s)
    if kind=='sumnum': return min(sum(NUM.get(x,0) for x in s),12)
    if kind=='maxw': return max(s)//100
    if kind=='lenpar': return L%2
    if kind=='run':
        best=cur=1
        for a,b in zip(s,s[1:]): cur=cur+1 if fam(a)==fam(b) else 1; best=max(best,cur)
        return best
    if kind=='fampos': i=p['i']; return fam(s[i]) if -L<=i<L else None
    if kind=='dist':
        a,b=p['a'],p['b']
        if a in s and b in s: return max(-3,min(3,s.index(b)-s.index(a)))
        return 'absent'
    if kind=='first2': return tuple(s[:2])
    if kind=='last2': return tuple(s[-2:])
    if kind=='contains': return p['s'] in s
    if kind=='posrel':
        if p['s'] not in s: return 'absent'
        return round(s.index(p['s'])/(L-1),1) if L>1 else 0
    # ---- new families ----
    if kind in ('midpos','midfam','midpar'):
        j=mid_index(L,p['i'])
        if j is None: return None
        return s[j] if kind=='midpos' else fam(s[j]) if kind=='midfam' else s[j]%2
    if kind=='parpos': i=p['i']; return s[i]%2 if -L<=i<L else None
    if kind=='xorpar':
        v=0
        for x in s: v^=x%2
        return v
    if kind=='npar': return min(sum(x%2 for x in s),6)
    if kind=='parfl': return (s[0]%2,s[-1]%2)
    if kind=='set3': return any(x in p['S'] for x in s)
    if kind=='set3fam': return any(fam(x) in p['S'] for x in s)
    if kind=='nset3': return min(sum(x in p['S'] for x in s),3)
    if kind=='nabove': return min(sum(x>=p['t'] for x in s),6)
    if kind=='fracabove': return round(sum(x>=p['t'] for x in s)/L,1)
    if kind=='nabove_bin': return sum(x>=p['t'] for x in s)>sum(x<p['t'] for x in s)
    if kind=='allabove': return all(x>=p['t'] for x in s)
    if kind=='flsame':
        g=gsim(s[0],s[-1]); return None if g is None else (fam(s[0])==fam(s[-1]),g>=GTHR)
    if kind=='flsim':
        g=gsim(s[0],s[-1]); return None if g is None else int(min(g,0.999)*5)
    if kind in ('adjsim','minadjsim'):
        gs=[g for g in (gsim(a,b) for a,b in zip(s,s[1:])) if g is not None]
        if not gs: return None
        v=(sum(gs)/len(gs)) if kind=='adjsim' else min(gs); return int(min(v,0.999)*5)
    if kind=='rarest': return rankbin(min(s,key=lambda x:FREQ[lv][x]),lv,6)
    if kind=='rarestfam': return fam(min(s,key=lambda x:FREQ[lv][x]))
    if kind=='rarepos':
        x=min(s,key=lambda x:FREQ[lv][x]) if p['which']=='rarest' else max(s,key=lambda x:FREQ[lv][x])
        return round(s.index(x)/(L-1),1)
    if kind=='nrare': return min(sum(FREQ[lv][x]<10 for x in s),4)
    if kind=='firstnum':
        for j,x in enumerate(s):
            if x in NUM: return min(j,4)
        return 'none'
    if kind=='lastnum':
        for j in range(L-1,-1,-1):
            if s[j] in NUM: return min(L-1-j,4)
        return 'none'
    if kind=='numval_first':
        for x in s:
            if x in NUM: return NUM[x]
        return 'none'
    if kind=='numgap':
        idx=[j for j,x in enumerate(s) if x in NUM]
        return 'none' if len(idx)<2 else min(idx[1]-idx[0],4)
    if kind in ('ascrun','descrun'):
        best=cur=1
        for a,b in zip(s,s[1:]):
            cur=cur+1 if ((b>a) if kind=='ascrun' else (b<a)) else 1; best=max(best,cur)
        return min(best,5)
    if kind=='nasc': return round(sum(b>a for a,b in zip(s,s[1:]))/(L-1),1)
    if kind=='monot':
        up=all(b>a for a,b in zip(s,s[1:])); dn=all(b<a for a,b in zip(s,s[1:]))
        return 'up' if up else 'down' if dn else 'mixed'
    if kind in ('modw','modfam','modrev'):
        seq=s if kind!='modrev' else s[::-1]; dig=fam if kind=='modfam' else (lambda x:x); v=0
        for x in seq: v=(v*p['base']+dig(x))%p['m']
        return v
    if kind=='digitsum': return sum(x%p['base'] for x in s)%p['m']
    if kind=='len': return min(L,12)
    if kind=='lenbin': return L>=p['cut']
    if kind=='lenfam': return min(sum(fam(x)==p['f'] for x in s),4)
    if kind in ('sumgap','maxgap','mingap','gapsign'):
        g=[b-a for a,b in zip(s,s[1:])]
        if kind=='sumgap': return int(np.sign(sum(g)))
        if kind=='maxgap': return min(max(g)//100,9)
        if kind=='mingap': return max(min(g)//100,-9)
        return tuple(int(np.sign(x)) for x in g[:3])
    if kind=='rankfirst': return rankbin(s[0],lv,p['bins'])
    if kind=='ranklast': return rankbin(s[-1],lv,p['bins'])
    if kind=='medrank': return int(np.median([rankbin(x,lv,p['bins']) for x in s]))
# ---------- conditional MI given strata, G screen, within-strata permutation ----------
def encode(xs,ys,st):
    xi=np.unique(np.asarray(xs),return_inverse=True)[1]; yi=np.unique(np.asarray(ys),return_inverse=True)[1]; si=np.unique(np.asarray(st),return_inverse=True)[1]
    return xi,yi,si,xi.max()+1,yi.max()+1,si.max()+1
def cmi(xi,yi,si,nx,ny,ns):
    """I(X;Y|S) = sum_s p(s) I(X;Y|S=s); also returns df for the G approximation."""
    n=len(xi); t=np.bincount((si*nx+xi)*ny+yi,minlength=ns*nx*ny).reshape(ns,nx,ny).astype(float)
    tot=0.0; df=0
    for s in range(ns):
        ts=t[s]; m=ts.sum()
        if m<2: continue
        ts=ts/m; px=ts.sum(1,keepdims=True); py=ts.sum(0,keepdims=True); nz=ts>0
        tot+=m/n*float((ts[nz]*np.log(ts[nz]/(px@py)[nz])).sum()); df+=(int((px>0).sum())-1)*(int((py>0).sum())-1)
    return tot,max(df,1)
def cmi_batch(xi,YY,si,nx,ny,ns):
    """CMI for a batch of B permuted y vectors (B,n) -> (B,)"""
    B,n=YY.shape; base=(si*nx+xi)*ny
    codes=(np.arange(B)[:,None]*(ns*nx*ny)+base[None,:]+YY).ravel()
    t=np.bincount(codes,minlength=B*ns*nx*ny).reshape(B,ns,nx,ny).astype(float)
    m=t.sum((2,3),keepdims=True); m[m==0]=1; ts=t/m
    px=ts.sum(3,keepdims=True); py=ts.sum(2,keepdims=True); pxy=px*py
    with np.errstate(divide='ignore',invalid='ignore'):
        term=np.where(ts>0,ts*np.log(ts/np.where(pxy>0,pxy,1)),0.0)
    return (term.sum((2,3))*(m[:,:,0,0]/n)).sum(1)
def perm_p(xi,yi,si,nx,ny,ns,nperm,rng,strata=None,stop=10,B=250):
    """within-strata permutation p for CMI; strata default to site; early stop once `stop` exceedances seen.
    Vectorised: rows sorted by stratum, random keys + stratum offset, argsort -> a within-stratum shuffle per row."""
    o,_=cmi(xi,yi,si,nx,ny,ns); r=np.random.default_rng(rng.randint(0,10**9))
    strata=si if strata is None else strata
    order=np.argsort(strata,kind='stable'); xs=xi[order]; ys=yi[order]; ss=si[order]; st=strata[order].astype(float)
    ge=0; done=0; n=len(xi)
    while done<nperm:
        b=min(B,nperm-done)
        perm=np.argsort(st[None,:]+r.random((b,n)),axis=1)
        YY=ys[perm]
        ge+=int((cmi_batch(xs,YY,ss,nx,ny,ns)>=o-1e-12).sum()); done+=b
        if ge>=stop: break
    return o,(ge+1)/(done+1),done,ge
def collect(kind,p,fact,objs,level,minn=40,strat=False):
    xs=[];ys=[];st=[];st2=[]
    for o in objs:
        y=o['f'][fact]
        if y is None: continue
        x=read(kind,p,o[level],level)
        if x is None: continue
        xs.append(str(x)); ys.append(str(y)); st.append(o['f']['site']); st2.append(o['f']['site']+'|'+o['f']['type']+'|'+str(min(len(o[level]),6)))
    if len(xs)<minn: return None
    cx=collections.Counter(xs); cy=collections.Counter(ys)   # collapse rare values (<5) to 'other'
    xs=[x if cx[x]>=5 else 'other' for x in xs]; ys=[y if cy[y]>=5 else 'other' for y in ys]
    xi,yi,si,nx,ny,ns=encode(xs,ys,st)
    if nx<2 or ny<2: return None
    return xi,yi,si,nx,ny,ns,(np.unique(np.asarray(st2),return_inverse=True)[1] if strat else None)
def evaluate(kind,p,fact,objs,level,nperm,rng,minn=40,strat=False):
    c=collect(kind,p,fact,objs,level,minn,strat)
    if c is None: return None
    xi,yi,si,nx,ny,ns,st=c
    o,pp,k,ge=perm_p(xi,yi,si,nx,ny,ns,nperm,rng,strata=st)
    return dict(mi=o,p=pp,n=len(xi),nx=nx,ny=ny,nperm=k,ge=ge)
# ---------- run ----------
t0=time.time(); log=[]
def say(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); log.append(s)
say(f'cycle {A.cycle} seed {A.seed} families {A.families} control {CONTROL} | objects {len(OBJ)} train {len(TRAIN)} held-out {len(TEST)} | kinds {len(KINDS)} facts {len(FACTS)} | statistic I(X;Y|site), null = facts shuffled within site')
arrows=[]
for h in range(NH):
    kind,p=make_reading(rng); fact=rng.choice(FACTS)
    c=collect(kind,p,fact,TRAIN,'seq_raw')
    if c is None: arrows.append(dict(kind=kind,p=p,fact=fact,p_screen=None)); continue
    xi,yi,si,nx,ny,ns,_=c; mi,df=cmi(xi,yi,si,nx,ny,ns); ps=float(chi2.sf(2*len(xi)*mi,df))
    arrows.append(dict(kind=kind,p=p,fact=fact,mi=mi,n=len(xi),nx=nx,ny=ny,p_screen=ps))
fired=sum(a['p_screen'] is not None for a in arrows); alpha_b=0.05/max(fired,1)
say(f'arrows fired (evaluable) {fired} of {NH}; Bonferroni alpha {alpha_b:.2e}')
# The G-test screen is calibrated on scrambled facts (1.3% at p<0.01 in a 300-arrow control), so the Bonferroni and
# BH corrections use the G-test p over all arrows; every corrected survivor must then also pass a within-site
# permutation null (2000 perms, p<0.005), the within site x type x length-bin null (rediscovery filter), held-out
# replication and the three merge levels.
evalu=[a for a in arrows if a['p_screen'] is not None]
pv=np.array([a['p_screen'] for a in evalu]); order=np.argsort(pv); m=len(pv); bh=np.zeros(m,bool); thr=0.05*np.arange(1,m+1)/m
ok=np.where(pv[order]<=thr)[0]
if len(ok): bh[order[:ok.max()+1]]=True
for a,b in zip(evalu,bh): a['bh']=bool(b); a['bonf']=a['p_screen']<alpha_b
say(f'Bonferroni survivors {sum(a["bonf"] for a in evalu)}; BH-FDR(q=0.05) survivors {sum(a["bh"] for a in evalu)} (G-test p; scrambled-fact null expectation for p<0.01 is ~{0.01*fired:.0f}, observed {sum(a["p_screen"]<0.01 for a in evalu)})')
surv=[a for a in evalu if a['bonf'] or a['bh']]
def rediscovery_flag(a):
    k=a['kind']; f=a['fact']
    frame_pos = k in ('pos','fampos','parpos','first2','last2','pair','famfl','parfl','flsame','flsim','rankfirst','ranklast','posrel','dist','midpos','midfam','midpar','rarepos')
    lenlike = k in ('len','lenbin','lenpar','nfam','rep','nnum','nabove','nset3','lenfam','npar','nrare','rarest','medrank','nasc','ascrun','descrun','run','even','odd','mod','modw','modfam','modrev','digitsum','sumgap','gapsign','xorpar')
    objclass = f in ('type','type2','sides','boss','xsec','shape','material','h','v','th','dir','preservation','condition','color')
    tags=[]
    if frame_pos and objclass: tags.append('frame-slot x object-class (opener/closer->seal/tablet?)')
    if lenlike and objclass: tags.append('length-like x object-class (length->type?)')
    return tags
# stage A: rediscovery filter = permutation within site x type x length bin
redis=[]; keep=[]
for a in surv:
    kind,p,fact=a['kind'],a['p'],a['fact']; a['flags']=rediscovery_flag(a)
    if fact in ('type','type2'):
        a['within_type']=None; a['flags'].append('fact is object type: text structure vs object type (GRAMMAR frame rules)'); redis.append(a); continue
    a['within_type']=evaluate(kind,p,fact,TRAIN,'seq_raw',1000,rng,strat=True)
    if a['within_type'] is None or a['within_type']['p']>0.05:
        a['flags'].append('vanishes within site x type x length strata -> rediscovery of type/length structure'); redis.append(a)
    else: keep.append(a)
say(f'stage A: {len(surv)} corrected survivors -> {len(redis)} are type/length rediscoveries (fact=type or vanish within site x type x length); {len(keep)} carry information beyond type and length')
# stage B: cheap asymptotic replication first (held-out G-test, merge levels G-test), then permutation confirmation
def gtest(kind,p,fact,objs,level,minn=40):
    c=collect(kind,p,fact,objs,level,minn)
    if c is None: return None
    xi,yi,si,nx,ny,ns,_=c; mi,df=cmi(xi,yi,si,nx,ny,ns); return dict(mi=mi,p=float(chi2.sf(2*len(xi)*mi,df)),n=len(xi))
for a in keep:
    kind,p,fact=a['kind'],a['p'],a['fact']
    a['held']=gtest(kind,p,fact,TEST,'seq_raw',minn=60)
    a['held_strong']=gtest(kind,p,fact,TEST,'seq_strong',minn=60); a['held_all']=gtest(kind,p,fact,TEST,'seq_all',minn=60)
    a['train_strong']=gtest(kind,p,fact,TRAIN,'seq_strong'); a['train_all']=gtest(kind,p,fact,TRAIN,'seq_all')
    a['ok_held']=bool(a['held'] and a['held']['p']<0.01)
    a['ok_levels']=all(x and x['p']<0.01 for x in (a['train_strong'],a['train_all'],a['held_strong'],a['held_all']))
    a['ok_perm']=False; a['p_perm']=None
    if a['held'] is None: a['flags'].append('held-out n<60: cannot replicate')
    elif not a['ok_held']: a['flags'].append('fails held-out sites')
    if not a['ok_levels']: a['flags'].append('depends on merge level')
cand=[a for a in keep if a['ok_held'] and a['ok_levels']]
say(f'stage B: {len(keep)} -> {len(cand)} replicate on held-out sites (G p<0.01) and on all three merge levels; now permutation-confirming these')
for a in cand:
    kind,p,fact=a['kind'],a['p'],a['fact']
    xi,yi,si,nx,ny,ns,_=collect(kind,p,fact,TRAIN,'seq_raw'); o,pp,k,ge=perm_p(xi,yi,si,nx,ny,ns,2000,rng); a['p_perm']=pp; a['nperm']=k; a['ge']=ge
    hp=evaluate(kind,p,fact,TEST,'seq_raw',2000,rng,minn=60); a['held_perm']=hp
    a['ok_perm']=pp<0.005 and bool(hp and hp['p']<0.01)
    if not a['ok_perm']: a['flags'].append('fails within-site permutation (train p %.3g, held-out p %s)'%(pp,'%.3g'%hp['p'] if hp else 'n/a'))
full=[a for a in keep if a['ok_perm'] and a['ok_held'] and a['ok_levels']]
say(f'full survivors (corrected + held-out p<0.01 + all three levels train and held-out): {len(full)}')
def fmt(r): return 'n/a' if r is None else 'CMI %.3f p %.2e n %d'%(r['mi'],r['p'],r['n'])
keep.sort(key=lambda a:a['p_screen'])
kc=collections.Counter((a['kind'],a['fact']) for a in keep)
say(f'beyond-type arrows by (reading, fact), top 20 of {len(kc)}: '+'; '.join(f'{k[0]}->{k[1]} x{v}' for k,v in kc.most_common(20)))
fc=collections.Counter(a['fact'] for a in keep); say('beyond-type arrows by fact: '+'; '.join(f'{k} x{v}' for k,v in fc.most_common()))
for a in [x for x in keep if x in full]+[x for x in keep if x not in full][:40]:
    say(('FULL ' if a in full else 'part ')+f"{a['kind']} {a['p']} -> {a['fact']} | train G-p {a['p_screen']:.2e} perm p {a['p_perm']} bonf {a['bonf']} bh {a['bh']} | held-out {fmt(a['held'])} | levels train {fmt(a['train_strong'])}; {fmt(a['train_all'])} | held levels {fmt(a['held_strong'])}; {fmt(a['held_all'])} | within site x type x len {fmt(a['within_type'])} | flags {a['flags'] or 'none'}")
redis.sort(key=lambda a:a['p_screen'])
rc=collections.Counter((a['kind'],a['fact']) for a in redis)
say(f'rediscoveries by (reading, fact), top 15 of {len(rc)}: '+'; '.join(f'{k[0]}->{k[1]} x{v}' for k,v in rc.most_common(15)))
for a in redis[:10]: say(f"  rediscovery {a['kind']} {a['p']} -> {a['fact']} G-p {a['p_screen']:.1e} n {a['n']} within-strata {fmt(a['within_type'])} | {a['flags']}")
byk=collections.defaultdict(list)
for a in evalu: byk[a['kind']].append(a['p_screen'])
say('family: arrows / frac screen p<0.01 (null 0.01):')
for k,v in sorted(byk.items(),key=lambda kv:-np.mean(np.array(kv[1])<0.01)): say(f'  {k:12s} {len(v):4d} {np.mean(np.array(v)<0.01):.3f}')
say(f'elapsed {time.time()-t0:.0f}s')
out=A.out or f"data/derived/dark/{A.tag}_cycle{A.cycle}{'_control' if CONTROL else ''}.txt"
os.makedirs(os.path.dirname(out),exist_ok=True)
open(out,'w').write('\n'.join(log)+'\n')
json.dump(dict(arrows=arrows),open(out.replace('.txt','.json'),'w'),default=str)
