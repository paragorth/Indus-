"""S-DARK-35: RANK THE CLOSER PARADIGM BY PHYSICAL COST.
If the closers (S289 paradigm: jar 740, arrow 520, 151, 156, 527, 226, 617, 154/158, 236, 700) are grades or authorities
in a credential code, the paradigm should have an ORDER, and that order should show in what the seal cost: face area
(h x v), thickness, material class, emblem class, boss, and use (text found on a sealing/tablet). The MI machine (S365,
S-DARK-1) tests association; here we test ORDINALITY: do independent splits and independent cost measures give the SAME
ranking of closers?
  cycle 1: log face area ~ closer class + text-length bin + site + material class + emblem class, random intercept for
           site x area-section (REML-profiled ridge), seals at Mohenjo-daro + Harappa. Closer effects with 95% CI.
           Ordinality: Spearman rank correlation of the closer effects between splits (MD vs Harappa; steatite vs other
           material; unicorn vs other emblem; random halves), against a null that permutes closer labels among seals
           within site x length bin (NP x), refitting both splits each time.
  cycle 2: the same engine for log thickness, material rank (clay/terracotta 0 < faience/paste 1 < steatite 2 <
           hard stone/metal/shell/ivory 3), unicorn (0/1), boss present (0/1), used (0/1; text on a TAG/TAB anywhere).
           Cost table; Kendall W of the closer rankings across measures vs the same label permutation.
  cycle 3: do the S303 qualifier sets rank by the same order? Per closer: mean number of TITLE signs in its closer unit,
           qualifier type/token ratio, mean rarity (-log freq) of the left partner, share of texts that are the bare
           closer unit. Spearman against the cycle-1/2 cost rank; exact permutation of the closer order (m! orders).
  cycle 4: out of sample: seals with size data at all other sites (held-out), same model with site fixed effects;
           Spearman of the held-out closer effects with the MD+H ranking vs label permutation within site x length.
           IM77 carries no dimensions (IM77_README), so it cannot replicate size; stated. Prediction: the top-ranked
           closer should almost never sit on the cheapest material class; count with upper bound (< 3/n).
Data: data/derived/merged-corpus-canonical.json (seq_raw / seq_strong / seq_all) aligned row by row to
data/raw/inscriptions.csv (horizontal(mm), vertical(mm), thickness(mm), material, boss, symbol, area-section).
Usage: python3 tools/dark_loop35.py <cycle 1|2|3|4> <seq_raw|seq_strong|seq_all> [nperm]
"""
import json,sys,csv,re,random,collections,math,itertools
import numpy as np
CY=int(sys.argv[1]); LV=sys.argv[2]; NP=int(sys.argv[3]) if len(sys.argv)>3 else 1000
rng=np.random.default_rng(35); rnd=random.Random(35)
C=json.load(open('data/derived/merged-corpus-canonical.json'))
R=list(csv.DictReader(open('data/raw/inscriptions.csv')))
BR=json.load(open('data/derived/bridge_extended.json'))
def Mno(w):
    m=BR.get(str(w)); return f'W{w}/M{"+".join(str(x) for x in m)}' if m else f'W{w}'
def bad(v): return v is None or str(v).strip() in ('-','--','- -','')
def fnum(x):
    try: v=float(x); return v if v>0 else None
    except: return None
# ---------- align canonical records to CSV rows (sequential; every record aligns, checked) ----------
def parse_csv_text(t):
    t=re.sub(r'[\[\]\+]','',t); out=[]
    for tok in re.split(r'[-/]',t):
        tok=tok.strip()
        if tok=='': continue
        try: out.append(int(tok))
        except: pass
    return out[::-1]
j=0; AL={}
for i,c in enumerate(C):
    k=(c['cisi'],c['site'],c['type'])
    for jj in range(j,min(j+400,len(R))):
        r=R[jj]
        if (r['cisi'],r['site'],r['type'])==k:
            p=parse_csv_text(r['text'])
            if p==c['seq_raw'] or [x for x in p if x!=0]==c['seq_raw']: AL[i]=jj; j=jj+1; break
assert len(AL)==len(C), 'alignment failed'
# ---------- closer classes ----------
OPEN={817,861,820,920,692}; MARK={2,60}; MJAR={741,742,745}; SUF={400,90}
CL=[740,520,151,156,527,226,617,154,158,236,700]
FISH={235,240,233,231,220}; NUM={1,3,4,5,16,17,18,31,32,33,34,55,56}
def closer_of(seq):
    s=list(seq)
    while len(s)>1 and s[-1] in SUF: s.pop()
    if s[-1] in CL: return s[-1]
    if s[-1] in (817,861,820) and len(s)>1: return 'OPN'   # S286 opener-final
    return 'none'
def matclass(m):
    m=(m or '').strip().lower()
    if m in ('clay','terracotta'): return 0
    if m in ('faience','paste'): return 1
    if m in ('steatite','soft stone','kaolinite'): return 2
    if m in ('agate','silver','copper','stone','limestone','metal','shell','ivory','bone'): return 3
    return None
def embclass(s):
    s=(s or '').strip()
    if s in ('','-'): return 'unk'
    s=s.split(':')[0]
    if s=='None': return 'none'
    if s=='Bull1': return 'unicorn'
    return 'other'
def bossclass(b):
    b=(b or '').strip()
    if b.startswith('P'): return 1
    if b.startswith('N'): return 0
    return None
def lenbin(L): return min(L,8) if L<8 else (8 if L<10 else 10)
# used = exact seal text appears on a TAG/TAB line anywhere (S-DARK-16.3)
TABTXT=set(tuple(c[LV]) for c in C if c['type'].split(':')[0] in ('TAB','TAG') and c[LV])
OBJ=[]
for i,c in enumerate(C):
    if not c['type'].startswith('SEAL'): continue
    s=c[LV]
    if not s or c['complete']!='Y': continue
    r=R[AL[i]]
    H=fnum(r['horizontal(mm)']); V=fnum(r['vertical(mm)']); T=fnum(r['thickness(mm)'])
    OBJ.append(dict(cisi=c['cisi'],site=c['site'],seq=tuple(s),L=len(s),clo=closer_of(s),
                    area=(H*V if H and V else None),th=T,mat=matclass(r['material']),emb=embclass(r['symbol']),
                    boss=bossclass(r['boss']),used=int(len(s)>=2 and tuple(s) in TABTXT),
                    sec=(c['site'],'?' if bad(c['area-section']) else c['area-section'].strip()),shape=r['shape'].strip()))
HOME={'Mohenjo-daro','Harappa'}
home=[o for o in OBJ if o['site'] in HOME]
cnt=collections.Counter(o['clo'] for o in home)
CLASSES=[k for k,n in cnt.most_common() if k not in ('none','OPN') and n>=8]   # paradigm closers with >= 8 home seals
print(f'== S-DARK-35 cycle {CY} level {LV} nperm {NP}: complete seal texts {len(OBJ)}, home {len(home)}; closer counts {dict(cnt)}')
print('closer classes used (>= 8 home seals):',[Mno(c) for c in CLASSES])
# ---------- mixed-model engine ----------
def design(objs,classes,ref=740,site_fe=True,use_mat=True,use_emb=True):
    """fixed: intercept, closer dummies (ref=jar; 'none' and 'OPN' and rare closers as own levels), length bins, site,
    material class, emblem class. random: site x area-section."""
    cols=['int']; X=[np.ones(len(objs))]
    levels=[c for c in classes if c!=ref]+['none','OPN','rare']
    for lv in levels:
        v=np.array([1.0 if (o['clo']==lv or (lv=='rare' and o['clo'] not in classes and o['clo'] not in ('none','OPN'))) else 0.0 for o in objs])
        if v.sum()>0: X.append(v); cols.append(f'clo:{lv}')
    for b in sorted(set(lenbin(o['L']) for o in objs))[1:]:
        X.append(np.array([1.0 if lenbin(o['L'])==b else 0.0 for o in objs])); cols.append(f'len:{b}')
    if site_fe:
        for s in sorted(set(o['site'] for o in objs))[1:]:
            X.append(np.array([1.0 if o['site']==s else 0.0 for o in objs])); cols.append(f'site:{s}')
    for m in (sorted(set(o['mat'] for o in objs),key=lambda x:(x is None,x)) if use_mat else []):
        if m==2: continue
        X.append(np.array([1.0 if o['mat']==m else 0.0 for o in objs])); cols.append(f'mat:{m}')
    for e in (('none','other','unk') if use_emb else []):
        v=np.array([1.0 if o['emb']==e else 0.0 for o in objs])
        if v.sum()>0: X.append(v); cols.append(f'emb:{e}')
    X=np.column_stack(X)
    # drop constant/collinear columns
    keep=[k for k in range(X.shape[1]) if k==0 or (X[:,k].std()>0)]
    X=X[:,keep]; cols=[cols[k] for k in keep]
    groups=sorted(set(o['sec'] for o in objs)); gi={g:k for k,g in enumerate(groups)}
    Z=np.zeros((len(objs),len(groups)))
    for k,o in enumerate(objs): Z[k,gi[o['sec']]]=1.0
    return X,cols,Z
def _parts(y,X,Z):
    ng=Z.sum(0); return dict(XtX=X.T@X,XtZ=X.T@Z,Xty=X.T@y,Zty=Z.T@y,yty=float(y@y),ng=ng,n=len(y),p=X.shape[1])
def _solve(P,lam):
    D=1.0/(lam+P['ng'])
    A=P['XtX']-(P['XtZ']*D)@P['XtZ'].T
    bv=P['Xty']-P['XtZ']@(D*P['Zty'])
    try: b=np.linalg.solve(A,bv)
    except np.linalg.LinAlgError: b=np.linalg.pinv(A)@bv
    yVy=P['yty']-float(P['Zty']@(D*P['Zty']))
    rVr=yVy-2*float(b@bv)+float(b@A@b)
    return b,A,rVr
def reml_fit(y,X,Z,lam=None,grid=np.logspace(-2,2.5,19)):
    """y = Xb + Zu + e, u ~ N(0, s2/lam), e ~ N(0, s2); Z'Z diagonal (one group per row). Woodbury forms, profile REML
    over lam. Returns b, se, lam, s2, lam (the last stands in for V^-1: with lam fixed, refits need only lam)."""
    P=_parts(y,X,Z); n,p=P['n'],P['p']
    def crit(lm):
        b,A,rVr=_solve(P,lm); s2=rVr/(n-p)
        ll=-0.5*(np.log(1+P['ng']/lm).sum()+np.linalg.slogdet(A)[1]+(n-p)*math.log(max(s2,1e-12)))
        return ll,b,s2,A
    if lam is None:
        best=None
        for lm in grid:
            ll,b,s2,A=crit(lm)
            if best is None or ll>best[0]: best=(ll,lm,b,s2,A)
        ll,lam,b,s2,A=best
    else: ll,b,s2,A=crit(lam)
    se=np.sqrt(np.maximum(np.diag(np.linalg.pinv(A)),0)*s2)
    return b,se,lam,s2,lam
def refit(y,X,Z,lam):
    b,A,rVr=_solve(_parts(y,X,Z),lam); return b
def closer_effects(objs,y,classes,lam=None,Vi=None,ref=740,use_mat=True,use_emb=True):
    X,cols,Z=design(objs,classes,ref,True,use_mat,use_emb)
    if Vi is None: b,se,lam,s2,Vi=reml_fit(y,X,Z,lam)
    else: b=refit(y,X,Z,Vi); se=np.full(len(b),np.nan); s2=np.nan
    eff={ref:0.0}; sef={ref:0.0}
    for c in classes:
        if c==ref: continue
        if f'clo:{c}' in cols: k=cols.index(f'clo:{c}'); eff[c]=float(b[k]); sef[c]=float(se[k])
    extra={lv:float(b[cols.index(f'clo:{lv}')]) for lv in ('none','OPN','rare') if f'clo:{lv}' in cols}
    return eff,sef,extra,lam,Vi,cols,b,se
def spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<3: return float('nan')
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b))
    if ra.std()==0 or rb.std()==0: return float('nan')
    return float(np.corrcoef(ra,rb)[0,1])
def kendall_w(rank_lists):
    """rank_lists: k lists of effects over the same m items; returns W."""
    Rk=np.array([np.argsort(np.argsort(np.asarray(r,float)))+1 for r in rank_lists],float)
    k,m=Rk.shape; S=((Rk.sum(0)-k*(m+1)/2)**2).sum()
    return float(12*S/(k*k*(m**3-m)))
def permute_labels(objs):
    """permute closer labels among seals within site x length bin; returns new list of dicts (shallow copies)."""
    strata=collections.defaultdict(list)
    for k,o in enumerate(objs): strata[(o['site'],lenbin(o['L']))].append(k)
    lab=[o['clo'] for o in objs]; new=list(lab)
    for idx in strata.values():
        vals=[lab[k] for k in idx]; rnd.shuffle(vals)
        for k,v in zip(idx,vals): new[k]=v
    return [dict(o,clo=v) for o,v in zip(objs,new)]
def pv(obs,null,hi=True):
    null=[x for x in null if x==x]
    if obs!=obs or not null: return float('nan')
    return (sum(1 for x in null if (x>=obs if hi else x<=obs))+1)/(len(null)+1)
def fmt_eff(eff,sef,classes,unit=''):
    out=[]
    for c in sorted(classes,key=lambda c:-eff.get(c,float('nan')) if c in eff else 0):
        if c in eff: out.append(f'{Mno(c)} {eff[c]:+.3f}'+(f' [{eff[c]-1.96*sef[c]:+.3f},{eff[c]+1.96*sef[c]:+.3f}]' if sef.get(c,0)==sef.get(c,0) and sef.get(c,0)>0 else ''))
    return '; '.join(out)
def ub(k,n): return f'{k}/{n}' if k>0 else (f'0/{n} (< {3/n:.3f})' if n else '0/0')

MEASURES=[('log_area',lambda o:(math.log(o['area']) if o['area'] else None)),
          ('log_thick',lambda o:(math.log(o['th']) if o['th'] else None)),
          ('mat_rank',lambda o:(float(o['mat']) if o['mat'] is not None else None)),
          ('unicorn',lambda o:(None if o['emb']=='unk' else float(o['emb']=='unicorn'))),
          ('boss',lambda o:(float(o['boss']) if o['boss'] is not None else None)),
          ('used',lambda o:float(o['used']))]
def subset(objs,meas,classes,minn=3):
    f=dict(MEASURES)[meas]
    S=[(o,f(o)) for o in objs if f(o) is not None]
    return [o for o,_ in S],np.array([v for _,v in S])

# =====================================================================================================================
if CY==1:
    S,y=subset(home,'log_area',CLASSES)
    print(f'\n--- cycle 1: log face area, home seals with H and V: n={len(S)}; by site {collections.Counter(o["site"] for o in S)}; closers {collections.Counter(o["clo"] for o in S)}')
    eff,sef,extra,lam,Vi,cols,b,se=closer_effects(S,y,CLASSES)
    print(f'REML lambda (s2_e/s2_u) = {lam:.3g}; n groups (site x area-section) = {len(set(o["sec"] for o in S))}')
    print('closer effects on log area (vs jar W740/M342 = 0), 95% CI:'); print('  '+fmt_eff(eff,sef,CLASSES))
    print('  other levels:',{k:round(v,3) for k,v in extra.items()})
    print('  covariates:',' '.join(f'{c}={v:+.2f}' for c,v in zip(cols,b) if not c.startswith('clo:')))
    # omnibus: variance of closer effects vs permutation
    def spread(e): v=[e[c] for c in CLASSES if c in e]; return float(np.var(v)) if len(v)>1 else float('nan')
    obs_spread=spread(eff)
    # splits
    SPLITS={'MD vs Harappa':(lambda o:o['site']=='Mohenjo-daro',lambda o:o['site']=='Harappa'),
            'steatite vs other-material':(lambda o:o['mat']==2,lambda o:o['mat'] is not None and o['mat']!=2),
            'unicorn vs other-emblem':(lambda o:o['emb']=='unicorn',lambda o:o['emb'] in('none','other')),
            'square vs non-square shape':(lambda o:o['shape']=='square',lambda o:o['shape'] not in('square','-',''))}
    for o in S: o['half']=rnd.random()<0.5
    SPLITS['random halves']=(lambda o:o['half'],lambda o:not o['half'])
    def split_rho(objs,yv,fa,fb,lamfix,Vis=None):
        A=[k for k,o in enumerate(objs) if fa(o)]; B=[k for k,o in enumerate(objs) if fb(o)]
        oa=[objs[k] for k in A]; ob=[objs[k] for k in B]
        ca=collections.Counter(o['clo'] for o in oa); cb=collections.Counter(o['clo'] for o in ob)
        common=[c for c in CLASSES if ca[c]>=3 and cb[c]>=3]
        if len(common)<4: return float('nan'),common,None,None,None
        ea,sa,_,la,Via,_,_,_=closer_effects(oa,yv[A],common,lamfix,None if Vis is None else Vis[0])
        eb,sb,_,lb,Vib,_,_,_=closer_effects(ob,yv[B],common,lamfix,None if Vis is None else Vis[1])
        return spearman([ea[c] for c in common],[eb[c] for c in common]),common,(ea,sa,eb,sb),(Via,Vib),(ca,cb)
    results={}
    for name,(fa,fb) in SPLITS.items():
        rho,common,effs,Vis,cc=split_rho(S,y,fa,fb,lam)
        if rho!=rho: print(f'\nsplit {name}: too few closers shared (common={[Mno(c) for c in common]})'); continue
        ea,sa,eb,sb=effs
        print(f'\nsplit {name}: common closers {len(common)}; counts A {[cc[0][c] for c in common]} B {[cc[1][c] for c in common]}')
        print('  A:',fmt_eff(ea,sa,common)); print('  B:',fmt_eff(eb,sb,common)); print(f'  Spearman rho between splits = {rho:+.3f}')
        results[name]=(rho,fa,fb,common,Vis)
    # permutation null (labels permuted within site x length; lam fixed, V fixed per split)
    null={k:[] for k in results}; null_spread=[]
    for t in range(NP):
        P=permute_labels(S)
        e,_,_,_,_,_,_,_=closer_effects(P,y,CLASSES,lam,Vi); null_spread.append(spread(e))
        for name,(rho,fa,fb,common,Vis) in results.items():
            r,_,_,_,_=split_rho(P,y,fa,fb,lam,Vis); null[name].append(r)
    print(f'\nomnibus: variance of closer effects (log area) obs {obs_spread:.4f}; null median {np.nanmedian(null_spread):.4f}, 95% {np.nanpercentile(null_spread,95):.4f}; P = {pv(obs_spread,null_spread):.3f}')
    for name,(rho,*_r) in results.items():
        nn=[x for x in null[name] if x==x]
        print(f'ordinality {name}: rho {rho:+.3f}; null median {np.median(nn):+.3f}, 95% {np.percentile(nn,95):+.3f}; P(one-sided) = {pv(rho,nn):.3f}  (null n={len(nn)})')
    # raw means per closer for the reader
    print('\nraw face area (mm2) by closer, home seals: median [n]')
    for c in CLASSES+['none','OPN']:
        v=[o['area'] for o in S if o['clo']==c]
        if v: print(f'  {Mno(c) if c not in("none","OPN") else c}: {np.median(v):.0f} [{len(v)}], mean text length {np.mean([o["L"] for o in S if o["clo"]==c]):.1f}')

# =====================================================================================================================
if CY==2:
    table={}; sefs={}; ns={}; lams={}; Vis={}; subsets={}
    for meas,_f in MEASURES:
        S,y=subset(home,meas,CLASSES)
        if meas in ('unicorn',): S=[o for o in S if True];
        eff,sef,extra,lam,Vi,cols,b,se=closer_effects(S,y,CLASSES,use_mat=(meas!='mat_rank'),use_emb=(meas!='unicorn'))
        table[meas]=eff; sefs[meas]=sef; ns[meas]=collections.Counter(o['clo'] for o in S); lams[meas]=lam; Vis[meas]=Vi; subsets[meas]=(S,y)
        print(f'\n--- {meas}: n={len(S)}, lambda {lam:.3g}; mean y {y.mean():.3f}')
        print('  '+fmt_eff(eff,sef,CLASSES)); print('  other levels:',{k:round(v,3) for k,v in extra.items()})
    # cost table
    print('\n=== COST TABLE (effect vs jar; + = costlier/used more) ===')
    hdr='closer'.ljust(16)+''.join(m.rjust(11) for m,_ in MEASURES)+'   mean rank'
    print(hdr)
    ranks={m:{c:r for r,c in enumerate(sorted(CLASSES,key=lambda c:table[m].get(c,0)),1)} for m,_ in MEASURES}
    meanrank={c:np.mean([ranks[m][c] for m,_ in MEASURES]) for c in CLASSES}
    for c in sorted(CLASSES,key=lambda c:-meanrank[c]):
        print(f'{Mno(c):16s}'+''.join(f'{table[m].get(c,float("nan")):+11.3f}' for m,_ in MEASURES)+f'   {meanrank[c]:.2f}  (n area {ns["log_area"][c]})')
    W=kendall_w([[table[m][c] for c in CLASSES] for m,_ in MEASURES])
    Wphys=kendall_w([[table[m][c] for c in CLASSES] for m in ('log_area','log_thick','mat_rank')])
    print(f'\nKendall W across {len(MEASURES)} measures = {W:.3f}; across the 3 physical measures (area, thickness, material) = {Wphys:.3f}')
    # pairwise spearman between measures
    for (m1,_),(m2,_) in itertools.combinations(MEASURES,2):
        print(f'  rho({m1},{m2}) = {spearman([table[m1][c] for c in CLASSES],[table[m2][c] for c in CLASSES]):+.2f}')
    # permutation null: permute labels on the home set, recompute each measure (fixed lambda/V), W
    nullW=[]; nullWp=[]
    for t in range(NP):
        P=permute_labels(home); pid={o['cisi']+str(k):v['clo'] for k,(o,v) in enumerate(zip(home,P))}
        lab={id(o):v['clo'] for o,v in zip(home,P)}
        effs=[]
        for meas,_f in MEASURES:
            S,y=subsets[meas]
            Pm=[dict(o,clo=lab[id(o)]) for o in S]
            e,*_=closer_effects(Pm,y,CLASSES,lams[meas],Vis[meas],use_mat=(meas!='mat_rank'),use_emb=(meas!='unicorn')); effs.append([e.get(c,0.0) for c in CLASSES])
        nullW.append(kendall_w(effs)); nullWp.append(kendall_w([effs[0],effs[1],effs[2]]))
    print(f'Kendall W null: median {np.median(nullW):.3f}, 95% {np.percentile(nullW,95):.3f}; P = {pv(W,nullW):.3f}')
    print(f'Kendall W (3 physical) null: median {np.median(nullWp):.3f}, 95% {np.percentile(nullWp,95):.3f}; P = {pv(Wphys,nullWp):.3f}')
    json.dump({'classes':CLASSES,'table':{m:{str(c):table[m].get(c) for c in CLASSES} for m,_ in MEASURES},'meanrank':{str(c):meanrank[c] for c in CLASSES},
               'W':W,'P_W':pv(W,nullW),'Wphys':Wphys,'P_Wphys':pv(Wphys,nullWp)},open(f'data/derived/dark/loop35_c2_{LV}.json','w'),indent=1)
    # raw material class x closer
    print('\nmaterial class by closer (home seals, all with material): 0 clay/terracotta, 1 faience/paste, 2 steatite, 3 hard stone/metal')
    for c in CLASSES+['none','OPN']:
        cc=collections.Counter(o['mat'] for o in home if o['clo']==c and o['mat'] is not None)
        print(f'  {Mno(c) if c not in("none","OPN") else c}: '+' '.join(f'{k}:{cc[k]}' for k in (0,1,2,3)))

# =====================================================================================================================
if CY==3:
    # cost rank from cycle 2 json (fall back to cycle-1-style area fit)
    try: J=json.load(open(f'data/derived/dark/loop35_c2_{LV}.json')); cost={int(k):v for k,v in J['meanrank'].items()}; area={int(k):v for k,v in J['table']['log_area'].items()}
    except Exception:
        S,y=subset(home,'log_area',CLASSES); eff,*_=closer_effects(S,y,CLASSES); cost=eff; area=eff
    # qualifier statistics per closer (all complete seal texts, home + other, like S303: sign before the closer)
    freq=collections.Counter(a for c in C if c[LV] for a in c[LV]); tot=sum(freq.values())
    def strip(seq):
        s=list(seq)
        while len(s)>1 and s[-1] in SUF: s.pop()
        return s
    # closer-unit length via the S310 parser's TITLE rule (qualifier sets from left-partner distributions, 60% mass)
    left=collections.defaultdict(collections.Counter)
    for o in OBJ:
        s=strip(o['seq'])
        if len(s)>=2 and s[-1] in CL: left[s[-1]][s[-2]]+=1
    Q={}
    for c in CLASSES:
        txt=[strip(o['seq']) for o in OBJ if o['clo']==c]
        n=len(txt); withq=[t for t in txt if len(t)>=2]
        parts=[t[-2] for t in withq]
        types=len(set(parts)); ttr=types/len(parts) if parts else float('nan')
        rarity=np.mean([-math.log(freq[a]/tot) for a in parts]) if parts else float('nan')
        # 'title phrase length' = how many signs before the closer belong to its 60%-mass qualifier set, scanning left
        qs=set(); acc=0; T=sum(left[c].values())
        for a,k in left[c].most_common():
            if acc/T>=0.6: break
            qs.add(a); acc+=k
        tl=[]
        for t in txt:
            k=0
            while len(t)-2-k>=0 and (t[-2-k] in qs or (k>0 and t[-2-k] in NUM)): k+=1
            tl.append(k)
        bare=sum(1 for t,k in zip(txt,tl) if len(t)==1+k)/n   # text is the closer unit alone
        lenall=np.mean([len(t) for t in txt])
        top=left[c].most_common(1)[0] if left[c] else (None,0)
        Q[c]=dict(n=n,title_len=float(np.mean(tl)),qual_types=types,ttr=ttr,rarity=float(rarity),bare=bare,text_len=float(lenall),top_share=top[1]/max(1,len(parts)),top=top[0])
    print('\n=== qualifier statistics per closer (complete seal texts, all sites) ===')
    print('closer'.ljust(16)+'   n  title_len  qual_types   TTR  rarity  bare  text_len  top_partner(share)   cost_rank  area_eff')
    for c in sorted(CLASSES,key=lambda c:-cost[c]):
        q=Q[c]; print(f'{Mno(c):16s} {q["n"]:4d}  {q["title_len"]:8.2f}  {q["qual_types"]:9d}  {q["ttr"]:.2f}  {q["rarity"]:5.2f}  {q["bare"]:.2f}  {q["text_len"]:7.2f}  {Mno(q["top"]) if q["top"] else "-":>14s}({q["top_share"]:.2f})   {cost[c]:.2f}   {area[c]:+.3f}')
    m=len(CLASSES); orders=list(itertools.permutations(range(m))) if m<=9 else None
    for stat in ('title_len','qual_types','ttr','rarity','bare','text_len','top_share'):
        v=[Q[c][stat] for c in CLASSES]
        for cname,cv in (('mean cost rank',[cost[c] for c in CLASSES]),('area effect',[area[c] for c in CLASSES])):
            rho=spearman(v,cv)
            if orders:
                null=[spearman(v,[cv[k] for k in p]) for p in orders]
                p2=sum(1 for x in null if x==x and abs(x)>=abs(rho)-1e-12)/len(null)
            else: p2=float('nan')
            print(f'  rho({stat}, {cname}) = {rho:+.3f}; exact two-sided P over {m}! orders = {p2:.3f}')

# =====================================================================================================================
if CY==4:
    try: J=json.load(open(f'data/derived/dark/loop35_c2_{LV}.json')); cost={int(k):v for k,v in J['meanrank'].items()}; area_home={int(k):v for k,v in J['table']['log_area'].items()}
    except Exception:
        S,y=subset(home,'log_area',CLASSES); eff,*_=closer_effects(S,y,CLASSES); cost=eff; area_home=eff
    other=[o for o in OBJ if o['site'] not in HOME and o['site']!='Unknown']
    S,y=subset(other,'log_area',CLASSES)
    cc=collections.Counter(o['clo'] for o in S)
    common=[c for c in CLASSES if cc[c]>=3]
    print(f'\n--- held-out sites: complete seal texts with H,V n={len(S)}; sites {collections.Counter(o["site"] for o in S).most_common(12)}')
    print(f'closer counts {dict(cc)}; closers with >= 3 seals: {[Mno(c) for c in common]}')
    if len(common)>=4:
        eff,sef,extra,lam,Vi,cols,b,se=closer_effects(S,y,common)
        print('held-out closer effects on log area:'); print('  '+fmt_eff(eff,sef,common))
        rho_a=spearman([eff[c] for c in common],[area_home[c] for c in common]); rho_c=spearman([eff[c] for c in common],[cost[c] for c in common])
        null_a=[]; null_c=[]
        for t in range(NP):
            P=permute_labels(S); e,*_=closer_effects(P,y,common,lam,Vi)
            null_a.append(spearman([e.get(c,0) for c in common],[area_home[c] for c in common])); null_c.append(spearman([e.get(c,0) for c in common],[cost[c] for c in common]))
        print(f'Spearman(held-out area effects, home area effects) = {rho_a:+.3f}; null median {np.nanmedian(null_a):+.3f}, 95% {np.nanpercentile(null_a,95):+.3f}; P = {pv(rho_a,null_a):.3f}')
        print(f'Spearman(held-out area effects, home mean cost rank) = {rho_c:+.3f}; null 95% {np.nanpercentile(null_c,95):+.3f}; P = {pv(rho_c,null_c):.3f}')
        # per-site sign check
        for site in ('Dholavira','Lothal','Kalibangan','Chanhu-daro'):
            Ss=[o for o in S if o['site']==site]
            if len(Ss)<15: continue
            med={c:np.median([o['area'] for o in Ss if o['clo']==c]) for c in common if sum(1 for o in Ss if o['clo']==c)>=2}
            print(f'  {site} (n={len(Ss)}): median area by closer '+', '.join(f'{Mno(c)} {v:.0f}' for c,v in sorted(med.items(),key=lambda x:-x[1])))
    else: print('too few closers at held-out sites for a ranking')
    # thickness and other measures at held-out
    for meas in ('log_thick','unicorn','boss','used'):
        S2,y2=subset(other,meas,CLASSES); c2=collections.Counter(o['clo'] for o in S2); com2=[c for c in CLASSES if c2[c]>=3]
        if len(com2)>=4:
            e2,s2,*_=closer_effects(S2,y2,com2,use_mat=(meas!='mat_rank'),use_emb=(meas!='unicorn'))
            print(f'  held-out {meas} (n={len(S2)}): '+fmt_eff(e2,s2,com2)+f'  | rho with home cost rank {spearman([e2[c] for c in com2],[cost[c] for c in com2]):+.2f}')
    # IM77: no dimensions
    hdr=open('data/im77/im77_corpus_lines.csv').readline().strip().split(',')
    print(f'\nIM77 columns: {hdr}; size fields present: {[h for h in hdr if any(k in h.lower() for k in ("mm","size","dim","width","height","thick"))]} -> IM77 cannot replicate size.')
    # prediction: top-cost closer on the cheapest material class
    top=max(CLASSES,key=lambda c:cost[c]); bot=min(CLASSES,key=lambda c:cost[c])
    print(f'\nprediction: top-ranked closer {Mno(top)} vs bottom {Mno(bot)} on the cheapest material classes (all sites, complete seal texts with material):')
    for c in (top,bot,740):
        mm=collections.Counter(o['mat'] for o in OBJ if o['clo']==c and o['mat'] is not None); n=sum(mm.values())
        print(f'  {Mno(c)}: clay/terracotta {ub(mm[0],n)}; faience/paste {ub(mm[1],n)}; steatite {mm[2]}/{n}; hard stone/metal {ub(mm[3],n)}')
    allm=collections.Counter(o['mat'] for o in OBJ if o['mat'] is not None); n=sum(allm.values())
    print(f'  all complete seals: clay/terracotta {allm[0]}/{n} ({allm[0]/n:.3f}), faience/paste {allm[1]}/{n} ({allm[1]/n:.3f}), hard {allm[3]}/{n}')
    # which seals of the cheapest classes carry which closers
    cheap=[o for o in OBJ if o['mat'] in (0,1)]
    print(f'  closers on cheap-material seals (n={len(cheap)}): {collections.Counter(Mno(o["clo"]) if o["clo"] not in ("none","OPN") else o["clo"] for o in cheap)}')
    print(f'  sites of cheap-material seals: {collections.Counter(o["site"] for o in cheap)}')
