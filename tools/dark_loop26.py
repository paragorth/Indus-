"""S-DARK-26: IS THE MIDDLE A LEXICON OR AN ATTRIBUTE CODE?
Middle = the NAME + COUNT residue of every text after the S310/S331 frame parser strips opener, connective, titles,
closer and suffix (same parser as tools/dark_loop22.py / parse_all.py).
  cycle 1 (a) independence census: observed co-occurrence of every pair of middle elements vs the stratified
              permutation expectation; share of pairs with |log2(O/E)| > 1 after null-calibration; held-out
              log-likelihood of an independence (unigram-given-length) model vs a pair (bigram) model; adjacent-pair MI.
  cycle 2 (b) position-specific marginals: MI(first element, last element) vs within-position shuffle; per-element
              partner prediction (KL of partner distribution vs the marginal); (d) quantity seals vs office seals.
  cycle 3 (e) slot inference from complementary distribution (mutual avoidance graph), inventory per slot;
              Chao1 for distinct middles and for middle elements; comparison with S309.
Controls in every cycle: Ur III seal-owner names (line 1 of distinct legends, syllables), a planted attribute code
(K independent slots with Zipf marginals), a planted lexicon (fixed word list, Zipf frequencies, middles = 1-3 words),
all generated at the Indus middle count and length distribution.
Usage: python3 tools/dark_loop26.py <cycle 1|2|3> <seq_raw|seq_strong|seq_all> [nperm]
"""
import json,sys,random,collections,math,itertools
C=json.load(open('data/derived/merged-corpus-canonical.json'))
BR=json.load(open('data/derived/bridge_extended.json'))
SP='/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
def M(w):
    m=BR.get(str(w)); return f'W{w}(M{"/".join(map(str,m))})' if m else f'W{w}'
CY=int(sys.argv[1]); LV=sys.argv[2]; NP=int(sys.argv[3]) if len(sys.argv)>3 else 200
rnd=random.Random(26)
def otype(t):
    t=t.split(':')[0]
    return {'SEAL':'seal','TAB':'tablet','POT':'pot','TAG':'sealing'}.get(t,'other')
OBJ=[]
for r in C:
    s=r[LV]
    if not s or len(s)<2 or r['complete']!='Y' or r['dir.'].strip()=='-': continue
    OBJ.append(dict(cisi=r['cisi'],site=r['site'],ot=otype(r['type']),seq=list(s),big=r['site'] in('Mohenjo-daro','Harappa')))
print(f'== S-DARK-26 cycle {CY} level {LV} nperm {NP}; objects {len(OBJ)} (complete, direction recorded, >=2 signs)')

# ---------------- frame parser (S310 parse_all.py + S331 openers, as in dark_loop19/22) ----------------
OPEN={817,861,820,920,692}; MARK={2,60}; MJAR={741,742,745}; SUF={400,90}; CL=[740,520,151,156,527,226,617,154,158,236,700]
FISH={235,240,233,231,220}; NUM={1,3,4,5,16,17,18,31,32,33,34,55,56}
left=collections.defaultdict(collections.Counter)
for o in OBJ:
    s=o['seq'][:]
    while len(s)>1 and s[-1] in SUF: s.pop()
    if len(s)>=2 and s[-1] in CL: left[s[-1]][s[-2]]+=1
QUAL={}
for c,cnt in left.items():
    tot=sum(cnt.values()); acc=0; q=set()
    for a,n in cnt.most_common():
        if acc/tot>=0.6: break
        q.add(a); acc+=n
    QUAL[c]=q
def parse(s):
    lab=['NAME']*len(s); i=0; j=len(s)
    if s[0] in OPEN:
        lab[0]='OPENER'; i=1
        if len(s)>1 and s[1] in MARK:
            lab[1]='MARKER'; i=2
            if s[0]==920 and len(s)>2 and s[2] in MJAR: lab[2]='MARKER'; i=3
    while j-1>i and s[j-1] in SUF and j>=2 and (s[j-2] in CL or s[j-2] in SUF): lab[j-1]='SUFFIX'; j-=1
    if j-1>=i and s[j-1] in CL:
        c=s[j-1]; lab[j-1]='CLOSER'; j-=1
        if c==520:
            if j-2>=i and s[j-1]==33 and s[j-2] in (705,706): lab[j-1]=lab[j-2]='TITLE'; j-=2
            while j-1>=i and s[j-1] in FISH: lab[j-1]='TITLE'; j-=1
        elif c==740:
            if j-1>=i and s[j-1]==100: lab[j-1]='TITLE'; j-=1
            if j-1>=i and s[j-1] in QUAL.get(c,()): lab[j-1]='TITLE'; j-=1
        elif j-1>=i and s[j-1] in QUAL.get(c,()): lab[j-1]='TITLE'; j-=1
        if j-1>=i and s[j-1] in NUM and lab[j]=='TITLE': lab[j-1]='TITLE'; j-=1
    for k in range(i,j-1):
        if s[k] in NUM and lab[k]=='NAME' and lab[k+1]=='NAME': lab[k]=lab[k+1]='COUNT'
    for k in range(i,j):
        if s[k] in NUM and lab[k]=='NAME': lab[k]='COUNT'
    return lab
for o in OBJ:
    o['lab']=parse(o['seq'])
    o['mid']=[a for a,l in zip(o['seq'],o['lab']) if l in('NAME','COUNT')]
    o['closer']=next((a for a,l in zip(o['seq'],o['lab']) if l=='CLOSER'),None)
    o['hascount']=any(l=='COUNT' for l in o['lab'])
    o['opener']=o['lab'][0]=='OPENER'
    o['stratum']=((o['site'] if o['big'] else 'other'),o['ot'])
PURGE=OPEN|MARK|set(CL)|SUF|{255,435,690,705,706,741,742,745}
PUR=[]
for o in OBJ:
    m=[a for a,l in zip(o['seq'],o['lab']) if l in('NAME','COUNT') and a not in PURGE]
    if m: PUR.append(dict(o,mid=m))
MID=[o for o in OBJ if len(o['mid'])>=1]
MID2=[o for o in MID if len(o['mid'])>=2]
LEN=collections.Counter(len(o['mid']) for o in MID)
ntok=sum(LEN[k]*k for k in LEN); V=len(set(a for o in MID for a in o['mid']))
print(f'   frame-purged middles: {len(PUR)} texts (signs of the opener/connective/closer/suffix classes and the 255-435-690, 705/706, marked-jar phrases removed wherever they stand)')
print(f'   middles: {len(MID)} texts with a middle, {len(MID2)} with >= 2 elements, {ntok} element tokens, {V} element types')
print('   middle length distribution',sorted(LEN.items()))
print('   by type',dict(collections.Counter(o["ot"] for o in MID2)),' big-city share %.2f'%(sum(o['big'] for o in MID2)/len(MID2)))

# ---------------- controls ----------------
def mk(seqs,label):
    out=[]
    for s in seqs:
        out.append(dict(mid=list(s),stratum=('c',rnd.random()<0.5),big=rnd.random()<0.7,ot='c',site='c',closer=None,hascount=False,opener=False))
    return out
def load_ur3(n):
    legs=json.load(open(SP+'ur3_legends.json')); seen=set(); names=[]
    TITLE={'dumu','dub','arad','arad2','ARAD2','sar','szagina','lu2','ensi2','sukkal','gudu4','sanga','nu','ku3','agrig','szabra','ugula','ra2','gal','di','kuruszda','sipa','szu','i3','du8','simug','aszgab','nagar','ad','kid','bur','e2','iri','ur','kaskal','nar','ab','ba','gu4','udu','ki','kisz','ma2','ensi'}
    for lg in legs:
        flat=tuple(x for ln in lg['lines'] for x in ln)
        if flat in seen: continue
        seen.add(flat)
        l1=tuple(lg['lines'][0])
        if 2<=len(l1)<=8 and 'x' not in l1: names.append(l1)
    rnd.shuffle(names)
    return mk(names[:n],'ur3'),len(names)
def zipf(n,a=1.0):
    w=[1/(i+1)**a for i in range(n)]; t=sum(w); return [x/t for x in w]
def planted_attr(n,lens,K=6,inv=None):
    # K slots, each with its own inventory (disjoint symbols), Zipf marginals; a middle fills k slots (k ~ Indus lengths)
    inv=inv or [max(8,V//K)]*K
    slots=[list(range(sum(inv[:k]),sum(inv[:k+1]))) for k in range(K)]
    pw=[zipf(len(sl),1.0) for sl in slots]
    seqs=[]
    for L in lens:
        k=min(L,K); which=sorted(rnd.sample(range(K),k))
        s=[rnd.choices(slots[q],pw[q])[0] for q in which]
        if L>K: s+=[rnd.choices(slots[q],pw[q])[0] for q in rnd.choices(range(K),k=L-K)]
        seqs.append(s)
    return mk(seqs,'attr'),slots
def planted_lex(n,lens,W=1500,S=150):
    # fixed word list: W words, each 1-3 syllables from S syllables; Zipf word frequencies; a middle = words until length reached
    sp=zipf(S,0.8); words=[]; seen=set()
    while len(words)<W:
        w=tuple(rnd.choices(range(S),sp,k=rnd.choice([1,2,2,3])))
        if w not in seen: seen.add(w); words.append(w)
    ww=zipf(W,1.0); seqs=[]
    for L in lens:
        s=[]
        while len(s)<L:
            s+=list(rnd.choices(words,ww)[0])
        seqs.append(s[:L] if len(s)>L and rnd.random()<0.5 else s)
    return mk(seqs,'lex'),words
lens=[len(o['mid']) for o in MID]
UR3,nur3=load_ur3(len(MID))
ATTR,ASLOTS=planted_attr(len(MID),lens)
LEX,LWORDS=planted_lex(len(MID),lens)
print(f'   controls: Ur III names {len(UR3)} of {nur3} distinct-legend line-1 names; planted attribute code K=6 slots x {len(ASLOTS[0])} symbols; planted lexicon {len(LWORDS)} words over 150 syllables')

# ---------------- helpers ----------------
def q(null,p): null=sorted(null); return null[min(len(null)-1,int(p*len(null)))]
def pval(obs,null,side='hi'):
    n=len(null)
    if side=='hi': return (sum(1 for x in null if x>=obs)+1)/(n+1)
    return (sum(1 for x in null if x<=obs)+1)/(n+1)
def shuffle_tokens(objs,r):
    # permute element tokens across middles within (stratum), keeping each middle's length
    by=collections.defaultdict(list)
    for i,o in enumerate(objs): by[o['stratum']].append(i)
    out=[None]*len(objs)
    for st,ids in by.items():
        pool=[a for i in ids for a in objs[i]['mid']]; r.shuffle(pool); k=0
        for i in ids:
            L=len(objs[i]['mid']); out[i]=pool[k:k+L]; k+=L
    return out
def pairs_of(mid):
    return set(itertools.combinations(sorted(set(mid)),2))
def pair_counts(mids):
    c=collections.Counter()
    for m in mids:
        if len(m)>=2: c.update(pairs_of(m))
    return c
def fmt(a,ind): return M(a) if ind else str(a)

# =========================== cycle 1: independence census ===========================
def census(objs,label,nperm,ind=False):
    mids=[o['mid'] for o in objs]
    obs=pair_counts(mids)
    acc=collections.Counter(); acc2=collections.Counter(); r=random.Random(1)
    reps=[]
    for k in range(nperm):
        sh=shuffle_tokens(objs,r); c=pair_counts(sh); reps.append(c)
        for p,n in c.items(): acc[p]+=n; acc2[p]+=n*n
    E={p:acc[p]/nperm for p in set(acc)|set(obs)}
    def stat(cnt,E,thr=3):
        # over pairs with E >= thr (well-expected) OR observed >= 5: share with |log2| > 1, mean |log2|, share obs >= 4E (compounds), share obs <= E/4 (exclusions)
        sel=[p for p in set(E)|set(cnt) if E.get(p,0)>=thr or cnt.get(p,0)>=5]
        if not sel: return (0,0,0,0,0)
        lr=[math.log2((cnt.get(p,0)+0.5)/(E.get(p,0)+0.5)) for p in sel]
        return (sum(1 for x in lr if abs(x)>1)/len(sel), sum(abs(x) for x in lr)/len(sel), sum(1 for x in lr if x>2)/len(sel), sum(1 for x in lr if x<-2)/len(sel), len(sel))
    so=stat(obs,E)
    # null calibration: the same statistic computed on each permuted corpus against E (leave-in; conservative, biased slightly low)
    sn=[stat(c,E) for c in reps[:min(nperm,60)]]
    print(f'\n--- {label}: pair census. {len(obs)} distinct observed pairs; {so[4]} pairs tested (E>=3 or O>=5)')
    print(f'    share |log2(O/E)|>1: {so[0]:.3f} vs permuted corpora {sum(x[0] for x in sn)/len(sn):.3f} [{q([x[0] for x in sn],0.025):.3f},{q([x[0] for x in sn],0.975):.3f}]  P={pval(so[0],[x[0] for x in sn]):.3f}')
    print(f'    mean |log2(O/E)|:     {so[1]:.3f} vs permuted {sum(x[1] for x in sn)/len(sn):.3f} [{q([x[1] for x in sn],0.025):.3f},{q([x[1] for x in sn],0.975):.3f}]  P={pval(so[1],[x[1] for x in sn]):.3f}')
    print(f'    share O > 4E (compound-like): {so[2]:.3f} vs permuted {sum(x[2] for x in sn)/len(sn):.3f};  share O < E/4 (exclusion-like): {so[3]:.3f} vs permuted {sum(x[3] for x in sn)/len(sn):.3f}')
    # token share: what share of element tokens sit in a pair with O>=5 and O>=4E
    comp=[p for p in obs if obs[p]>=5 and obs[p]>=4*max(E.get(p,0),0.25)]
    cs=set(comp); tok=0; tot=0
    for m in mids:
        if len(m)<2: continue
        ps=pairs_of(m); tot+=len(m)
        inc=set(a for p in ps if p in cs for a in p); tok+=len([a for a in m if a in inc])
    print(f'    strong compounds (O>=5, O>=4E): {len(comp)} pairs covering {tok/max(tot,1):.3f} of element tokens in >=2-element middles')
    top=sorted(comp,key=lambda p:-obs[p])[:12]
    print('    top compounds:',', '.join(f'{fmt(a,ind)}+{fmt(b,ind)} {obs[(a,b)]}:{E[(a,b)]:.1f}' for a,b in top))
    excl=[p for p in E if E[p]>=6 and obs.get(p,0)==0]
    print(f'    strong exclusions (E>=6, O=0): {len(excl)} of {sum(1 for p in E if E[p]>=6)} pairs with E>=6')
    print('    top exclusions:',', '.join(f'{fmt(a,ind)}|{fmt(b,ind)} 0:{E[(a,b)]:.1f}' for a,b in sorted(excl,key=lambda p:-E[p])[:10]))
    return so,sn,E,obs
def wb_models(train):
    uni=collections.Counter(); big=collections.defaultdict(collections.Counter)
    for m in train:
        p='<s>'
        for a in m: uni[a]+=1; big[p][a]+=1; p=a
    return uni,big
def heldout(train,test,label):
    uni,big=wb_models(train); N=sum(uni.values()); Vt=len(uni)+1
    def pu(a): # add-one unigram over the seen types + one unseen class
        return (uni.get(a,0)+1)/(N+Vt)
    def pb(p,a):
        c=big.get(p);
        if not c: return pu(a)
        n=sum(c.values()); t=len(c); lam=n/(n+t)
        return lam*c.get(a,0)/n+(1-lam)*pu(a)
    lu=lb=0; n=0
    for m in test:
        p='<s>'
        for a in m:
            lu+=-math.log2(pu(a)); lb+=-math.log2(pb(p,a)); p=a; n+=1
    print(f'    {label}: held-out bits/element independence {lu/n:.3f} vs pair(bigram) {lb/n:.3f}; gain {(lu-lb)/n:.3f} bits = {(lu-lb)/lu*100:.1f}%  (n={n} tokens, {len(test)} middles)')
    return (lu-lb)/n,(lu-lb)/lu
def adj_mi(objs,nperm):
    def mi(mids):
        j=collections.Counter(); a1=collections.Counter(); a2=collections.Counter()
        for m in mids:
            for x,y in zip(m,m[1:]): j[(x,y)]+=1; a1[x]+=1; a2[y]+=1
        n=sum(j.values())
        return sum(c/n*math.log2(c*n/(a1[x]*a2[y])) for (x,y),c in j.items())
    o=mi([x['mid'] for x in objs]); r=random.Random(2)
    nl=[mi(shuffle_tokens(objs,r)) for _ in range(nperm)]
    print(f'    adjacent-pair MI {o:.3f} bits vs shuffle {sum(nl)/len(nl):.3f} [{q(nl,0.025):.3f},{q(nl,0.975):.3f}]; excess {o-sum(nl)/len(nl):.3f}')
    return o-sum(nl)/len(nl)
if CY==1:
    R={}
    for objs,label,ind in [(MID,'Indus '+LV,True),(PUR,'Indus frame-purged',True),(UR3,'Ur III names',False),(ATTR,'planted attribute code',False),(LEX,'planted lexicon',False)]:
        so,sn,E,obs=census(objs,label,NP,ind)
        # held-out: Indus fit MD+HP, test other sites; controls 70/30 by 'big' flag; plus 5-fold random for all
        tr=[o['mid'] for o in objs if o['big']]; te=[o['mid'] for o in objs if not o['big']]
        g1=heldout(tr,te,'fit big / test held-out')
        r=random.Random(3); idx=list(range(len(objs))); r.shuffle(idx); gs=[]
        for f in range(5):
            te_i=set(idx[f::5]); gs.append(heldout([objs[i]['mid'] for i in idx if i not in te_i],[objs[i]['mid'] for i in te_i],f'5-fold {f}')[1])
        print(f'    5-fold mean gain {sum(gs)/5*100:.1f}%')
        mi=adj_mi(objs,min(NP,100))
        R[label]=dict(share=so[0],share_null=sum(x[0] for x in sn)/len(sn),meanlr=so[1],meanlr_null=sum(x[1] for x in sn)/len(sn),comp=so[2],excl=so[3],gain_heldout=g1[1],gain_5fold=sum(gs)/5,mi=mi)
    print('\n=== SUMMARY cycle 1 (share |log2|>1 above null; mean|log2| above null; held-out pair gain %; 5-fold gain %; adjacent MI excess bits)')
    for k,v in R.items():
        print(f'    {k:28s} share {v["share"]:.3f}-{v["share_null"]:.3f}={v["share"]-v["share_null"]:+.3f}  mean {v["meanlr"]:.3f}-{v["meanlr_null"]:.3f}={v["meanlr"]-v["meanlr_null"]:+.3f}  compounds {v["comp"]:.3f} exclusions {v["excl"]:.3f}  gain {v["gain_heldout"]*100:5.1f}% / {v["gain_5fold"]*100:5.1f}%  MI {v["mi"]:+.3f}')
    # seals only and NAME-only variants for Indus
    SE=[o for o in MID if o['ot']=='seal']
    print('\n--- Indus seals only')
    so,sn,E,obs=census(SE,'Indus seals',NP//2,True); heldout([o['mid'] for o in SE if o['big']],[o['mid'] for o in SE if not o['big']],'seals fit big / test held-out'); adj_mi(SE,50)
    NM=[]
    for o in OBJ:
        m=[a for a,l in zip(o['seq'],o['lab']) if l=='NAME']
        if m: NM.append(dict(o,mid=m))
    print('\n--- Indus NAME-only residue (COUNT stripped)')
    so,sn,E,obs=census(NM,'Indus NAME only',NP//2,True); heldout([o['mid'] for o in NM if o['big']],[o['mid'] for o in NM if not o['big']],'NAME fit big / test held-out'); adj_mi(NM,50)

# =========================== cycle 2: position-specific marginals ===========================
def mi_fl(objs,nperm,label,ind=False,minlen=2):
    sel=[o['mid'] for o in objs if len(o['mid'])>=minlen]
    def mi(pairs):
        j=collections.Counter(pairs); a1=collections.Counter(x for x,y in pairs); a2=collections.Counter(y for x,y in pairs); n=len(pairs)
        return sum(c/n*math.log2(c*n/(a1[x]*a2[y])) for (x,y),c in j.items())
    pr=[(m[0],m[-1]) for m in sel]; o=mi(pr)
    # within-position shuffle: permute the first elements among middles of the same stratum and length class
    strat=[(ob['stratum'],min(len(ob['mid']),4)) for ob in objs if len(ob['mid'])>=minlen]
    r=random.Random(4); nl=[]
    for _ in range(nperm):
        by=collections.defaultdict(list)
        for i,s in enumerate(strat): by[s].append(i)
        first=[None]*len(pr)
        for s,ids in by.items():
            f=[pr[i][0] for i in ids]; r.shuffle(f)
            for i,x in zip(ids,f): first[i]=x
        nl.append(mi([(first[i],pr[i][1]) for i in range(len(pr))]))
    # normalised: excess MI / H(first)
    a1=collections.Counter(x for x,y in pr); n=len(pr); Hf=-sum(c/n*math.log2(c/n) for c in a1.values())
    ex=o-sum(nl)/len(nl)
    print(f'    {label}: MI(first,last) {o:.3f} bits vs within-position shuffle {sum(nl)/len(nl):.3f} [{q(nl,0.025):.3f},{q(nl,0.975):.3f}] P={pval(o,nl):.3f}; excess {ex:.3f} = {ex/Hf*100:.1f}% of H(first) ({Hf:.2f} bits); n={n}')
    return ex/Hf
def partner_pred(objs,nperm,label,ind=False,minn=15):
    # for each element X with >= minn middles (len>=2): KL( partner distribution | marginal partner distribution ) vs shuffle null z-score
    mids=[o['mid'] for o in objs if len(o['mid'])>=2]
    def kls(mids):
        part=collections.defaultdict(collections.Counter); marg=collections.Counter()
        for m in mids:
            s=set(m)
            for x in s:
                for y in s:
                    if y!=x: part[x][y]+=1; marg[y]+=1
        N=sum(marg.values()); out={}
        for x,c in part.items():
            n=sum(c.values())
            if n<minn: continue
            # smoothed KL
            out[x]=sum(v/n*math.log2((v/n)/((marg[y]+0.5)/(N+0.5*len(marg)))) for y,v in c.items())
        return out
    o=kls(mids); r=random.Random(5); nulls=collections.defaultdict(list)
    for _ in range(nperm):
        sh=shuffle_tokens(objs,r); k=kls([m for m in sh if len(m)>=2])
        for x,v in k.items(): nulls[x].append(v)
    z={}
    for x,v in o.items():
        nl=nulls.get(x,[])
        if len(nl)<10: continue
        mu=sum(nl)/len(nl); sd=(sum((y-mu)**2 for y in nl)/len(nl))**0.5 or 1e-9
        z[x]=((v-mu)/sd, v, mu, (sum(1 for y in nl if y>=v)+1)/(len(nl)+1))
    bonf=0.05/max(len(z),1)
    sig=[x for x,(zz,v,mu,p) in z.items() if p<=bonf or (p<=1/(nperm+1)+1e-12 and zz>4)]
    print(f'    {label}: {len(z)} elements with >= {minn} partner tokens; mean z {sum(v[0] for v in z.values())/max(len(z),1):.2f}; share with partner-KL above Bonferroni null (P<={bonf:.4f} or z>4 at floor) {len(sig)/max(len(z),1):.3f} ({len(sig)}/{len(z)}); mean KL excess {sum(v[1]-v[2] for v in z.values())/max(len(z),1):.3f} bits')
    top=sorted(z.items(),key=lambda kv:-kv[1][0])[:10]
    print('      strongest partner-predictors:',', '.join(f'{fmt(x,ind)} z={v[0]:.1f}' for x,v in top))
    low=sorted(z.items(),key=lambda kv:kv[1][0])[:6]
    print('      weakest:',', '.join(f'{fmt(x,ind)} z={v[0]:.1f}' for x,v in low))
    return len(sig)/max(len(z),1), sum(v[0] for v in z.values())/max(len(z),1)
if CY==2:
    R={}
    for objs,label,ind in [(MID,'Indus '+LV,True),(PUR,'Indus frame-purged',True),(UR3,'Ur III names',False),(ATTR,'planted attribute code',False),(LEX,'planted lexicon',False)]:
        print(f'\n--- {label}')
        a=mi_fl(objs,NP,label,ind); b=mi_fl(objs,NP,label+' (len>=3)',ind,3); c=partner_pred(objs,min(NP,100),label,ind)
        R[label]=(a,b,c)
    print('\n=== SUMMARY cycle 2 (MI(first,last) excess as % of H(first), len>=2 / len>=3; share of elements predicting partners; mean z)')
    for k,(a,b,c) in R.items(): print(f'    {k:28s} {a*100:5.1f}% / {b*100:5.1f}%   partner-predictors {c[0]:.3f}  mean z {c[1]:.2f}')
    # (d) quantity seals vs office (jar) seals vs other, Indus
    print('\n--- (d) Indus subsets')
    SE=[o for o in MID if o['ot']=='seal']
    QTY=[o for o in SE if o['hascount']]
    OFF=[o for o in SE if o['closer']==740 and not o['hascount']]
    OTH=[o for o in SE if o['closer']!=740 and not o['hascount']]
    NOCL=[o for o in SE if o['closer'] is None]
    for objs,label in [(SE,'all seals'),(QTY,'quantity seals (COUNT in middle)'),(OFF,'office seals (jar closer, no count)'),(OTH,'other seals (no jar, no count)'),(NOCL,'seals without closer'),([o for o in MID if o['ot']=='tablet'],'tablets'),([o for o in MID if not o['big']],'held-out sites (all types)'),([o for o in MID if o['site']=='Mohenjo-daro'],'Mohenjo-daro'),([o for o in MID if o['site']=='Harappa'],'Harappa')]:
        n2=sum(1 for o in objs if len(o['mid'])>=2)
        print(f'  [{label}] {len(objs)} middles, {n2} with >=2 elements')
        if n2<40: print('    too few'); continue
        mi_fl(objs,NP,label,True); partner_pred(objs,min(NP,100),label,True,minn=12)
        # pair census summary for the subset
        so,sn,E,obs=census(objs,label,min(NP,60),True)

# =========================== cycle 3: slots and inventory ===========================
def chao1(counts):
    f1=sum(1 for c in counts if c==1); f2=sum(1 for c in counts if c==2); S=len(counts)
    return S+(f1*f1/(2*f2) if f2 else f1*(f1-1)/2), S, f1, f2
def slots_from_avoidance(objs,label,nperm,ind=False,minn=10,thr=3.0):
    mids=[o['mid'] for o in objs]
    tok=collections.Counter(a for m in mids for a in m)
    freq=[a for a,n in tok.items() if n>=minn]
    obs=pair_counts(mids); r=random.Random(6); acc=collections.Counter()
    for _ in range(nperm):
        for p,n in pair_counts(shuffle_tokens(objs,r)).items(): acc[p]+=n
    E={p:acc[p]/nperm for p in acc}
    fs=set(freq)
    # avoidance edge: E >= thr and O == 0 ; also 'strong avoidance' O <= E/5
    avoid=set(); testable=0
    for (a,b),e in E.items():
        if a in fs and b in fs and e>=thr:
            testable+=1
            if obs.get((a,b),0)<=e/5: avoid.add((a,b))
    # greedy clique cover of the avoidance graph = candidate slots (members never/rarely co-occur)
    adj=collections.defaultdict(set)
    for a,b in avoid: adj[a].add(b); adj[b].add(a)
    order=sorted(freq,key=lambda a:-tok[a]); assigned={}; slots=[]
    for a in order:
        placed=False
        for si,sl in enumerate(slots):
            if all(b in adj[a] for b in sl): sl.append(a); assigned[a]=si; placed=True; break
        if not placed: slots.append([a]); assigned[a]=len(slots)-1
    big=[s for s in slots if len(s)>=3]
    # explanatory power: share of >=2-element middles (over frequent signs) with at most one member per slot
    ok=0; tot=0
    for m in mids:
        f=[assigned[a] for a in m if a in assigned]
        if len(f)>=2:
            tot+=1; ok+= (len(f)==len(set(f)))
    # null expectation of 'one per slot' given the same assignment: shuffle
    nl=[]
    for _ in range(min(nperm,30)):
        sh=shuffle_tokens(objs,r); o2=t2=0
        for m in sh:
            f=[assigned[a] for a in m if a in assigned]
            if len(f)>=2: t2+=1; o2+=(len(f)==len(set(f)))
        nl.append(o2/max(t2,1))
    print(f'\n--- {label}: {len(freq)} elements with >= {minn} tokens; {testable} pairs testable (E>={thr}); avoidance edges (O <= E/5): {len(avoid)} = {len(avoid)/max(testable,1):.3f} of testable')
    # null share of avoidance edges: computed on one permuted corpus
    sh=shuffle_tokens(objs,random.Random(7)); c=pair_counts(sh); na=sum(1 for (a,b),e in E.items() if a in fs and b in fs and e>=thr and c.get((a,b),0)<=e/5)
    print(f'    avoidance share on a permuted corpus {na/max(testable,1):.3f}')
    print(f'    greedy clique cover: {len(slots)} groups, {len(big)} with >= 3 members; sizes {sorted((len(s) for s in slots),reverse=True)[:15]}')
    print(f'    one-member-per-group holds in {ok/max(tot,1):.3f} of {tot} middles vs shuffle {sum(nl)/len(nl):.3f} [{min(nl):.3f},{max(nl):.3f}]')
    for s in sorted(big,key=len,reverse=True)[:8]:
        print('      group:',', '.join(fmt(a,ind) for a in s[:12]),('...' if len(s)>12 else ''))
    return len(slots),len(big),ok/max(tot,1),sum(nl)/len(nl),len(avoid)/max(testable,1),na/max(testable,1)
if CY==3:
    R={}
    for objs,label,ind in [(MID,'Indus '+LV,True),(PUR,'Indus frame-purged',True),(UR3,'Ur III names',False),(ATTR,'planted attribute code',False),(LEX,'planted lexicon',False)]:
        R[label]=slots_from_avoidance(objs,label,min(NP,100),ind)
        mids=[tuple(o['mid']) for o in objs]
        cm=collections.Counter(mids); ce=collections.Counter(a for m in mids for a in m)
        c1=chao1(list(cm.values())); c2=chao1(list(ce.values()))
        bg=collections.Counter(p for m in mids for p in zip(m,m[1:])); c3=chao1(list(bg.values()))
        print(f'    Chao1: distinct middles {c1[1]} -> est {c1[0]:.0f} (f1={c1[2]}, f2={c1[3]}); elements {c2[1]} -> est {c2[0]:.0f} (f1={c2[2]}, f2={c2[3]}); adjacent bigrams {c3[1]} -> est {c3[0]:.0f}')
        if label.startswith('Indus'):
            for sub,lab in [([o for o in objs if o['ot']=='seal'],'seals'),([o for o in objs if o['big']],'MD+Harappa'),([o for o in objs if not o['big']],'held-out sites')]:
                mids=[tuple(o['mid']) for o in sub]; cm=collections.Counter(mids); ce=collections.Counter(a for m in mids for a in m)
                c1=chao1(list(cm.values())); c2=chao1(list(ce.values()))
                print(f'      {lab}: distinct middles {c1[1]} -> {c1[0]:.0f}; elements {c2[1]} -> {c2[0]:.0f}')
            print('    held-out replication of the avoidance structure:')
            slots_from_avoidance([o for o in objs if o['big']],'Indus MD+Harappa',min(NP,60),True)
            slots_from_avoidance([o for o in objs if not o['big']],'Indus held-out sites',min(NP,60),True,minn=6,thr=2.0)
            slots_from_avoidance([o for o in objs if o['ot']=='seal'],'Indus seals',min(NP,60),True)
    print('\n=== SUMMARY cycle 3 (groups, groups>=3, one-per-group share obs/null, avoidance share obs/null)')
    for k,v in R.items(): print(f'    {k:28s} groups {v[0]:3d} big {v[1]:2d}  one-per-group {v[2]:.3f}/{v[3]:.3f}  avoidance {v[4]:.3f}/{v[5]:.3f}')
