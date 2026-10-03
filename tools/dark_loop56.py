"""S-DARK-56: THE 'NOT NAMES' CLAIM WITH THE RIGHT COMPARATOR.
Our name contrast (S321, S-DARK-18/41/53) used Ur III seal legends, where names repeat because the same families and
offices recur. A one-person-per-entry name list (personnel register) has uniqueness ~1 by construction, and S-DARK-53
put the Indus middles next to the Linear B personnel list on uniqueness. Here every comparator is DEDUPLICATED to one
copy per distinct name, so uniqueness is uninformative, and the comparison rests on NAME-DIAGNOSTIC element statistics:
  el_ttr      element types / element tokens (reuse of elements across names; low = heavy reuse)
  el_reuse    share of element tokens whose type occurs in >= 2 names
  el_zipf     Zipf slope of element frequencies; el_gini Gini of element frequencies; el_heaps Heaps exponent (types vs tokens)
  pos_excess  positional morphology: mean over frequent elements (>= 5 tokens, names >= 2 long) of
              |P(initial) - P(final)| minus the same under element shuffle (prefix/suffix/theophoric slot preference)
  pos_bound   share of frequent elements whose initial/final split lies outside the 95% band of 60 shuffles
  big_reuse   share of adjacent element pairs (bigrams) that recur in >= 2 names (compound structure), minus shuffle null
  rev_pairs   share of recurring bigrams (a,b) whose reverse (b,a) also occurs ('Nabu-X / X-Nabu'), minus shuffle null
  mi_ex       adjacent-pair mutual information minus shuffle null (bits)
  pair_pred   1 - H(next | prev) / H(next) (S-DARK-26 pair predictability) minus shuffle null
  substr      share of names that are contiguous substrings of another name, minus shuffle null
  cmp_share   share of >= 2-element names that share an element with another name (always high; reported for completeness)
Null 1 = within-corpus shuffle (element tokens permuted across names, lengths kept: kills order, keeps frequencies).
Null 2 = frequency-matched random strings (iid draws from the element unigram, lengths kept).
Null 3 = uniform IDs over the same alphabet size and lengths.
Corpora: data/derived/dark/loop56_corpora (tools/dark_loop56_prep.py) + Indus middles (S-DARK-26 parser, as in
tools/dark_loop53.py), deduplicated one per site x object type x text, then one per distinct middle.
Usage: python3 tools/dark_loop56.py <1|2|3> [ndraw]
"""
import json,sys,random,collections,math,csv,os
C=json.load(open('data/derived/merged-corpus-canonical.json'))
BR=json.load(open('data/derived/bridge_extended.json'))
PROP=json.load(open('data/derived/dark/bridge_proposals.json'))
CORP='data/derived/dark/loop56_corpora/'
CY=int(sys.argv[1]) if len(sys.argv)>1 else 0; NB=int(sys.argv[2]) if len(sys.argv)>2 else 100
rnd=random.Random(56)
LOG=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); LOG.append(s)

# ---------------- frame parser (S310 parse_all.py + S331 openers, identical to tools/dark_loop26.py / dark_loop53.py) ----------------
OPEN={817,861,820,920,692}; MARK={2,60}; MJAR={741,742,745}; SUF={400,90}; CL=[740,520,151,156,527,226,617,154,158,236,700]
FISH={235,240,233,231,220}; NUM={1,3,4,5,16,17,18,31,32,33,34,55,56}
def otype(t):
    t=t.split(':')[0]
    return {'SEAL':'seal','TAB':'tablet','POT':'pot','TAG':'sealing'}.get(t,'other')
def build_qual(objs):
    left=collections.defaultdict(collections.Counter)
    for o in objs:
        s=o['seq'][:]
        while len(s)>1 and s[-1] in SUF: s.pop()
        if len(s)>=2 and s[-1] in CL: left[s[-1]][s[-2]]+=1
    Q={}
    for c,cnt in left.items():
        tot=sum(cnt.values()); acc=0; q=set()
        for a,n in cnt.most_common():
            if acc/tot>=0.6: break
            q.add(a); acc+=n
        Q[c]=q
    return Q
def parse(s,QUAL):
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
def annotate(objs):
    QUAL=build_qual(objs)
    for o in objs:
        o['lab']=parse(o['seq'],QUAL)
        o['mid']=tuple(a for a,l in zip(o['seq'],o['lab']) if l in('NAME','COUNT'))
        o['name']=tuple(a for a,l in zip(o['seq'],o['lab']) if l=='NAME')
        o['closer']=next((a for a,l in zip(o['seq'],o['lab']) if l=='CLOSER'),None)
        o['hascount']=any(l=='COUNT' for l in o['lab'])
    return objs
def load_indus(LV,dedup=True):
    objs=[]; seen=set()
    for r in C:
        s=r[LV]
        if not s or len(s)<2 or r['complete']!='Y' or r['dir.'].strip()=='-': continue
        key=(r['site'],otype(r['type']),tuple(s))
        if dedup and key in seen: continue
        seen.add(key)
        objs.append(dict(cisi=r['cisi'],site=r['site'],ot=otype(r['type']),seq=list(s),big=r['site'] in('Mohenjo-daro','Harappa')))
    return annotate(objs)
def im77_objects():
    M2W={}
    for w,ms in BR.items():
        for m in ms: M2W.setdefault(m,[]).append(int(w))
    for p in PROP['proposals']: M2W.setdefault(p['M'],[]).append(p['W'])
    for m in M2W: M2W[m]=min(M2W[m])
    rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
    by=collections.defaultdict(list)
    for r in rows: by[r['text_no']].append(r)
    objs=[]; seen=set(); unb=0; tot=0
    for tn,rs in by.items():
        rs=sorted(rs,key=lambda r:(int(r['side']),int(r['line'])))
        rs0=[r for r in rs if r['side']=='0'] or rs[:1]
        ms=[int(x) for r in rs0 for x in r['signs_clean'].split() if x.strip() and x!='0']
        if len(ms)<2: continue
        seq=[]
        for m in ms:
            tot+=1
            if m in M2W: seq.append(M2W[m])
            else: unb+=1; seq.append(10000+m)
        site={'Mohenjodaro':'Mohenjo-daro','Harappa':'Harappa'}.get(rs0[0]['site'],rs0[0]['site'])
        ot={'seal':'seal','sealing':'sealing','miniature tablet':'tablet','copper tablet':'tablet','pottery graffito':'pot'}.get(rs0[0]['object_type'],'other')
        key=(site,ot,tuple(seq))
        if key in seen: continue
        seen.add(key)
        objs.append(dict(cisi='IM'+tn,site=site,ot=ot,seq=seq,big=site in('Mohenjo-daro','Harappa')))
    P(f'  IM77: {len(objs)} distinct objects; unbridged M tokens {unb}/{tot}={unb/tot:.3f} kept as opaque elements')
    return annotate(objs)
def indus_names(LV,minlen=2,field='mid',subset=None):
    objs=load_indus(LV) if LV!='im77' else im77_objects()
    if subset: objs=[o for o in objs if subset(o)]
    return sorted(set(o[field] for o in objs if len(o[field])>=minlen))

# ---------------- statistics on a deduplicated list of names ----------------
def H(cnt):
    t=sum(cnt.values()); return -sum(v/t*math.log2(v/t) for v in cnt.values() if v>0) if t else 0.0
def gini(counts):
    x=sorted(counts); n=len(x); s=sum(x)
    return (2*sum((i+1)*v for i,v in enumerate(x))/(n*s))-(n+1)/n if n and s else 0.0
def zipf(counts):
    x=sorted(counts,reverse=True)
    if len(x)<3: return float('nan')
    xs=[math.log(i+1) for i in range(len(x))]; ys=[math.log(v) for v in x]
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys); sxx=sum((a-mx)**2 for a in xs)
    return sum((a-mx)*(b-my) for a,b in zip(xs,ys))/sxx if sxx>0 else float('nan')
def heaps_el(names,r,norders=5):
    toks=[a for n in names for a in n]; N=len(toks)
    grid=sorted(set(int(round(N*0.05*1.25**i)) for i in range(40) if N*0.05*1.25**i<=N)|{N})
    acc=collections.defaultdict(float)
    for _ in range(norders):
        idx=list(range(len(names))); r.shuffle(idx); seen=set(); m=0; gi=0
        for i in idx:
            for a in names[i]:
                seen.add(a); m+=1
                while gi<len(grid) and m>=grid[gi]: acc[grid[gi]]+=len(seen); gi+=1
    xs=[math.log(g) for g in grid if g>=max(20,N*0.05)]; ys=[math.log(acc[g]/norders) for g in grid if g>=max(20,N*0.05)]
    if len(xs)<3: return float('nan')
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    return sum((a-mx)*(b-my) for a,b in zip(xs,ys))/sum((a-mx)**2 for a in xs)
def raw_stats(names,freq_min=5):
    """order-dependent statistics (no null subtraction)."""
    long=[n for n in names if len(n)>=2]
    el=collections.Counter(a for n in names for a in n)
    # positional morphology
    pos=collections.defaultdict(lambda:[0,0,0])
    for n in long:
        L=len(n)
        for i,a in enumerate(n): pos[a][0 if i==0 else 2 if i==L-1 else 1]+=1
    freq={a:v for a,v in pos.items() if sum(v)>=freq_min}
    skew={a:abs(v[0]-v[2])/sum(v) for a,v in freq.items()}
    pos_mean=sum(skew.values())/len(skew) if skew else float('nan')
    # bigrams
    bg=collections.Counter((n[i],n[i+1]) for n in long for i in range(len(n)-1))
    nb=sum(bg.values())
    big_reuse=sum(v for v in bg.values() if v>=2)/nb if nb else float('nan')
    rec=[p for p,v in bg.items() if v>=2]
    rev=sum(1 for (a,b) in rec if (b,a) in bg and a!=b)/len(rec) if rec else float('nan')
    # MI and pair predictability
    A=collections.Counter(); B=collections.Counter(); byfirst=collections.defaultdict(collections.Counter)
    for (a,b),v in bg.items(): A[a]+=v; B[b]+=v; byfirst[a][b]+=v
    mi=sum(v/nb*math.log2(v*nb/(A[a]*B[b])) for (a,b),v in bg.items()) if nb else float('nan')
    hnext=H(B); hcond=sum(A[a]/nb*H(byfirst[a]) for a in A) if nb else 0
    pred=1-hcond/hnext if hnext else float('nan')
    # substring share: names that occur as a proper contiguous substring of another name
    subs=set()
    for m in names:
        L=len(m)
        for l in range(1,L):
            for i in range(L-l+1): subs.add(m[i:i+l])
    substr=sum(1 for n in names if n in subs)/len(names)
    # compound share
    byel=collections.defaultdict(set)
    for i,n in enumerate(long):
        for a in n: byel[a].add(i)
    cmp=sum(1 for i,n in enumerate(long) if any(len(byel[a])>1 for a in n))/len(long) if long else float('nan')
    return dict(pos_mean=pos_mean,skew=skew,freq=freq,big_reuse=big_reuse,rev=rev,mi=mi,pred=pred,substr=substr,cmp=cmp,el=el)
def shuffle_names(names,r,mode='shuf'):
    lens=[len(n) for n in names]
    if mode=='shuf':
        toks=[a for n in names for a in n]; r.shuffle(toks)
    elif mode=='freq':
        pool=[a for n in names for a in n]; toks=[r.choice(pool) for _ in range(len(pool))]
    elif mode=='unif':
        k=len(set(a for n in names for a in n)); toks=[r.randrange(k) for _ in range(sum(lens))]
    out=[]; i=0
    for L in lens: out.append(tuple(toks[i:i+L])); i+=L
    return out
def metrics(names,r=None,nnull=20,nullmode='shuf'):
    r=r or rnd
    names=[tuple(n) for n in names]
    el=collections.Counter(a for n in names for a in n); T=sum(el.values()); K=len(el)
    obs=raw_stats(names)
    nulls=[raw_stats(shuffle_names(names,r,nullmode)) for _ in range(nnull)]
    def nm(k): v=[x[k] for x in nulls if not math.isnan(x[k])]; return sum(v)/len(v) if v else float('nan')
    # pos_bound: share of frequent elements whose observed |pi-pf| exceeds the 95th pct of its own null distribution
    bound=0; tot=0
    for a,s in obs['skew'].items():
        nv=sorted(x['skew'].get(a,0.0) for x in nulls)
        if len(nv)>=10:
            tot+=1
            if s>nv[int(0.95*len(nv))-1]: bound+=1
    return dict(n=len(names),k=K,T=T,meanL=T/len(names),el_ttr=K/T,
                el_reuse=sum(v for v in el.values() if v>=2)/T,el_gini=gini(list(el.values())),el_zipf=zipf(list(el.values())),
                el_heaps=heaps_el(names,r),
                pos_excess=obs['pos_mean']-nm('pos_mean'),pos_null=nm('pos_mean'),pos_obs=obs['pos_mean'],
                pos_bound=bound/tot if tot else float('nan'),nfreq=tot,
                big_reuse=obs['big_reuse']-nm('big_reuse'),big_obs=obs['big_reuse'],big_null=nm('big_reuse'),
                rev_pairs=obs['rev']-nm('rev'),rev_obs=obs['rev'],
                mi_ex=obs['mi']-nm('mi'),pair_pred=obs['pred']-nm('pred'),pred_obs=obs['pred'],
                substr=obs['substr']-nm('substr'),substr_obs=obs['substr'],cmp_share=obs['cmp'])
KEYS=['n','k','T','meanL','el_ttr','el_reuse','el_gini','el_zipf','el_heaps','pos_excess','pos_bound','big_reuse','rev_pairs','mi_ex','pair_pred','substr','cmp_share']
LADDER=['el_ttr','el_reuse','el_gini','el_zipf','el_heaps','pos_excess','pos_bound','big_reuse','rev_pairs','mi_ex','pair_pred','substr']
def fmt(m,keys=KEYS):
    return ' '.join(f'{k}={m[k]:.3f}' if isinstance(m[k],float) else f'{k}={m[k]}' for k in keys)
def q(vals,p):
    v=sorted(x for x in vals if not (isinstance(x,float) and math.isnan(x)))
    return v[min(len(v)-1,int(p*len(v)))] if v else float('nan')

def jl(name): return [tuple(json.loads(l)['seq']) for l in open(CORP+name+'.jsonl')]
def length_match(pool,target_lens,n,r):
    """draw n names from pool matching the target length distribution where possible (without replacement per length)."""
    byL=collections.defaultdict(list)
    for s in pool: byL[len(s)].append(s)
    tl=collections.Counter(target_lens); tot=sum(tl.values()); out=[]; short=0
    for L,c in tl.items():
        want=int(round(n*c/tot)); have=byL.get(L,[])
        if len(have)>=want: out+=r.sample(have,want)
        else: out+=have; short+=want-len(have)
    return out,short
def draws(pool,n,label,ndraw=NB,lens=None,nullmode='shuf'):
    runs=[]; shorts=[]
    for b in range(ndraw):
        r=random.Random(3000+b)
        if lens is not None:
            sub,sh=length_match(pool,lens,n,r); shorts.append(sh)
        else: sub=r.sample(pool,min(n,len(pool)))
        runs.append(metrics(sub,r,nnull=10,nullmode=nullmode))
    med={k:q([x[k] for x in runs],0.5) for k in KEYS}
    lo={k:q([x[k] for x in runs],0.025) for k in KEYS}; hi={k:q([x[k] for x in runs],0.975) for k in KEYS}
    P(f'  [{label}] pool={len(pool)} draw n={med["n"]} x{ndraw}'+(f' length-matched (median shortfall {q(shorts,0.5):.0f})' if lens is not None else ''))
    P('    median '+fmt(med))
    P('    CI: '+'; '.join(f'{k} {med[k]:.3f} [{lo[k]:.3f},{hi[k]:.3f}]' for k in LADDER))
    return dict(med=med,lo=lo,hi=hi,n=med['n'],pool=len(pool))
def boot(names,label,nboot=NB,frac=0.8,nullmode='shuf'):
    m=metrics(names,nnull=20,nullmode=nullmode); runs=[]
    for b in range(nboot):
        r=random.Random(1000+b); sub=r.sample(names,int(frac*len(names))); runs.append(metrics(sub,r,nnull=10,nullmode=nullmode))
    lo={}; hi={}
    for k in KEYS:
        v=[x[k] for x in runs]; md=q(v,0.5)
        lo[k]=m[k]+q(v,0.025)-md if isinstance(m[k],float) else m[k]; hi[k]=m[k]+q(v,0.975)-md if isinstance(m[k],float) else m[k]
    P(f'  [{label}] n={len(names)}'); P('    '+fmt(m))
    P('    CI(0.8-subsample spread centred on full-n): '+'; '.join(f'{k} {m[k]:.3f} [{lo[k]:.3f},{hi[k]:.3f}]' for k in LADDER))
    return dict(med=m,lo=lo,hi=hi,n=len(names))

def save(name,obj):
    json.dump(obj,open(f'data/derived/dark/{name}.json','w'),indent=1)
    open(f'data/derived/dark/{name}_log.txt','w').write('\n'.join(LOG)+'\n')

# ================= cycles =================
if CY==1:
    P(f'== S-DARK-56 cycle 1: Indus middles vs ONE-PER-PERSON name lists, element statistics; ndraw {NB}')
    RES={}
    IND={}
    for LV in ['seq_raw','seq_strong','seq_all','im77']:
        names=indus_names(LV,2)
        IND[LV]=names
        lens=collections.Counter(len(n) for n in names)
        P(f'\n##### Indus {LV}: {len(names)} distinct middles with >= 2 elements; lengths {sorted(lens.items())}')
        RES['indus_'+LV]=boot(names,f'Indus {LV} middles >=2 dedup')
    n0=len(IND['seq_raw']); L0=[len(n) for n in IND['seq_raw']]
    P(f'\n##### reference name lists, each deduplicated to one per distinct name; subsampled to n={n0} (or full pool if smaller)')
    for nm in ['ur3_names_dedup','ur3_names_elem','ob_names_dedup','oa_names_dedup','linb_personnel_dedup','linb_personnel_KN','linb_personnel_PY','latin_names_dedup','latin_names_letters']:
        pool=[s for s in jl(nm) if len(s)>=2]
        RES[nm]=draws(pool,n0,nm)
    P('\n=== cycle 1 summary')
    P('  corpus                        '+' '.join(f'{k:>10s}' for k in LADDER))
    for k,v in RES.items():
        m=v['med']; P(f'  {k:28s} '+' '.join(f'{m[x]:10.3f}' for x in LADDER)+f'   n={m["n"]} k={m["k"]} meanL={m["meanL"]:.2f}')
    save('loop56_cycle1',RES)
