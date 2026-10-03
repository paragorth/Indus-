"""S-DARK-53: CAPACITY UTILISATION OF THE IDENTIFIER SLOT.
Does the Indus middle (S-DARK-26 definition: NAME + COUNT residue of the S310/S331 frame parser) use its code space
the way a designed identifier code does (dense, even, near-uniform, Heaps exponent ~1) or the way a name stock does
(sparse, Zipfian, heavy reuse, Heaps exponent well below 1)?
  cycle 1  Indus middles, duplicates collapsed to one per site x object type x full text; seq_raw / seq_strong / seq_all;
           combinatorial capacity (k^L summed over observed lengths; product of per-position observed alphabets; entropy
           capacity 2^sum H_i), realised distinct, Chao1 and Good-Turing, utilisation, Gini and Zipf slope of middle
           frequencies, per-position evenness, Heaps exponent; CIs by 0.8-subsampling.
  cycle 2  the same code on the S-DARK-32 reference corpora subsampled to the Indus n (designed codes, name stocks,
           accounting, grammar) plus a uniform random ID generator and a Zipfian name generator at the Indus k and lengths.
  cycle 3  Indus subsets: quantity seals, office seals, other seals, tablets, Mohenjo-daro, Harappa, held-out sites;
           IM77 replication through the W<->M bridge (+ S-DARK-27 proposals); all three merge levels.
Usage: python3 tools/dark_loop53.py <1|2|3> [nboot]
"""
import json,sys,random,collections,math,csv,os
C=json.load(open('data/derived/merged-corpus-canonical.json'))
BR=json.load(open('data/derived/bridge_extended.json'))
PROP=json.load(open('data/derived/dark/bridge_proposals.json'))
CORP='data/derived/dark/loop32_corpora/'
CY=int(sys.argv[1]); NB=int(sys.argv[2]) if len(sys.argv)>2 else 200
rnd=random.Random(53)
LOG=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); LOG.append(s)

# ---------------- frame parser (S310 parse_all.py + S331 openers, identical to tools/dark_loop26.py) ----------------
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

# ---------------- statistics ----------------
def gini(counts):
    x=sorted(counts); n=len(x); s=sum(x)
    if n==0 or s==0: return 0.0
    return (2*sum((i+1)*v for i,v in enumerate(x))/(n*s))-(n+1)/n
def zipf_slope(counts):
    x=sorted(counts,reverse=True)
    if len(x)<3: return float('nan')
    xs=[math.log(i+1) for i in range(len(x))]; ys=[math.log(v) for v in x]
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    sxx=sum((a-mx)**2 for a in xs)
    return sum((a-mx)*(b-my) for a,b in zip(xs,ys))/sxx if sxx>0 else float('nan')
def H(counter):
    t=sum(counter.values())
    return -sum(v/t*math.log2(v/t) for v in counter.values() if v>0) if t else 0.0
def chao1(cnt):
    f1=sum(1 for v in cnt.values() if v==1); f2=sum(1 for v in cnt.values() if v==2); D=len(cnt)
    return D+(f1*f1/(2*f2) if f2>0 else f1*(f1-1)/2), f1, f2
def heaps(texts,norders=10,r=None):
    r=r or rnd; n=len(texts); grid=sorted(set(int(round(n*0.05*1.25**i)) for i in range(40) if n*0.05*1.25**i<=n)|{n})
    acc=collections.defaultdict(float)
    for _ in range(norders):
        idx=list(range(n)); r.shuffle(idx); seen=set(); gi=0; m=0
        for i in idx:
            seen.add(texts[i]); m+=1
            if gi<len(grid) and m==grid[gi]: acc[m]+=len(seen); gi+=1
    xs=[math.log(m) for m in grid if m>=max(10,n*0.05)]; ys=[math.log(acc[m]/norders) for m in grid if m>=max(10,n*0.05)]
    if len(xs)<3: return float('nan')
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    return sum((a-mx)*(b-my) for a,b in zip(xs,ys))/sum((a-mx)**2 for a in xs)
def metrics(texts,r=None):
    """texts: list of tuples (identifiers). Returns dict of statistics."""
    r=r or rnd
    texts=[tuple(t) for t in texts if len(t)>=1]
    n=len(texts); cnt=collections.Counter(texts); D=len(cnt)
    ch,f1,f2=chao1(cnt)
    el=collections.Counter(a for t in texts for a in t); k=len(el)
    lens=collections.Counter(len(t) for t in texts)
    # capacities (log10)
    cap_naive=sum(k**L for L in lens)
    cap_pos=0.0; cap_ent=0.0; even_pos=[]; even_k=[]; wts=[]
    for L,nL in lens.items():
        sub=[t for t in texts if len(t)==L]
        prodK=1; sumH=0.0
        for i in range(L):
            c=collections.Counter(t[i] for t in sub); Ki=len(c); h=H(c)
            prodK*=Ki; sumH+=h
            if Ki>1 and nL>=20:
                even_pos.append(h/math.log2(Ki)); even_k.append(h/math.log2(k)); wts.append(nL)
        cap_pos+=prodK; cap_ent+=2**sumH
    ev_pos=sum(a*w for a,w in zip(even_pos,wts))/sum(wts) if wts else float('nan')
    ev_k=sum(a*w for a,w in zip(even_k,wts))/sum(wts) if wts else float('nan')
    # adjacent MI excess vs shuffle of second element
    pairs=[(t[i],t[i+1]) for t in texts for i in range(len(t)-1)]
    def mi(pp):
        a=collections.Counter(x for x,_ in pp); b=collections.Counter(y for _,y in pp); j=collections.Counter(pp); N=len(pp)
        return sum(v/N*math.log2(v*N/(a[x]*b[y])) for (x,y),v in j.items()) if N else 0.0
    mi_obs=mi(pairs); null=[]
    for _ in range(5):
        ys=[y for _,y in pairs]; r.shuffle(ys); null.append(mi([(x,y) for (x,_),y in zip(pairs,ys)]))
    mi_ex=mi_obs-sum(null)/len(null) if pairs else float('nan')
    return dict(n=n,D=D,uniq=D/n,f1=f1,f2=f2,chao1=ch,gt_new=f1/n,k=k,meanL=sum(L*c for L,c in lens.items())/n,
                lcap_naive=math.log10(cap_naive),lcap_pos=math.log10(cap_pos),lcap_ent=math.log10(cap_ent),
                lutil_naive=math.log10(ch)-math.log10(cap_naive),lutil_pos=math.log10(ch)-math.log10(cap_pos),
                lutil_ent=math.log10(ch)-math.log10(cap_ent),
                gini_id=gini(list(cnt.values())),zipf_id=zipf_slope(list(cnt.values())),
                gini_el=gini(list(el.values())),zipf_el=zipf_slope(list(el.values())),
                even_pos=ev_pos,even_k=ev_k,heaps=heaps(texts,10,r),mi_ex=mi_ex,
                top10=sum(v for _,v in cnt.most_common(10))/n)
KEYS=['n','D','uniq','chao1','gt_new','k','meanL','lcap_naive','lcap_pos','lcap_ent','lutil_naive','lutil_pos','lutil_ent','gini_id','zipf_id','gini_el','zipf_el','even_pos','even_k','heaps','mi_ex','top10']
def fmt(m):
    return ' '.join(f'{k}={m[k]:.3f}' if isinstance(m[k],float) else f'{k}={m[k]}' for k in KEYS)
def ci(vals,p):
    v=sorted(vals); return v[min(len(v)-1,int(p*len(v)))]
def with_ci(texts,label,nboot=NB,frac=0.8):
    m=metrics(texts); n=len(texts); boots=[]
    for b in range(nboot):
        r=random.Random(1000+b); sub=r.sample(texts,int(frac*n)); boots.append(metrics(sub,r))
    P(f'  [{label}] n={n}')
    P('    '+fmt(m))
    s=[]
    for k in ['uniq','chao1','lutil_pos','lutil_ent','gini_id','zipf_id','even_pos','heaps','gini_el','zipf_el','mi_ex']:
        v=[x[k] for x in boots]; md=ci(v,0.5); lo=m[k]+ci(v,0.025)-md; hi=m[k]+ci(v,0.975)-md
        s.append(f'{k} {m[k]:.3f} [{lo:.3f},{hi:.3f}]')
    P('    CI(0.8-subsample spread, centred on the full-n estimate): '+'; '.join(s))
    return m,boots
def subsampled(pool,n,label,ndraw=NB):
    """reference corpus: draw n texts without replacement ndraw times; report median and 2.5/97.5."""
    runs=[]
    for b in range(ndraw):
        r=random.Random(2000+b); sub=r.sample(pool,min(n,len(pool))); runs.append(metrics(sub,r))
    med={k:ci([x[k] for x in runs],0.5) for k in KEYS}
    P(f'  [{label}] pool={len(pool)} draw n={min(n,len(pool))} x{ndraw}')
    P('    median '+fmt(med))
    s=[]
    for k in ['uniq','chao1','lutil_pos','lutil_ent','gini_id','zipf_id','even_pos','heaps','gini_el','zipf_el','mi_ex']:
        s.append(f'{k} {med[k]:.3f} [{ci([x[k] for x in runs],0.025):.3f},{ci([x[k] for x in runs],0.975):.3f}]')
    P('    CI: '+'; '.join(s))
    return med,runs

# ---------------- reference corpora ----------------
def jl(name): return [tuple(json.loads(l)['seq']) for l in open(CORP+name+'.jsonl')]
UR3_TITLE={'dumu','dub-sar','lugal','arad2','arad2-zu','arad','lu2','ensi2','ugula','nu-banda3','gudu4','dam-gar3','sanga','sukkal','szabra','agrig','sipa','lunga','simug','aszgab','nagar','kuruszda','szagina','ra2-gab','gal5-la2','muhaldim','kiszib3','dam','szesz','nin','ama','ab-ba','lu2-kin-gi4-a','i3-du8','gudu4-abzu','ensi','sagi','gal','nar','azlag2','ma2-lah5','szu-i','ad-kup4','asz-gab','bahar2','szitim','engar','mu'}
def ur3_names():
    out_w=[]; out_s=[]
    for s in jl('ur3_words'):
        w=s[0]
        if w in UR3_TITLE or w.startswith('_') or 'x' in w.split('-'): continue
        out_w.append((w,)); out_s.append(tuple(x for x in w.split('-') if x))
    return out_w,out_s
def linb_names():
    out=[]
    for s in jl('linb_words'):
        if ('VIR' in s or 'MUL' in s) and s[0] not in('NUM','VIR','MUL') and '-' in s[0] and not any(ch in s[0] for ch in '[]?*ạẹịọụṃṇṛṣṭḳẉ'):
            out.append(tuple(s[0].split('-')))
    return out
LAT_STOP={'hic','situs','est','et','sacrum','qui','quae','vixit','filio','filiae','filius','filia','coniugi','uxori','patri','matri','fratri','sorori','libertae','liberto','carissimae','carissimo','piissimae','piissimo','bene','merenti','dulcissimae','dulcissimo','sanctissimae','optimae','optimo','annis','annorum','vix','sibi','suis','posterisque','libertis','libertabusque','in','ex','pro','dis','manibus','memoriae','infelicissimae','infelicissimo','pientissimae','pientissimo','incomparabili','rarissimae','rarissimo','sanctae','sancto','posuit','fecit','fecerunt','parentes','parentibus','mater','pater','frater','soror','marito','coniunx','uxor','filii','liberti','liberta','libertus'}
def latin_names():
    out=[]
    for s in jl('latin_edh'):
        if len(s)>=4 and s[0]=='dis' and s[1]=='manibus':
            nm=[]
            for w in s[2:6]:
                if w in LAT_STOP or len(w)<3: break
                nm.append(w)
                if len(nm)==3: break
            if 1<=len(nm)<=3: out.append(tuple(nm))
    return out
def pe_middles():
    out=[]
    for s in jl('proto_elamite'):
        m=tuple(t for t in s if not (t[0]=='N' and len(t)>1 and t[1].isdigit()))
        if m: out.append(m)
    return out
def synth_uniform(n,lens,k,r):
    return [tuple(r.randrange(k) for _ in range(r.choice(lens))) for _ in range(n)]
def synth_zipf_names(n,lens,r,stock=3000,syll=400,a=1.0):
    names=[]
    while len(names)<stock:
        t=tuple(r.randrange(syll) for _ in range(r.choice(lens)))
        if t not in names: names.append(t)
    w=[1/(i+1)**a for i in range(stock)]
    return r.choices(names,weights=w,k=n)

# ---------------- IM77 via the bridge ----------------
def im77_objects():
    M2W={}
    for w,ms in BR.items():
        for m in ms:
            M2W.setdefault(m,[]).append(int(w))
    for p in PROP['proposals']:
        M2W.setdefault(p['M'],[]).append(p['W'])
    for m in M2W: M2W[m]=min(M2W[m])
    rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
    by=collections.defaultdict(list)
    for r in rows: by[r['text_no']].append(r)
    objs=[]; unb=0; tot=0; seen=set()
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
    P(f'  IM77: {len(objs)} distinct (site, type, text) objects with >= 2 signs on the text face; unbridged M tokens {unb}/{tot} = {unb/tot:.3f} (kept as opaque elements)')
    return annotate(objs)

# ================= cycles =================
OUT=f'data/derived/dark/loop53_cycle{CY}_log.txt'
if CY==1:
    P(f'== S-DARK-53 cycle 1: Indus middle capacity utilisation; nboot {NB}')
    RES={}
    for LV in ['seq_raw','seq_strong','seq_all']:
        objs=load_indus(LV)
        mids=[o['mid'] for o in objs if len(o['mid'])>=1]
        mids2=[o['mid'] for o in objs if len(o['mid'])>=2]
        P(f'\n##### {LV}: {len(objs)} distinct (site, type, text) objects; {len(mids)} middles (>=1 element), {len(mids2)} (>=2)')
        lens=collections.Counter(len(m) for m in mids); P('   middle length distribution',sorted(lens.items()))
        m,_=with_ci(mids,f'{LV} middles >=1'); RES[LV]=m
        m2,_=with_ci(mids2,f'{LV} middles >=2',nboot=max(20,NB//4)); RES[LV+'_ge2']=m2
        if LV=='seq_raw':
            # no-dedup variant and physical-seal variant for comparison with S-DARK-18/26
            raw=load_indus(LV,dedup=False); mr=[o['mid'] for o in raw if len(o['mid'])>=1]
            mm=metrics(mr); P(f'  [seq_raw NO dedup] n={mm["n"]}'); P('    '+fmt(mm))
            sl=[o['mid'] for o in raw if o['ot']=='seal' and len(o['mid'])>=1]
            mm=metrics(sl); P(f'  [seq_raw seals, one per object row] n={mm["n"]}'); P('    '+fmt(mm))
            # whole texts (not just middles) for reference
            wt=[tuple(o['seq']) for o in objs]; mm=metrics(wt); P(f'  [seq_raw WHOLE texts dedup] n={mm["n"]}'); P('    '+fmt(mm))
    P('\n=== cycle 1 summary (dedup site x type x text, middles >= 1)')
    for k,m in RES.items(): P(f'  {k:16s} n {m["n"]} D {m["D"]} uniq {m["uniq"]:.3f} Chao1 {m["chao1"]:.0f} GTnew {m["gt_new"]:.3f} k {m["k"]} log10cap naive/pos/ent {m["lcap_naive"]:.1f}/{m["lcap_pos"]:.1f}/{m["lcap_ent"]:.1f} log10util naive/pos/ent {m["lutil_naive"]:.1f}/{m["lutil_pos"]:.1f}/{m["lutil_ent"]:.1f} Gini_id {m["gini_id"]:.3f} Zipf_id {m["zipf_id"]:.2f} even {m["even_pos"]:.3f} Heaps {m["heaps"]:.3f} Gini_el {m["gini_el"]:.3f} Zipf_el {m["zipf_el"]:.2f}')
    json.dump(RES,open('data/derived/dark/loop53_cycle1.json','w'),indent=1)

if CY==2:
    P(f'== S-DARK-53 cycle 2: reference ladder at the Indus n; ndraw {NB}')
    objs=load_indus('seq_raw'); mids=[o['mid'] for o in objs if len(o['mid'])>=1]
    n=len(mids); lens=[len(m) for m in mids]; k=len(set(a for m in mids for a in m))
    P(f'  Indus seq_raw dedup: n={n}, k={k}')
    ind=metrics(mids); P('  [Indus seq_raw] '+fmt(ind))
    uw,us=ur3_names()
    pools={
     'D aircraft_reg (chars)':jl('aircraft_reg'),'D hts (2-digit groups)':jl('hts'),'D icd10 (chars)':jl('icd10'),
     'D unicode_names (words)':jl('unicode_names'),'G chess_eco (moves)':jl('chess_eco'),
     'L ur3 names, 1 per legend (words)':uw,'L ur3 names, 1 per legend (syllables)':us,'L ur3 names per impression (syll)':jl('ur3_names_syll'),
     'L linear B personnel names (syll)':linb_names(),'L latin EDH names after dis manibus (words)':latin_names(),
     'L linear B lines (words)':jl('linb_words'),'A proto-elamite entry middles (signs)':pe_middles(),
     'A proto-cuneiform lines':jl('proto_cuneiform'),'A khipu clusters':jl('khipu'),'G chords sections':jl('chords'),
     'D heraldry blazons (words)':jl('heraldry'),
    }
    r=random.Random(7)
    pools['S uniform ID generator (k=%d, Indus lengths)'%k]=synth_uniform(20000,lens,k,r)
    pools['S uniform ID generator (k=36, L=5-6)']=synth_uniform(20000,[5,6],36,r)
    pools['S zipf name stock 3000 (a=1.0)']=synth_zipf_names(20000,lens,r,3000,400,1.0)
    pools['S zipf name stock 3000 (a=0.6)']=synth_zipf_names(20000,lens,r,3000,400,0.6)
    pools['S zipf name stock 30000 (a=1.0)']=synth_zipf_names(20000,lens,r,30000,400,1.0)
    RES={'Indus seq_raw':ind}
    for lab,pool in pools.items():
        if len(pool)<200: P(f'  [{lab}] pool too small ({len(pool)}), skipped'); continue
        P(f'\n--- {lab}: pool {len(pool)} texts, {len(set(pool))} distinct, k={len(set(a for t in pool for a in t))}, mean L {sum(len(t) for t in pool)/len(pool):.2f}')
        med,_=subsampled(pool,n,lab,ndraw=max(20,NB//4)); RES[lab]=med
    P('\n=== cycle 2 LADDER at n=%d (median over draws): sorted by Heaps exponent'%n)
    P('  %-48s %5s %6s %6s %6s %6s %6s %6s %6s %6s'%('corpus','uniq','GTnew','Gini','Zipf','even','Heaps','lutilP','lutilE','MIex'))
    for lab,m in sorted(RES.items(),key=lambda x:-x[1]['heaps']):
        P('  %-48s %.3f %.3f %.3f %6.2f %.3f %.3f %6.1f %6.1f %6.2f'%(lab[:48],m['uniq'],m['gt_new'],m['gini_id'],m['zipf_id'],m['even_pos'],m['heaps'],m['lutil_pos'],m['lutil_ent'],m['mi_ex']))
    json.dump(RES,open('data/derived/dark/loop53_cycle2.json','w'),indent=1)

if CY==3:
    P(f'== S-DARK-53 cycle 3: Indus subsets, sites, IM77 replication, three levels; nboot {NB}')
    RES={}
    for LV in ['seq_raw','seq_strong','seq_all']:
        objs=load_indus(LV)
        SE=[o for o in objs if o['ot']=='seal' and len(o['mid'])>=1]
        subsets={'all middles':[o for o in objs if len(o['mid'])>=1],'seals':SE,
                 'quantity seals (COUNT in middle)':[o for o in SE if o['hascount']],
                 'office seals (jar closer 740, no count)':[o for o in SE if o['closer']==740 and not o['hascount']],
                 'other seals (no jar, no count)':[o for o in SE if o['closer']!=740 and not o['hascount']],
                 'tablets':[o for o in objs if o['ot']=='tablet' and len(o['mid'])>=1],
                 'Mohenjo-daro (all types)':[o for o in objs if o['site']=='Mohenjo-daro' and len(o['mid'])>=1],
                 'Harappa (all types)':[o for o in objs if o['site']=='Harappa' and len(o['mid'])>=1],
                 'Mohenjo-daro seals':[o for o in SE if o['site']=='Mohenjo-daro'],
                 'Harappa seals':[o for o in SE if o['site']=='Harappa'],
                 'held-out sites (all types)':[o for o in objs if not o['big'] and len(o['mid'])>=1]}
        P(f'\n##### {LV}')
        for lab,ss in subsets.items():
            mids=[o['mid'] for o in ss]
            if len(mids)<60: P(f'  [{lab}] too few ({len(mids)})'); continue
            m,_=with_ci(mids,f'{LV} {lab}',nboot=max(20,NB//4)); RES[f'{LV} | {lab}']=m
    P('\n##### IM77 (Mahadevan 1977) mapped M -> W through bridge_extended + S-DARK-27 proposals, same parser')
    im=im77_objects()
    SE=[o for o in im if o['ot']=='seal' and len(o['mid'])>=1]
    for lab,ss in {'IM77 all middles':[o for o in im if len(o['mid'])>=1],'IM77 seals':SE,
                   'IM77 quantity seals':[o for o in SE if o['hascount']],'IM77 office seals (740)':[o for o in SE if o['closer']==740 and not o['hascount']],
                   'IM77 other seals':[o for o in SE if o['closer']!=740 and not o['hascount']],
                   'IM77 Mohenjo-daro seals':[o for o in SE if o['site']=='Mohenjo-daro'],'IM77 Harappa seals':[o for o in SE if o['site']=='Harappa'],
                   'IM77 small sites (all types)':[o for o in im if not o['big'] and len(o['mid'])>=1]}.items():
        mids=[o['mid'] for o in ss]
        if len(mids)<60: P(f'  [{lab}] too few ({len(mids)})'); continue
        m,_=with_ci(mids,lab,nboot=max(20,NB//4)); RES[lab]=m
    # size-matched synthetic anchors for the subsets (same n, k, lengths): uniform ID and zipf names
    P('\n##### size-matched anchors per subset (uniform ID at the subset k and lengths; Zipf-1.0 stock 3000)')
    for key in list(RES):
        m=RES[key]
        if m['n']<150: continue
        r=random.Random(11); lens=[2,3,4] if m['meanL']>2.2 else [1,2,3]
        u=metrics(synth_uniform(m['n'],lens,m['k'],r),r); z=metrics(synth_zipf_names(m['n'],lens,r,3000,400,1.0),r)
        P(f'  {key[:52]:52s} n={m["n"]:4d} uniq {m["uniq"]:.3f} (U {u["uniq"]:.3f} Z {z["uniq"]:.3f}) Gini {m["gini_id"]:.3f} (U {u["gini_id"]:.3f} Z {z["gini_id"]:.3f}) Heaps {m["heaps"]:.3f} (U {u["heaps"]:.3f} Z {z["heaps"]:.3f}) even {m["even_pos"]:.3f} (U {u["even_pos"]:.3f} Z {z["even_pos"]:.3f}) lutilE {m["lutil_ent"]:.1f} (U {u["lutil_ent"]:.1f} Z {z["lutil_ent"]:.1f})')
    P('\n=== cycle 3 summary')
    P('  %-58s %5s %5s %6s %6s %6s %6s %6s %6s'%('subset','n','uniq','Gini','Zipf','even','Heaps','lutilE','GTnew'))
    for k,m in RES.items(): P('  %-58s %5d %.3f %.3f %6.2f %.3f %.3f %6.1f %.3f'%(k[:58],m['n'],m['uniq'],m['gini_id'],m['zipf_id'],m['even_pos'],m['heaps'],m['lutil_ent'],m['gt_new']))
    json.dump(RES,open('data/derived/dark/loop53_cycle3.json','w'),indent=1)

open(OUT,'w').write('\n'.join(LOG)+'\n')
