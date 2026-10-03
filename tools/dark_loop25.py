"""S-DARK-25: PHONETIC COMPLEMENTS AND DETERMINATIVES without sound values.
Two structural signatures of a logo-syllabic script:
 (1) complements: a word-sign X is followed/preceded by one of a SMALL set of signs; the set is SHARED across many X
     (a complement inventory) and the complement signs are ALSO used as free signs elsewhere with a different profile.
 (2) determinatives: a class sign that never stands alone, attaches at a fixed side to MANY different hosts, and the
     hosts form a coherent class (vs a frame word, which attaches to everything).
Identical tests on Indus (seq_raw / seq_strong / seq_all), Ur III seal legends (syllables; 'd' and 'ki' are real
determinatives), Linear B tablet lines (words, ideograms, numerals), Proto-Elamite lines, and two planted corpora:
a logo-syllabic one with known complements + determinatives, and a frame-only one (opener / name / qualifier / closer).
Null: tokens shuffled across texts within strata (site x object type x length bin), which keeps text lengths and
per-stratum sign frequencies and destroys adjacency. Indus also gets a frame-kept null (shuffle within S310 slot labels).
Usage: python3 tools/dark_loop25.py <cycle 1|2|3> <seq_raw|seq_strong|seq_all> [nperm]
"""
import json,sys,random,collections,math,re
C=json.load(open('data/derived/merged-corpus-canonical.json'))
BR=json.load(open('data/derived/bridge_extended.json'))
SP='/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
def M(w):
    m=BR.get(str(w)); return f'W{w}(M{"/".join(map(str,m))})' if m else f'W{w}'
CY=int(sys.argv[1]); LV=sys.argv[2]; NP=int(sys.argv[3]) if len(sys.argv)>3 else 300
rnd=random.Random(25)
def otype(t):
    t=t.split(':')[0]
    return {'SEAL':'seal','TAB':'tablet','POT':'pot','TAG':'sealing'}.get(t,'other')
OBJ=[]
for r in C:
    s=r[LV]
    if not s or len(s)<2 or r['complete']!='Y' or r['dir.'].strip()=='-': continue
    OBJ.append(dict(cisi=r['cisi'],site=r['site'],ot=otype(r['type']),seq=list(s),big=r['site'] in('Mohenjo-daro','Harappa')))
print(f'== S-DARK-25 cycle {CY} level {LV} nperm {NP}; Indus objects {len(OBJ)} (complete, direction recorded, >=2 signs)')

# ---------------- controls ----------------
def load_ur3():
    legs=json.load(open(SP+'ur3_legends.json')); seen=set(); out=[]
    for lg in legs:
        flat=tuple(x for ln in lg['lines'] for x in ln)
        if len(flat)<2 or flat in seen or 'x' in flat: continue
        seen.add(flat); out.append(dict(site='ur3',ot='legend',seq=list(flat),big=rnd.random()<0.7))
    return out
def load_linb():
    out=[]; seen=set()
    for l in open('other-scripts/linear-a/data/damos_items.jsonl'):
        d=json.loads(l); c=d.get('content') or ''
        for ln in c.split('\n'):
            m=re.match(r'\s*\.(\d+[a-z]?)\s+(.*)',ln)
            if not m: continue
            toks=[]
            for t in re.split(r'\s+',m.group(2).strip()):
                t=t.strip('[],')
                if not t or t in ('/','vac.','vacat','vest.') or t.startswith('vac'): continue
                if re.fullmatch(r'\d+',t): t='NUM'
                elif re.fullmatch(r"'?[a-z0-9*]+(-[a-z0-9*]+)+'?",t): t='w:'+t.strip("'")
                elif re.fullmatch(r'[A-Z][A-Z0-9*+]*',t): t='I:'+t
                elif t in ('S','V','T','Z','M','N','P','Q','L'): t='U:'+t
                else: continue
                toks.append(t)
            if len(toks)>=2 and tuple(toks) not in seen:
                seen.add(tuple(toks)); out.append(dict(site='linb',ot='line',seq=toks,big=rnd.random()<0.7))
    return out
def load_pe():
    pe=json.load(open('other-scripts/proto-elamite/data/pe_corpus.json')); out=[]; seen=set()
    for t in pe:
        for ln in t['lines']:
            s=[x.split('~')[0] for x in ln['signs'] if x!='x' and not x.startswith('x')]
            if ln.get('lacuna'): continue
            if ln['numerals']: s=s+['N:'+ln['numerals'][0][1]]
            if len(s)>=2 and tuple(s) not in seen:
                seen.add(tuple(s)); out.append(dict(site='pe',ot=t.get('object_type','tablet'),seq=s,big=rnd.random()<0.7))
    return out
def zipf(n,a=1.0):
    w=[1/(i+1)**a for i in range(n)]; z=sum(w); return [x/z for x in w]
def planted_logosyll(ntext=3200,seed=7):
    """logo-syllabic: 60 logograms (Zipf); each has a final syllable from a 12-sign complement inventory, written as a
    phonetic complement after the logogram with p=0.6; 40 syllable signs spell names (2-4 syllables, Zipf);
    3 determinatives, each before its own class of 10 logograms (p=0.9); numerals."""
    R=random.Random(seed); syl=[f's{i}' for i in range(40)]; ws=zipf(40,0.9)
    logo=[f'L{i}' for i in range(60)]; wl=zipf(60,1.0)
    fin={L:R.choices(syl[:12],zipf(12,0.8))[0] for L in logo}
    det={}
    for d in range(3):
        for L in logo[d*10+3:(d+1)*10+3]: det[L]=f'D{d}'
    out=[]
    for _ in range(ntext):
        s=[]
        for _ in range(R.choice([1,2,2,3,3,4])):
            u=R.random()
            if u<0.45:
                L=R.choices(logo,wl)[0]
                if L in det and R.random()<0.9: s.append(det[L])
                s.append(L)
                if R.random()<0.6: s.append(fin[L])
            elif u<0.85:
                s+=R.choices(syl,ws,k=R.choice([2,3,3,4]))
            else: s.append('NUM')
        if len(s)>=2: out.append(dict(site=R.choice(['A','B','C']),ot=R.choice(['seal','tablet']),seq=s,big=R.random()<0.7))
    return out
def planted_frame(ntext=3200,seed=8):
    """frame only: [opener+marker] name(1-4 of 150 signs) [qualifier from the closer's own set] closer [suffix].
    No complements: qualifier pools and name pools are disjoint; 25 qualifier signs shared unevenly among 10 closers."""
    R=random.Random(seed); names=[f'n{i}' for i in range(150)]; wn=zipf(150,1.0)
    closers=[f'c{i}' for i in range(10)]; wc=zipf(10,0.9); quals=[f'q{i}' for i in range(25)]
    QS={c:R.sample(quals,3) for c in closers}
    out=[]
    for _ in range(ntext):
        s=[]
        if R.random()<0.45: s+= [R.choice(['o0','o0','o1','o2']),'mk']
        s+=R.choices(names,wn,k=R.choice([1,2,2,3,3,4]))
        if R.random()<0.8:
            c=R.choices(closers,wc)[0]
            if R.random()<0.6: s.append(R.choice(QS[c]))
            s.append(c)
            if R.random()<0.15: s.append('sf')
        if len(s)>=2: out.append(dict(site=R.choice(['A','B','C']),ot=R.choice(['seal','tablet']),seq=s,big=R.random()<0.7))
    return out

# ---------------- frame parser (S310, as in dark_loop19/22) for Indus slot labels ----------------
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
for o in OBJ: o['lab']=parse(o['seq'])
slotvotes=collections.defaultdict(collections.Counter)
for o in OBJ:
    for a,l in zip(o['seq'],o['lab']): slotvotes[a][l]+=1
SLOT={a:c.most_common(1)[0][0] for a,c in slotvotes.items()}
def crude_slot(objs):
    """for controls: a sign's modal position class first / mid / last"""
    v=collections.defaultdict(collections.Counter)
    for o in objs:
        L=len(o['seq'])
        for k,a in enumerate(o['seq']): v[a]['first' if k==0 else 'last' if k==L-1 else 'mid']+=1
    return {a:c.most_common(1)[0][0] for a,c in v.items()}

def H(cnt):
    n=sum(cnt.values()); return -sum(c/n*math.log2(c/n) for c in cnt.values() if c) if n else 0.0
def pval(obs,null,side='hi'):
    n=len(null)
    if side=='hi': return (sum(1 for x in null if x>=obs)+1)/(n+1)
    return (sum(1 for x in null if x<=obs)+1)/(n+1)
def cos(a,b):
    ks=set(a)|set(b); na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
    return sum(a.get(k,0)*b.get(k,0) for k in ks)/(na*nb) if na and nb else 0.0
def jsd(p,q):
    ks=set(p)|set(q); sp=sum(p.values()); sq=sum(q.values())
    def kl(a,sa,m): return sum(a[k]/sa*math.log2((a[k]/sa)/m[k]) for k in a if a[k])
    m={k:0.5*(p.get(k,0)/sp+q.get(k,0)/sq) for k in ks}
    return 0.5*kl(p,sp,m)+0.5*kl(q,sq,m)

def strat_key(o):
    L=len(o['seq']); lb=L if L<=6 else 7 if L<=8 else 9
    return (o['site'],o['ot'],lb)
def shuffled(objs,R,keep_slot=False):
    """stratified token shuffle: pool tokens per stratum (site x type x length bin), reassign; keeps lengths + per-stratum sign counts.
    keep_slot: pool per stratum x slot label (Indus only), so the frame is kept and only within-slot identity moves."""
    pools=collections.defaultdict(list); idx=collections.defaultdict(list)
    for i,o in enumerate(objs):
        k=strat_key(o)
        for p,a in enumerate(o['seq']):
            kk=(k,o['lab'][p]) if keep_slot else k
            pools[kk].append(a); idx[kk].append((i,p))
    new=[list(o['seq']) for o in objs]
    for kk,pl in pools.items():
        R.shuffle(pl)
        for (i,p),a in zip(idx[kk],pl): new[i][p]=a
    return [dict(o,seq=s) for o,s in zip(objs,new)]

# =========================== cycle 1: complement signature ===========================
def neighbour_tables(objs,minn):
    toks=collections.Counter(a for o in objs for a in o['seq'])
    signs=[a for a,n in toks.items() if n>=minn]; sset=set(signs)
    R_=collections.defaultdict(collections.Counter); L_=collections.defaultdict(collections.Counter)
    for o in objs:
        s=o['seq']
        for k,a in enumerate(s):
            if a in sset:
                if k+1<len(s): R_[a][s[k+1]]+=1
                if k>0: L_[a][s[k-1]]+=1
    return toks,signs,R_,L_
def k80(cnt,cov=0.8):
    tot=sum(cnt.values()); acc=0; core=[]
    for a,n in cnt.most_common():
        core.append(a); acc+=n
        if acc/tot>=cov: break
    return len(core),core
def complement_stats(objs,minn=30,kmax=4,minnb=20,minhosts=3,slot=None,verbose=False,name=''):
    toks,signs,R_,L_=neighbour_tables(objs,minn)
    small={}  # (sign, side) -> core set
    for a in signs:
        for side,T in (('R',R_),('L',L_)):
            cnt=T[a]
            if sum(cnt.values())<minnb: continue
            k,core=k80(cnt)
            if k<=kmax: small[(a,side)]=core
    n_small=len(small)
    # hubs: a neighbour c that is in the core set of >= minhosts hosts on the same side
    hostsof=collections.defaultdict(set)
    for (a,side),core in small.items():
        for c in core: hostsof[(c,side)].add(a)
    hubs={k:v for k,v in hostsof.items() if len(v)>=minhosts}
    n_hubs=len(hubs); n_att=sum(len(v) for v in hubs.values())
    # per hub: free fraction, far-side profile cosine attached vs free, host slot entropy, pair end-offset entropy
    rows=[]; Ntok=sum(toks.values())
    for (c,side),hosts in sorted(hubs.items(),key=lambda kv:-len(kv[1])):
        att=0; free=0; far_att=collections.Counter(); far_free=collections.Counter(); pos_att=collections.Counter(); pos_free=collections.Counter(); alone=0
        st_att=collections.Counter(); st_free=collections.Counter(); unif=collections.Counter(); ustart=collections.Counter()
        for o in objs:
            s=o['seq']; L=len(s)
            for k,a in enumerate(s):
                if a!=c: continue
                hostpos=k-1 if side=='R' else k+1   # side R means c is the RIGHT neighbour of the host
                farpos=k+1 if side=='R' else k-1
                attached=0<=hostpos<L and s[hostpos] in hosts
                far=s[farpos] if 0<=farpos<L else '#'
                if attached:
                    att+=1; far_att[far]+=1; pos_att[L-1-k]+=1; st_att[k]+=1
                    for j in range(L): unif[L-1-j]+=1; ustart[j]+=1   # uniform-position baseline over the same texts
                else: free+=1; far_free[far]+=1; pos_free[L-1-k]+=1; st_free[k]+=1
        tot=att+free; ff=free/tot if tot else 0
        cs=cos(far_att,far_free) if att>=10 and free>=10 else float('nan')
        js=max(jsd(pos_att,pos_free),jsd(st_att,st_free)) if att>=10 and free>=10 else float('nan')
        hu=min(H(unif),H(ustart)); mob=min(H(pos_att),H(st_att))/hu if hu else float('nan')
        hs=collections.Counter((slot or {}).get(h,'?') for h in hosts); hse=H(hs)
        rows.append(dict(c=c,side=side,nhost=len(hosts),hosts=sorted(hosts,key=lambda h:-toks[h]),tok=tot,free=ff,cos=cs,jsd=js,hslotH=hse,hslots=dict(hs),
                         pairendH=H(pos_att),mob=mob,share=tot/Ntok))
    return dict(n_signs=len(signs),n_small=n_small,n_hubs=n_hubs,n_att=n_att,hubs=rows,small=small)
def fmt_hub(r,nm):
    cs='nan' if r['cos']!=r['cos'] else '%.2f'%r['cos']; js='nan' if r['jsd']!=r['jsd'] else '%.2f'%r['jsd']
    return (f"  {nm(r['c'])} side {r['side']}: hosts {r['nhost']} [{' '.join(nm(h) for h in r['hosts'][:8])}{' ...' if r['nhost']>8 else ''}] tok {r['tok']} "
            f"free {r['free']:.2f} farcos(att,free) {cs} posJSD {js} hostslotH {r['hslotH']:.2f} {r['hslots']} pairEndH {r['pairendH']:.2f}")
def run_cycle1(objs,name,nperm,slot,nm,minn=30,keep_slot=False):
    print(f'\n--- {name}: {len(objs)} texts, {sum(len(o["seq"]) for o in objs)} tokens')
    S=complement_stats(objs,minn=minn,slot=slot)
    print(f'  signs >= {minn} tokens: {S["n_signs"]}; (sign,side) with a small neighbour set (<= 4 signs cover 80%, >= 20 neighbours): {S["n_small"]}; '
          f'hubs (a neighbour in >= 3 hosts\' small sets, same side): {S["n_hubs"]}; host-hub attachments {S["n_att"]}')
    R=random.Random(1)
    null=[]; nullh=[]; nulla=[]
    for _ in range(nperm):
        sh=shuffled(objs,R,keep_slot=keep_slot); T=complement_stats(sh,minn=minn,slot=slot)
        null.append(T['n_small']); nullh.append(T['n_hubs']); nulla.append(T['n_att'])
    print(f'  null ({"frame kept" if keep_slot else "stratified shuffle"} x{nperm}): small sets {sum(null)/len(null):.1f} (max {max(null)}) P={pval(S["n_small"],null):.4f}; '
          f'hubs {sum(nullh)/len(nullh):.1f} (max {max(nullh)}) P={pval(S["n_hubs"],nullh):.4f}; attachments {sum(nulla)/len(nulla):.1f} P={pval(S["n_att"],nulla):.4f}')
    rows=S['hubs']
    for r in rows[:25]: print(fmt_hub(r,nm))
    # summary signature numbers
    if rows:
        ff=[r['free'] for r in rows]; hs=[r['hslotH'] for r in rows]; pe=[r['pairendH'] for r in rows]; mb=[r['mob'] for r in rows if r['mob']==r['mob']]
        cs=[r['cos'] for r in rows if r['cos']==r['cos']]
        nfree=sum(1 for r in rows if r['free']>=0.3)
        print(f'  SIGNATURE: hubs {len(rows)}; median free fraction {sorted(ff)[len(ff)//2]:.2f}; hubs with free >= 0.3: {nfree}; '
              f'mean host-slot entropy {sum(hs)/len(hs):.2f} bits; mean pair end-offset entropy {sum(pe)/len(pe):.2f} bits; mean mobility (nearer-edge offset entropy / uniform) {sum(mb)/len(mb) if mb else float("nan"):.2f}; '
              f'mean far-side cosine attached vs free {sum(cs)/len(cs) if cs else float("nan"):.2f} (n={len(cs)})')
    # small-set hosts: how many are explained by hubs
    hubsigns={k for k in [(r['c'],r['side']) for r in rows]}
    expl=sum(1 for (a,side),core in S['small'].items() if any((c,side) in hubsigns for c in core))
    print(f'  small-set (sign,side) pairs whose core contains a hub: {expl} of {S["n_small"]}')
    if name.startswith('Indus') and not keep_slot:
        for (a,side),core in sorted(S['small'].items(),key=lambda kv:str(kv[0])):
            print(f'    small set: {nm(a)} {side} -> [{" ".join(nm(c) for c in core)}]  slot {(slot or {}).get(a,"?")}')
    return S

# =========================== cycle 2: determinative signature ===========================
def determinative_stats(objs,minn=30,slot=None,nm=str,nperm=200,verbose=True,name=''):
    toks=collections.Counter(a for o in objs for a in o['seq']); signs=[a for a,n in toks.items() if n>=minn]; sset=set(signs)
    NUMLIKE=lambda a: (isinstance(a,int) and a in NUM) or (isinstance(a,str) and (a.startswith('N') or a=='NUM'))
    alone=collections.Counter(); onlynon=collections.Counter(); final=collections.Counter(); initial=collections.Counter()
    Rn=collections.defaultdict(collections.Counter); Ln=collections.defaultdict(collections.Counter)
    singles=collections.Counter(a for r in C for a in (r[LV] if len(r[LV])==1 else [])) if name.startswith('Indus') else collections.Counter()
    for o in objs:
        s=o['seq']; L=len(s); nonnum=[a for a in s if not NUMLIKE(a)]
        for k,a in enumerate(s):
            if a not in sset: continue
            if k==L-1: final[a]+=1
            else: Rn[a][s[k+1]]+=1
            if k==0: initial[a]+=1
            else: Ln[a][s[k-1]]+=1
            if len(nonnum)==1 and nonnum[0]==a: onlynon[a]+=1
    # other-side context profiles for coherence
    ctxR=collections.defaultdict(collections.Counter); ctxL=collections.defaultdict(collections.Counter)
    for o in objs:
        s=o['seq']
        for k,a in enumerate(s):
            if k+1<len(s): ctxR[a][s[k+1]]+=1
            if k>0: ctxL[a][s[k-1]]+=1
    rows=[]
    allsigns=[a for a,n in toks.items() if n>=5]; wts=[toks[a] for a in allsigns]
    R=random.Random(2)
    for a in signs:
        n=toks[a]; nal=singles.get(a,0); non=onlynon[a]
        for side,T,edge in (('R',Rn,final),('L',Ln,initial)):
            # side R: a precedes its hosts (a's right neighbours are hosts); side L: a follows its hosts
            att=sum(T[a].values()); fix=att/(att+edge[a]) if att+edge[a] else 0
            hosts=[h for h,c in T[a].items() if c>=2]
            if fix<0.9 or len(hosts)<8 or H(T[a])<3.0: continue
            # coherence: hosts' profile on the side AWAY from a (a precedes -> host's right context; excluding a itself)
            far=ctxR if side=='R' else ctxL
            hosts=sorted(hosts,key=lambda h:-T[a][h])[:25]   # cap at the 25 commonest hosts (speed)
            memo={}
            def cc(x,y):
                k=(x,y) if str(x)<=str(y) else (y,x)
                if k not in memo: memo[k]=cos(far[x],far[y])
                return memo[k]
            def coh(hs):
                hs=[h for h in hs if sum(far[h].values())>=3]
                if len(hs)<3: return float('nan')
                ps=[cc(hs[i],hs[j]) for i in range(len(hs)) for j in range(i+1,len(hs))]
                return sum(ps)/len(ps)
            obs=coh(hosts)
            if obs!=obs: continue
            # frequency-matched random host sets
            nullc=[]
            for _ in range(nperm):
                hs=set()
                while len(hs)<len(hosts): hs.add(R.choices(allsigns,wts)[0])
                v=coh(list(hs))
                if v==v: nullc.append(v)
            mu=sum(nullc)/len(nullc); sd=math.sqrt(sum((x-mu)**2 for x in nullc)/len(nullc)) or 1e-9
            # host loyalty: P(c on this side | host), token-weighted over hosts, and share of hosts with P >= 0.5
            hl=[T[a][h]/toks[h] for h in hosts]; loy=sum(T[a][h] for h in hosts)/sum(toks[h] for h in hosts); loy50=sum(1 for x in hl if x>=0.5)/len(hl)
            rows.append(dict(c=a,side=side,n=n,alone=nal,onlynon=non,fix=fix,nhost=len(hosts),partH=H(T[a]),coh=obs,cohnull=mu,z=(obs-mu)/sd,loy=loy,loy50=loy50,
                             p=pval(obs,nullc),slot=(slot or {}).get(a,'?'),hosts=sorted(T[a],key=lambda h:-T[a][h])[:6]))
    rows.sort(key=lambda r:-r['nhost'])
    print(f'\n--- {name}: candidates (never-edge on the host side >= 0.90, >= 8 hosts with >= 2 tokens, partner entropy >= 3 bits): {len(rows)} of {len(signs)} signs')
    print('   sign side n alone onlyNonNum sideFix nHosts partnerH hostCoherence(null) z P loyalty loy>=.5 slot | top hosts')
    for r in rows[:40]:
        print(f"   {nm(r['c'])} {r['side']} {r['n']} {r['alone']} {r['onlynon']} {r['fix']:.2f} {r['nhost']} {r['partH']:.2f} {r['coh']:.3f}({r['cohnull']:.3f}) z{r['z']:+.1f} P={r['p']:.3f} loy {r['loy']:.2f} {r['loy50']:.2f} {r['slot']} | {' '.join(nm(h) for h in r['hosts'])}")
    loyal=[r for r in rows if r['alone']+r['onlynon']<3 and r['loy']>=0.5]
    print(f'   => loyal class markers (never alone, hosts take it >= 50% of the time): {len(loyal)} [{" ".join(nm(r["c"])+"/"+r["side"]+"(%.2f)"%r["loy"] for r in loyal)}]')
    det=[r for r in rows if r['alone']+r['onlynon']<3 and r['z']>=3]
    frame=[r for r in rows if r['alone']+r['onlynon']<3 and r['z']<3]
    print(f'   => determinative-like (never alone, coherent host class z>=3): {len(det)} [{" ".join(nm(r["c"])+"/"+r["side"] for r in det)}]')
    print(f'   => frame-word-like (never alone, hosts incoherent z<3): {len(frame)} [{" ".join(nm(r["c"])+"/"+r["side"] for r in frame)}]')
    return rows

# =========================== driver ===========================
nmI=lambda a:M(a)
CONTROLS=[('Ur III legends',load_ur3,None),('Linear B lines',load_linb,None),('Proto-Elamite lines',load_pe,None),
          ('Planted logo-syllabic',planted_logosyll,None),('Planted frame-only',planted_frame,None)]
if CY==1:
    run_cycle1(OBJ,f'Indus {LV}',NP,SLOT,nmI)
    run_cycle1(OBJ,f'Indus {LV} frame-kept null',NP,SLOT,nmI,keep_slot=True)
    for nm_,ld,_ in CONTROLS:
        ob=ld(); run_cycle1(ob,nm_,min(NP,100),crude_slot(ob),str)
elif CY==2:
    determinative_stats(OBJ,slot=SLOT,nm=nmI,nperm=NP,name=f'Indus {LV}')
    for nm_,ld,_ in CONTROLS:
        ob=ld(); determinative_stats(ob,slot=crude_slot(ob),nm=str,nperm=NP,name=nm_)
elif CY==3:
    # held-out replication: hubs and determinatives fitted on MD+HP, tested at other sites against the stratified null
    fit=[o for o in OBJ if o['big']]; test=[o for o in OBJ if not o['big']]
    print(f'\n--- held-out: fit {len(fit)} MD+HP texts, test {len(test)} other-site texts')
    S=complement_stats(fit,slot=SLOT)
    hubs={(r['c'],r['side']):set(r['hosts']) for r in S['hubs']}
    print(f'  fitted hubs: {len(hubs)}: '+' '.join(f"{M(c)}/{sd}({len(h)})" for (c,sd),h in hubs.items()))
    def att_rate(objs):
        a=0; t=0
        for o in objs:
            s=o['seq']
            for k in range(len(s)-1):
                for (c,sd),hs in hubs.items():
                    if sd=='R' and s[k] in hs: t+=1; a+= s[k+1]==c
                    if sd=='L' and s[k+1] in hs: t+=1; a+= s[k]==c
        return a/t if t else 0,t
    obs,t=att_rate(test); R=random.Random(3); null=[att_rate(shuffled(test,R))[0] for _ in range(NP)]
    print(f'  held-out attachment rate (host token followed/preceded by its fitted hub): {obs:.3f} over {t} host tokens; null {sum(null)/len(null):.3f} max {max(null):.3f} P={pval(obs,null):.4f}')
    nullk=[att_rate(shuffled(test,R,keep_slot=True))[0] for _ in range(NP)]
    print(f'  frame-kept null: {sum(nullk)/len(nullk):.3f} max {max(nullk):.3f} P={pval(obs,nullk):.4f}')
    # per-hub held-out
    for (c,sd),hs in hubs.items():
        a=0;t=0
        for o in test:
            s=o['seq']
            for k in range(len(s)-1):
                if sd=='R' and s[k] in hs: t+=1; a+= s[k+1]==c
                if sd=='L' and s[k+1] in hs: t+=1; a+= s[k]==c
        print(f'    {M(c)}/{sd}: held-out {a}/{t}')
    # held-out free use of hubs: fraction of hub tokens at other sites not adjacent to a fitted host
    for (c,sd),hs in hubs.items():
        att=0;fr=0
        for o in test:
            s=o['seq']
            for k,x in enumerate(s):
                if x!=c: continue
                hp=k-1 if sd=='R' else k+1
                if 0<=hp<len(s) and s[hp] in hs: att+=1
                else: fr+=1
        print(f'    {M(c)}/{sd}: held-out tokens attached {att} free {fr}')
    # determinatives: fitted on MD+HP, test coherence of host class at other sites
    print('\n--- determinative candidates fitted on MD+HP, re-tested on other sites')
    rowsF=determinative_stats(fit,slot=SLOT,nm=nmI,nperm=NP,name=f'Indus {LV} MD+HP')
    rowsT=determinative_stats(test,minn=12,slot=SLOT,nm=nmI,nperm=NP,name=f'Indus {LV} other sites (minn 12)')
    cf={(r['c'],r['side']):r for r in rowsF}; ct={(r['c'],r['side']):r for r in rowsT}
    both=[k for k in cf if k in ct]
    print(f'  candidates in both: {len(both)}: '+' '.join(f"{M(k[0])}/{k[1]} z_fit {cf[k]['z']:+.1f} z_test {ct[k]['z']:+.1f}" for k in both))
    missing=[k for k in cf if k not in ct]
    print(f'  fitted candidates not qualifying at other sites: {len(missing)} ({" ".join(M(k[0])+"/"+k[1] for k in missing)})')
print('\n== done')
