"""S-DARK-19: is an Indus text a SEQUENCE or a SET?  (arrow-in-the-dark loop 19)
(a) anagram census  (b) order entropy per multiset  (c) fixed vs free sign-pair order and the partial order it builds
(d) co-occurrence of free pairs  (e) controls: Ur III legends, planted free-order and planted strict-order corpora.
Usage: python3 tools/dark_loop19.py <cycle 1|2|3> <seq_raw|seq_strong|seq_all> [nperm]
Writes nothing itself; the caller redirects stdout to data/derived/dark/loop19_cycle<N>_<level>.txt
"""
import json,sys,random,collections,math,itertools
from math import comb
C=json.load(open('data/derived/merged-corpus-canonical.json'))
BR=json.load(open('data/derived/bridge_extended.json'))
def M(w):
    m=BR.get(str(w)); return f'W{w}(M{"/".join(map(str,m))})' if m else f'W{w}(M?)'
CY=int(sys.argv[1]); LV=sys.argv[2]; NP=int(sys.argv[3]) if len(sys.argv)>3 else 1000
rnd=random.Random(19)
def otype(t):
    t=t.split(':')[0]
    return {'SEAL':'seal','TAB':'tablet','POT':'pot','TAG':'sealing'}.get(t,'other')
# objects: complete, direction recorded (S-DARK-8.4: dir '-' is direction-uncertain), >= 2 signs
OBJ=[]
for r in C:
    s=r[LV]
    if not s or len(s)<2 or r['complete']!='Y' or r['dir.'].strip()=='-': continue
    OBJ.append(dict(cisi=r['cisi'],site=r['site'],ot=otype(r['type']),seq=list(s),big=r['site'] in('Mohenjo-daro','Harappa')))
print(f'== S-DARK-19 cycle {CY} level {LV} nperm {NP}; objects {len(OBJ)} (complete, direction recorded, >=2 signs)')
print('   by type',dict(collections.Counter(o["ot"] for o in OBJ)))
# ---------- frame parser (copied from tools/parse_all.py, S310), QUAL learned on all texts of this level ----------
OPEN={817,861,820}; MARK={2,60}; SUF={400,90}; CL=[740,520,151,156,527,226,617,154,158,236,700]
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
        if len(s)>1 and s[1] in MARK: lab[1]='MARKER'; i=2
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
SLOTRANK={'OPENER':0,'MARKER':1,'NAME':2,'COUNT':2,'TITLE':3,'CLOSER':4,'SUFFIX':5}
for o in OBJ: o['lab']=parse(o['seq'])
# majority frame slot per sign
slotvotes=collections.defaultdict(collections.Counter)
for o in OBJ:
    for a,l in zip(o['seq'],o['lab']): slotvotes[a][l]+=1
SLOT={a:c.most_common(1)[0][0] for a,c in slotvotes.items()}
def shuffle_within_slot(seq,lab,r):
    out=list(seq)
    for l in set(lab):
        pos=[k for k in range(len(seq)) if lab[k]==l]
        vals=[seq[k] for k in pos]; r.shuffle(vals)
        for k,v in zip(pos,vals): out[k]=v
    return out
def shuffle_all(seq,lab,r):
    out=list(seq); r.shuffle(out); return out
def pval(obs,null,side='hi'):
    n=len(null)
    if side=='hi': return (sum(1 for x in null if x>=obs)+1)/(n+1)
    return (sum(1 for x in null if x<=obs)+1)/(n+1)
def q(null,p): null=sorted(null); return null[min(len(null)-1,int(p*len(null)))]
# ============================ (a) anagram census ============================
def anagram_stats(objs,seqs):
    """seqs: list of sign lists parallel to objs. Returns dict of counts."""
    groups=collections.defaultdict(list)
    for o,s in zip(objs,seqs): groups[tuple(sorted(s))].append(tuple(s))
    g2=[v for v in groups.values() if len(v)>=2]
    anag=sum(1 for v in g2 if len(set(v))>1)
    pairs_same=pairs_diff=0
    for v in g2:
        c=collections.Counter(v); n=len(v)
        same=sum(comb(k,2) for k in c.values()); tot=comb(n,2)
        pairs_same+=same; pairs_diff+=tot-same
    return dict(groups=len(g2),anagram_groups=anag,pairs_same=pairs_same,pairs_diff=pairs_diff,
                rate=anag/max(1,len(g2)),pair_rate=pairs_diff/max(1,pairs_same+pairs_diff))
def run_a(objs,label,nperm):
    seqs=[o['seq'] for o in objs]; labs=[o['lab'] for o in objs]
    obs=anagram_stats(objs,seqs)
    print(f'\n-- (a) anagram census [{label}] objects {len(objs)}; multisets on >=2 objects {obs["groups"]}; in >1 order {obs["anagram_groups"]} ({obs["rate"]:.3f}); object pairs same order {obs["pairs_same"]} vs different {obs["pairs_diff"]} (diff share {obs["pair_rate"]:.3f})')
    for name,fn in (('within-slot shuffle',shuffle_within_slot),('whole-text shuffle',shuffle_all)):
        r=random.Random(7); nr=[];npr=[]
        for _ in range(nperm):
            st=anagram_stats(objs,[fn(s,l,r) for s,l in zip(seqs,labs)]); nr.append(st['rate']); npr.append(st['pair_rate'])
        print(f'   null {name:20s}: anagram-group rate median {q(nr,.5):.3f} [5% {q(nr,.05):.3f}, 95% {q(nr,.95):.3f}], P(obs<=null) {pval(obs["rate"],nr,"lo"):.4f}; pair-diff share median {q(npr,.5):.3f} [{q(npr,.05):.3f},{q(npr,.95):.3f}] P_lo {pval(obs["pair_rate"],npr,"lo"):.4f}; order information retained = 1 - obs/null = {1-obs["pair_rate"]/max(1e-9,q(npr,.5)):.2f}')
    # where do the orders differ
    groups=collections.defaultdict(list)
    for o in objs: groups[tuple(sorted(o['seq']))].append(o)
    swaps=collections.Counter(); adj=0; nonadj=0; examples=[]; sitespan=collections.Counter()
    for k,v in groups.items():
        orders=sorted(set(tuple(o['seq']) for o in v))
        if len(orders)<2: continue
        sitespan['multi-site' if len(set(o['site'] for o in v))>1 else 'one site']+=1
        ex=' | '.join('-'.join(map(str,x))+f" x{sum(1 for o in v if tuple(o['seq'])==x)}" for x in orders)
        examples.append((len(v),ex,sorted(set(o['site'][:3]+'/'+o['ot'] for o in v))))
        for x,y in itertools.combinations(orders,2):
            if len(set(x))<len(x): continue
            px={a:i for i,a in enumerate(x)}; py={a:i for i,a in enumerate(y)}
            lx=dict(zip(x,parse(list(x))))
            inv=[(a,b) for a,b in itertools.combinations(x,2) if (px[a]<px[b])!=(py[a]<py[b])]
            for a,b in inv:
                swaps[tuple(sorted((lx[a],lx[b])))]+=1
            # adjacent transposition?
            if len(inv)==1 and abs(px[inv[0][0]]-px[inv[0][1]])==1: adj+=1
            else: nonadj+=1
    print(f'   anagram groups: {dict(sitespan)}; order-pairs that are a single adjacent transposition {adj}, other {nonadj}')
    print(f'   inverted sign pairs by frame-slot pair: {swaps.most_common()}')
    for n,ex,src in sorted(examples,reverse=True)[:25]: print(f'     n={n} {ex}   [{", ".join(src)}]')
# ============================ (b) order entropy ============================
def H(c):
    n=sum(c.values()); return -sum(k/n*math.log2(k/n) for k in c.values())
def run_b(objs,nperm):
    groups=collections.defaultdict(list)
    for o in objs:
        if len(o['seq'])>=3 and len(set(o['seq']))==len(o['seq']): groups[tuple(sorted(o['seq']))].append(o)
    g=[v for v in groups.values() if len(v)>=2]
    print(f'\n-- (b) order entropy: multisets of >=3 distinct signs on >=2 objects: {len(g)} groups, {sum(len(v) for v in g)} objects')
    def ent_by(keyf,v_list,shuf=None,r=None):
        acc=collections.defaultdict(lambda:[0.0,0,0])
        for v in v_list:
            key=keyf(v)
            seqs=[tuple(shuf(o['seq'],o['lab'],r)) if shuf else tuple(o['seq']) for o in v]
            c=collections.Counter(seqs); acc[key][0]+=H(c); acc[key][1]+=len(set(seqs))-1; acc[key][2]+=len(v)-1
        return acc
    for name,keyf in (('object type',lambda v:collections.Counter(o['ot'] for o in v).most_common(1)[0][0]),
                      ('site',lambda v:collections.Counter(o['site'] for o in v).most_common(1)[0][0]),
                      ('all',lambda v:'all')):
        obs=ent_by(keyf,g); r=random.Random(3)
        nulls=collections.defaultdict(list); nullw=collections.defaultdict(list)
        for _ in range(nperm//5):
            na=ent_by(keyf,g,shuffle_all,r)
            for k,x in na.items(): nulls[k].append(x[1]/max(1,x[2]))
            nw=ent_by(keyf,g,shuffle_within_slot,r)
            for k,x in nw.items(): nullw[k].append(x[1]/max(1,x[2]))
        print(f'   split by {name}:')
        ng=collections.Counter(keyf(v) for v in g)
        for k,(h,kd,nd) in sorted(obs.items(),key=lambda kv:-kv[1][2]):
            if nd<5: continue
            d=kd/nd; na=q(nulls[k],.5); nw=q(nullw[k],.5)
            print(f'     {k:14s} groups {ng[k]:4d} extra-objects {nd:4d}: distinct-order excess (k-1)/(n-1) = {d:.3f}; whole-shuffle null {na:.3f} [{q(nulls[k],.05):.3f},{q(nulls[k],.95):.3f}]; within-slot null {nw:.3f}; order retained vs free {1-d/max(na,1e-9):.2f}; summed H {h:.2f} bits')
    # pots vs seals test: permute type labels within site
    def diversity(v_list,types):
        acc=collections.defaultdict(lambda:[0,0])
        for v in v_list:
            t=collections.Counter(types[id(o)] for o in v).most_common(1)[0][0]
            seqs=[tuple(o['seq']) for o in v]; acc[t][0]+=len(set(seqs))-1; acc[t][1]+=len(v)-1
        return {k:a/max(1,b) for k,(a,b) in acc.items()}
    types={id(o):o['ot'] for o in objs}; obs=diversity(g,types)
    allobjs=[o for v in g for o in v]
    bysite=collections.defaultdict(list)
    for o in allobjs: bysite[o['site']].append(o)
    r=random.Random(5); nd=[]
    for _ in range(nperm):
        t2={}
        for s,lst in bysite.items():
            labs=[o['ot'] for o in lst]; r.shuffle(labs)
            for o,l in zip(lst,labs): t2[id(o)]=l
        dv=diversity(g,t2); nd.append(dv.get('pot',0)-dv.get('seal',0))
    dobs=obs.get('pot',0)-obs.get('seal',0)
    print(f'   pots vs seals: distinct-order excess pot {obs.get("pot",0):.3f} seal {obs.get("seal",0):.3f} tablet {obs.get("tablet",0):.3f}; pot-seal diff {dobs:.3f}, type-label permutation within site null median {q(nd,.5):.3f} [{q(nd,.05):.3f},{q(nd,.95):.3f}], P_hi {pval(dobs,nd,"hi"):.4f}')
# ============================ (c) pair order census ============================
def pair_counts(texts):
    """texts: list of sign lists. For each unordered pair in texts where neither sign repeats: counts of A-before-B."""
    cnt=collections.Counter()
    for s in texts:
        c=collections.Counter(s)
        u=[a for a in s if c[a]==1]
        for i in range(len(u)):
            for j in range(i+1,len(u)):
                a,b=u[i],u[j]
                if a<b: cnt[(a,b,0)]+=1
                else: cnt[(b,a,1)]+=1
    out={}
    for (a,b,d),n in cnt.items():
        x=out.setdefault((a,b),[0,0]); x[d]+=n
    return out
def binom_one_sided(k,n):  # P(X<=k) for X~Bin(n,.5), k = minority count
    return sum(comb(n,i) for i in range(k+1))/2**n
def classify(pc,nmin=5,alpha=0.05):
    fixed=[];free=[];amb=[]
    for (a,b),(nab,nba) in pc.items():
        n=nab+nba
        if n<nmin: continue
        mn=min(nab,nba); p=binom_one_sided(mn,n)
        if p<=alpha: fixed.append((a,b,nab,nba,p))
        elif mn/n>=0.3 and n>=8: free.append((a,b,nab,nba,p))
        else: amb.append((a,b,nab,nba,p))
    return fixed,free,amb
def bh(fixed,allpairs_n,alpha=0.05):
    """Benjamini-Hochberg over all tested pairs (p for untested-as-fixed treated as 1)."""
    ps=sorted(f[4] for f in fixed); m=allpairs_n; k=0
    for i,p in enumerate(ps,1):
        if p<=alpha*i/m: k=i
    thr=ps[k-1] if k else -1
    return [f for f in fixed if f[4]<=thr],thr
def distinct_texts(objs):
    seen=set(); T=[]
    for o in objs:
        key=(o['site'],tuple(o['seq']))
        if key in seen: continue
        seen.add(key); T.append(o)
    return T
def run_c(objs,nperm):
    T=distinct_texts(objs); texts=[o['seq'] for o in T]; labs=[o['lab'] for o in T]
    pc=pair_counts(texts); tested={k:v for k,v in pc.items() if sum(v)>=5}
    fixed,free,amb=classify(pc); fixedbh,thr=bh(fixed,len(tested))
    print(f'\n-- (c) pair order census: distinct (site,text) {len(T)}; unordered sign pairs co-occurring in >=5 texts {len(tested)}; fixed-order (one-sided binomial p<=.05) {len(fixed)}, after BH(0.05) {len(fixedbh)}; free (minority >=30%, n>=8) {len(free)}; ambiguous {len(amb)}')
    # null: how many pairs pass the cut under shuffles
    for name,fn in (('whole-text shuffle',shuffle_all),('within-slot shuffle',shuffle_within_slot)):
        r=random.Random(11); nf=[];nfb=[];nsame=[]
        for _ in range(max(50,nperm//10)):
            pcn=pair_counts([fn(s,l,r) for s,l in zip(texts,labs)])
            f,fr,am=classify(pcn); fb,_=bh(f,sum(1 for v in pcn.values() if sum(v)>=5)); nf.append(len(f)); nfb.append(len(fb))
            nsame.append(sum(1 for a,b,x,y,p in fb if SLOTRANK[SLOT[a]]==SLOTRANK[SLOT[b]]))
        print(f'   null {name:20s}: fixed pairs median {q(nf,.5)} [max {max(nf)}]; after BH median {q(nfb,.5)} [max {max(nfb)}]; same-frame-slot fixed pairs after BH median {q(nsame,.5)} [max {max(nsame)}]')
    # partial order from BH-fixed pairs
    edges=set()
    for a,b,nab,nba,p in fixedbh:
        edges.add((a,b) if nab>nba else (b,a))
    nodes=set(x for e in edges for x in e)
    succ=collections.defaultdict(set)
    for a,b in edges: succ[a].add(b)
    # cycles? longest path (levels) via DFS with memo, detect cycles
    sys.setrecursionlimit(10000)
    color={}; cyc=[0]
    def dfs(u):
        color[u]=1
        for v in succ[u]:
            if color.get(v)==1: cyc[0]+=1
            elif v not in color: dfs(v)
        color[u]=2
    for u in list(nodes):
        if u not in color: dfs(u)
    # longest path ignoring cycle edges (approximate: Kahn levels)
    indeg=collections.Counter()
    for a,b in edges: indeg[b]+=1
    level={}; frontier=[u for u in nodes if indeg[u]==0]; cur=0; remaining=set(nodes)
    while frontier:
        nxt=[]
        for u in frontier:
            level[u]=cur; remaining.discard(u)
            for v in succ[u]:
                indeg[v]-=1
                if indeg[v]==0: nxt.append(v)
        frontier=nxt; cur+=1
    print(f'   partial order: {len(nodes)} signs, {len(edges)} fixed edges, back-edges (cycles) {cyc[0]}, signs left in cycles {len(remaining)}, Kahn levels (longest chain) {cur}')
    # frame explanation
    expl=0;same=[];contra=[]
    for a,b in edges:
        ra,rb=SLOTRANK[SLOT[a]],SLOTRANK[SLOT[b]]
        if ra<rb: expl+=1
        elif ra==rb: same.append((a,b))
        else: contra.append((a,b))
    print(f'   fixed edges explained by frame slot order (slot(A)<slot(B)) {expl}; both in the same frame slot {len(same)}; against the frame {len(contra)}')
    pcd=dict(pc)
    def show(lst,title):
        print(f'   {title} ({len(lst)}):')
        rows=[]
        for a,b in lst:
            x=pcd[(min(a,b),max(a,b))]; nab=x[0] if a<b else x[1]; nba=sum(x)-nab
            rows.append((nab+nba,f'{M(a)} [{SLOT[a]}] -> {M(b)} [{SLOT[b]}] {nab}:{nba}'))
        for n,s in sorted(rows,reverse=True)[:40]: print('     ',s)
    show(same,'same-slot fixed pairs (ordering rules the frame does not give)')
    show(contra,'fixed pairs AGAINST the frame slot order')
    # level membership by frame slot
    lv=collections.defaultdict(collections.Counter)
    for u,l in level.items(): lv[l][SLOT[u]]+=1
    print('   Kahn level -> frame slots of its signs:',{l:dict(c) for l,c in sorted(lv.items())})
    print('   free pairs (both orders >=30%, n>=8), top 40 by n:')
    for a,b,nab,nba,p in sorted(free,key=lambda f:-(f[2]+f[3]))[:40]:
        print(f'      {M(a)} [{SLOT[a]}] <-> {M(b)} [{SLOT[b]}] {nab}:{nba}')
    # share of order information: fraction of co-occurring (text,pair) instances covered by fixed pairs
    tot=sum(sum(v) for v in pc.values()); fx=sum(f[2]+f[3] for f in fixedbh); fr=sum(f[2]+f[3] for f in free)
    print(f'   pair instances: total {tot}; in BH-fixed pairs {fx} ({fx/tot:.3f}); in free pairs {fr} ({fr/tot:.3f}); in pairs seen <5 times {sum(sum(v) for v in pc.values() if sum(v)<5)/tot:.3f}')
    return fixedbh,free,pc
def heldout_c(objs,nperm):
    big=[o for o in objs if o['big']]; small=[o for o in objs if not o['big']]
    Tb=distinct_texts(big); Ts=distinct_texts(small)
    pcb=pair_counts([o['seq'] for o in Tb]); f,_,_=classify(pcb); fb,_=bh(f,sum(1 for v in pcb.values() if sum(v)>=5))
    direc={(a,b):(0 if nab>nba else 1) for a,b,nab,nba,p in fb}
    pcs=pair_counts([o['seq'] for o in Ts])
    agree=dis=0; sameslot_agree=sameslot_dis=0
    for (a,b),d in direc.items():
        if (a,b) in pcs:
            x=pcs[(a,b)]; agree+=x[d]; dis+=x[1-d]
            if SLOTRANK[SLOT[a]]==SLOTRANK[SLOT[b]]: sameslot_agree+=x[d]; sameslot_dis+=x[1-d]
    r=random.Random(13); na=[]; ns=[]
    for _ in range(max(100,nperm//5)):
        pcn=pair_counts([shuffle_within_slot(o['seq'],o['lab'],r) for o in Ts]); ag=0; tt=0; sa=0; st=0
        for (a,b),d in direc.items():
            if (a,b) in pcn:
                x=pcn[(a,b)]; ag+=x[d]; tt+=sum(x)
                if SLOTRANK[SLOT[a]]==SLOTRANK[SLOT[b]]: sa+=x[d]; st+=sum(x)
        na.append(ag/max(1,tt)); ns.append(sa/max(1,st))
    print(f'\n-- (c) held-out replication: {len(fb)} BH-fixed pairs from Mohenjo-daro+Harappa ({len(Tb)} texts) tested on other sites ({len(Ts)} texts): direction agrees {agree}, disagrees {dis} ({agree/max(1,agree+dis):.3f}); within-slot-shuffle null agreement median {q(na,.5):.3f} [95% {q(na,.95):.3f}]')
    print(f'   same-frame-slot pairs only: agree {sameslot_agree}, disagree {sameslot_dis} ({sameslot_agree/max(1,sameslot_agree+sameslot_dis):.3f}); null median {q(ns,.5):.3f} [95% {q(ns,.95):.3f}]')
# ============================ (d) co-occurrence of free pairs ============================
def run_d(objs,fixed,free,nperm):
    T=distinct_texts(objs); texts=[o['seq'] for o in T]
    strata=collections.defaultdict(list)
    for i,o in enumerate(T): strata[(o['site'],o['ot'])].append(i)
    def cooc(tx,pairs):
        c=collections.Counter()
        for s in tx:
            st=set(s)
            for a,b in pairs:
                if a in st and b in st: c[(a,b)]+=1
        return c
    fp=[(a,b) for a,b,*_ in fixed]; frp=[(a,b) for a,b,*_ in free]
    allp=fp+frp
    obs=cooc(texts,allp)
    r=random.Random(17); nul=collections.defaultdict(list)
    for _ in range(nperm//5):
        tx=[None]*len(texts)
        for k,idx in strata.items():
            pool=[a for i in idx for a in texts[i]]; r.shuffle(pool); p=0
            for i in idx: L=len(texts[i]); tx[i]=pool[p:p+L]; p+=L
        cn=cooc(tx,allp)
        for pr in allp: nul[pr].append(cn[pr])
    def summarize(pairs,title):
        oe=[];above=below=0
        for pr in pairs:
            e=sum(nul[pr])/len(nul[pr]); oe.append((obs[pr]+0.5)/(e+0.5))
            if obs[pr]>q(nul[pr],.975): above+=1
            if obs[pr]<q(nul[pr],.025): below+=1
        oe.sort(); med=oe[len(oe)//2] if oe else float('nan')
        print(f'   {title}: {len(pairs)} pairs; median O/E co-occurrence {med:.2f}; pairs above null 97.5% {above}, below 2.5% {below} (expected ~{0.025*len(pairs):.1f} each); mean log2 O/E {sum(math.log2(x) for x in oe)/max(1,len(oe)):.2f}')
        return oe
    print(f'\n-- (d) co-occurrence vs a sign permutation null stratified by site x object type ({nperm//5} reps), distinct texts {len(T)}')
    oef=summarize(fp,'fixed-order pairs'); oefr=summarize(frp,'free-order pairs')
    # adjacency of free pairs: are they adjacent when co-occurring?
    def adjshare(pairs):
        adj=tot=0
        for s in texts:
            pos={a:i for i,a in enumerate(s)}
            for a,b in pairs:
                if a in pos and b in pos: tot+=1; adj+= abs(pos[a]-pos[b])==1
        return adj/max(1,tot),tot
    print(f'   adjacency when co-occurring: fixed pairs {adjshare(fp)[0]:.2f} (n {adjshare(fp)[1]}), free pairs {adjshare(frp)[0]:.2f} (n {adjshare(frp)[1]})')
    # free pairs: coordinate members (same slot, both-present enriched) vs alternatives
    sameslot=[p for p in frp if SLOTRANK[SLOT[p[0]]]==SLOTRANK[SLOT[p[1]]]]
    print(f'   free pairs in the same frame slot {len(sameslot)} of {len(frp)}; slot pairs: {collections.Counter(tuple(sorted((SLOT[a],SLOT[b]))) for a,b in frp).most_common()}')
    rows=[]
    for pr in frp:
        e=sum(nul[pr])/len(nul[pr]); rows.append((obs[pr],e,pr))
    print('   free pairs with O/E (top 25 by observed):')
    for o_,e,(a,b) in sorted(rows,reverse=True)[:25]: print(f'      {M(a)} [{SLOT[a]}] ~ {M(b)} [{SLOT[b]}]: obs {o_} exp {e:.1f} O/E {(o_+.5)/(e+.5):.2f}')
# ============================ (e) controls ============================
def generic_census(texts,label,nperm):
    """anagram census + pair census on a plain list of token sequences (no frame)."""
    objs=[dict(seq=list(s),lab=['X']*len(s),site='x',ot='x') for s in texts]
    seqs=[o['seq'] for o in objs]; labs=[o['lab'] for o in objs]
    obs=anagram_stats(objs,seqs); r=random.Random(23); npr=[]
    for _ in range(nperm//5):
        npr.append(anagram_stats(objs,[shuffle_all(s,l,r) for s,l in zip(seqs,labs)])['pair_rate'])
    print(f'\n-- (e) control [{label}] texts {len(texts)}: multisets on >=2 texts {obs["groups"]}, in >1 order {obs["anagram_groups"]} ({obs["rate"]:.3f}); pair-diff share {obs["pair_rate"]:.3f} vs whole-shuffle median {q(npr,.5):.3f}; order retained {1-obs["pair_rate"]/max(1e-9,q(npr,.5)):.2f}')
    pc=pair_counts(texts); tested=sum(1 for v in pc.values() if sum(v)>=5)
    fixed,free,amb=classify(pc); fb,_=bh(fixed,max(1,tested))
    tot=sum(sum(v) for v in pc.values()); fx=sum(f[2]+f[3] for f in fb); fr=sum(f[2]+f[3] for f in free)
    r=random.Random(29); nfb=[]
    for _ in range(max(20,nperm//25)):
        pcn=pair_counts([shuffle_all(s,None,r) for s in texts]); f,_,_=classify(pcn); b,_=bh(f,max(1,sum(1 for v in pcn.values() if sum(v)>=5))); nfb.append(len(b))
    print(f'   pairs tested (n>=5) {tested}: BH-fixed {len(fb)} (shuffle null median {q(nfb,.5)}, max {max(nfb)}), free {len(free)}, ambiguous {len(amb)}; pair-instance share fixed {fx/max(1,tot):.3f}, free {fr/max(1,tot):.3f}')
    for a,b,nab,nba,p in sorted(fb,key=lambda f:-(f[2]+f[3]))[:12]: print(f'      fixed {a if nab>nba else b} -> {b if nab>nba else a} {max(nab,nba)}:{min(nab,nba)}')
    for a,b,nab,nba,p in sorted(free,key=lambda f:-(f[2]+f[3]))[:8]: print(f'      free  {a} <-> {b} {nab}:{nba}')
    return obs,fb,free
def ur3():
    D=json.load(open('/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/ur3_legends.json'))
    full=[];names=[]
    for d in D:
        toks=[t for l in d['lines'] for t in l if t and 'x' not in t.lower().split('(')[0:1]]
        if 2<=len(toks)<=12: full.append(toks)
        n=d['lines'][0]
        if 2<=len(n)<=6: names.append(list(n))
    return full,names
# ============================ main ============================
if CY==1:
    run_a(OBJ,'all sites',NP)
    run_a([o for o in OBJ if len(o['seq'])>=3],'texts of >=3 signs',NP)
    run_a([o for o in OBJ if len(o['seq'])==2],'2-sign texts',NP//2)
    run_a([o for o in OBJ if not (len(o['seq'])==2 and 700 in o['seq'])],'all minus the N-700 voucher tablets',NP)
    # medium of the reversed voucher tablets
    vc=collections.Counter()
    for r in C:
        s=r[LV]
        if len(s)==2 and 700 in s and r['complete']=='Y' and r['dir.'].strip()!='-' and (set(s)-{700})<=set(NUM):
            vc[(r['type'],'700 first' if s[0]==700 else 'N first')]+=1
    print('\n   N-700 voucher tablets by medium and order:',sorted(vc.items()))
    run_a([o for o in OBJ if o['big']],'Mohenjo-daro + Harappa',NP//2)
    run_a([o for o in OBJ if not o['big']],'held-out: other sites',NP//2)
    run_a([o for o in OBJ if o['ot']=='seal'],'seals',NP//2)
    run_a([o for o in OBJ if o['ot']=='tablet'],'tablets',NP//2)
    run_a([o for o in OBJ if o['ot']=='pot'],'pots',NP//2)
elif CY==2:
    run_b(OBJ,NP)
    fixed,free,pc=run_c(OBJ,NP)
    heldout_c(OBJ,NP)
elif CY==3:
    fixed,free,pc=run_c(OBJ,NP//5)
    run_d(OBJ,fixed,free,NP)
    full,names=ur3()
    generic_census(full,'Ur III legends, all lines concatenated',NP)
    generic_census(names,'Ur III legends, line 1 (owner names, syllables)',NP)
    T=distinct_texts(OBJ); r=random.Random(31)
    generic_census([shuffle_all(o['seq'],None,r) for o in T],'planted FREE order: Indus texts shuffled within text',NP)
    order={a:r.random() for o in T for a in o['seq']}
    generic_census([sorted(o['seq'],key=lambda a:order[a]) for o in T],'planted STRICT order: Indus multisets sorted by one global sign order',NP)
    generic_census([o['seq'] for o in T],'Indus distinct texts, same generic census (no frame)',NP)
