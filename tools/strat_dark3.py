"""S-DARK-3: 'maybe we are reading it wrong'. Apply random TRANSFORMATIONS to the whole corpus, then run four
established tests, and ask whether any transformation makes the corpus behave like a known system.
Tests: (1) name-calibration ratio = uniqueness of texts / uniqueness under a bigram model trained on the same texts
(Ur III owner names 0.62, Indus 0.97; tools/strat_namecalib*.py); (2) within-text repetition vs slot-preserving
shuffle (S88); (3) slot-entropy edge drop (frame profile); (4) arithmetic closure among numeral values in a text.
Arrow = (transformation, test). Each arrow gets >=20 random parameterisations; the arrow statistic is the median over
parameterisations of D = S(T(X)) - S(X). Null: sign-shuffle across texts (lengths kept; for the slot test, shuffle
within text), same transformation, same parameterisations. Correction: Bonferroni over all arrows fired. Replication:
train = Mohenjo-daro + Harappa, held-out = all other sites; seq_raw / seq_strong / seq_all.
Usage: python3 tools/strat_dark3.py screen|confirm|ur3|pipelines [seed]
Outputs go to data/derived/dark/ (loop3_*.txt)."""
import json,random,math,sys,collections,statistics as st,time,os
C=json.load(open('data/derived/merged-corpus-canonical.json'))
OUT='data/derived/dark/'; os.makedirs(OUT,exist_ok=True)
# ---- sign classes (Wells numbers; bridge_extended.json maps to Mahadevan) ----
OPEN={817,861,820,920}; MARK={2,60,741}; MIDINIT={235,31}
CLOSE={740,390,405,406,407,156,527,526,151}; SUFFIX={400,90}
FRAME=OPEN|MARK|MIDINIT|CLOSE|SUFFIX
NUMVAL={1:1,3:3,4:4,5:5,16:6,17:7,18:8,31:1,32:2,33:3,34:4,55:12}
def is_num(s,with2=False): return s<60 and (with2 or s!=2)
# ---- corpus ----
def texts(split,var):
    out=[]
    for r in C:
        s=r.get(var) or []
        if not s: continue
        home=r['site'] in ('Mohenjo-daro','Harappa')
        if split=='train' and not home: continue
        if split=='heldout' and home: continue
        if split=='heldout' and r['site']=='Unknown': continue
        out.append({'seq':list(s),'emblem':(r.get('symbol') or '').split(':')[0] or 'NONE','type':r['type'].split(':')[0]})
    return out
# ---- statistics ----
def uniq(ms):
    c=collections.Counter(ms); return sum(1 for m in ms if c[m]==1)/len(ms)
def bigram_uniq(seqs,rng,G=1):
    big=collections.defaultdict(list)
    for m in seqs:
        p='S'
        for c in m: big[p].append(c); p=c
    L=[len(m) for m in seqs]; vals=[]
    for _ in range(G):
        gen=[]
        for n in L:
            out=[];p='S'
            for _ in range(n):
                src=big.get(p) or big['S']; c=rng.choice(src); out.append(c); p=c
            gen.append(tuple(out))
        vals.append(uniq(gen))
    return st.mean(vals)
def S_namecalib(T,rng):
    seqs=[tuple(t) for t in T if 2<=len(t)<=6]
    if len(seqs)<50: return None
    b=bigram_uniq(seqs,rng)
    return uniq(seqs)/b if b>0 else None
def S_repeat(T,rng,K=1):
    seqs=[t for t in T if len(t)>=3]
    if len(seqs)<50: return None
    obs=sum(1 for t in seqs if len(set(t))<len(t))/len(seqs)
    exp=[]
    for _ in range(K):
        bypos=collections.defaultdict(list)
        for t in seqs:
            for i,s in enumerate(t): bypos[i].append(s)
        for i in bypos: rng.shuffle(bypos[i])
        idx=collections.Counter(); r=0
        for t in seqs:
            u=set()
            for i in range(len(t)): u.add(bypos[i][idx[i]]); idx[i]+=1
            if len(u)<len(t): r+=1
        exp.append(r/len(seqs))
    e=st.mean(exp)
    return obs/e if e>0 else None
def H(cnt):
    n=sum(cnt.values()); return -sum(v/n*math.log2(v/n) for v in cnt.values())
def S_slot(T,rng):
    seqs=[t for t in T if len(t)>=4]
    if len(seqs)<50: return None
    first=collections.Counter(t[0] for t in seqs); last=collections.Counter(t[-1] for t in seqs)
    mid=collections.Counter(s for t in seqs for s in t[1:-1])
    return H(mid)-(H(first)+H(last))/2
def S_arith(T,rng):
    rows=[[NUMVAL[s] for s in t if s in NUMVAL] for t in T]
    rows=[r for r in rows if len(r)>=2]
    if len(rows)<20: return None
    def closed(r):
        r=sorted(r); return sum(r[:-1])==r[-1] or any(r[i]+r[j] in r for i in range(len(r)) for j in range(i+1,len(r)))
    return sum(closed(r) for r in rows)/len(rows)
TESTS={'namecalib':S_namecalib,'repeat':S_repeat,'slot':S_slot,'arith':S_arith}
# ---- transformations: f(T, meta, rng, theta) -> list of sequences ----
def t_identity(T,M,rng,th): return [list(t) for t in T]
def t_reverse(T,M,rng,th): return [t[::-1] for t in T]
def t_interleave(T,M,rng,th):
    out=[]
    for t in T:
        if len(t)<2: out.append(list(t)); continue
        a=t[th['off']::2]; b=t[1-th['off']::2]
        out.append(list(a)); out.append(list(b))
    return out
def t_middleout(T,M,rng,th):
    out=[]
    for t in T:
        n=len(t)
        if n==0: out.append([]); continue
        m=n//2 if th['start']=='floor' else (n-1)//2
        seq=[t[m]]; l=m-1; r=m+1; left=th['left_first']
        while l>=0 or r<n:
            if left:
                if l>=0: seq.append(t[l]); l-=1
                elif r<n: seq.append(t[r]); r+=1
            else:
                if r<n: seq.append(t[r]); r+=1
                elif l>=0: seq.append(t[l]); l-=1
            left=not left
        out.append(seq)
    return out
def t_sort(T,M,rng,th):
    key=th['key']; return [sorted(t,key=lambda s:key.get(s,s)) for t in T]
def t_family(T,M,rng,th):
    fm=th['fam']; return [[fm.get(s//100 if s>=100 else 0, s//100) for s in t] for t in T]
def t_numonly(T,M,rng,th): return [[s for s in t if is_num(s,th['with2'])] for t in T]
def t_nonnum(T,M,rng,th): return [[s for s in t if not is_num(s,th['with2'])] for t in T]
def t_dropframe(T,M,rng,th):
    lead=th['lead']; trail=th['trail']; out=[]
    for t in T:
        t=list(t)
        while t and t[0] in lead: t.pop(0)
        while t and t[-1] in trail: t.pop()
        out.append(t)
    return out
def t_bpe(T,M,rng,th):
    k=th['k']; T=[list(t) for t in T]; nxt=100000
    for _ in range(k):
        c=collections.Counter()
        for t in T:
            for i in range(len(t)-1): c[(t[i],t[i+1])]+=1
        if not c: break
        (a,b),n=c.most_common(1)[0]
        if n<3: break
        new=[]
        for t in T:
            o=[];i=0
            while i<len(t):
                if i<len(t)-1 and t[i]==a and t[i+1]==b: o.append(nxt); i+=2
                else: o.append(t[i]); i+=1
            new.append(o)
        T=new; nxt+=1
    return T
def t_splitnum(T,M,rng,th):
    out=[]
    for t in T:
        cur=[]
        for s in t:
            if is_num(s,th['with2']):
                if th['mode']=='left': cur.append(s); out.append(cur); cur=[]
                elif th['mode']=='right': out.append(cur); cur=[s]
                else: out.append(cur); cur=[]
            else: cur.append(s)
        out.append(cur)
    return [o for o in out if o]
def t_rotate(T,M,rng,th):
    r=th['r']; out=[]
    for t in T:
        if len(t)<2: out.append(list(t)); continue
        k=r if r>0 else rng.randrange(1,len(t)); k%=len(t); out.append(list(t[k:])+list(t[:k]))
    return out
def t_emblem(T,M,rng,th):
    code={}; out=[]
    for t,m in zip(T,M):
        e=m['emblem']; c=code.setdefault(e,200000+len(code))
        out.append([c]+list(t) if th['pos']=='start' else list(t)+[c])
    return out
def t_type(T,M,rng,th):
    code={}; out=[]
    for t,m in zip(T,M):
        c=code.setdefault(m['type'],300000+len(code))
        out.append([c]+list(t) if th['pos']=='start' else list(t)+[c])
    return out
def t_randfamily(T,M,rng,th):
    fm=th['map']; return [[fm.get(s,0) for s in t] for t in T]
def t_relabel(T,M,rng,th):
    p=th['perm']; return [[p.get(s,s) for s in t] for t in T]
def t_dropsign(T,M,rng,th):
    p=th['p']; r=random.Random(th['seed']); return [[s for s in t if r.random()>p] for t in T]
TRANS={'identity':t_identity,'reverse':t_reverse,'interleave':t_interleave,'middleout':t_middleout,'sort':t_sort,
 'family':t_family,'numonly':t_numonly,'nonnum':t_nonnum,'dropframe':t_dropframe,'bpe':t_bpe,'splitnum':t_splitnum,
 'rotate':t_rotate,'emblem':t_emblem,'objtype':t_type,'relabel':t_relabel,'randfamily':t_randfamily,'dropsign':t_dropsign}
def draw_theta(name,rng,signs):
    if name=='interleave': return {'off':rng.randrange(2)}
    if name=='middleout': return {'start':rng.choice(['floor','ceil']),'left_first':rng.random()<0.5}
    if name=='sort':
        k=list(signs); rng.shuffle(k); return {'key':{s:i for i,s in enumerate(k)}}
    if name=='family':
        fams=list(range(0,10)); p=rng.choice([0,0,0.2,0.4]); fm={}
        for f in fams: fm[f]=rng.choice(fams) if rng.random()<p else f
        return {'fam':fm}
    if name in ('numonly','nonnum'): return {'with2':rng.random()<0.5}
    if name=='dropframe':
        lead=set(s for s in OPEN|MARK|MIDINIT if rng.random()<0.8); trail=set(s for s in CLOSE|SUFFIX if rng.random()<0.8)
        return {'lead':lead,'trail':trail}
    if name=='bpe': return {'k':rng.choice([5,10,20,40,80])}
    if name=='splitnum': return {'with2':rng.random()<0.5,'mode':rng.choice(['left','right','drop'])}
    if name=='rotate': return {'r':rng.choice([1,1,2,0])}
    if name in ('emblem','objtype'): return {'pos':rng.choice(['start','end'])}
    if name=='randfamily':
        # control for 'family': random partition of the signs into classes with the SAME sizes as the W-hundreds
        k=list(signs); sizes=collections.Counter((s//100 if s>=100 else 0) for s in k); rng.shuffle(k); mp={}; i=0
        for fam,n in sizes.items():
            for s in k[i:i+n]: mp[s]=fam
            i+=n
        return {'map':mp}
    if name=='relabel':
        k=list(signs); v=list(signs); rng.shuffle(v); return {'perm':dict(zip(k,v))}
    if name=='dropsign': return {'p':rng.choice([0.1,0.2,0.3]),'seed':rng.randrange(10**9)}
    return {}
def shuffle_cross(T,rng):
    pool=[s for t in T for s in t]; rng.shuffle(pool); out=[]; i=0
    for t in T: out.append(pool[i:i+len(t)]); i+=len(t)
    return out
def shuffle_within(T,rng):
    out=[]
    for t in T:
        t=list(t); rng.shuffle(t); out.append(t)
    return out
def arrow_multi(tname,tests,X,M,thetas,P,rng):
    """One transformation, several tests. Returns {test: (median D over thetas, null Ds, S(T(X)) per theta, S(X))}.
    The transformed corpora are computed once per theta and shared by the tests."""
    f=TRANS[tname]; out={}
    TX=[f(X,M,rng,th) for th in thetas]
    base={sn:TESTS[sn](X,rng) for sn in tests}
    vals={sn:[v for v in (TESTS[sn](T,rng) for T in TX) if v is not None] for sn in tests}
    null={sn:[] for sn in tests}
    for _ in range(P):
        for shuf,group in ((shuffle_cross,[sn for sn in tests if sn!='slot']),(shuffle_within,[sn for sn in tests if sn=='slot'])):
            if not group: continue
            Xs=shuf(X,rng); TXs=[f(Xs,M,rng,th) for th in thetas]
            for sn in group:
                b=TESTS[sn](Xs,rng); vs=[v for v in (TESTS[sn](T,rng) for T in TXs) if v is not None]
                if b is None or not vs: continue
                null[sn].append(st.median(v-b for v in vs))
    for sn in tests:
        if base[sn] is None or len(vals[sn])<len(thetas)//2: out[sn]=None; continue
        out[sn]=(st.median(v-base[sn] for v in vals[sn]),null[sn],vals[sn],base[sn])
    return out
def arrow(tname,sname,X,M,thetas,P,rng): return arrow_multi(tname,[sname],X,M,thetas,P,rng)[sname]
def pval(D,null):
    if not null: return 1.0
    return (1+sum(1 for d in null if abs(d)>=abs(D)))/(len(null)+1)
def zval(D,null):
    if len(null)<3: return 0.0
    sd=st.pstdev(null); return (D-st.mean(null))/sd if sd>0 else (0.0 if D==st.mean(null) else math.copysign(99,D-st.mean(null)))
# ---- reference corpora (known systems) ----
def refs():
    out={}
    def load(f): return [r['seq'] for r in map(json.loads,open('data/codelib/'+f+'.jsonl'))]
    STOP={'dub-sar','dumu','arad2','arad','ir3','ir11','dam','szabra','ensi2','sukkal','kiszib3','kiszib','sanga','nu-banda3','gudu4','sipa','nar','ugula','_dub-sar_','_dumu_','_arad_','_arad2_','_dam_'}
    names=[]
    for s in load('ur3_legends'):
        n=[]
        for w in s:
            if w in STOP: break
            n.append(w)
        if n:
            syl=[x for x in n[0].replace('{','-').replace('}','-').split('-') if x]
            if 2<=len(syl)<=6: names.append(syl)
    out['ur3_names_syll']=names
    out['ur3_legends_words']=load('ur3_legends')
    for f in ('linear_b','latin_edh','proto_elamite','khipu','heraldry'): out[f]=load(f)
    return out
