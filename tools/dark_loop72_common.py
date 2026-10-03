"""S-DARK-72 common code: IS THE DESIGNATION A SORTED LIST OR A NAME?
A sorted list writes its elements in one global order (every pair of elements always in the same order, and the pairwise
orders are transitive, so one ranking of the whole inventory reproduces the list). A name list fixes the order of each
name but not across names (pair a,b can be 'a b' in one name and 'b a' in another; no global ranking exists beyond
positional tendencies). Statistics, identical code for every list (one copy per distinct string, >= 2 elements):
  anag        share of strings whose element multiset recurs in another string in a different order (anagram rate)
  consist     share of tested pairs (co-occurring in >= 3 strings) written in one order only
  contra      frequency-weighted contradiction: sum over pairs (>= 2 strings) of the minority-order count / all pair instances
  fas         minimum feedback arc set (heuristic: net-score start + insertion local search), share of ALL pair instances
              that the best single global ranking of the elements must violate; fit = 1 - fas
  fas_cyc     fas minus the pairwise floor (sum of minority counts over all pairs) / all instances: violations forced
              by cycles, i.e. by intransitivity of the majority orders
  str_sorted  share of strings whose element order agrees exactly with the fitted global ranking
  ho_seen     2-fold held-out: ranking fitted on half A; share of pair instances in half B ordered as the ranking says,
              pairs co-observed in A (5 random splits)
  ho_unseen   same, pairs NOT co-observed in A (both elements ranked): transitive prediction, the 'global order' test
  rho_freq    Spearman(rank position in the fitted ranking, log element frequency), elements >= 5 tokens
Direction of a pair inside one string = order of first occurrence (repeated elements counted once).
"""
import json,random,collections,math,re,sys
import numpy as np

def clean(seqs):
    out=[]
    for s in seqs:
        s=tuple(a for a in s if a not in ('',None))
        if len(set(s))>=2: out.append(s)
    return sorted(set(out))

def firstorder(s):
    seen=[];
    for a in s:
        if a not in seen: seen.append(a)
    return seen

def pair_table(strings):
    W=collections.Counter()
    for s in strings:
        f=firstorder(s)
        for i in range(len(f)):
            for j in range(i+1,len(f)): W[(f[i],f[j])]+=1
    return W

def anag(strings):
    g=collections.defaultdict(set)
    for s in strings: g[tuple(sorted(collections.Counter(s).items(),key=lambda x:str(x)))].add(s)
    n=sum(len(v) for v in g.values() if len(v)>=2)
    return n/len(strings)

def pair_stats(W):
    seen=set(); tot2=0; min2=0; nt=0; nc=0; totall=0; minall=0
    for (a,b),v in W.items():
        key=(a,b) if str(a)<str(b) else (b,a)
        if key in seen: continue
        seen.add(key); x=W.get(key,0); y=W.get((key[1],key[0]),0); t=x+y
        totall+=t; minall+=min(x,y)
        if t>=2: tot2+=t; min2+=min(x,y)
        if t>=3:
            nt+=1; nc+=(min(x,y)==0)
    return dict(consist=nc/nt if nt else float('nan'),n_tested=nt,contra=min2/tot2 if tot2 else float('nan'),floor=minall/totall if totall else float('nan'),tot=totall)

def matrix(W,elems=None):
    if elems is None: elems=sorted({a for k in W for a in k},key=str)
    ix={a:i for i,a in enumerate(elems)}; A=np.zeros((len(elems),len(elems)))
    for (a,b),v in W.items():
        if a in ix and b in ix: A[ix[a],ix[b]]+=v
    return elems,A

def fas_rank(A,passes=30,seed=0):
    """Order (list of indices) minimising sum of A[x,y] with x ranked AFTER y (x was written before y)."""
    k=A.shape[0]
    if k==0: return []
    net=(A.sum(1)-A.sum(0))/np.maximum(A.sum(1)+A.sum(0),1)
    order=list(np.argsort(-net,kind='stable'))
    rng=random.Random(seed)
    for p in range(passes):
        improved=0
        idx=list(range(k)); rng.shuffle(idx)
        for x in idx:
            pos=order.index(x); r=order[:pos]+order[pos+1:]
            ra=np.array(r,dtype=int)
            a=A[x,ra]   # x before r_t in data: violated if x placed after r_t
            b=A[ra,x]   # r_t before x in data: violated if x placed before r_t
            ca=np.concatenate([[0],np.cumsum(a)]); cb=np.concatenate([[0],np.cumsum(b)])
            cost=ca+(cb[-1]-cb)       # insert at q: violations = sum_{t<q} a + sum_{t>=q} b
            q=int(np.argmin(cost))
            if cost[q]+1e-9<cost[pos]:
                r.insert(q,x); order=r; improved+=1
        if not improved: break
    return order

def cost_of(A,order):
    k=len(order); pos=np.empty(k,dtype=int); pos[np.array(order)]=np.arange(k)
    later=pos[:,None]>pos[None,:]   # later[x,y]: x ranked after y
    return float((A*later).sum())

def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<5: return float('nan')
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
    return float(np.corrcoef(rx,ry)[0,1])

def fit_rank(strings,seed=0):
    W=pair_table(strings); elems,A=matrix(W); order=fas_rank(A,seed=seed)
    rank={elems[i]:r for r,i in enumerate(order)}
    return W,elems,A,order,rank

def battery(strings,seed=0,nsplit=5,extra=False):
    strings=clean(strings)
    W,elems,A,order,rank=fit_rank(strings,seed)
    ps=pair_stats(W); tot=A.sum()
    fas=cost_of(A,order)/tot if tot else float('nan')
    ss=0
    for s in strings:
        f=firstorder(s); r=[rank[a] for a in f]; ss+=all(r[i]<r[i+1] for i in range(len(r)-1))
    freq=collections.Counter(a for s in strings for a in s)
    el=[a for a in elems if freq[a]>=5]
    rho=spearman([rank[a]/len(rank) for a in el],[math.log(freq[a]) for a in el])
    # mean relative position ranking (alternative, no optimisation)
    hs=[];hu=[]
    for sp in range(nsplit):
        r=random.Random(1000*seed+sp); idx=list(range(len(strings))); r.shuffle(idx)
        h=len(idx)//2; SA=[strings[i] for i in idx[:h]]; SB=[strings[i] for i in idx[h:]]
        WA,eA,AA,oA,rA=fit_rank(SA,seed)
        seenA=set(frozenset(k) for k in WA)
        WB=pair_table(SB); gs=ns=gu=nu=0
        for (a,b),v in WB.items():
            if a not in rA or b not in rA: continue
            ok=rA[a]<rA[b]
            if frozenset((a,b)) in seenA: gs+=v*ok; ns+=v
            else: gu+=v*ok; nu+=v
        hs.append(gs/ns if ns else float('nan')); hu.append(gu/nu if nu else float('nan'))
    out=dict(n=len(strings),k=len(elems),meanL=float(np.mean([len(s) for s in strings])),anag=anag(strings),
             consist=ps['consist'],n_tested=ps['n_tested'],contra=ps['contra'],fas=fas,fit=1-fas,fas_cyc=fas-ps['floor'],
             str_sorted=ss/len(strings),ho_seen=float(np.nanmean(hs)),ho_unseen=float(np.nanmean(hu)),rho_freq=rho)
    if extra: out['_rank']=rank; out['_freq']=freq
    return out

KEYS=['anag','consist','contra','fas','fit','fas_cyc','str_sorted','ho_seen','ho_unseen','rho_freq']

# ---------- nulls and anchors ----------
def shuffle_null(strings,r):
    out=[]
    for s in strings:
        s=list(s); r.shuffle(s); out.append(tuple(s))
    return out

def markov2(strings,r,n=None):
    n=n or len(strings); T=collections.defaultdict(collections.Counter)
    for s in strings:
        x=('^','^')+tuple(s)+('$',)
        for i in range(2,len(x)): T[(x[i-2],x[i-1])][x[i]]+=1
    TT={k:(list(v.keys()),np.cumsum(list(v.values()))) for k,v in T.items()}
    out=set(); tries=0
    while len(out)<n and tries<200*n:
        tries+=1; a,b='^','^'; s=[]
        while True:
            ks,cs=TT[(a,b)]; u=r.random()*cs[-1]; c=ks[int(np.searchsorted(cs,u,side='right'))]
            if c=='$' or len(s)>=40: break
            s.append(c); a,b=b,c
        if len(set(s))>=2: out.add(tuple(s))
    return sorted(out)

def sorted_anchor(strings,r):
    el=sorted({a for s in strings for a in s},key=str); r.shuffle(el); rk={a:i for i,a in enumerate(el)}
    return sorted({tuple(sorted(s,key=lambda a:rk[a])) for s in strings})

def pername_anchor(strings,r):
    g={}
    for s in strings:
        key=tuple(sorted(s,key=str))
        if key not in g:
            t=list(s); r.shuffle(t); g[key]=tuple(t)
    return sorted(set(g.values()))

def q(v,p):
    v=sorted(x for x in v if x==x)
    if not v: return float('nan')
    i=p*(len(v)-1); lo=int(math.floor(i)); hi=min(lo+1,len(v)-1)
    return v[lo]+(v[hi]-v[lo])*(i-lo)

def summarize(runs):
    return {k:(q([x[k] for x in runs],0.5),q([x[k] for x in runs],0.025),q([x[k] for x in runs],0.975)) for k in KEYS}

def boot(strings,nb=20,frac=0.8,seed=0):
    strings=clean(strings); full=battery(strings,seed)
    runs=[battery(random.Random(7000+b).sample(strings,int(frac*len(strings))),seed+b,nsplit=2) for b in range(nb)]
    out={}
    for k in KEYS:
        v=[x[k] for x in runs]; md=q(v,0.5); out[k]=(full[k],full[k]+q(v,0.025)-md,full[k]+q(v,0.975)-md)
    out['_n']=full['n']; out['_k']=full['k']; out['_meanL']=full['meanL']
    return out

def length_match(pool,lens,n,r):
    by=collections.defaultdict(list)
    for s in pool: by[len(s)].append(s)
    want=collections.Counter(r.choice(lens) for _ in range(n)); out=[]
    for L,c in want.items():
        src=by.get(L) or by[min(by,key=lambda x:abs(x-L))]
        out+=r.sample(src,min(c,len(src)))
    return out

def draws(pool,n,nd=10,lens=None,seed=0):
    pool=clean(pool); runs=[]
    for b in range(nd):
        r=random.Random(5000+b+seed)
        sub=length_match(pool,lens,n,r) if lens is not None else (r.sample(pool,n) if len(pool)>n else pool)
        runs.append(battery(sub,b,nsplit=2))
        if len(pool)<=n and lens is None: break
    s=summarize(runs); s['_n']=runs[0]['n']; s['_k']=runs[0]['k']; s['_meanL']=runs[0]['meanL']
    return s

def t3(v): return f'{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]'
