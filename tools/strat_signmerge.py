"""S268: merge hand-drawn variants across the whole Wells sign list (user's point: one script,
many hands, a big region, a long period).
Rules: (1) numeral/stroke signs W1-W60 never merged by shape (value matters);
(2) candidate pair = glyph similarity >= G (dilated overlap on normalised renders);
(3) evidence from use: if both signs have >=5 tokens, context cosine (prev/next incl. text
boundaries) >= C; if either is rarer, glyph similarity must be >= G_RARE;
(4) veto: the two signs stand next to each other (a,b or b,a) in >=2 texts (distinct signs);
(5) complete linkage: a sign joins a class only if it passes with every member.
Plus bridge-confirmed M-equivalences (same Mahadevan number, non-numeral).
Validation: gain in (i) distinct texts shared by a seal and a tablet, (ii) distinct texts
shared across >=2 sites; control = the same class structure filled with frequency-matched
random signs, 200x."""
import sys,json,collections,random,math,numpy as np
sys.path.insert(0,'tools')
from sign_similarity import load
m=load()
T=[(str(r['type']),r['site'],tuple(r['seq'])) for r in m if r.get('seq')]
freq=collections.Counter(x for _,_,s in T for x in s if x!=999)
signs=json.load(open('data/derived/glyph_sim_signs.json')); S=np.load('data/derived/glyph_sim.npy'); idx={w:i for i,w in enumerate(signs)}
ctx=collections.defaultdict(collections.Counter); adj=collections.Counter()
for _,_,s in T:
    for i,x in enumerate(s):
        ctx[x]['p'+str(s[i-1] if i else '^')]+=1; ctx[x]['n'+str(s[i+1] if i+1<len(s) else '$')]+=1
    for a,b in zip(s,s[1:]):
        if a!=b: adj[frozenset((a,b))]+=1
def cos(a,b):
    A=ctx[a];B=ctx[b]; num=sum(A[k]*B[k] for k in A if k in B)
    return num/math.sqrt(sum(v*v for v in A.values())*sum(v*v for v in B.values()))
def ok(a,b,G,C,GR):
    if a<=60 or b<=60: return False
    if adj[frozenset((a,b))]>=2: return False
    g=S[idx[a],idx[b]]
    if g<G: return False
    if min(freq[a],freq[b])>=5: return cos(a,b)>=C
    return g>=GR
b=json.load(open('data/derived/bridge_extended.json'))
def bridge_pairs():
    M2W=collections.defaultdict(set)
    for w in freq:
        for mm in (b.get(str(w)) or []): M2W[mm].add(w)
    P=set()
    for mm,ws in M2W.items():
        ws=[w for w in ws if w>60]
        if 1<len(ws)<=3:
            for i in range(len(ws)):
                for j in range(i+1,len(ws)): P.add((ws[i],ws[j]))
    return P
def classes(G,C,GR,use_bridge=True):
    cand=[]
    for i,a in enumerate(signs):
        for bb in signs[i+1:]:
            if ok(a,bb,G,C,GR): cand.append((S[idx[a],idx[bb]],a,bb))
    if use_bridge:
        for a,bb in bridge_pairs():
            if adj[frozenset((a,bb))]<2: cand.append((1.01,a,bb))
    cand.sort(reverse=True)
    cls={w:{w} for w in signs}
    for g,a,bb in cand:
        A=cls[a];B=cls[bb]
        if A is B: continue
        # complete linkage (bridge pairs exempt from glyph test but not from the veto)
        if all(adj[frozenset((x,y))]<2 for x in A for y in B) and (g>1 or all(S[idx[x],idx[y]]>=G-0.03 for x in A for y in B)):
            U=A|B
            for x in U: cls[x]=U
    rep={}
    for w,c in cls.items(): rep[w]=max(c,key=lambda x:freq[x])
    return rep
def shared(rep):
    se=set();tb=set();site=collections.defaultdict(set)
    for t,st,s in T:
        s2=tuple(rep.get(x,x) for x in s)
        if t.startswith('SEAL'): se.add(s2)
        elif t.startswith('TAB'): tb.add(s2)
        site[s2].add(st)
    return len(se&tb), sum(1 for v in site.values() if len(v)>=2)
def random_like(rep):
    groups=collections.defaultdict(list)
    for w,r in rep.items():
        if w!=r: groups[r].append(w)
    pool=[w for w in signs if w>60]; used=set(); new={}
    def pick(f):
        c=[y for y in pool if 0.5*f<=freq[y]<=2*f and y not in used] or [y for y in pool if y not in used]
        y=random.choice(c); used.add(y); return y
    for r,ms in groups.items():
        rr=pick(freq[r])
        for x in ms: new[pick(freq[x])]=rr
    return new
if __name__=='__main__' and len(sys.argv)==1:
    base=shared({})
    print('base shared (seal&tablet, cross-site):',base)
    random.seed(7)
    for G,C,GR in [(0.97,0.6,0.99),(0.95,0.5,0.98),(0.93,0.5,0.97),(0.90,0.4,0.96)]:
        rep=classes(G,C,GR); merged=sum(1 for w,r in rep.items() if w!=r)
        ob=shared(rep); gain=(ob[0]-base[0],ob[1]-base[1])
        null=[shared(random_like(rep)) for _ in range(200)]
        n0=[x[0]-base[0] for x in null]; n1=[x[1]-base[1] for x in null]
        p0=sum(v>=gain[0] for v in n0)/len(n0); p1=sum(v>=gain[1] for v in n1)/len(n1)
        print(f'G={G} C={C} GR={GR}: signs merged away {merged}, classes<-{len(set(rep.values()))} | gain seal&tab {gain[0]} (null mean {np.mean(n0):.1f}, P={p0:.3f}) | gain cross-site {gain[1]} (null {np.mean(n1):.1f}, P={p1:.3f})')

def tiers(G=0.90,C=0.5,GR=0.98):
    """Tier A: usage-backed (both >=5 tokens, glyph>=G, context>=C) + bridge; Tier B: rare, shape-only (glyph>=GR)."""
    A=[];B=[]
    for i,a in enumerate(signs):
        for bb in signs[i+1:]:
            if a<=60 or bb<=60 or adj[frozenset((a,bb))]>=2: continue
            g=S[idx[a],idx[bb]]
            if min(freq[a],freq[bb])>=5:
                if g>=G and cos(a,bb)>=C: A.append((a,bb))
            elif g>=GR: B.append((a,bb))
    for a,bb in bridge_pairs():
        if adj[frozenset((a,bb))]<2 and S[idx[a],idx[bb]]>=0.6: A.append((a,bb))   # drop bridge pairs with unlike shapes (e.g. W790 oval / W625 square)
    return A,B
def union(pairs):
    par={}
    def f(x):
        par.setdefault(x,x)
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for a,b in pairs: par[f(a)]=f(b)
    grp=collections.defaultdict(set)
    for x in list(par): grp[f(x)].add(x)
    rep={}
    for g in grp.values():
        r=max(g,key=lambda x:freq[x])
        for x in g: rep[x]=r
    return rep
def test(rep,label,n=300):
    base=shared({}); ob=shared(rep); gain=(ob[0]-base[0],ob[1]-base[1])
    null=[shared(random_like(rep)) for _ in range(n)]
    n0=[x[0]-base[0] for x in null]; n1=[x[1]-base[1] for x in null]
    print(f'{label}: merged away {sum(1 for w,r in rep.items() if w!=r)} | seal&tablet gain {gain[0]} (null {np.mean(n0):.1f}, P={sum(v>=gain[0] for v in n0)/n:.3f}) | cross-site gain {gain[1]} (null {np.mean(n1):.1f}, P={sum(v>=gain[1] for v in n1)/n:.3f})')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='tiers':
    random.seed(11)
    A,B=tiers(); ra=union(A); rb=union(B); rab=union(A+B)
    test(ra,'Tier A (usage/bridge)'); test(rb,'Tier B (rare, shape only)'); test(rab,'A+B')
    g=collections.defaultdict(list)
    for w,r in ra.items(): g[r].append(w)
    print('Tier A classes:'); [print('  ',' '.join(f'W{x}({freq[x]})' for x in sorted(v,key=lambda x:-freq[x]))) for r,v in sorted(g.items(),key=lambda t:-sum(freq[x] for x in t[1])) if len(v)>1]
    json.dump({'tierA':{str(k):v for k,v in ra.items() if k!=v},'tierB':{str(k):v for k,v in rb.items() if k!=v}},open('data/derived/sign_variant_classes.json','w'),indent=1)

from scipy.stats import chi2_contingency
def medium(t): return 'seal' if t.startswith('SEAL') else 'tab' if t.startswith('TAB') else 'other'
cell=collections.defaultdict(collections.Counter)   # sign -> Counter over (site,medium)
for r in m:
    for x in r.get('seq') or []: cell[x][(r['site'],medium(str(r['type'])))]+=1
def complementary(a,b,alpha=0.01):
    keys=sorted(set(cell[a])|set(cell[b]))
    tab=np.array([[cell[a][k] for k in keys],[cell[b][k] for k in keys]])
    tab=tab[:,tab.sum(0)>0]
    if tab.shape[1]<2: return False, 1.0
    chi,p,_,_=chi2_contingency(tab)
    return p<alpha, p
def allographs(G=0.90,C=0.5,alpha=0.01):
    cand=[]
    for i,a in enumerate(signs):
        for bb in signs[i+1:]:
            if a<=60 or bb<=60 or min(freq[a],freq[bb])<5 or adj[frozenset((a,bb))]>=2: continue
            g=S[idx[a],idx[bb]]
            if g<G: continue
            c=cos(a,bb)
            if c<C: continue
            comp,p=complementary(a,bb,alpha)
            if comp: cand.append((g*c,a,bb,round(g,2),round(c,2),p))
    # medium variants validated one by one in S262-S266 (looser shape match, same slot, split seal/tablet)
    for a,bb in ((390,405),(154,158),(320,318),(527,525),(527,526)):
        cand.append((3,a,bb,round(float(S[idx[a],idx[bb]]),2),round(cos(a,bb),2),'S262-S266'))
    for a,bb in bridge_pairs():
        if adj[frozenset((a,bb))]<2 and S[idx[a],idx[bb]]>=0.6: cand.append((2,a,bb,round(float(S[idx[a],idx[bb]]),2),round(cos(a,bb),2),'bridge'))
    cand.sort(reverse=True)
    ok_pair={frozenset((a,bb)) for _,a,bb,*_ in cand}
    cls={w:{w} for w in signs}
    for _,a,bb,*_ in cand:
        A=cls[a];B=cls[bb]
        if A is B: continue
        if all(frozenset((x,y)) in ok_pair for x in A for y in B):   # complete linkage
            U=A|B
            for x in U: cls[x]=U
    rep={}
    for w,c in cls.items(): rep[w]=max(c,key=lambda x:freq[x])
    return rep,cand
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='allo':
    random.seed(13)
    rep,cand=allographs()
    for c in cand: print('  pair',c[1:])
    test(rep,'Allograph classes (complementary distribution + shape + slot, or bridge)')
    g=collections.defaultdict(list)
    for w,r in rep.items(): g[r].append(w)
    cl=[sorted(v,key=lambda x:-freq[x]) for v in g.values() if len(v)>1]
    for v in sorted(cl,key=lambda v:-sum(freq[x] for x in v)): print('  class',' '.join(f'W{x}({freq[x]})' for x in v))
    json.dump({str(k):v for k,v in rep.items() if k!=v},open('data/derived/sign_allographs.json','w'),indent=1)
