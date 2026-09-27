"""S193: Do pottery graffiti (2+ signs) occur as contiguous strings inside seal middles more than chance, and more at the same site?
Control: graffiti with sign order shuffled (keeps signs & length) - 2000 draws; and site-label permutation."""
import json,random,collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
OP={817,861,820}
def mid(s):
    s=list(s)
    if len(s)>1 and s[0] in OP and s[1]==2: s=s[2:]
    if s and s[-1]==740: s=s[:-2]
    return s
M=[(x['site'],tuple(mid(x['seq']))) for x in d if x['type'].startswith('SEAL') and len(mid(x['seq']))>=2]
G=[(x['site'],tuple(x['seq'])) for x in d if x['type'].upper()=='POT:T:G' and len(x['seq'])>=2 and 0 not in x['seq']]
def contains(m,g):
    L=len(g); return any(m[i:i+L]==g for i in range(len(m)-L+1))
subs=C.defaultdict(set)
for site,m in M:
    for L in (2,3,4):
        for i in range(len(m)-L+1): subs[m[i:i+L]].add(site)
def hits(Gs):
    a=sum(1 for s,g in Gs if g in subs); b=sum(1 for s,g in Gs if g in subs and s in subs[g]); return a,b
obs=hits(G); print('graffiti',len(G),'found in seal middles',obs[0],'same site',obs[1])
random.seed(0); A=[];B=[]
for _ in range(2000):
    Gs=[(s,tuple(random.sample(g,len(g)))) for s,g in G]; a,b=hits(Gs); A.append(a); B.append(b)
print('order-shuffled: mean %.1f (P=%.4f); same-site mean %.1f (P=%.4f)'%(sum(A)/2000,sum(x>=obs[0] for x in A)/2000,sum(B)/2000,sum(x>=obs[1] for x in B)/2000))
sites=[s for s,g in G]; B2=[]
for _ in range(2000):
    random.shuffle(sites); B2.append(sum(1 for s,(_,g) in zip(sites,G) if g in subs and s in subs[g]))
print('site-permuted same-site mean %.1f (P=%.4f)'%(sum(B2)/2000,sum(x>=obs[1] for x in B2)/2000))
print([ (s,g) for s,g in G if g in subs][:12])
