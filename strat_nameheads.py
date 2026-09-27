"""S182: Are Indus 'names' (seal middles) built with fixed final elements (vs Sumerian names with fixed initial elements)?
Control: shuffle sign order within each middle (keeps signs and lengths), 2000 draws: expected top-5 share at first/last."""
import json,random,collections as C,math,re
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
OP={817,861,820}
def mid(s):
    s=list(s)
    if len(s)>1 and s[0] in OP and s[1]==2: s=s[2:]
    if s and s[-1]==740: s=s[:-2] if len(s)>=2 else []
    return s
M=[m for m in (mid(x['seq']) for x in d if x['type']=='SEAL:S') if len(m)>=2]
def top5(c,N): return sum(v for _,v in c.most_common(5))/N
N=len(M)
of=top5(C.Counter(m[0] for m in M),N); ol=top5(C.Counter(m[-1] for m in M),N)
random.seed(0); sf=[];sl=[]
for _ in range(2000):
    S=[random.sample(m,len(m)) for m in M]
    sf.append(top5(C.Counter(m[0] for m in S),N)); sl.append(top5(C.Counter(m[-1] for m in S),N))
print(f"Indus middles n={N}: first top5 {of:.3f} (shuffled {sum(sf)/2000:.3f}, P_low={sum(x<=of for x in sf)/2000:.4f}); last top5 {ol:.3f} (shuffled {sum(sl)/2000:.3f}, P_high={sum(x>=ol for x in sl)/2000:.4f})")
# same for Ur III names
L=[]
for l in open('/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/seal_line1.txt'):
    l=l.strip()
    if not l or '[' in l or '...' in l or ' x' in l: continue
    w=re.sub(r'\{[^}]*\}','',l.split()[0]); s=[x for x in re.split(r'[-.]',w) if x and not re.search(r'[#?!]',x)]
    if len(s)>=2: L.append(s)
N2=len(L); of2=top5(C.Counter(m[0] for m in L),N2); ol2=top5(C.Counter(m[-1] for m in L),N2)
sf2=[];sl2=[]
for _ in range(300):
    S=[random.sample(m,len(m)) for m in L]
    sf2.append(top5(C.Counter(m[0] for m in S),N2)); sl2.append(top5(C.Counter(m[-1] for m in S),N2))
print(f"Ur III names n={N2}: first top5 {of2:.3f} (shuffled {sum(sf2)/300:.3f}); last top5 {ol2:.3f} (shuffled {sum(sl2)/300:.3f})")
