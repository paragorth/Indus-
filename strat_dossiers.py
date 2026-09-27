"""S195: Seal 'names' (middles of 3+ signs, frame removed via M-code closer family) reappearing on non-seal objects.
Same-site vs other-site; control: permute site labels of non-seal objects (2000x)."""
import json,random,collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
b=json.load(open('data/derived/bridge_extended.json'))
FR_M={267,391,99,123,343,293,65,86,342,211,15,12,254,162,169,176,1}
FR={int(w) for w,ms in b.items() if any(m in FR_M for m in ms)}
NUM=set(range(1,8))|set(range(12,21))|set(range(25,30))|set(range(31,38))|{39,42,43,55,56}
def core(s):
    s=list(s)
    while s and (s[0] in FR or s[0] in NUM): s=s[1:]
    while s and (s[-1] in FR or s[-1] in NUM): s=s[:-1]
    return tuple(s)
cores=[(x['site'],core(x['seq'])) for x in d if x['type'].startswith('SEAL')]
cores=[(s,c) for s,c in cores if len(c)>=3]
other=[(x['site'],tuple(x['seq']),x['type']) for x in d if not x['type'].startswith('SEAL') and len(x['seq'])>=3]
def contains(t,c): L=len(c); return any(t[i:i+L]==c for i in range(len(t)-L+1))
hits=[]
for s,c in set(cores):
    for so,t,ty in other:
        if contains(t,c): hits.append((s,so,c,ty))
same=sum(1 for s,so,c,ty in hits if s==so)
print('distinct seal cores',len(set(cores)),'hits',len(hits),'same-site',same)
sites=[so for so,_,_ in other]; random.seed(0); null=[]
idx={}
for s,c in set(cores):
    for j,(so,t,ty) in enumerate(other):
        if contains(t,c): idx.setdefault((s,c),[]).append(j)
for _ in range(2000):
    random.shuffle(sites); null.append(sum(1 for (s,c),js in idx.items() for j in js if sites[j]==s))
print('site-permuted mean %.1f P=%.4f'%(sum(null)/2000,sum(v>=same for v in null)/2000))
print(C.Counter(ty for s,so,c,ty in hits))
for h in hits[:15]: print(h)
