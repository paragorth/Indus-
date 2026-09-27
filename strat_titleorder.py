"""S213: Order of counted title (numeral + counted sign, S198) relative to the rest of the text core on seals: before or after?
Does it differ between Mohenjo-daro and Harappa (a regional convention)? Frame signs identified via M-codes (CLAUDE.md rule).
Control: permute site labels among qualifying seals 5000x."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
b=json.load(open('data/derived/bridge_extended.json'))
FR_M={267,391,99,123,343,293,65,86,342,211,15,12,254,162,169,176,1}
FR={int(w) for w,ms in b.items() if any(m in FR_M for m in ms)}
NUM=(set(range(1,8))|set(range(12,21))|set(range(25,30))|set(range(32,38))|{39,55,56})-{2}
CT={220,700,900,405,390}
rows=[]; seen=set()
for x in d:
    if not x['type'].startswith('SEAL') or x['site'] not in ('Mohenjo-daro','Harappa'): continue
    s=tuple(x['seq'])
    if (s,x['site']) in seen: continue
    seen.add((s,x['site']))
    pos=[i for i in range(1,len(s)) if s[i] in CT and s[i-1] in NUM]
    if len(pos)!=1: continue
    i=pos[0]
    core=[j for j,w in enumerate(s) if j not in (i-1,i) and w not in FR and w not in NUM and w!=740]
    if len(core)<2: continue
    before=sum(j<i-1 for j in core); after=sum(j>i for j in core)
    if before and after: o='mid'
    elif after: o='title-first'
    else: o='title-last'
    rows.append((x['site'],o))
def tvd(rows):
    a=C.Counter(o for s,o in rows if s=='Mohenjo-daro'); h=C.Counter(o for s,o in rows if s=='Harappa')
    return 0.5*sum(abs(a[k]/sum(a.values())-h[k]/sum(h.values())) for k in set(a)|set(h)),a,h
r,a,h=tvd(rows); labs=[s for s,o in rows]; random.seed(37); nl=[]
for _ in range(5000):
    random.shuffle(labs); nl.append(tvd([(l,o) for l,(s,o) in zip(labs,rows)])[0])
print('Mohenjo-daro',dict(a)); print('Harappa',dict(h))
print('TVD %.3f null %.3f P=%.4f'%(r,sum(nl)/5000,sum(v>=r for v in nl)/5000))
