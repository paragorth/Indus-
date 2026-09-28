"""S262: on script-only seals, do W1/W405/W806 fill the same slots that W2/W390/W741 fill on animal seals?
Slot = (previous sign, next sign) context. Compare the context distributions of each pair."""
import json,collections
m=json.load(open('data/derived/merged-corpus-reading-order.json'))
m=m if isinstance(m,list) else list(m.values())[0]
S=[r for r in m if str(r.get('type','')).startswith('SEAL') and r['symbol']!='-' and r.get('seq')]
def ctx(sign,so):
    P=collections.Counter(); N=collections.Counter(); pos=collections.Counter(); n=0
    for r in S:
        if (r['symbol']=='')!=so: continue
        s=r['seq']
        for i,x in enumerate(s):
            if x==sign:
                n+=1; P[s[i-1] if i else '^']+=1; N[s[i+1] if i+1<len(s) else '$']+=1
                pos['first' if i==0 else 'last' if i==len(s)-1 else 'mid']+=1
    return n,P,N,pos
for a,b in ((2,1),(390,405),(741,806)):
    for sg,so,lab in ((a,False,'animal'),(a,True,'script'),(b,False,'animal'),(b,True,'script')):
        n,P,N,pos=ctx(sg,so)
        print(f'W{sg} on {lab}-seals n={n} pos={dict(pos)} prev={P.most_common(4)} next={N.most_common(4)}')
    print()
