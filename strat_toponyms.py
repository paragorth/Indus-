"""Strategy 11: site-localised signs (candidate place/institution names). Seals only,
deduplicated (site, text); one-sided Fisher per sign x site; BH-FDR 5%."""
import json, math, collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
seen=set(); seals=[]
for x in d:
    if not x['type'].startswith('SEAL'): continue
    q=tuple(s for s in x['seq'] if s)
    if len(q)<2 or (x['site'],q) in seen: continue
    seen.add((x['site'],q)); seals.append((x['site'],set(q)))
N=len(seals); sites=C.Counter(s for s,_ in seals); df=C.Counter(g for _,q in seals for g in q)
def fisher(a,b,c,dd):
    n1,n2,k,T=a+b,c+dd,a+c,a+b+c+dd
    return sum(math.comb(n1,x)*math.comb(n2,k-x) for x in range(a,min(n1,k)+1))/math.comb(T,k)
tests=[]
for site,ns in sites.items():
    if ns<30: continue
    for g,ng in df.items():
        if ng<5: continue
        a=sum(1 for s,q in seals if s==site and g in q)
        if a<3: continue
        tests.append((fisher(a,ns-a,ng-a,N-ns-ng+a),site,g,a,ns,ng))
tests.sort(); m=len(tests); surv=[]
for i,t in enumerate(tests,1):
    if t[0]<=.05*i/m: surv=tests[:i]
print('seals',N,'sites>=30:',{s:n for s,n in sites.items() if n>=30},'tests',m,'survive FDR',len(surv))
for p,site,g,a,ns,ng in surv[:30]:
    print(f'  sign {g:4} {site:12s} {a}/{ns} seals there vs {ng}/{N} overall  (share there {a/ng:.2f})  p={p:.1e}')
