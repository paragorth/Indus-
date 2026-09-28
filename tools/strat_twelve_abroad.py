"""S284: is the fixed number 12 (W55) enriched in overseas texts, like the twins (S275)?
Share of W55 texts (site+text dedup) found abroad vs the same share for signs of similar frequency."""
import json,collections
m=json.load(open('data/derived/merged-corpus-canonical.json'))
ABROAD={'Failaka','Hajar','Kish','Luristan','Nippur',"Qala'at al-Bahrain","Ra's al-Junayz",'Susa','Tell Umma','Tello','Tepe Yahya','Ur','Gonur Depe','Salut','Girsu','Karzakan','Saar','Janabiyah','Kalba','Dilmun','Altyn Depe','Shortughai','Miri Qalat'}
seen=set(); T=[]
for r in m:
    s=tuple(r.get('seq') or [])
    if s and (r['site'],s) not in seen: seen.add((r['site'],s)); T.append((r['site'],set(s),s))
f=collections.Counter(x for _,s,_ in T for x in s)
def share(sg): n=[st for st,s,_ in T if sg in s]; return sum(st in ABROAD for st in n)/len(n),len(n)
for sg in (55,91):
    o,n=share(sg); peers=[w for w,c in f.items() if 0.5*f[sg]<=c<=2*f[sg] and w!=sg]
    ps=[share(w)[0] for w in peers]
    print(f'W{sg}: abroad {o:.3f} (n={n}) | peers n={len(peers)} mean {sum(ps)/len(ps):.3f}, share of peers >= : {sum(p>=o for p in ps)/len(ps):.3f}')
for st,s,seq in T:
    if 55 in s and st in ABROAD: print('  ',st,seq)
# adjacency of 12 to person signs abroad vs home
P={90,91,93,99,121}
ab=[seq for st,_,seq in T if st in ABROAD and 55 in seq]; hm=[seq for st,_,seq in T if st not in ABROAD and 55 in seq]
adj=lambda q:any((a==55 and b in P) or (b==55 and a in P) for a,b in zip(q,q[1:]))
print('12 next to a person sign: abroad',sum(map(adj,ab)),'/',len(ab),'| home',sum(map(adj,hm)),'/',len(hm))
