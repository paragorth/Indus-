"""S275: is the 'twins' sign W91 a sea-trade marker? Share of its (site-deduplicated) texts from
coastal Indus sites or the Gulf/Mesopotamia/Iran vs the same share for every sign of similar frequency."""
import json,collections
m=json.load(open('data/derived/merged-corpus-canonical.json'))
COAST={'Lothal','Dholavira','Chanhu-daro','Gola Dhoro (Bagasra)','Kanmer','Rojdi','Rodji','Bala-kot','Nageshwar','Khirsara','Surkotada','Desalpur','Juna Khatiya','Pabumath','Kuntasi','Nuhato'}
ABROAD={'Failaka','Hajar','Kish','Luristan','Nippur',"Qala'at al-Bahrain","Ra's al-Junayz",'Susa','Tell Umma','Tello','Tepe Yahya','Ur','Gonur Depe','Salut','Girsu','Karzakan','Saar','Janabiyah','Kalba','Dilmun','Altyn Depe','Shortughai','Miri Qalat'}
seen=set(); T=[]
for r in m:
    s=tuple(r.get('seq') or [])
    if s and (r['site'],s) not in seen: seen.add((r['site'],s)); T.append((r['site'],set(s)))
f=collections.Counter(x for _,s in T for x in s)
def share(sg,grp): n=[st for st,s in T if sg in s]; return sum(st in grp for st in n)/len(n), len(n)
for lab,grp in (('coast',COAST),('abroad',ABROAD),('coast+abroad',COAST|ABROAD)):
    o,n=share(91,grp)
    peers=[w for w,c in f.items() if 0.5*f[91]<=c<=2*f[91] and w!=91]
    ps=[share(w,grp)[0] for w in peers]
    print(f'{lab}: W91 {o:.2f} (n={n}) | peers (n={len(peers)}) mean {sum(ps)/len(ps):.2f}, share of peers >= W91: {sum(p>=o for p in ps)/len(ps):.3f}')
print('W91 sites',collections.Counter(st for st,s in T if 91 in s).most_common())
