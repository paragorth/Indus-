"""S316: test of the C gloss 'twins W91 = partners'. Predictions: (a) twins texts sit on trade objects (round/Gulf
seals, tags, tablets, or found abroad) more than other texts of similar length; (b) twins avoid the home frame
(opener/closer titles). Control group: texts with the single man W90 (a person sign without the 'two'), and all
texts; length-matched resampling (10,000x). seq_raw, site+text dedup."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
ABROAD={'Failaka','Hajar','Kish','Luristan','Nippur',"Qala'at al-Bahrain","Ra's al-Junayz",'Susa','Tell Umma','Tello','Tepe Yahya','Ur','Gonur Depe','Salut','Girsu','Karzakan','Saar','Janabiyah','Kalba','Dilmun','Altyn Depe','Shortughai','Miri Qalat'}
OPEN={817,861,820}; CL={740,520,151,156,527,226,617,154,158,236,700}
def trade(r): return r['site'] in ABROAD or r['type'] in ('SEAL:R','SEAL:C','SEAL:CY') or r['type'].startswith('TAG')
seen=set(); R=[]
for r in C:
    s=r['seq_raw']
    if s and len(s)>=2 and (r['site'],tuple(s)) not in seen: seen.add((r['site'],tuple(s))); R.append((r,s))
byL=collections.defaultdict(list)
for r,s in R: byL[min(len(s),8)].append((r,s))
def feats(G): return (sum(trade(r) for r,_ in G)/len(G), sum(s[0] in OPEN or s[-1] in CL for _,s in G)/len(G))
rnd=random.Random(26)
for sign,name in ((91,'twins W91'),(90,'man W90')):
    G=[(r,s) for r,s in R if sign in s]; t,f=feats(G); st=[];sf=[]
    for _ in range(10000):
        smp=[rnd.choice(byL[min(len(s),8)]) for _,s in G]; a,b=feats(smp); st.append(a); sf.append(b)
    print(f'{name}: n={len(G)}; trade-object share {t:.3f} vs matched {sum(st)/len(st):.3f} P(high)={sum(x>=t for x in st)/len(st):.4f}; home-frame share {f:.3f} vs {sum(sf)/len(sf):.3f} P(low)={sum(x<=f for x in sf)/len(sf):.4f}')
