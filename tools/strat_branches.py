"""S326: five 'trade-logic' tests (user: be creative).
1 Branch map: texts (>= 3 signs) on 2+ objects: how many animals / sites per text; control = animal labels permuted
  among seals (2,000x): are repeated texts spread over MORE emblems than chance (firm with branches) or fewer (one house)?
2 Trade direction: for sealings (TAG) matching a seal text, origin site of the seal -> find site of the sealing.
3 Business volume: sealings per matched text.
4 Seal size by emblem: mean max side (mm) of square seals per emblem, and how many seals each emblem has.
5 Gulf link: Gulf/abroad seal texts matching any home sealing, tablet or seal text (exact, >= 3 signs; also sub-strings of 3)."""
import json,csv,collections,random,statistics as st
C=json.load(open('data/derived/merged-corpus-canonical.json'))
raw={x['cisi']:x for x in csv.DictReader(open('data/raw/inscriptions.csv')) if x['cisi'] not in ('-','')}
emb=lambda r:(r.get('symbol') or '').split(':')[0]
print('== 1 branch map')
seals=[r for r in C if r['type'].startswith('SEAL') and r['seq_raw'] and len(r['seq_raw'])>=3 and emb(r) not in ('','-')]
by=collections.defaultdict(list)
for r in seals: by[tuple(r['seq_raw'])].append(r)
rep={t:L for t,L in by.items() if len(L)>=2}
def spread(lab): return sum(len({lab[id(r)] for r in L}) for L in rep.values())/len(rep)
lab={id(r):emb(r) for r in seals}; obs=spread(lab)
ids=[id(r) for r in seals]; vals=[lab[i] for i in ids]; rnd=random.Random(33); null=[]
for _ in range(2000):
    rnd.shuffle(vals); null.append(spread(dict(zip(ids,vals))))
null.sort(); print(f'  repeated seal texts {len(rep)}; mean distinct emblems per text {obs:.2f}; permuted median {null[1000]:.2f} [{null[50]:.2f},{null[1950]:.2f}]; P(fewer)={sum(x<=obs for x in null)/2000:.4f}')
multi=[(t,sorted({(r['site'],emb(r)) for r in L})) for t,L in rep.items() if len({emb(r) for r in L})>1]
for t,x in multi[:8]: print('   ',t,x)
print('== 2/3 trade direction and volume')
sealtxt=collections.defaultdict(set)
for r in C:
    if r['type'].startswith('SEAL') and r['seq_raw'] and len(r['seq_raw'])>=3: sealtxt[tuple(r['seq_raw'])].add(r['site'])
flow=collections.Counter(); vol=collections.Counter()
for r in C:
    t=tuple(r['seq_raw'] or [])
    if r['type'].startswith('TAG') and len(t)>=3 and t in sealtxt:
        for o in sealtxt[t]: flow[(o,r['site'])]+=1
        vol[t]+=1
print('  origin(seal site) -> sealing site:',dict(flow)); print('  sealings per text:',vol.most_common(5))
print('== 4 seal size by emblem (square seals, max side mm)')
sz=collections.defaultdict(list)
for r in C:
    x=raw.get(r['cisi'])
    if r['type']=='SEAL:S' and x:
        try: h=float(x['horizontal(mm)']); v=float(x['vertical(mm)'])
        except: continue
        if h>0 and v>0: sz[emb(r) or 'none'].append(max(h,v))
for e,v in sorted(sz.items(),key=lambda x:-len(x[1])):
    if len(v)>=8: print(f'   {e:8s} n={len(v):4d} mean {st.mean(v):.1f} mm  median {st.median(v):.1f}')
print('== 5 Gulf link')
ABROAD={'Failaka','Hajar','Kish','Luristan','Nippur',"Qala'at al-Bahrain","Ra's al-Junayz",'Susa','Tell Umma','Tello','Tepe Yahya','Ur','Gonur Depe','Salut','Girsu','Karzakan','Saar','Janabiyah','Kalba','Dilmun','Altyn Depe','Shortughai','Miri Qalat'}
home=collections.defaultdict(list); home3=collections.defaultdict(set)
for r in C:
    s=r['seq_raw'] or []
    if r['site'] in ABROAD or len(s)<2: continue
    home[tuple(s)].append((r['site'],r['type'])) 
    for i in range(len(s)-2): home3[tuple(s[i:i+3])].add((r['site'],r['type']))
for r in C:
    s=r['seq_raw'] or []
    if r['site'] not in ABROAD or len(s)<2: continue
    ex=home.get(tuple(s)); tri=[(tuple(s[i:i+3]),len(home3[tuple(s[i:i+3])])) for i in range(len(s)-2) if tuple(s[i:i+3]) in home3]
    if ex or tri: print(f'   {r["site"]} {r["cisi"]} {s}: exact home match {ex}; shared 3-sign runs {tri}')
