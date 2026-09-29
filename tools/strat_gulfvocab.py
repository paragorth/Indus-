"""S314 (user idea): the Gulf round seals use the same signs with a different syntax, as if the signs were shared,
intuitive (language-independent) symbols. Prediction if signs were borrowed across a language boundary: the signs
the Gulf texts keep are the picture/number ones (numerals, persons, fish, counted items), while the home
grammar words (opener, markers, closers, title qualifiers) and name elements (S311) drop out.
Texts found abroad (S284 list) vs home seals; class shares of sign tokens; control: 10,000 length-matched samples
of home seal texts. Also text length. seq_raw, site+text dedup."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
ABROAD={'Failaka','Hajar','Kish','Luristan','Nippur',"Qala'at al-Bahrain","Ra's al-Junayz",'Susa','Tell Umma','Tello','Tepe Yahya','Ur','Gonur Depe','Salut','Girsu','Karzakan','Saar','Janabiyah','Kalba','Dilmun','Altyn Depe','Shortughai','Miri Qalat'}
CLS={'numeral':{1,2,3,4,5,16,17,18,31,32,33,34,55,56,13},'person':{90,91,93,99,100,121,125,140,142,151,156,176},
     'fish':{220,240,235,233,231,226},'counted item':{390,405,407,900,700,904,645,384},
     'frame word':{817,861,820,60,740,400,520,527,617,154,158,236,760,923,690,482,752},
     'name element':{692,575,413,416,920,495,840,460,70,35,440,435,717}}
def cls(a):
    for k,v in CLS.items():
        if a in v: return k
    return 'other'
seen=set(); A=[]; H=collections.defaultdict(list)
for r in C:
    s=r['seq_raw']
    if not s or (r['site'],tuple(s)) in seen: continue
    seen.add((r['site'],tuple(s)))
    if r['site'] in ABROAD: A.append(s)
    elif r['type'].startswith('SEAL'): H[len(s)].append(s)
def shares(T):
    c=collections.Counter(cls(a) for s in T for a in s); n=sum(c.values()); return {k:c[k]/n for k in list(CLS)+['other']}
obs=shares(A); rnd=random.Random(24); sims=collections.defaultdict(list)
for _ in range(10000):
    smp=[rnd.choice(H[len(s)] or H[3]) for s in A]
    for k,v in shares(smp).items(): sims[k].append(v)
allH=[s for v in H.values() for s in v]
print(f'abroad texts {len(A)}, mean length {sum(map(len,A))/len(A):.2f}; home seals {len(allH)}, mean length {sum(map(len,allH))/len(allH):.2f}')
for k in list(CLS)+['other']:
    v=sorted(sims[k]); m=sum(v)/len(v)
    print(f'  {k:13s} abroad {obs[k]:.3f} vs home length-matched {m:.3f}  P(high)={sum(x>=obs[k] for x in v)/len(v):.4f} P(low)={sum(x<=obs[k] for x in v)/len(v):.4f}')
shared=collections.Counter(a for s in A for a in s)
homeset=collections.Counter(a for s in allH for a in s)
print('abroad sign types',len(shared),'of which also used at home',sum(1 for a in shared if homeset[a]),'; abroad-only',[a for a in shared if not homeset[a]])
