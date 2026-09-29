"""S305: which closing title phrases travel abroad? Texts found outside the Indus Valley (ABROAD list of S284) vs
home. For each closer head (S289 paradigm): share of texts ending in it, abroad vs home, with the home texts
length-matched (10,000 resamples of home texts with the same lengths as the abroad set). Also: any fish words or
arrow phrase abroad."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
ABROAD={'Failaka','Hajar','Kish','Luristan','Nippur',"Qala'at al-Bahrain","Ra's al-Junayz",'Susa','Tell Umma','Tello','Tepe Yahya','Ur','Gonur Depe','Salut','Girsu','Karzakan','Saar','Janabiyah','Kalba','Dilmun','Altyn Depe','Shortughai','Miri Qalat'}
CL=[740,520,151,156,527,226,617,154,158,236,700]; SUF={400,90}; FISH={220,240,235,233,231}
seen=set(); A=[]; H=collections.defaultdict(list)
def close(s):
    core=list(s)
    while len(core)>1 and core[-1] in SUF: core.pop()
    return core[-1] if core[-1] in CL else None
for r in C:
    s=r['seq_raw']
    if not s or len(s)<2 or (r['site'],tuple(s)) in seen: continue
    seen.add((r['site'],tuple(s)))
    (A.append(s) if r['site'] in ABROAD else H[len(s)].append(s))
print('abroad texts',len(A))
obs=collections.Counter(close(s) for s in A); fishA=sum(any(a in FISH for a in s) for s in A)
rnd=random.Random(16); sims=collections.defaultdict(list); fsim=[]
for _ in range(10000):
    samp=[rnd.choice(H[len(s)] or H[3]) for s in A]
    c=collections.Counter(close(s) for s in samp)
    for k in CL+[None]: sims[k].append(c[k])
    fsim.append(sum(any(a in FISH for a in s) for s in samp))
for k in CL+[None]:
    v=sorted(sims[k]); m=sum(v)/len(v)
    lo=sum(x<=obs[k] for x in v)/len(v); hi=sum(x>=obs[k] for x in v)/len(v)
    if m>=0.5 or obs[k]: print(f'  closer {k}: abroad {obs[k]} vs home length-matched mean {m:.1f}; P(low)={lo:.4f} P(high)={hi:.4f}')
v=sorted(fsim); print(f'  texts with a fish word: abroad {fishA} vs expected {sum(v)/len(v):.1f}; P(low)={sum(x<=fishA for x in v)/len(v):.4f}')
for s in A:
    if close(s) or any(a in FISH for a in s): print('   ',s)
