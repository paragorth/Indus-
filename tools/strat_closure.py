"""S359: arithmetic closure by FIND-GROUP. Harappa voucher tablets (count + W700) found in the same trench + grid square:
if a find-group is one transaction's set of tokens, its counts should (a) sum to a 'round' value (multiple of 12, or 4, or a
weight-series value 1,2,4,8,16,32,64,160,200,320,640) or (b) form complete runs (2,3,4 each once) more than chance, and
(c) group sums should cluster (low variance) if each group is a fixed quota. Null: permute values across tablets, keeping
group sizes (5,000x)."""
import json,collections,random,statistics as st
C=json.load(open('data/derived/merged-corpus-canonical.json'))
NUM={1:1,3:3,4:4,5:5,16:6,17:7,18:8,31:1,32:2,33:3,34:4}
def val(s):
    for i in range(len(s)-1):
        if s[i] in NUM and s[i+1]==700: return NUM[s[i]]
T=[(r.get('area-section'),r.get('room-grid'),val(r['seq_raw'])) for r in C if r['type'].startswith('TAB') and r['site']=='Harappa' and r.get('seq_raw')]
T=[t for t in T if t[2] is not None and t[1] not in (None,'--','-') ]
G=collections.defaultdict(list)
for a,g,v in T: G[(a,g)].append(v)
G={k:v for k,v in G.items() if len(v)>=3}
vals=[v for vs in G.values() for v in vs]; sizes=[len(v) for v in G.values()]
print('groups',len(G),'tablets',len(vals),'sizes',collections.Counter(sizes).most_common(8))
W={1,2,4,8,16,32,64,160,200,320,640}
def stats(groups):
    sums=[sum(g) for g in groups]
    return dict(div12=sum(s%12==0 for s in sums), div4=sum(s%4==0 for s in sums), weight=sum(s in W for s in sums),
                run234=sum(sorted(collections.Counter(g).items())==[(2,1),(3,1),(4,1)] or (set(g)>={2,3,4}) for g in groups),
                allsame=sum(len(set(g))==1 for g in groups), sumvar=st.pvariance(sums))
obs=stats(list(G.values()))
rng=random.Random(1); sims=collections.defaultdict(list)
for _ in range(5000):
    rng.shuffle(vals); k=0; gs=[]
    for n in sizes: gs.append(vals[k:k+n]); k+=n
    for a,b in stats(gs).items(): sims[a].append(b)
for a,o in obs.items():
    s=sims[a]; m=st.mean(s); hi=sum(x>=o for x in s)/5000; lo=sum(x<=o for x in s)/5000
    print(f'{a:8s} obs {o:8.2f}  null {m:8.2f}  p_high {hi:.3f}  p_low {lo:.3f}')
print('group sums:',sorted(sum(g) for g in G.values()))
