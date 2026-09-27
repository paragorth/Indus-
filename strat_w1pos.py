"""S235: Where does the single-stroke marker W1 sit? Relative position in the text (0 = start, 1 = end) and whether it stands between
the name core and the title zone: left neighbour a core sign and right neighbour a title/closer/numeral. Frame via M-codes (CLAUDE.md rule).
Control: same statistics for every other sign of similar frequency (100-300 tokens)."""
import json,collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
b=json.load(open('data/derived/bridge_extended.json'))
FR_M={267,391,99,123,343,293,65,86,342,211,15,12,254,162,169,176,1}
FR={int(w) for w,ms in b.items() if any(m in FR_M for m in ms)}-{1}
NUM=(set(range(1,8))|set(range(12,21))|set(range(25,30))|set(range(32,38))|{39,55,56})-{1,2}
TITLE=FR|NUM|{740,590,390,405,435,690,840,585,100,760,220,900,700}
seen=set(); stats=C.defaultdict(lambda:[0,0,0.0]); cnt=C.Counter()
for x in d:
    s=tuple(x['seq'])
    if (s,x['site']) in seen or len(s)<3: continue
    seen.add((s,x['site']))
    for i in range(1,len(s)-1):
        w=s[i]; cnt[w]+=1
        st=stats[w]; st[0]+=1
        if s[i-1] not in TITLE and s[i-1] not in (817,861,820,2) and s[i+1] in TITLE: st[1]+=1
        st[2]+=i/(len(s)-1)
r1=stats[1]; print('W1 medial tokens',r1[0],'boundary share %.2f'%(r1[1]/r1[0]),'mean rel pos %.2f'%(r1[2]/r1[0]))
peers=[(v[1]/v[0],w) for w,v in stats.items() if 60<=v[0]<=400 and w!=1]
peers.sort(reverse=True)
print('rank of W1 among',len(peers),'peers:',sum(p>r1[1]/r1[0] for p,w in peers)+1)
print('top boundary signs:',[(w,round(p,2)) for p,w in peers[:8]])
