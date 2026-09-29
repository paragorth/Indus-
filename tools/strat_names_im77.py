"""S311b: replicate the name-internal positions in IM77 with signs fixed in advance from Wells via the bridge.
Name-initial set: W692=M150, W575=M197, W125=M28, W416=M173. Name-final set: W840=M403, W460=M230, W435=M130,
W440=M127, W717=M341. Names in IM77 = text minus the frame (opener M267/M391 + marker M99/M123 at start;
suffix M176/M1, closer M342/M211/M12/M15/M254/M60/M328 and its preceding sign at the end; numerals M87-M112 and
M121 removed). Statistic: mean name-initial share of the initial set minus that of the final set, for names of
2-4 signs. Control: shuffle within names, 2,000x."""
import csv,collections,random
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv'))); sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
OPEN={267,391}; MARK={99,100,123}; SUF={176,1}; CL={342,211,12,15,254,60,328}; NUM=set(range(86,122))
INI={150,197,28,173}; FIN={403,230,130,127,341}
seen=set(); names=[]
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    s=[int(x) for x in s if x.isdigit() and x not in ('0','000')]
    if not s or (ls[0]['site'],tuple(s)) in seen: continue
    seen.add((ls[0]['site'],tuple(s)))
    if s[0] in OPEN: s=s[1:]
    if s and s[0] in MARK: s=s[1:]
    while s and s[-1] in SUF: s=s[:-1]
    if s and s[-1] in CL: s=s[:-2]
    s=[a for a in s if a not in NUM]
    if 2<=len(s)<=4: names.append(s)
def stat(N):
    pos=collections.defaultdict(lambda:[0,0,0])
    for s in N:
        for i,a in enumerate(s):
            pos[a][2]+=1
            if i==0: pos[a][0]+=1
            if i==len(s)-1: pos[a][1]+=1
    ini=[pos[a][0]/pos[a][2] for a in INI if pos[a][2]>=5]; fin=[pos[a][0]/pos[a][2] for a in FIN if pos[a][2]>=5]
    return sum(ini)/len(ini)-sum(fin)/len(fin), pos
obs,pos=stat(names); rnd=random.Random(22); null=sorted(stat([rnd.sample(s,len(s)) for s in names])[0] for _ in range(2000))
print(f'IM77 names 2-4 signs: {len(names)}; initial-share difference (initial set - final set) {obs:.3f}; null median {null[1000]:.3f}, max {null[-1]:.3f}; P={(sum(x>=obs for x in null)+1)/2001:.4f}')
for a in sorted(INI|FIN): print(f'  M{a} ({"INI" if a in INI else "FIN"}): n={pos[a][2]} first {pos[a][0]} last {pos[a][1]}')
