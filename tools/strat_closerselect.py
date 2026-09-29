"""S303: do closing words select their own qualifier classes? For each paradigm closer, the distribution of the
sign directly before it (its left partner). Statistic: mean pairwise overlap (sum of min shares, 'histogram
intersection') between closers' left-partner distributions, over closers with >= 15 texts. Low overlap = each
closer takes its own qualifiers. Control: closer labels permuted over the texts (2,000x), keeping the left partners.
Wells seq_raw and IM77."""
import json,csv,collections,random,itertools
def run(T,CL,SUF,label):
    rows=[]
    for s in T:
        core=list(s)
        while len(core)>1 and core[-1] in SUF: core.pop()
        if len(core)>=2 and core[-1] in CL: rows.append((core[-1],core[-2]))
    cnt=collections.Counter(c for c,_ in rows); use=[c for c in CL if cnt[c]>=15]
    rows=[r for r in rows if r[0] in use]
    def stat(R):
        d=collections.defaultdict(collections.Counter)
        for c,l in R: d[c][l]+=1
        ov=[]
        for a,b in itertools.combinations(use,2):
            na,nb=sum(d[a].values()),sum(d[b].values())
            ov.append(sum(min(d[a][k]/na,d[b][k]/nb) for k in d[a]))
        return sum(ov)/len(ov),d
    obs,d=stat(rows); rnd=random.Random(14); le=0
    labs=[c for c,_ in rows]; lefts=[l for _,l in rows]
    for _ in range(2000):
        rnd.shuffle(labs); le+=stat(list(zip(labs,lefts)))[0]<=obs
    print(f'{label}: closers {use}; texts {len(rows)}; mean overlap {obs:.3f}; P(lower)={(le+1)/2001:.4f}')
    for c in use: print('   ',c,d[c].most_common(4))
C=json.load(open('data/derived/merged-corpus-canonical.json')); seen=set(); T=[]
for r in C:
    s=r['seq_raw']
    if s and (r['site'],tuple(s)) not in seen: seen.add((r['site'],tuple(s))); T.append(s)
run(T,[740,520,151,156,527,226,617,154,158,236,700],{400,90},'Wells')
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv'))); sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
seen=set(); T=[]
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    s=[int(x) for x in s if x.isdigit() and x not in ('0','000')]
    if s and (ls[0]['site'],tuple(s)) not in seen: seen.add((ls[0]['site'],tuple(s))); T.append(s)
run(T,[342,211,12,15,254,60,328],{176,1},'IM77')
