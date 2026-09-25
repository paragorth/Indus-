"""Strategy 138: apply the S135 insight to REAL Indus data (no anchors available): remove the frame /
grammatical signs identified in GRAMMAR.md (openers, markers, closers, suffixes, numerals) as word-signs,
and measure how the remaining 'middle' material looks: how many sign types, how long the runs, and how
much of the corpus a future syllabic key would have to cover. This sizes the problem for an anchor search."""
import json, collections, statistics
d=json.load(open("data/derived/merged-corpus-reading-order.json"))
FRAME={817,861,820,2,60,740,520,390,405,407,400,90,151,156,100,176,760,1,3,4,5,16,17,18,31,32,33,34,55,700,220,226,900}
U=list({tuple(s for s in x['seq'] if s) for x in d})
runs=[]; toks=collections.Counter()
for t in U:
    cur=[]
    for s in t:
        if s in FRAME:
            if cur: runs.append(cur); cur=[]
        else: cur.append(s); toks[s]+=1
    if cur: runs.append(cur)
n=sum(toks.values()); srt=[v for _,v in toks.most_common()]
print('texts',len(U),'frame signs removed',len(FRAME))
print('residual tokens',n,'types',len(toks),'hapax',sum(v==1 for v in srt))
print('runs',len(runs),'median run length',statistics.median(map(len,runs)),'runs of 3+',sum(len(r)>=3 for r in runs))
for k in (20,40,80,150): print(f'top-{k} residual signs cover {sum(srt[:k])/n:.2f} of residual tokens')
