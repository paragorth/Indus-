"""S255: do scribes' line breaks fall at frame-slot boundaries?
IM77 sides with 2+ lines (excluding single-sign label lines). For each break, record
the sign just after it and just before it. Control: same texts, break moved to every
other internal position (uniform), repeated per text."""
import csv, collections, random
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
sides=collections.defaultdict(list)
for r in rows:
    sides[(r['text_no'],r['side'])].append(r)
OPEN={'267','391'}; MARK={'99','65','86'}; JAR='342'
obs=collections.Counter(); ctl=collections.Counter(); nb=0
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line']))
    ls=[l for l in ls if int(l['n_signs'] or 0)>=2]   # drop one-sign label lines
    if len(ls)<2: continue
    seq=[];cuts=[]
    for l in ls:
        seq+=l['signs_clean'].split(); cuts.append(len(seq))
    cuts=cuts[:-1]
    if any(s in ('0','000') for s in seq): pass
    inner=list(range(1,len(seq)))
    for c in cuts:
        nb+=1
        def tag(c,C):
            C['after_jar_start']+= seq[c]==JAR
            C['after_marker']+= seq[c-1] in MARK
            C['after_opener']+= seq[c-1] in OPEN
            C['before_marker']+= seq[c] in MARK
        tag(c,obs)
        for c2 in inner: 
            for key in ('after_jar_start','after_marker','after_opener','before_marker'):
                pass
        tmp=collections.Counter()
        for c2 in inner: tag(c2,tmp)
        for key,v in tmp.items(): ctl[key]+=v/len(inner)
print('breaks',nb)
for key in ('after_jar_start','after_marker','after_opener','before_marker'):
    print(f'{key:16s} obs {obs[key]:4d}  expected {ctl[key]:6.1f}  ratio {obs[key]/max(ctl[key],1e-9):.2f}')

# diagnostics: jar just before the break; what starts line 2; permutation p for jar-start
jb=0; starts=collections.Counter(); ends=collections.Counter(); pairs=[]
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); ls=[l for l in ls if int(l['n_signs'] or 0)>=2]
    if len(ls)<2: continue
    seq=[];cuts=[]
    for l in ls: seq+=l['signs_clean'].split(); cuts.append(len(seq))
    for c in cuts[:-1]:
        jb+= seq[c-1]==JAR; starts[seq[c]]+=1; ends[seq[c-1]]+=1
        pairs.append((seq,c))
print('jar ends line before break',jb)
print('line-2 starts',starts.most_common(8)); print('line-1 ends',ends.most_common(8))
random.seed(1); hits=0; N=20000
for _ in range(N):
    s=sum(seq[random.randrange(1,len(seq))]==JAR for seq,c in pairs)
    hits+= s<=0
print('P(jar-start count<=0 under random break)',hits/N)
# jar positions in these texts: how many jars are not final?
nf=sum(1 for seq,c in pairs for i,x in enumerate(seq) if x==JAR and i<len(seq)-1)
print('non-final jars in these texts',nf, 'total jars',sum(seq.count(JAR) for seq,c in pairs))
exp=sum(sum(seq[c2-1]==JAR for c2 in range(1,len(seq)))/(len(seq)-1) for seq,c in pairs)
print('jar ends line1: obs',jb,'expected',round(exp,1))
# is jar at end of the LAST line too (i.e. both lines close with jar)?
both=sum(1 for seq,c in pairs if seq[c-1]==JAR and seq[-1]==JAR); print('jar at end of both lines',both)
