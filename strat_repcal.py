"""Strategy 16: calibrate Indus repetition avoidance (S9) against other early *logographic* systems.
Unit = one entry: the sign string of a Proto-Elamite / proto-cuneiform account line (numerals removed),
or a whole seal legend (Ur III, Old Akkadian). Ratio = observed share of 4-6 sign strings with a repeated
sign / share expected (a) drawing signs iid from that corpus's own frequencies and (b) shuffling signs
across strings of the same length at the same position (slot-preserving).
Indus: IM77 lines (seals and all objects)."""
import csv, re, random, collections, sys
SP='/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
csv.field_size_limit(10**9)
NUM=re.compile(r'^(\d+|n)\((N|n|disz|u|asz|gesz|barig|ban|szar)', re.I)
def clean(t): return re.sub(r'[#?!\[\]<>*]|~\w+','',t)
cat={}
for r in csv.DictReader(open(SP+'cdli_cat.csv')):
    cat[r['id_text'].zfill(6)]=(r['period'],r['object_type'],r['genre'],r['language'])
def groups():
    out=collections.defaultdict(list); pid=None; seal=[]; in_seal=False
    def flush():
        if pid and seal: out[('seal',)+cat.get(pid,('',)*4)[:1]].append(seal[:])
    for line in open(SP+'cdli.atf',errors='ignore'):
        if line.startswith('&P'):
            flush(); pid=line[2:8]; seal=[]; in_seal=False; continue
        if line.startswith('@seal') or line.startswith('@envelope'): in_seal=line.startswith('@seal'); continue
        if line.startswith('@'): 
            if line.startswith(('@obverse','@reverse','@left','@right','@top','@bottom','@edge','@column','@surface','@face')): pass
            continue
        m=re.match(r'^\d+\'?\.\s+(.*)',line)
        if not m or pid is None: continue
        per,ot,gen,lang=cat.get(pid,('','','',''))
        body=m.group(1)
        if 'Proto-Elamite' in per or per.startswith('Uruk'):
            left=body.split(',')[0] if 'Proto-Elamite' in per else body
            toks=[clean(t) for t in left.split() if not NUM.match(clean(t)) and clean(t) not in ('','...','X','x')]
            key='PE' if 'Proto-Elamite' in per else 'proto-cuneiform'
            if toks: out[key].append(toks)
        elif ot=='seal' or in_seal:
            toks=[clean(s) for s in re.split(r'[-\s.{}]+',body) if clean(s) and not re.match(r'^(x|\.\.\.|\d.*)$',clean(s))]
            seal.extend(toks)
    flush()
    return out
def ratio(strs,rng,R=20000):
    res={}
    freq=collections.Counter(x for s in strs for x in s); pool=list(freq.elements())
    for L in (4,5,6):
        S=[s for s in strs if len(s)==L]
        if len(S)<30: continue
        obs=sum(len(set(s))<L for s in S)/len(S)
        iid=sum(len(set(rng.choices(pool,k=L)))<L for _ in range(R))/R
        cols=[[s[i] for s in S] for i in range(L)]; hit=0
        for _ in range(200):
            for c in cols: rng.shuffle(c)
            hit+=sum(len(set(t))<L for t in zip(*cols))
        slot=hit/(200*len(S))
        res[L]=(len(S),obs,obs/iid if iid else float('nan'),obs/slot if slot else float('nan'))
    return res
if __name__=='__main__':
    rng=random.Random(0); G=groups()
    im=collections.defaultdict(list)
    for r in csv.DictReader(open('data/im77/im77_corpus_lines.csv')):
        s=[x for x in r['signs_clean'].split() if x!='0']
        if s: im['Indus all lines'].append(s)
        if s and r['object_type']=='seal': im['Indus seals'].append(s)
    todo=[(k,v) for k,v in im.items()]+[('Proto-Elamite entries',G['PE']),('Proto-cuneiform lines',G['proto-cuneiform'])]
    for k,v in G.items():
        if k[0]=='seal' and len(v)>=100: todo.append(('seal legends '+k[1][:24],v))
    for name,strs in todo:
        r=ratio(strs,rng)
        print(f"{name:40s} n_strings={len(strs):6d} "+"  ".join(f"L{L}: n={n} obs={o:.3f} /iid={a:.2f} /slot={b:.2f}" for L,(n,o,a,b) in r.items()))
