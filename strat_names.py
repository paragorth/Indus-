"""Strategy 19: are the unique 'middles' of Indus seal texts (S18) built like compound personal names?
Reference: Ur III seal legends, line 1 = owner name (CDLI), split into signs.
Head productivity: the share of strings covered by the 10 most common initial signs, and for those heads
the number of distinct continuations per occurrence. Compositional (theophoric) naming = a few heads
covering much of the corpus, each with many different, mostly unique tails."""
import csv, re, collections, random
import strat_repcal as S
SP=S.SP
def ur3_names():
    names=[]; pid=None; in_seal=False; first=True
    for line in open(SP+'cdli.atf',errors='ignore'):
        if line.startswith('&P'): pid=line[2:8]; in_seal=False; continue
        if line.startswith('@'):
            in_seal = line.startswith('@seal') or (S.cat.get(pid,('','',''))[1]=='seal' and in_seal)
            if line.startswith('@seal') or S.cat.get(pid,('','',''))[1]=='seal': first=True
            if S.cat.get(pid,('','',''))[1]=='seal': in_seal=True
            continue
        m=re.match(r'^1\.\s+(.*)',line)
        if m and in_seal and first and 'Ur III' in S.cat.get(pid,('',))[0]:
            toks=[S.clean(s) for s in re.split(r'[-\s.{}]+',m.group(1)) if S.clean(s) and not re.match(r'^(x|\.\.\.|\d.*)$',S.clean(s))]
            if toks and 'kiszib3' not in toks: names.append(tuple(toks))
            first=False
    return names
def indus_middles():
    objs=collections.defaultdict(list); typ={}
    for r in csv.DictReader(open('data/im77/im77_corpus_lines.csv')):
        objs[r['text_no']].append(r['signs_clean']); typ[r['text_no']]=r['object_type']
    out=[]
    for k,v in objs.items():
        if typ[k]!='seal' or len(v)!=1: continue
        t=v[0].split()
        if '0' in t or len(t)<4: continue
        if t[:2]==['267','99']: t=t[2:]
        if t and t[-1] in {'342','162','169','15','254','12'}: t=t[:-2]
        if len(t)>=2: out.append(tuple(t))
    return list(set(out))
def profile(name,strs):
    strs=list(set(strs)); heads=collections.Counter(s[0] for s in strs)
    top=heads.most_common(10); cov=sum(n for _,n in top)/len(strs)
    tails={h:collections.Counter(s[1:] for s in strs if s[0]==h) for h,_ in top}
    uniq=sum(sum(1 for c in tails[h].values() if c==1) for h,_ in top)/sum(n for _,n in top)
    L=collections.Counter(len(s) for s in strs)
    print(f"{name}: {len(strs)} distinct strings; lengths {dict(sorted(L.items()))}")
    print(f"  top-10 heads cover {cov:.2f}; unique tails after a top head {uniq:.2f}; heads {[h for h,_ in top]}")
    # same measure on the last sign
    tl=collections.Counter(s[-1] for s in strs).most_common(10)
    print(f"  top-10 final signs cover {sum(n for _,n in tl)/len(strs):.2f}; finals {[h for h,_ in tl]}")
    # shuffled control: same strings, signs permuted within each string
    rng=random.Random(0); sh=[tuple(rng.sample(s,len(s))) for s in strs]
    hc=collections.Counter(s[0] for s in sh).most_common(10)
    print(f"  control (signs shuffled within strings): top-10 heads cover {sum(n for _,n in hc)/len(sh):.2f}")
if __name__=='__main__':
    profile('Ur III seal owner names',ur3_names())
    profile('Indus seal middles',indus_middles())
