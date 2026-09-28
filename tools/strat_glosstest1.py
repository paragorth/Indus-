"""S292: first quick tests of grade-C glosses in WORKING-DICTIONARY.md, each with a stratified control.
G1 'porter/carrier' (W151, W156 as closer): predicted to sit more on handled-goods objects (TAG sealings,
   tablets) than other closing words. Control: closer labels permuted within site (5,000x).
G2 'chief' (W176), 'overseer' (W100), 'the Twelve' (W55): predicted to be on LARGER square seals than seals
   of the same site and text length. Control: label permuted within site x length strata (5,000x).
G3 'fish = food ration' (W220 with a number directly before): predicted more often on tablets/tags than on
   seals, compared with other counted items (W390/405 tree, W900, W700). Control: permutation within site.
Canonical corpus, all three sequence levels, site+text dedup."""
import json,csv,random,collections,statistics as st
C=json.load(open('data/derived/merged-corpus-canonical.json'))
raw={x['cisi']:x for x in csv.DictReader(open('data/raw/inscriptions.csv')) if x['cisi'] not in ('-','')}
SUF={400,90}; CL={740,154,151,158,527,520,156,226,617,236,700}
NUMS={1,2,3,4,5,16,17,18,31,32,33,34}
def num(v):
    try: return float(v)
    except: return 0.0
def handled(t): return t.startswith('TAG') or t.startswith('TAB')
def perm(labels,vals,strata,stat,n=5000,seed=0):
    rnd=random.Random(seed); obs=stat(labels,vals); ge=0
    groups=collections.defaultdict(list)
    for i,s in enumerate(strata): groups[s].append(i)
    for _ in range(n):
        L=labels[:]
        for idx in groups.values():
            sub=[labels[i] for i in idx]; rnd.shuffle(sub)
            for i,v in zip(idx,sub): L[i]=v
        ge+=stat(L,vals)>=obs
    return obs,(ge+1)/(n+1)
for key in ['seq_raw','seq_strong','seq_all']:
    seen=set(); R=[]
    for r in C:
        s=r.get(key)
        if not s or (r['site'],tuple(s)) in seen: continue
        seen.add((r['site'],tuple(s))); R.append((r,s))
    print('==',key)
    # G1
    lab=[];val=[];strat=[]
    for r,s in R:
        core=list(s)
        while len(core)>1 and core[-1] in SUF: core.pop()
        if core[-1] in CL and core[-1]!=700 and len(core)>=2:
            lab.append(core[-1] in (151,156)); val.append(handled(r['type'])); strat.append(r['site'])
    f=lambda L,V: sum(v for l,v in zip(L,V) if l)/max(1,sum(L))
    o,p=perm(lab,val,strat,f)
    base=sum(v for l,v in zip(lab,val) if not l)/sum(1 for l in lab if not l)
    print(f'G1 porter closers: handled-object share {o:.2f} (n={sum(lab)}) vs other closers {base:.2f}; P={p:.4f}')
    for w in (151,156):
        n=[v for (r,s),l,v in zip([x for x in R],lab,val)] # placeholder
    # G2
    rows=[]
    for r,s in R:
        x=raw.get(r['cisi'])
        if not x or r['type']!='SEAL:S' or len(s)<2: continue
        h,v=num(x['horizontal(mm)']),num(x['vertical(mm)'])
        if h<=0 or v<=0: continue
        rows.append((max(h,v),s,r['site'],min(len(s),7)))
    for w,g in ((176,'chief'),(100,'overseer'),(55,'the Twelve')):
        lab=[w in s for _,s,_,_ in rows]; val=[z for z,*_ in rows]; strat=[(a,b) for _,_,a,b in rows]
        f=lambda L,V: st.mean([v for l,v in zip(L,V) if l]) if any(L) else 0
        o,p=perm(lab,val,strat,f,seed=1)
        print(f'G2 {g} (W{w}): {sum(lab)} square seals, mean size {o:.1f} mm vs all {st.mean(val):.1f}; P(larger)={p:.4f}')
    # G3
    lab=[];val=[];strat=[]
    for r,s in R:
        for i in range(1,len(s)):
            if s[i-1] in NUMS and s[i] in (220,390,405,900,700):
                lab.append(s[i]==220); val.append(handled(r['type'])); strat.append(r['site'])
    f=lambda L,V: sum(v for l,v in zip(L,V) if l)/max(1,sum(L))
    o,p=perm(lab,val,strat,f,seed=2)
    base=sum(v for l,v in zip(lab,val) if not l)/max(1,sum(1 for l in lab if not l))
    print(f'G3 counted fish: handled share {o:.2f} (n={sum(lab)}) vs other counted items {base:.2f}; P={p:.4f}')
    # also excluding W700 (Harappa-tablet unit)
    keep=[i for i in range(len(lab))]
