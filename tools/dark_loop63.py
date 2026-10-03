"""S-DARK-63: THE LOGOGRAPHIC-NAME COMPARATOR.
All earlier name comparators (Ur III, OB, Linear B, Latin) are phonetically written names of inflecting languages with a
closed edge set (S-DARK-56.3d). Names written LOGOGRAPHICALLY, one word-sign per element, free element order, no inflection
(Chinese, Japanese kanji, Sino-Vietnamese syllables) are the right comparator for the hypothesis 'Indus middle = name
written with word-signs'. Corpora: data/derived/dark/loop63_corpora (tools/dark_loop63_prep.py).
Cycle 1: S-DARK-56 battery + closed-slot (56.3d) + S-DARK-53 capacity statistics, Indus vs logographic names vs the old
         comparators vs frequency-matched random strings, at the Indus n, with bootstrap CIs.
Cycle 2: logographic signatures: (a) two partially disjoint element pools (gender) - spectral bipartition with a permutation
         null, calibrated on the gender-tagged lists; (b) head slot vs surname slot: closed-set size, coverage, MI(head, middle
         element) vs permutation; (c) common-element free combination.
Cycle 3: replication on held-out sites, Mohenjo-daro vs Harappa, IM77, seq_raw / seq_strong / seq_all.
Usage: python3 tools/dark_loop63.py <1|2|3> [ndraw]
"""
import sys,os
CY=int(sys.argv[1]) if len(sys.argv)>1 else 0; NB=int(sys.argv[2]) if len(sys.argv)>2 else 40
sys.argv=['x','0']; sys.path.insert(0,'tools')
import dark_loop56 as L56
from dark_loop56 import *            # parser, load_indus, im77_objects, indus_names, metrics, draws, boot, H, gini, zipf, KEYS, LADDER, P, LOG, q
import dark_loop53 as L53            # capacity statistics (metrics -> uniq, chao1, gini_id, heaps, lutil_ent ...)
import numpy as np
C63='data/derived/dark/loop63_corpora/'
def jl63(name,minlen=1,maxlen=99):
    out=[]
    for l in open(C63+name+'.jsonl'):
        d=json.loads(l)
        if minlen<=len(d['seq'])<=maxlen: out.append((tuple(d['seq']),d.get('g','U'),d.get('sur')))
    return out
def seqs(rows): return [r[0] for r in rows]
LOGO=['cn_given','cn_ancient','jp_given','jp_person_given','vi_given']
OLD=['ur3_names_dedup','ob_names_dedup','linb_personnel_dedup','latin_names_dedup']
def save(name,obj):
    json.dump(obj,open(f'data/derived/dark/{name}.json','w'),indent=1,default=str)
    open(f'data/derived/dark/{name}_log.txt','w').write('\n'.join(LOG)+'\n')

# ---- closed-slot statistics (S-DARK-56.3d) ----
def cov(names):
    names=[t for t in names if len(t)>=2]
    I=collections.Counter(t[0] for t in names); F=collections.Counter(t[-1] for t in names); M=collections.Counter(a for t in names for a in t[1:-1])
    c10=lambda C:sum(v for _,v in C.most_common(10))/sum(C.values()) if C else float('nan')
    hn=lambda C:H(C)/math.log2(len(C)) if len(C)>1 else float('nan')
    return dict(init10=c10(I),fin10=c10(F),mid10=c10(M),Hinit=hn(I),Hfin=hn(F),kI=len(I),kF=len(F),
                top10el=sum(v for _,v in collections.Counter(a for t in names for a in t).most_common(10))/sum(len(t) for t in names))
COVK=['init10','fin10','mid10','Hinit','Hfin','top10el']
def cov_draws(pool,n,ndraw,lens=None):
    runs=[]
    for b in range(ndraw):
        r=random.Random(5000+b)
        sub=L56.length_match(pool,lens,n,r)[0] if lens is not None else r.sample(pool,min(n,len(pool)))
        runs.append(cov(sub))
    return {k:(q([x[k] for x in runs],0.5),q([x[k] for x in runs],0.025),q([x[k] for x in runs],0.975)) for k in COVK}
def cov_boot(names,nboot,frac=0.8):
    full=cov(names); runs=[]
    for b in range(nboot):
        r=random.Random(7000+b); runs.append(cov(r.sample(names,int(frac*len(names)))))
    out={}
    for k in COVK:
        v=[x[k] for x in runs]; md=q(v,0.5); out[k]=(full[k],full[k]+q(v,0.025)-md,full[k]+q(v,0.975)-md)
    return out
def cap_draws(pool,n,ndraw):
    runs=[]
    for b in range(ndraw):
        r=random.Random(9000+b); sub=r.sample(pool,min(n,len(pool))); runs.append(L53.metrics(sub,r))
    return {k:(q([x[k] for x in runs],0.5),q([x[k] for x in runs],0.025),q([x[k] for x in runs],0.975)) for k in CAPK}
CAPK=['uniq','gt_new','gini_id','heaps','gini_el','zipf_el','even_pos','lutil_ent','top10']
def cap_boot(names,nboot,frac=0.8):
    full=L53.metrics(names,random.Random(1)); runs=[]
    for b in range(nboot):
        r=random.Random(8000+b); runs.append(L53.metrics(r.sample(names,int(frac*len(names))),r))
    out={}
    for k in CAPK:
        v=[x[k] for x in runs]; md=q(v,0.5); out[k]=(full[k],full[k]+q(v,0.025)-md,full[k]+q(v,0.975)-md)
    return out
def freq_random(base,r,mult=6):
    pool=[a for n in base for a in n]; L0=[len(n) for n in base]
    return sorted(set(tuple(r.choice(pool) for _ in range(r.choice(L0))) for _ in range(mult*len(base))))
def t3(v): return f'{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]'

if CY==1:
    P(f'== S-DARK-63 cycle 1: Indus middles vs LOGOGRAPHICALLY written names (Chinese, Japanese kanji, Sino-Vietnamese) vs the phonetic name lists; ndraw {NB}')
    R={}
    # Indus
    IND={}
    for LV in ['seq_raw','seq_strong','seq_all','im77']:
        nm=indus_names(LV,2); IND[LV]=nm
        P(f'\n##### Indus {LV}: {len(nm)} distinct middles >= 2 elements; lengths {sorted(collections.Counter(len(n) for n in nm).items())}')
        R['indus_'+LV]=dict(bat=boot(nm,f'Indus {LV} middles',nboot=NB),cov=cov_boot(nm,NB))
        nm1=indus_names(LV,1); R['indus_'+LV]['cap']=cap_boot(nm1,NB); R['indus_'+LV]['n1']=len(nm1)
        P('    closed-slot: '+'; '.join(f'{k} {t3(R["indus_"+LV]["cov"][k])}' for k in COVK))
        P('    capacity (>=1 el, n=%d): '%len(nm1)+'; '.join(f'{k} {t3(R["indus_"+LV]["cap"][k])}' for k in CAPK))
    base=IND['seq_raw']; n0=len(base); L0=[len(n) for n in base]; n1=R['indus_seq_raw']['n1']
    L1=[len(n) for n in indus_names('seq_raw',1)]
    # logographic names
    P(f'\n##### logographic name lists, one per distinct name, drawn to n={n0} (>=2 elements; order battery) and n={n1} (>=1 element; capacity); also LENGTH-MATCHED to the Indus middle lengths where the list allows')
    for nm in LOGO+['cn_full','jp_person','vi_full']:
        rows=jl63(nm); pool2=[s for s in seqs(rows) if len(s)>=2]; pool1=seqs(rows)
        R[nm]=dict(bat=draws(pool2,n0,nm,ndraw=NB),cov=cov_draws(pool2,n0,NB),cap=cap_draws(pool1,n1,NB),k_pool=len(set(a for s in pool1 for a in s)),pool=len(pool1))
        P('    closed-slot: '+'; '.join(f'{k} {t3(R[nm]["cov"][k])}' for k in COVK))
        P('    capacity: '+'; '.join(f'{k} {t3(R[nm]["cap"][k])}' for k in CAPK))
        if nm in ('jp_given','vi_given','cn_ancient','jp_person_given'):
            R[nm+'_lenmatch']=dict(bat=draws(pool2,n0,nm+' length-matched',ndraw=NB,lens=L0),cov=cov_draws(pool2,n0,NB,lens=L0))
            P('    closed-slot (length-matched): '+'; '.join(f'{k} {t3(R[nm+"_lenmatch"]["cov"][k])}' for k in COVK))
        # frequency-matched random strings on the SAME list's element unigram and lengths (the logographic null)
        r=random.Random(11); fr=freq_random(pool2,r,2)
        R[nm+'_freqrand']=dict(bat=draws(fr,n0,nm+' freq-matched random',ndraw=max(10,NB//2)),cov=cov_draws(fr,n0,max(10,NB//2)))
        P('    closed-slot (freq-random): '+'; '.join(f'{k} {t3(R[nm+"_freqrand"]["cov"][k])}' for k in COVK))
    # old comparators + Indus freq-random
    P('\n##### phonetic name lists (S-DARK-56) and the Indus frequency-matched random null')
    for nm in OLD:
        pool2=[s for s in jl(nm) if len(s)>=2]
        R[nm]=dict(bat=draws(pool2,n0,nm,ndraw=NB),cov=cov_draws(pool2,n0,NB),cap=cap_draws(jl(nm),n1,NB))
        P('    closed-slot: '+'; '.join(f'{k} {t3(R[nm]["cov"][k])}' for k in COVK))
    fr=freq_random(base,random.Random(7))
    R['indus_freqrand']=dict(bat=draws(fr,n0,'Indus freq-matched random',ndraw=NB),cov=cov_draws(fr,n0,NB),cap=cap_draws(fr,n1,NB))
    P('    closed-slot: '+'; '.join(f'{k} {t3(R["indus_freqrand"]["cov"][k])}' for k in COVK))
    # ladder
    P('\n=== cycle 1 ladder (median; Indus seq_raw 95% CI). Verdict per statistic: LOGO if the Indus CI overlaps the logographic given-name range (+-0.01) ; PHON if it overlaps the phonetic range; RAND if it overlaps the Indus freq-random CI')
    allk=[('bat',k) for k in LADDER]+[('cov',k) for k in COVK]+[('cap',k) for k in CAPK]
    def val(c,g,k):
        v=R[c][g]
        if g=='bat': return v['med'][k],v['lo'][k],v['hi'][k]
        return v[k]
    verd={}
    for g,k in allk:
        iv,lo,hi=val('indus_seq_raw',g,k)
        lg=[val(c,g,k)[0] for c in LOGO if g in R[c]]; ph=[val(c,g,k)[0] for c in OLD if g in R[c]]; rr=val('indus_freqrand',g,k)
        tags=[]
        if min(lg)-0.01<=hi and lo<=max(lg)+0.01: tags.append('LOGO')
        if min(ph)-0.01<=hi and lo<=max(ph)+0.01: tags.append('PHON')
        if rr[1]-0.01<=hi and lo<=rr[2]+0.01: tags.append('RAND')
        verd[k]='+'.join(tags) or 'NONE'
        P(f'  {k:10s} Indus {iv:7.3f} [{lo:.3f},{hi:.3f}]  logographic {min(lg):7.3f}..{max(lg):7.3f}  phonetic {min(ph):7.3f}..{max(ph):7.3f}  Indus-freq-random {rr[0]:7.3f} [{rr[1]:.3f},{rr[2]:.3f}] -> {verd[k]}')
    P('\n  full table (median)')
    cols=LADDER+COVK+CAPK
    P('  corpus                      '+' '.join(f'{k:>9s}' for k in cols))
    for c in R:
        row=[]
        for g,k in allk:
            try: row.append(f'{val(c,g,k)[0]:9.3f}')
            except Exception: row.append(f'{"-":>9s}')
        P(f'  {c:27s} '+' '.join(row))
    R['_verdict']=verd
    save('loop63_cycle1',R)

# ---- cycle 2: logographic signatures ----
def spectral_split(names,fmin=5,r=None):
    """elements with >= fmin tokens; co-occurrence graph within names; Fiedler bipartition of the normalised Laplacian.
    Returns side dict and the within-name same-side share among pairs of frequent elements."""
    el=collections.Counter(a for n in names for a in n); E=[a for a,v in el.items() if v>=fmin]
    if len(E)<6: return None,float('nan'),0
    ix={a:i for i,a in enumerate(E)}; W=np.zeros((len(E),len(E)))
    for n in names:
        f=[ix[a] for a in n if a in ix]
        for i in range(len(f)):
            for j in range(i+1,len(f)):
                if f[i]!=f[j]: W[f[i],f[j]]+=1; W[f[j],f[i]]+=1
    d=W.sum(1); keep=d>0
    if keep.sum()<6: return None,float('nan'),0
    Wk=W[keep][:,keep]; dk=Wk.sum(1); Dm=np.diag(1/np.sqrt(dk)); Ls=np.eye(len(dk))-Dm@Wk@Dm
    vals,vecs=np.linalg.eigh(Ls); f=vecs[:,1]
    side={}; kept=[a for a,kp in zip(E,keep) if kp]
    for a,v in zip(kept,f): side[a]=int(v>0)
    same=0; tot=0
    for n in names:
        s=[side[a] for a in n if a in side]
        for i in range(len(s)):
            for j in range(i+1,len(s)): tot+=1; same+=(s[i]==s[j])
    return side,(same/tot if tot else float('nan')),tot
def bipartition_test(names,label,nperm=50,fmin=5,gender=None,r=None):
    r=r or random.Random(63)
    names=[tuple(n) for n in names if len(n)>=2]
    side,obs,tot=spectral_split(names,fmin)
    if side is None: P(f'  {label}: too few frequent elements'); return None
    nul=[]
    for _ in range(nperm):
        sh=L56.shuffle_names(names,r,'shuf'); _,v,_=spectral_split(sh,fmin); nul.append(v)
    mu=sum(nul)/len(nul); sd=(sum((x-mu)**2 for x in nul)/len(nul))**0.5
    # balance of the split and purity against a known label if given
    bal=sum(side.values())/len(side)
    out=dict(n=len(names),nel=len(side),pairs=tot,same=obs,null=mu,sd=sd,z=(obs-mu)/sd if sd else None,balance=bal)
    if gender is not None:
        # element majority gender from names with known gender; agreement of the spectral side with gender (max over the two labelings)
        eg=collections.defaultdict(collections.Counter)
        for n,g in zip(names,gender):
            if g in('M','F'):
                for a in n: eg[a][g]+=1
        pur=[]; agree=0; cnt=0
        for a,s in side.items():
            c=eg.get(a)
            if c and sum(c.values())>=3:
                pur.append(max(c.values())/sum(c.values())); cnt+=1; agree+=(s==(c['M']>c['F']))
        agr=max(agree,cnt-agree)/cnt if cnt else float('nan')
        # gender purity of the shuffle (elements keep their names' genders in the shuffle? no: shuffle breaks element-name link, so purity under
        # shuffle = baseline from the gender mix)
        out.update(gender_purity=sum(pur)/len(pur) if pur else float('nan'),side_gender_agreement=agr,n_gendered_el=cnt,
                   same_gender_share=sum(1 for n,g in zip(names,gender) if g in('M','F'))/len(names))
    P(f'  {label}: n={len(names)} frequent elements {len(side)} (balance {bal:.2f}); same-side share of within-name pairs {obs:.3f} vs shuffle {mu:.3f} +- {sd:.3f} (z={out["z"]:.1f})'
      +(f'; element gender purity {out["gender_purity"]:.3f}, spectral side = gender for {out["side_gender_agreement"]:.3f} of {cnt} gendered elements' if gender is not None else ''))
    return out
def head_test(heads,mids,label,nperm=200,r=None):
    """heads: list of head tokens (surname / closer), mids: list of middle tuples (same length). Closed-set size, coverage,
    MI(head, middle element) as % of H(head) vs heads permuted among texts; also mean partner-set Jaccard between heads."""
    r=r or random.Random(64)
    hc=collections.Counter(heads); n=len(heads)
    top10=sum(v for _,v in hc.most_common(10))/n; top1=hc.most_common(1)[0][1]/n
    def mi(hs):
        J=collections.Counter(); A=collections.Counter(); B=collections.Counter(); N=0
        for h,m in zip(hs,mids):
            for a in m: J[(h,a)]+=1; A[h]+=1; B[a]+=1; N+=1
        return sum(v/N*math.log2(v*N/(A[h]*B[a])) for (h,a),v in J.items())
    obs=mi(heads); hs=list(heads); nul=[]
    for _ in range(nperm):
        r.shuffle(hs); nul.append(mi(hs))
    mu=sum(nul)/nperm; sd=(sum((x-mu)**2 for x in nul)/nperm)**0.5
    Hh=H(hc)
    # element exclusivity: share of middle-element tokens whose element occurs with only one head type (among elements >= 5 tokens)
    byel=collections.defaultdict(collections.Counter)
    for h,m in zip(heads,mids):
        for a in m: byel[a][h]+=1
    freq=[c for c in byel.values() if sum(c.values())>=5]
    excl=sum(1 for c in freq if len(c)==1)/len(freq) if freq else float('nan')
    out=dict(n=n,k_head=len(hc),top1=top1,top10=top10,Hnorm=Hh/math.log2(len(hc)) if len(hc)>1 else 0,mi=obs,mi_null=mu,mi_sd=sd,mi_z=(obs-mu)/sd if sd else None,
             mi_excess_pct=100*(obs-mu)/Hh if Hh else float('nan'),el_exclusive=excl,nfreq_el=len(freq))
    P(f'  {label}: n={n} head types {len(hc)} top1 {top1:.3f} top10 {top10:.3f} Hnorm {out["Hnorm"]:.3f}; MI(head, middle element) {obs:.3f} vs permuted {mu:.3f} +- {sd:.3f} (z {out["mi_z"]:.1f}; excess {out["mi_excess_pct"]:.1f}% of H(head)); elements (>=5) exclusive to one head {excl:.3f} of {len(freq)}')
    return out
def indus_head_sets(LV):
    objs=load_indus(LV) if LV!='im77' else im77_objects()
    seen=set(); heads=[]; mids=[]; heads2=[]; mids2=[]
    for o in objs:
        if len(o['mid'])<1: continue
        key=(tuple(o['mid']),o['closer'],o['seq'][-1])
        if key in seen: continue
        seen.add(key)
        if o['closer'] is not None: heads.append(o['closer']); mids.append(o['mid'])
        # closer-agnostic: head = last sign of the text, middle = everything before it (texts >= 2)
        if len(o['seq'])>=2: heads2.append(o['seq'][-1]); mids2.append(tuple(o['seq'][:-1]))
    return heads,mids,heads2,mids2

if CY==2:
    P(f'== S-DARK-63 cycle 2: logographic-name signatures (gender pools, surname slot, free combination of common elements); nperm {NB}')
    R={}
    n0=len(indus_names('seq_raw',2))
    P(f'\n##### (a) two partially disjoint element pools: spectral bipartition of the element co-occurrence graph (elements >= 5 tokens), same-side share of within-name pairs vs token shuffle (lengths kept), {NB} permutations; calibration = gender-tagged lists at n={n0}')
    for nm in ['cn_given','jp_given','vi_given']:
        rows=[x for x in jl63(nm,2) if x[1] in('M','F')]; r=random.Random(21); sub=r.sample(rows,min(n0,len(rows)))
        R['bip_'+nm]=bipartition_test(seqs(sub),nm+' (gendered names only)',NB,gender=[x[1] for x in sub])
        # mixed-gender list as drawn (includes U)
        rows=jl63(nm,2); sub=r.sample(rows,min(n0,len(rows)))
        R['bip_'+nm+'_all']=bipartition_test(seqs(sub),nm+' (all)',NB,gender=[x[1] for x in sub])
    for nm in ['cn_ancient','jp_person_given']:
        rows=jl63(nm,2); sub=random.Random(22).sample(rows,min(n0,len(rows)))
        R['bip_'+nm]=bipartition_test(seqs(sub),nm,NB)
    for nm in OLD:
        pool=[s for s in jl(nm) if len(s)>=2]; sub=random.Random(23).sample(pool,min(n0,len(pool)))
        R['bip_'+nm]=bipartition_test(sub,nm,NB)
    for LV in ['seq_raw','seq_strong','seq_all','im77']:
        nm=indus_names(LV,2); R['bip_indus_'+LV]=bipartition_test(nm,f'Indus {LV} middles',NB)
        if LV=='seq_raw':
            R['bip_indus_NAMEonly']=bipartition_test(indus_names(LV,2,field='name'),'Indus seq_raw NAME-only',NB)
            R['bip_indus_seals']=bipartition_test(indus_names(LV,2,subset=lambda o:o['ot']=='seal'),'Indus seq_raw seals',NB)
            # object type as the 'gender' label for Indus: seal vs tablet
            objs=load_indus(LV); d={}
            for o in objs:
                if len(o['mid'])>=2 and o['ot'] in('seal','tablet'): d.setdefault(o['mid'],set()).add(o['ot'])
            nms=sorted(d); lab=['M' if d[m]=={'seal'} else 'F' if d[m]=={'tablet'} else 'U' for m in nms]
            R['bip_indus_objtype']=bipartition_test(nms,'Indus seq_raw middles with seal/tablet as the label',NB,gender=lab)
    fr=freq_random(indus_names('seq_raw',2),random.Random(7)); R['bip_indus_freqrand']=bipartition_test(random.Random(1).sample(fr,n0),'Indus freq-matched random',NB)
    P(f'\n##### (b) head slot: Indus closer (parser) and last-sign (closer-agnostic) vs the surname slot of Chinese / Japanese / Vietnamese full names; MI(head, middle element) vs heads permuted (200x)')
    for nm in ['cn_full','jp_person','vi_full']:
        rows=jl63(nm); rows=[x for x in rows if x[2] and len(x[0])>len(x[2] if isinstance(x[2],str) and nm!='vi_full' else [0])]
        r=random.Random(31); sub=r.sample(rows,min(2000,len(rows)))
        if nm=='vi_full': heads=[x[2] for x in sub]; mids=[x[0][1:] for x in sub]
        else: heads=[x[2] for x in sub]; mids=[x[0][len(x[2]):] for x in sub]
        R['head_'+nm]=head_test(heads,mids,nm+' surname -> given')
        # control: given-name LAST element as a fake head (open slot)
        R['head_'+nm+'_lastel']=head_test([m[-1] for m in mids if len(m)>=2],[m[:-1] for m in mids if len(m)>=2],nm+' last given element as head (open-slot control)')
    for LV in ['seq_raw','seq_strong','seq_all','im77']:
        heads,mids,heads2,mids2=indus_head_sets(LV)
        R['head_indus_closer_'+LV]=head_test(heads,mids,f'Indus {LV} parser closer -> middle')
        R['head_indus_last_'+LV]=head_test(heads2,mids2,f'Indus {LV} last sign -> rest of text')
    for nm in ['ur3_names_dedup','linb_personnel_dedup']:
        pool=[s for s in jl(nm) if len(s)>=3]; sub=random.Random(32).sample(pool,min(2000,len(pool)))
        R['head_'+nm+'_lastel']=head_test([s[-1] for s in sub],[s[:-1] for s in sub],nm+' last element as head (suffix slot)')
        R['head_'+nm+'_firstel']=head_test([s[0] for s in sub],[s[1:] for s in sub],nm+' first element as head (prefix slot)')
    P(f'\n##### (c) common elements combined freely: top-10 element token share; for the top-10 elements, number of distinct partners / tokens and share of partners that are hapax partners; pair-model gain (S-DARK-26) is in cycle 1 pair_pred')
    def free_comb(names,label):
        names=[n for n in names if len(n)>=2]; el=collections.Counter(a for n in names for a in n); T=sum(el.values())
        top=[a for a,_ in el.most_common(10)]; part=collections.defaultdict(collections.Counter)
        for n in names:
            for a in n:
                if a in top:
                    for b in n:
                        if b!=a: part[a][b]+=1
        pr=[len(part[a])/sum(part[a].values()) for a in top if part[a]]
        hap=[sum(1 for v in part[a].values() if v==1)/len(part[a]) for a in top if part[a]]
        pos=collections.defaultdict(lambda:[0,0])
        for n in names:
            for i,a in enumerate(n):
                if a in top: pos[a][0 if i==0 else 1]+=1
        skew=[abs(v[0]-v[1])/sum(v) for v in pos.values()]
        out=dict(n=len(names),top10_share=sum(el[a] for a in top)/T,partners_per_token=sum(pr)/len(pr),hapax_partner_share=sum(hap)/len(hap),top10_edge_skew=sum(skew)/len(skew))
        P(f'  {label}: top-10 elements carry {out["top10_share"]:.3f} of tokens; distinct partners per token {out["partners_per_token"]:.3f}; hapax partners {out["hapax_partner_share"]:.3f}; |P(first)-P(not first)| of the top-10 {out["top10_edge_skew"]:.3f}')
        return out
    for nm in LOGO:
        pool=[s for s in seqs(jl63(nm,2))]; R['free_'+nm]=free_comb(random.Random(41).sample(pool,min(n0,len(pool))),nm)
    for nm in OLD:
        pool=[s for s in jl(nm) if len(s)>=2]; R['free_'+nm]=free_comb(random.Random(42).sample(pool,min(n0,len(pool))),nm)
    for LV in ['seq_raw','seq_all','im77']: R['free_indus_'+LV]=free_comb(indus_names(LV,2),f'Indus {LV}')
    R['free_indus_freqrand']=free_comb(random.Random(1).sample(fr,n0),'Indus freq-matched random')
    save('loop63_cycle2',R)

if CY==3:
    P(f'== S-DARK-63 cycle 3: replication on held-out sites, Mohenjo-daro vs Harappa, IM77, three merge levels; ndraw {NB}')
    R={}
    KEY=['pos_excess','pos_bound','mi_ex','pair_pred','big_reuse','el_gini','el_heaps']
    def block(names,label,gender=None):
        names=sorted(set(tuple(n) for n in names if len(n)>=2)); n=len(names)
        b=boot(names,label,nboot=max(10,NB//2)); c=cov_boot(names,max(10,NB//2))
        bp=bipartition_test(names,label+' bipartition',max(10,NB//2),gender=gender)
        P('    closed-slot: '+'; '.join(f'{k} {t3(c[k])}' for k in COVK))
        return dict(n=n,bat={k:(b['med'][k],b['lo'][k],b['hi'][k]) for k in LADDER},cov=c,bip=bp)
    for LV in ['seq_raw','seq_strong','seq_all','im77']:
        objs=load_indus(LV) if LV!='im77' else im77_objects()
        groups={'MD':[o['mid'] for o in objs if o['site']=='Mohenjo-daro'],'Harappa':[o['mid'] for o in objs if o['site']=='Harappa'],
                'heldout':[o['mid'] for o in objs if not o['big']],'seals':[o['mid'] for o in objs if o['ot']=='seal'],'tablets':[o['mid'] for o in objs if o['ot']=='tablet']}
        for g,nm in groups.items():
            P(f'\n##### Indus {LV} {g}')
            R[f'indus_{LV}_{g}']=block(nm,f'Indus {LV} {g}')
    # logographic comparators at the held-out n (~300) and at the MD n, with gender as positive control for the bipartition
    nh=R['indus_seq_raw_heldout']['n']; nmd=R['indus_seq_raw_MD']['n']
    P(f'\n##### logographic lists drawn at the held-out n={nh} and the Mohenjo-daro n={nmd} (same code), gendered rows only for the purity figure')
    for nm in ['cn_given','jp_given','vi_given']:
        rows=jl63(nm,2); r=random.Random(51)
        for n,tag in [(nh,'heldout_n'),(nmd,'MD_n')]:
            sub=r.sample(rows,n); P(f'\n  {nm} at n={n}')
            R[f'{nm}_{tag}']=block(seqs(sub),f'{nm} n={n}',gender=[x[1] for x in sub])
    fr=freq_random(indus_names('seq_raw',2),random.Random(7))
    for n,tag in [(nh,'heldout_n'),(nmd,'MD_n')]:
        R[f'indus_freqrand_{tag}']=block(random.Random(2).sample(fr,n),f'Indus freq-random n={n}')
    P('\n=== cycle 3 summary (median [CI]); bipartition z')
    P('  group                          n    '+' '.join(f'{k:>10s}' for k in KEY)+'     init10     fin10   bip_z')
    for k,v in R.items():
        P(f'  {k:30s} {v["n"]:5d} '+' '.join(f'{v["bat"][x][0]:10.3f}' for x in KEY)+f'   {v["cov"]["init10"][0]:7.3f}   {v["cov"]["fin10"][0]:7.3f}   {(v["bip"] or {}).get("z",float("nan")):5.1f}')
    save('loop63_cycle3',R)
