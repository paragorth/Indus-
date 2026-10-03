"""S-DARK-72: is the designation (the deduplicated middle of S-DARK-26/56) a SORTED LIST or a NAME?
Statistics in tools/dark_loop72_common.py. Corpora reused, nothing downloaded:
  Indus middles: tools/dark_loop56.py parser (seq_raw / seq_strong / seq_all / IM77 via bridge + S-DARK-27 proposals)
  logographic given names: data/derived/dark/loop63_corpora; phonetic names, ICD-10, HTS, Proto-Elamite middles:
  data/derived/dark/loop56_corpora; aircraft registrations: data/derived/dark/loop32_corpora.
Cycle 1: battery on Indus (4 levels, bootstrap CIs), within-text shuffle and Markov-2 nulls, sorted / per-name-fixed
         synthetic anchors, every comparator drawn to the Indus n; ladder placement.
Cycle 2: length-matched comparators; Indus variants (NAME only, seals, no numerals, lengths 2-5); does the fitted global
         order track frequency, graphic complexity, Wells catalogue band (graphic class) or numeral class?
Cycle 3: replication: ranking fitted on Mohenjo-daro + Harappa scored on other sites, MD -> Harappa, IM77; Wells vs IM77
         ranking agreement through the bridge.
Usage: python3 tools/dark_loop72.py <1|2|3> [ndraw]
"""
import sys,os,json,random,collections,math,re
CY=int(sys.argv[1]) if len(sys.argv)>1 else 1; ND=int(sys.argv[2]) if len(sys.argv)>2 else 10
sys.argv=['x','0']; sys.path.insert(0,'tools')
import dark_loop56 as L56
from dark_loop72_common import *
import numpy as np
from multiprocessing import Pool
OUT='data/derived/dark/'
LOG=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s,flush=True); LOG.append(s)
def jl(path):
    return [tuple(json.loads(l)['seq']) for l in open(path)]
C56='data/derived/dark/loop56_corpora/'; C63='data/derived/dark/loop63_corpora/'; C32='data/derived/dark/loop32_corpora/'
def load_comp(name):
    if name in ('cn_given','cn_ancient','jp_given','jp_person_given','vi_given'): return jl(C63+name+'.jsonl')
    if name=='aircraft_reg': return jl(C32+'aircraft_reg.jsonl')
    s=jl(C56+name+'.jsonl')
    if name.startswith('ur3'): s=[x for x in s if not any('$' in a or a=='blank' for a in x)]
    if name=='proto_elamite_mid': s=[tuple(a for a in x if not re.match(r'^\d+\(N',a)) for x in s]
    return s
GROUPS=[('logographic names',['cn_given','jp_given','jp_person_given','vi_given','cn_ancient']),
        ('phonetic names',['ur3_names_dedup','ur3_names_elem','ob_names_dedup','linb_personnel_dedup','latin_names_dedup']),
        ('designed codes',['icd10','hts','aircraft_reg']),
        ('Proto-Elamite middles',['proto_elamite_mid'])]
COMPS=[c for _,cs in GROUPS for c in cs]
def fmtrow(lab,s):
    return f'  {lab:34s} n={s["_n"]:5d} k={s["_k"]:5d} L={s["_meanL"]:.2f} '+' '.join(f'{k} {s[k][0]:.3f}' for k in KEYS)
def save(tag,obj):
    json.dump(obj,open(OUT+f'loop72_{tag}.json','w'),indent=1,default=lambda o:float(o) if isinstance(o,np.floating) else str(o))
    open(OUT+f'loop72_{tag}_log.txt','w').write('\n'.join(LOG)+'\n')

def _draw(args):
    name,n,nd,lens=args
    return name,draws(load_comp(name),n,nd,lens)
def _null(args):
    kind,strings,b=args; r=random.Random(300+b)
    s={'shuffle':shuffle_null,'markov2':lambda x,r:markov2(x,r,len(x)),'sorted':sorted_anchor,'pername':pername_anchor}[kind](strings,r)
    return kind,battery(s,b,nsplit=2)
def _boot(args):
    lab,strings,nb=args; return lab,boot(strings,nb)

def indus_levels(field='mid',subset=None,minlen=2):
    return {LV:L56.indus_names(LV,minlen,field,subset) for LV in ['seq_raw','seq_strong','seq_all','im77']}

def ladder(R,ind,keys=KEYS):
    """place Indus between the per-name-fixed anchor (0) and the sorted anchor (1); list every corpus median"""
    for k in keys:
        a0=R['null_pername'][k][0]; a1=R['null_sorted'][k][0]; iv=ind[k]
        sc=lambda v:(v-a0)/(a1-a0) if abs(a1-a0)>1e-9 else float('nan')
        P(f'  {k:10s} Indus {iv[0]:.3f} [{iv[1]:.3f},{iv[2]:.3f}] scaled {sc(iv[0]):.2f} [{sc(iv[1]):.2f},{sc(iv[2]):.2f}] | '+
          ' '.join(f'{c}={R[c][k][0]:.3f}' for c in R if not c.startswith('indus')))
        # which corpora overlap the Indus CI
        ov=[c for c in R if not c.startswith('indus') and iv[1]-0.005<=R[c][k][0]<=iv[2]+0.005]
        P(f'  {"":10s} corpora whose median lies in the Indus CI: {ov}')

if __name__=='__main__' and CY==1:
    P(f'== S-DARK-72 cycle 1: sorted list or name? battery on deduplicated middles; comparators drawn to the Indus n, {ND} draws')
    IND=indus_levels(); n0=len(clean(IND['seq_raw']))
    R={}
    with Pool(3) as pool:
        for lab,res in pool.map(_boot,[(LV,IND[LV],20) for LV in IND]): R['indus_'+lab]=res
        for LV in IND: P(fmtrow('Indus '+LV,R['indus_'+LV]))
        for LV in IND:
            P('    CI: '+'; '.join(f'{k} {t3(R["indus_"+LV][k])}' for k in KEYS))
        base=clean(IND['seq_raw'])
        nul=pool.map(_null,[(k,base,b) for k in ['shuffle','markov2'] for b in range(20)]+[(k,base,b) for k in ['sorted','pername'] for b in range(10)])
        for kind in ['shuffle','markov2','sorted','pername']:
            runs=[x for k,x in nul if k==kind]; s=summarize(runs); s['_n']=runs[0]['n']; s['_k']=runs[0]['k']; s['_meanL']=runs[0]['meanL']
            s['_sd']={k:float(np.nanstd([x[k] for x in runs])) for k in KEYS}; R['null_'+kind]=s
            P(fmtrow('null '+kind+' (seq_raw)',s))
        for name,res in pool.map(_draw,[(c,n0,ND,None) for c in COMPS]):
            R[name]=res; P(fmtrow(name,res))
    P('\n=== z of Indus seq_raw vs the two nulls (null sd)')
    for k in KEYS:
        iv=R['indus_seq_raw'][k][0]
        P(f'  {k:10s} Indus {iv:.3f}  shuffle {R["null_shuffle"][k][0]:.3f} (z {(iv-R["null_shuffle"][k][0])/max(R["null_shuffle"]["_sd"][k],1e-9):.1f})  markov2 {R["null_markov2"][k][0]:.3f} (z {(iv-R["null_markov2"][k][0])/max(R["null_markov2"]["_sd"][k],1e-9):.1f})')
    P('\n=== ladder: scaled = (value - per-name-fixed anchor)/(sorted anchor - per-name-fixed anchor); 0 = name-like free global order, 1 = sorted list')
    ladder(R,R['indus_seq_raw'])
    P('\n=== group ranges (median of draws) per statistic')
    for k in KEYS:
        P(f'  {k:10s} '+' | '.join(f'{g}: {min(R[c][k][0] for c in cs):.3f}..{max(R[c][k][0] for c in cs):.3f}' for g,cs in GROUPS)+
          ' | Indus levels '+' '.join(f'{R["indus_"+LV][k][0]:.3f}' for LV in IND))
    save('cycle1',R)

# ======================= cycle 2 =======================
CPX={int(k):v for k,v in json.load(open('data/derived/sign-complexity.json')).items()}
def track(strings,seed=0):
    """Does the fitted global ranking track frequency, complexity, Wells band, numeral class, mean relative position?"""
    strings=clean(strings); W,elems,A,order,rank=fit_rank(strings,seed)
    freq=collections.Counter(a for s in strings for a in s)
    relp=collections.defaultdict(list)
    for s in strings:
        f=firstorder(s)
        for i,a in enumerate(f): relp[a].append(i/(len(f)-1))
    el=[a for a in elems if freq[a]>=5 and isinstance(a,(int,np.integer)) and a<10000]
    rk=np.array([rank[a]/len(rank) for a in el])
    out=dict(n_el=len(el))
    out['rho_freq']=spearman(rk,[math.log(freq[a]) for a in el])
    ec=[a for a in el if a in CPX]; out['rho_cpx']=spearman([rank[a]/len(rank) for a in ec],[CPX[a] for a in ec])
    out['rho_relpos']=spearman(rk,[np.mean(relp[a]) for a in el])
    band=np.array([a//100 for a in el]); out['eta_band']=eta2(rk,band)
    isnum=np.array([a in L56.NUM for a in el])
    out['auc_num']=auc(rk[isnum],rk[~isnum]) if isnum.any() and (~isnum).any() else float('nan')
    out['_el']=el; out['_rk']=rk.tolist(); out['_band']=band.tolist()
    return out
def eta2(y,g):
    y=np.asarray(y,float); m=y.mean(); ss=((y-m)**2).sum()
    sb=sum(((y[g==b].mean()-m)**2)*(g==b).sum() for b in set(g.tolist()))
    return float(sb/ss) if ss else float('nan')
def auc(a,b):
    a=np.asarray(a); b=np.asarray(b)
    return float(((a[:,None]<b[None,:]).sum()+0.5*(a[:,None]==b[None,:]).sum())/(len(a)*len(b)))  # P(numeral ranked EARLIER)
TK=['rho_freq','rho_cpx','rho_relpos','eta_band','auc_num']
def _track_null(args):
    kind,strings,b=args; r=random.Random(900+b)
    s=shuffle_null(strings,r) if kind=='shuffle' else markov2(strings,r,len(strings))
    return kind,track(s,b)
def _bat(args):
    lab,strings=args; return lab,boot(strings,10)

if __name__=='__main__' and CY==2:
    P(f'== S-DARK-72 cycle 2: length-matched comparators, Indus variants, and what the global order tracks; {ND} draws')
    R1=json.load(open(OUT+'loop72_cycle1.json'))
    IND=indus_levels(); base=clean(IND['seq_raw']); n0=len(base); L0=[len(s) for s in base]
    R={}
    nomid=lambda LV:[tuple(a for a in s if a not in L56.NUM) for s in IND[LV]]
    VAR={}
    for LV in ['seq_raw','seq_strong','seq_all']:
        VAR[f'{LV} NAME only']=L56.indus_names(LV,2,'name')
        VAR[f'{LV} seals']=L56.indus_names(LV,2,'mid',lambda o:o['ot']=='seal')
        VAR[f'{LV} no numerals']=nomid(LV)
        VAR[f'{LV} len 2-5']=[s for s in IND[LV] if 2<=len(s)<=5]
    with Pool(3) as pool:
        for name,res in pool.map(_draw,[(c,n0,ND,L0) for c in COMPS]):
            R[name+'_lenmatch']=res; P(fmtrow(name+' (length-matched)',res))
        for lab,res in pool.map(_bat,list(VAR.items())):
            R['indus '+lab]=res; P(fmtrow('Indus '+lab,res)); P('    CI: '+'; '.join(f'{k} {t3(res[k])}' for k in KEYS))
        P('\n=== what does the global order track? (fitted ranking position 0 = first; elements >= 5 tokens)')
        T={}
        for LV in ['seq_raw','seq_strong','seq_all','im77']:
            t=track(IND[LV]); T[LV]=t
            nul=pool.map(_track_null,[(k,clean(IND[LV]),b) for k in ['shuffle','markov2'] for b in range(20)])
            for kind in ['shuffle','markov2']:
                runs=[x for k,x in nul if k==kind]
                T[LV+'_'+kind]={k:(q([x[k] for x in runs],0.5),q([x[k] for x in runs],0.025),q([x[k] for x in runs],0.975)) for k in TK}
            # label permutation null for band / numeral (ranking fixed, labels permuted among elements)
            r=random.Random(5); rk=np.array(t['_rk']); band=np.array(t['_band']); el=t['_el']
            pe=[eta2(rk,np.array(r.sample(band.tolist(),len(band)))) for _ in range(1000)]
            T[LV+'_bandperm_p']=float(np.mean([x>=t['eta_band'] for x in pe])); T[LV+'_bandperm_q95']=q(pe,0.95)
            P(f'  {LV}: {t["n_el"]} elements; '+'; '.join(f'{k} {t[k]:.3f} (shuffle {T[LV+"_shuffle"][k][0]:.3f} [{T[LV+"_shuffle"][k][1]:.3f},{T[LV+"_shuffle"][k][2]:.3f}], markov2 {T[LV+"_markov2"][k][0]:.3f} [{T[LV+"_markov2"][k][1]:.3f},{T[LV+"_markov2"][k][2]:.3f}])' for k in TK))
            P(f'      band eta2 label-permutation P = {T[LV+"_bandperm_p"]:.3f} (95th {T[LV+"_bandperm_q95"]:.3f}); mean rank by Wells band: '+
              ', '.join(f'{b*100}s {np.mean(rk[band==b]):.2f} (n={int((band==b).sum())})' for b in sorted(set(band.tolist()))))
            if LV=='seq_raw':
                o=sorted(zip(rk.tolist(),el)); P('      first 25 in the fitted order: '+' '.join(f'W{a}' for _,a in o[:25]))
                P('      last 25: '+' '.join(f'W{a}' for _,a in o[-25:]))
        R['track']={k:({kk:vv for kk,vv in v.items() if not kk.startswith('_')} if isinstance(v,dict) else v) for k,v in T.items()}
    P('\n=== comparators: rho_freq (rank vs log frequency) cycle-1 draws; length-matched ladder (median)')
    for c in COMPS: P(f'  {c:22s} rho_freq {R1[c]["rho_freq"][0]:.3f}  lenmatch: '+' '.join(f'{k} {R[c+"_lenmatch"][k][0]:.3f}' for k in KEYS))
    P('\n=== length-matched ladder vs Indus seq_raw CI (cycle 1)')
    for k in KEYS:
        iv=R1['indus_seq_raw'][k]
        P(f'  {k:10s} Indus {iv[0]:.3f} [{iv[1]:.3f},{iv[2]:.3f}] | '+' | '.join(f'{g}: {min(R[c+"_lenmatch"][k][0] for c in cs):.3f}..{max(R[c+"_lenmatch"][k][0] for c in cs):.3f}' for g,cs in GROUPS))
    save('cycle2',R)

# ======================= cycle 3 =======================
def transfer(train,test,nb=200,seed=0):
    train=clean(train); test=clean(test)
    WA,eA,AA,oA,rA=fit_rank(train,seed); seenA=set(frozenset(k) for k in WA)
    per=[]
    for s in test:
        f=[a for a in firstorder(s) if a in rA]; gs=ns=gu=nu=0
        for i in range(len(f)):
            for j in range(i+1,len(f)):
                ok=rA[f[i]]<rA[f[j]]
                if frozenset((f[i],f[j])) in seenA: gs+=ok; ns+=1
                else: gu+=ok; nu+=1
        per.append((gs,ns,gu,nu))
    per=np.array(per,float)
    def st(p): return (p[:,0].sum()/max(p[:,1].sum(),1), p[:,2].sum()/max(p[:,3].sum(),1))
    full=st(per); r=np.random.default_rng(seed); bs=[st(per[r.integers(0,len(per),len(per))]) for _ in range(nb)]
    return dict(n_train=len(train),n_test=len(test),seen=full[0],seen_ci=(q([b[0] for b in bs],0.025),q([b[0] for b in bs],0.975)),
                unseen=full[1],unseen_ci=(q([b[1] for b in bs],0.025),q([b[1] for b in bs],0.975)),n_seen=int(per[:,1].sum()),n_unseen=int(per[:,3].sum()))
def ptr(lab,t): P(f'  {lab:44s} train {t["n_train"]:5d} test {t["n_test"]:5d}: seen pairs {t["seen"]:.3f} [{t["seen_ci"][0]:.3f},{t["seen_ci"][1]:.3f}] (n={t["n_seen"]}); unseen {t["unseen"]:.3f} [{t["unseen_ci"][0]:.3f},{t["unseen_ci"][1]:.3f}] (n={t["n_unseen"]})')

if __name__=='__main__' and CY==3:
    P('== S-DARK-72 cycle 3: does the order fitted at one place predict another? (held-out sites, MD -> Harappa, IM77, comparators)')
    R={}
    for LV in ['seq_raw','seq_strong','seq_all','im77']:
        big=L56.indus_names(LV,2,'mid',lambda o:o['big']); oth=L56.indus_names(LV,2,'mid',lambda o:not o['big'])
        md=L56.indus_names(LV,2,'mid',lambda o:o['site']=='Mohenjo-daro'); ha=L56.indus_names(LV,2,'mid',lambda o:o['site']=='Harappa')
        othset=set(clean(oth)); bigc=[s for s in clean(big) if s not in othset]
        R[LV+' MD+H -> other sites']=t=transfer(bigc,oth); ptr(f'{LV} MD+H -> other sites',t)
        R[LV+' MD -> Harappa']=t=transfer(md,[s for s in clean(ha) if s not in set(clean(md))]); ptr(f'{LV} MD -> Harappa (new strings)',t)
        R[LV+' Harappa -> MD']=t=transfer(ha,[s for s in clean(md) if s not in set(clean(ha))]); ptr(f'{LV} Harappa -> MD (new strings)',t)
        r=random.Random(3); R[LV+' MD+H shuffled -> other']=t=transfer(shuffle_null(bigc,r),oth); ptr(f'{LV} control: MD+H order shuffled -> other',t)
        r=random.Random(4); R[LV+' MD+H markov2 -> other']=t=transfer(markov2(bigc,r,len(bigc)),oth); ptr(f'{LV} control: Markov-2 clone of MD+H -> other',t)
    # Wells vs IM77 global ranking through the bridge
    P('\n=== Wells (seq_raw) vs IM77 fitted rankings on shared elements (>= 5 tokens in both); transcription-robust, not replication (S312)')
    def rk_of(s):
        s=clean(s); W,elems,A,order,rank=fit_rank(s); fr=collections.Counter(a for x in s for a in x); return rank,fr,len(rank)
    rw,fw,nw=rk_of(L56.indus_names('seq_raw',2)); ri,fi,ni=rk_of(L56.indus_names('im77',2))
    sh=[a for a in rw if a in ri and fw[a]>=5 and fi[a]>=5]
    rho=spearman([rw[a]/nw for a in sh],[ri[a]/ni for a in sh]); R['wells_im77_rho']=rho; R['wells_im77_n']=len(sh)
    nr=[]
    for b in range(20):
        r=random.Random(70+b); rs,fs,ns=rk_of(shuffle_null(clean(L56.indus_names('im77',2)),r)); nr.append(spearman([rw[a]/nw for a in sh],[rs[a]/ns for a in sh if a in rs]) if all(a in rs for a in sh) else float('nan'))
    R['wells_im77_null']=(q(nr,0.5),q(nr,0.025),q(nr,0.975))
    P(f'  Spearman {rho:.3f} on {len(sh)} elements; IM77 order shuffled: {R["wells_im77_null"][0]:.3f} [{R["wells_im77_null"][1]:.3f},{R["wells_im77_null"][2]:.3f}]')
    # held-out halves within each level (seen pairs only vs unseen) also reported in cycle 1 (ho_seen / ho_unseen)
    P('\n=== comparator transfers (same code): Linear B Knossos -> Pylos; Chinese male -> female given names; Japanese given F -> M; ICD-10 chapters A-M -> N-Z; HTS chapters 01-49 -> 50-99; Proto-Elamite random halves')
    def g63(name,g): return [tuple(json.loads(l)['seq']) for l in open(C63+name+'.jsonl') if json.loads(l).get('g')==g]
    rr=random.Random(9)
    comp=[('Linear B KN -> PY',jl(C56+'linb_personnel_KN.jsonl'),jl(C56+'linb_personnel_PY.jsonl')),
          ('Chinese given M -> F',rr.sample(g63('cn_given','M'),3000),rr.sample(g63('cn_given','F'),3000)),
          ('Japanese given F -> M',rr.sample(g63('jp_given','F'),3000),rr.sample(g63('jp_given','M'),3000)),
          ('ICD-10 A-M -> N-Z',[s for s in rr.sample(load_comp('icd10'),20000) if s[0]<'N'][:3000],[s for s in rr.sample(load_comp('icd10'),20000) if s[0]>='N'][:3000]),
          ('HTS 01-49 -> 50-99',[s for s in rr.sample(load_comp('hts'),20000) if s[0]<'50'][:3000],[s for s in rr.sample(load_comp('hts'),20000) if s[0]>='50'][:3000])]
    pe=clean(load_comp('proto_elamite_mid')); rr.shuffle(pe); comp.append(('Proto-Elamite random halves',pe[:len(pe)//2],pe[len(pe)//2:]))
    ub=clean(load_comp('ur3_names_dedup')); rr.shuffle(ub); comp.append(('Ur III names random halves',ub[:len(ub)//2],ub[len(ub)//2:]))
    lt=clean(load_comp('latin_names_dedup')); rr.shuffle(lt); comp.append(('Latin names random halves',lt[:len(lt)//2],lt[len(lt)//2:]))
    for lab,a,b in comp:
        R['comp '+lab]=t=transfer(a,b); ptr(lab,t)
    save('cycle3',R)

# ======================= cycle 4 =======================
def triads(strings,minn=2,maj=2/3):
    """Decisive pairs: co-occur in >= minn strings with majority share >= maj. Among triads whose three pairs are all
    decisive, share that are cyclic (a>b, b>c, c>a). Sorted list 0; a random tournament 0.25."""
    strings=clean(strings); W=pair_table(strings); D=collections.defaultdict(set); dec={}
    for (a,b),x in W.items():
        y=W.get((b,a),0)
        if x+y>=minn and x/(x+y)>=maj: D[a].add(b); dec[frozenset((a,b))]=(a,b)
    nb=collections.defaultdict(set)
    for k in dec:
        a,b=tuple(k); nb[a].add(b); nb[b].add(a)
    tot=cyc=0; ex=collections.Counter()
    for a in nb:
        for b in nb[a]:
            if str(b)<=str(a): continue
            for c in nb[a]&nb[b]:
                if str(c)<=str(b): continue
                tot+=1
                win=collections.Counter(dec[frozenset(p)][0] for p in ((a,b),(b,c),(a,c)))
                if max(win.values())==1:
                    cyc+=1; ex[(a,b,c)]=min(W.get((x,y),0)+W.get((y,x),0) for x,y in ((a,b),(b,c),(a,c)))
    return dict(n_dec=len(dec),triads=tot,cyc=cyc/tot if tot else float('nan'),top=ex.most_common(8))
def _tri_null(args):
    kind,strings,b=args; r=random.Random(1300+b)
    s={'shuffle':shuffle_null,'markov2':lambda x,r:markov2(x,r,len(x)),'sorted':sorted_anchor,'pername':pername_anchor}[kind](strings,r)
    return kind,triads(s)
def _tri_draw(args):
    name,n,lens,b=args; r=random.Random(5000+b); pool=clean(load_comp(name))
    sub=length_match(pool,lens,n,r) if len(pool)>n else pool
    return name,triads(sub)

if __name__=='__main__' and CY==4:
    P('== S-DARK-72 cycle 4: transitivity of decisive triads (pairs in >= 2 strings, majority >= 2/3) and the NAME-only frequency effect')
    R={}
    IND=indus_levels(); base=clean(IND['seq_raw']); n0=len(base); L0=[len(s) for s in base]
    with Pool(3) as pool:
        for LV in IND:
            t=triads(IND[LV]); R['indus_'+LV]=t
            P(f'  Indus {LV:10s}: decisive pairs {t["n_dec"]}, decisive triads {t["triads"]}, cyclic share {t["cyc"]:.3f}')
        tn=triads(L56.indus_names('seq_raw',2,'name')); R['indus_name']=tn; P(f'  Indus seq_raw NAME only: {tn["n_dec"]} / {tn["triads"]} / {tn["cyc"]:.3f}')
        P('    strongest cyclic triads (min pair support): '+'; '.join(f'W{a}-W{b}-W{c} ({n})' for (a,b,c),n in t['top'][:0]+R['indus_seq_raw']['top']))
        nul=pool.map(_tri_null,[(k,base,b) for k in ['shuffle','markov2','sorted','pername'] for b in range(10)])
        for kind in ['shuffle','markov2','sorted','pername']:
            v=[x['cyc'] for k,x in nul if k==kind]; tr=[x['triads'] for k,x in nul if k==kind]
            R['null_'+kind]=(q(v,0.5),q(v,0.025),q(v,0.975)); P(f'  null {kind:8s}: cyclic share {q(v,0.5):.3f} [{q(v,0.025):.3f},{q(v,0.975):.3f}], triads ~{int(np.median(tr))}')
        res=pool.map(_tri_draw,[(c,n0,L0,b) for c in COMPS for b in range(5)])
        for c in COMPS:
            v=[x['cyc'] for k,x in res if k==c]; tr=[x['triads'] for k,x in res if k==c]
            R[c]=(q(v,0.5),q(v,0.025),q(v,0.975),int(np.median(tr))); P(f'  {c:22s} (length-matched): cyclic share {q(v,0.5):.3f} [{q(v,0.025):.3f},{q(v,0.975):.3f}] of ~{int(np.median(tr))} triads')
        # Indus bootstrap CI
        bs=[triads(random.Random(77+b).sample(base,int(0.8*len(base))))['cyc'] for b in range(20)]
        md=q(bs,0.5); R['indus_ci']=(R['indus_seq_raw']['cyc']+q(bs,0.025)-md,R['indus_seq_raw']['cyc']+q(bs,0.975)-md)
        P(f'  Indus seq_raw cyclic share CI [{R["indus_ci"][0]:.3f},{R["indus_ci"][1]:.3f}]')
        P('\n=== NAME-only field: does the global order track frequency? (shuffle / Markov-2 nulls, 20x)')
        for LV in ['seq_raw','seq_strong','seq_all','im77']:
            nm=clean(L56.indus_names(LV,2,'name')); t=track(nm)
            nul=pool.map(_track_null,[(k,nm,b) for k in ['shuffle','markov2'] for b in range(20)])
            s={kind:[x for k,x in nul if k==kind] for kind in ['shuffle','markov2']}
            P(f'  {LV} NAME: '+'; '.join(f'{k} {t[k]:.3f} (shuffle {q([x[k] for x in s["shuffle"]],0.5):.3f} [{q([x[k] for x in s["shuffle"]],0.025):.3f},{q([x[k] for x in s["shuffle"]],0.975):.3f}], markov2 {q([x[k] for x in s["markov2"]],0.5):.3f} [{q([x[k] for x in s["markov2"]],0.025):.3f},{q([x[k] for x in s["markov2"]],0.975):.3f}])' for k in ['rho_freq','rho_cpx','eta_band']))
            R['name_track_'+LV]={k:t[k] for k in TK}
    save('cycle4',R)
