"""S-DARK-22: FIXED-WIDTH FIELDS.  Does the Indus text behave like a designed record format (fields at fixed offsets
from one edge) or like language (no fixed offsets)?
  cycle 1 (a) start- vs end-anchoring entropy per sign, vs within-text shuffle null; controls Ur III legends, Linear B lines
  cycle 2 (b) offset grammar (P(sign | offset from end/start, length)) vs order-2 Markov vs slot-class grammar, held out
  cycle 3 (c) slot-width signatures before the closer vs a generative null; (d) hole test: placeholder signs
Usage: python3 tools/dark_loop22.py <cycle 1|2|3> <seq_raw|seq_strong|seq_all> [nperm]
stdout is redirected by the caller to data/derived/dark/loop22_cycle<N>_<level>.txt
"""
import json,sys,random,collections,math,re
C=json.load(open('data/derived/merged-corpus-canonical.json'))
BR=json.load(open('data/derived/bridge_extended.json'))
SP='/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
def M(w):
    m=BR.get(str(w)); return f'W{w}(M{"/".join(map(str,m))})' if m else f'W{w}'
CY=int(sys.argv[1]); LV=sys.argv[2]; NP=int(sys.argv[3]) if len(sys.argv)>3 else 300
rnd=random.Random(22)
def otype(t):
    t=t.split(':')[0]
    return {'SEAL':'seal','TAB':'tablet','POT':'pot','TAG':'sealing'}.get(t,'other')
OBJ=[]
for r in C:
    s=r[LV]
    if not s or len(s)<2 or r['complete']!='Y' or r['dir.'].strip()=='-': continue
    OBJ.append(dict(cisi=r['cisi'],site=r['site'],ot=otype(r['type']),seq=list(s),big=r['site'] in('Mohenjo-daro','Harappa')))
print(f'== S-DARK-22 cycle {CY} level {LV} nperm {NP}; objects {len(OBJ)} (complete, direction recorded, >=2 signs)')
print('   by type',dict(collections.Counter(o["ot"] for o in OBJ)),' big-city share %.2f'%(sum(o['big'] for o in OBJ)/len(OBJ)))

# ---------------- controls ----------------
def load_ur3():
    legs=json.load(open(SP+'ur3_legends.json')); seen=set(); out=[]
    for lg in legs:
        flat=tuple(x for ln in lg['lines'] for x in ln)
        if len(flat)<2 or flat in seen: continue
        seen.add(flat); out.append(dict(site='ur3',ot='legend',seq=list(flat),big=rnd.random()<0.7))
    return out
def load_linb():
    out=[]; seen=set()
    for l in open('other-scripts/linear-a/data/damos_items.jsonl'):
        d=json.loads(l); c=d.get('content') or ''
        for ln in c.split('\n'):
            m=re.match(r'\s*\.(\d+[a-z]?)\s+(.*)',ln)
            if not m: continue
            toks=[]
            for t in re.split(r'\s+',m.group(2).strip()):
                t=t.strip('[],')
                if not t or t in ('/','vac.','vacat','vest.') or t.startswith('vac'): continue
                if re.fullmatch(r'\d+',t): t='NUM'
                elif re.fullmatch(r"'?[a-z0-9*]+(-[a-z0-9*]+)+'?",t): t='w:'+t.strip("'")
                elif re.fullmatch(r'[A-Z][A-Z0-9*+]*',t): t='I:'+t
                elif t in ('S','V','T','Z','M','N','P','Q','L'): t='U:'+t
                else: continue
                toks.append(t)
            if len(toks)>=2 and tuple(toks) not in seen:
                seen.add(tuple(toks)); out.append(dict(site='linb',ot='line',seq=toks,big=rnd.random()<0.7))
    return out

# ---------------- frame parser (S310 parse_all.py, as in dark_loop19) ----------------
OPEN={817,861,820,920,692}; MARK={2,60}; MJAR={741,742,745}; SUF={400,90}; CL=[740,520,151,156,527,226,617,154,158,236,700]
FISH={235,240,233,231,220}; NUM={1,3,4,5,16,17,18,31,32,33,34,55,56}
left=collections.defaultdict(collections.Counter)
for o in OBJ:
    s=o['seq'][:]
    while len(s)>1 and s[-1] in SUF: s.pop()
    if len(s)>=2 and s[-1] in CL: left[s[-1]][s[-2]]+=1
QUAL={}
for c,cnt in left.items():
    tot=sum(cnt.values()); acc=0; q=set()
    for a,n in cnt.most_common():
        if acc/tot>=0.6: break
        q.add(a); acc+=n
    QUAL[c]=q
def parse(s):
    lab=['NAME']*len(s); i=0; j=len(s)
    if s[0] in OPEN:
        lab[0]='OPENER'; i=1
        if len(s)>1 and s[1] in MARK:
            lab[1]='MARKER'; i=2
            if s[0]==920 and len(s)>2 and s[2] in MJAR: lab[2]='MARKER'; i=3
    while j-1>i and s[j-1] in SUF and j>=2 and (s[j-2] in CL or s[j-2] in SUF): lab[j-1]='SUFFIX'; j-=1
    if j-1>=i and s[j-1] in CL:
        c=s[j-1]; lab[j-1]='CLOSER'; j-=1
        if c==520:
            if j-2>=i and s[j-1]==33 and s[j-2] in (705,706): lab[j-1]=lab[j-2]='TITLE'; j-=2
            while j-1>=i and s[j-1] in FISH: lab[j-1]='TITLE'; j-=1
        elif c==740:
            if j-1>=i and s[j-1]==100: lab[j-1]='TITLE'; j-=1
            if j-1>=i and s[j-1] in QUAL.get(c,()): lab[j-1]='TITLE'; j-=1
        elif j-1>=i and s[j-1] in QUAL.get(c,()): lab[j-1]='TITLE'; j-=1
        if j-1>=i and s[j-1] in NUM and lab[j]=='TITLE': lab[j-1]='TITLE'; j-=1
    for k in range(i,j-1):
        if s[k] in NUM and lab[k]=='NAME' and lab[k+1]=='NAME': lab[k]=lab[k+1]='COUNT'
    for k in range(i,j):
        if s[k] in NUM and lab[k]=='NAME': lab[k]='COUNT'
    return lab
for o in OBJ: o['lab']=parse(o['seq'])
slotvotes=collections.defaultdict(collections.Counter)
for o in OBJ:
    for a,l in zip(o['seq'],o['lab']): slotvotes[a][l]+=1
SLOT={a:c.most_common(1)[0][0] for a,c in slotvotes.items()}
def q(null,p): null=sorted(null); return null[min(len(null)-1,int(p*len(null)))]
def pval(obs,null,side='hi'):
    n=len(null)
    if side=='hi': return (sum(1 for x in null if x>=obs)+1)/(n+1)
    return (sum(1 for x in null if x<=obs)+1)/(n+1)
def H(cnt):
    n=sum(cnt.values()); return -sum(c/n*math.log2(c/n) for c in cnt.values() if c)

# =========================== (a) anchoring entropy ===========================
def anchoring(objs,label,nperm,minn=30,top=25):
    ind=label.startswith('Indus'); NM=(lambda a:M(a)) if ind else (lambda a:str(a))
    toks=collections.Counter(a for o in objs for a in o['seq'])
    signs=[a for a,n in toks.items() if n>=minn]
    def stats(seqs):
        ps=collections.defaultdict(collections.Counter); pe=collections.defaultdict(collections.Counter)
        for s in seqs:
            L=len(s)
            for k,a in enumerate(s):
                if a in sset: ps[a][k]+=1; pe[a][L-1-k]+=1
        return {a:(H(ps[a]),H(pe[a])) for a in signs}
    sset=set(signs)
    obs=stats([o['seq'] for o in objs])
    r=random.Random(5); null=collections.defaultdict(list)
    for _ in range(nperm):
        sh=[]
        for o in objs:
            s=o['seq'][:]; r.shuffle(s); sh.append(s)
        st=stats(sh)
        for a in signs: null[a].append(st[a])
    rows=[]
    for a in signs:
        hs,he=obs[a]; ns=[x[0] for x in null[a]]; ne=[x[1] for x in null[a]]; nd=[x[1]-x[0] for x in null[a]]
        d=he-hs   # negative = tighter from the END
        rows.append(dict(sign=a,n=toks[a],hs=hs,he=he,d=d,gs=q(ns,.5)-hs,ge=q(ne,.5)-he,
                         p_end=pval(d,nd,'lo'),p_start=pval(d,nd,'hi'),slot=SLOT.get(a,'-') if ind else '-',z=(d-sum(nd)/len(nd))/max(1e-9,(sum((x-sum(nd)/len(nd))**2 for x in nd)/len(nd))**.5)))
    # summary
    nend=sum(1 for x in rows if x['d']<0); nstart=sum(1 for x in rows if x['d']>0)
    alpha=0.05/(2*len(rows))
    import statistics
    from math import erf,sqrt
    zcrit=None
    # Bonferroni via normal approximation of the null difference (permutation P floors at 1/(nperm+1))
    def z2p(z): return 0.5*(1-erf(abs(z)/sqrt(2)))
    sig_end=[x for x in rows if x['z']<0 and z2p(x['z'])<alpha]; sig_start=[x for x in rows if x['z']>0 and z2p(x['z'])<alpha]
    tot_gs=sum(x['gs']*x['n'] for x in rows)/sum(x['n'] for x in rows); tot_ge=sum(x['ge']*x['n'] for x in rows)/sum(x['n'] for x in rows)
    print(f'\n-- (a) ANCHORING [{label}] texts {len(objs)}, signs with >= {minn} tokens: {len(rows)}; '
          f'tighter from END {nend}, from START {nstart}; Bonferroni alpha {alpha:.2e}: END-anchored {len(sig_end)}, START-anchored {len(sig_start)}')
    print(f'   token-weighted entropy gain over shuffle null (bits): from start {tot_gs:.3f}, from end {tot_ge:.3f}; ratio end/start {tot_ge/max(1e-9,tot_gs):.2f}')
    mean_abs_d=sum(abs(x['d']) for x in rows)/len(rows)
    print(f'   mean |H_end - H_start| {mean_abs_d:.3f} bits (null ~0 by construction)')
    rows.sort(key=lambda x:x['d'])
    print(f'   strongest END-anchored (H_end - H_start, bits; H_end; gain_end; n; slot):')
    for x in rows[:top]:
        print(f'     {NM(x["sign"]):>22s} d {x["d"]:+.2f}  H_end {x["he"]:.2f} gain {x["ge"]:.2f}  n {x["n"]:4d}  {x["slot"]}  z {x["z"]:+.1f}')
    print(f'   strongest START-anchored:')
    for x in rows[-top:][::-1]:
        print(f'     {NM(x["sign"]):>22s} d {x["d"]:+.2f}  H_start {x["hs"]:.2f} gain {x["gs"]:.2f}  n {x["n"]:4d}  {x["slot"]}  z {x["z"]:+.1f}')
    print(f'   (sig counts use the normal approximation to the permutation null; z shown below)')
    if ind:
        by=collections.defaultdict(list)
        for x in rows: by[x['slot']].append(x)
        print('   by majority frame slot: slot, n signs, mean d, share end-tighter:')
        for sl,xs in sorted(by.items()):
            print(f'     {sl:8s} {len(xs):3d}  mean d {sum(x["d"] for x in xs)/len(xs):+.3f}  end-tighter {sum(1 for x in xs if x["d"]<0)}/{len(xs)}')
    return rows,dict(nend=nend,nstart=nstart,sig_end=len(sig_end),sig_start=len(sig_start),gs=tot_gs,ge=tot_ge,mad=mean_abs_d)

# =========================== (b) predictive models ===========================
class WB:
    """Witten-Bell backoff over a list of context functions (most specific first). Predicts s given (seq, i, L)."""
    def __init__(self,ctxfns,V):
        self.f=ctxfns; self.V=V; self.cnt=[collections.defaultdict(collections.Counter) for _ in ctxfns]; self.uni=collections.Counter()
    def fit(self,texts):
        for s in texts:
            L=len(s)
            for i in range(L):
                self.uni[s[i]]+=1
                for k,f in enumerate(self.f): self.cnt[k][f(s,i,L)][s[i]]+=1
        self.N=sum(self.uni.values())
    def p(self,s,i,L):
        pr=(self.uni[s[i]]+0.5)/(self.N+0.5*self.V)
        for k in range(len(self.f)-1,-1,-1):
            c=self.cnt[k][self.f[k](s,i,L)]
            if not c: continue
            n=sum(c.values()); t=len(c)
            pr=(c[s[i]]+t*pr)/(n+t)
        return pr
    def nll(self,texts):
        tot=0;n=0;per=[]
        for s in texts:
            L=len(s); x=0
            for i in range(L): lp=-math.log(self.p(s,i,L)); tot+=lp; x+=lp
            n+=L; per.append(x)
        return tot/n,per
P1=lambda s,i,L: s[i-1] if i>0 else '<s>'
P2=lambda s,i,L: (s[i-2] if i>1 else '<s>', s[i-1] if i>0 else '<s>')
def mk_models(slot):
    cls=lambda a: slot.get(a,'NAME')
    return {
     'Markov-2'            : lambda V: WB([P1,P2],V),
     'Markov-2 + L,rem'    : lambda V: WB([P1,P2,lambda s,i,L:(P2(s,i,L),L-1-i)],V),
     'Offset-from-END'     : lambda V: WB([lambda s,i,L:L-1-i, lambda s,i,L:(L-1-i,L)],V),
     'Offset-from-START'   : lambda V: WB([lambda s,i,L:i, lambda s,i,L:(i,L)],V),
     'Offset-both (i,L)'   : lambda V: WB([lambda s,i,L:L-1-i, lambda s,i,L:(i,L)],V),
     'Slot-class bigram'   : lambda V: WB([lambda s,i,L: cls(s[i-1]) if i>0 else '<s>'],V),
     'Slot-class + END off': lambda V: WB([lambda s,i,L: cls(s[i-1]) if i>0 else '<s>', lambda s,i,L:(cls(s[i-1]) if i>0 else '<s>',L-1-i)],V),
     'Markov-1'            : lambda V: WB([P1],V),
     'Markov-1 + END off'  : lambda V: WB([P1,lambda s,i,L:(P1(s,i,L),L-1-i)],V),
    }
def eval_models(objs,label,slot,folds=5):
    V=len(set(a for o in objs for a in o['seq']))
    models=mk_models(slot)
    res={}
    # held-out: fit big-city (or random 70%) test rest
    fit=[o['seq'] for o in objs if o['big']]; test=[o['seq'] for o in objs if not o['big']]
    r=random.Random(3); idx=list(range(len(objs))); r.shuffle(idx)
    print(f'\n-- (b) PREDICTIVE MODELS [{label}] V {V}; held-out split fit {len(fit)} / test {len(test)} texts; plus {folds}-fold on all {len(objs)}')
    print(f'   {"model":22s} {"held-out nats/tok":>18s} {"5-fold nats/tok":>16s}')
    per_held={}
    for name,mk in models.items():
        m=mk(V); m.fit(fit); ho,per=m.nll(test); per_held[name]=per
        cv=0;n=0
        for f in range(folds):
            te=[objs[k]['seq'] for j,k in enumerate(idx) if j%folds==f]; tr=[objs[k]['seq'] for j,k in enumerate(idx) if j%folds!=f]
            m=mk(V); m.fit(tr); x,_=m.nll(te); cv+=x*sum(len(t) for t in te); n+=sum(len(t) for t in te)
        res[name]=(ho,cv/n)
        print(f'   {name:22s} {ho:18.3f} {cv/n:16.3f}')
    # paired bootstrap: offset-END vs Markov-2, slot vs Markov-2, on held-out texts
    def boot(a,b,nb=1000):
        n=len(a); rr=random.Random(9); ds=[]
        for _ in range(nb):
            ii=[rr.randrange(n) for _ in range(n)]; ds.append(sum(a[k]-b[k] for k in ii)/sum(len(test[k]) for k in ii))
        return q(ds,.025),q(ds,.975)
    for a,b in (('Offset-from-END','Markov-2'),('Offset-from-END','Slot-class bigram'),('Markov-2 + L,rem','Markov-2'),('Offset-from-END','Offset-from-START'),('Slot-class + END off','Slot-class bigram')):
        lo,hi=boot(per_held[a],per_held[b]); print(f'   held-out difference {a} - {b}: 95% CI [{lo:+.3f}, {hi:+.3f}] nats/tok')
    return res

def per_sign_gain(objs,slot,label):
    """Per sign: held-out log-prob under Offset-from-END minus under Markov-2 (positive = offset predicts it better)."""
    V=len(set(a for o in objs for a in o['seq'])); models=mk_models(slot)
    fit=[o['seq'] for o in objs if o['big']]; test=[o['seq'] for o in objs if not o['big']]
    mE=models['Offset-from-END'](V); mE.fit(fit); mM=models['Markov-2'](V); mM.fit(fit); mS=models['Offset-from-START'](V); mS.fit(fit)
    g=collections.defaultdict(float); gS=collections.defaultdict(float); n=collections.Counter()
    for s in test:
        L=len(s)
        for i in range(L):
            g[s[i]]+=math.log(mE.p(s,i,L))-math.log(mM.p(s,i,L)); gS[s[i]]+=math.log(mS.p(s,i,L))-math.log(mM.p(s,i,L)); n[s[i]]+=1
    rows=[(a,n[a],g[a]/n[a],gS[a]/n[a]) for a in n if n[a]>=15]
    rows.sort(key=lambda x:-x[2])
    print(f'\n   per-sign held-out gain of Offset-from-END over Markov-2 (nats/token; >= 15 held-out tokens); also START-offset gain:')
    for a,k,ge,gs in rows[:15]: print(f'     {M(a) if label=="Indus" else a:>22s} n {k:4d} END-gain {ge:+.3f}  START-gain {gs:+.3f}  slot {slot.get(a,"-")}')
    print('   ... worst for the offset model:')
    for a,k,ge,gs in rows[-8:]: print(f'     {M(a) if label=="Indus" else a:>22s} n {k:4d} END-gain {ge:+.3f}  START-gain {gs:+.3f}  slot {slot.get(a,"-")}')
    print(f'   signs better predicted by END offset than Markov-2: {sum(1 for r in rows if r[2]>0)} / {len(rows)}')

# =========================== (c) slot-width signatures ===========================
def widths(o):
    lab=o['lab']; w=collections.Counter(lab)
    return (w['OPENER'],w['MARKER'],w['NAME'],w['COUNT'],w['TITLE'],w['CLOSER'],w['SUFFIX'])
def sig_table(objs,label):
    closed=[o for o in objs if 'CLOSER' in o['lab']]
    print(f'\n-- (c) FIELD WIDTHS [{label}]: texts with a closer {len(closed)} of {len(objs)}')
    for sl,ix in (('OPENER',0),('MARKER',1),('NAME',2),('COUNT',3),('TITLE',4),('SUFFIX',6)):
        c=collections.Counter(widths(o)[ix] for o in closed); n=len(closed)
        print(f'   {sl:7s} width: '+'  '.join(f'{k}:{v/n:.3f}' for k,v in sorted(c.items())))
    full=collections.Counter(widths(o) for o in closed)
    print(f'   distinct signatures (O,M,N,C,T,Cl,S): {len(full)}; top 12 of {len(closed)}:')
    for k,v in full.most_common(12): print(f'     {k}  {v}  ({v/len(closed):.3f})')
    return closed,full

def generative_null(objs,nrep,label):
    """Slot-class bigram generative model (sign classes fixed per sign, class bigram + emission), same length distribution, parsed with the same parser.
    Compare entropy of the width signature and the TITLE/COUNT width distributions."""
    cls=lambda a: SLOT.get(a,'NAME')
    cb=collections.defaultdict(collections.Counter); em=collections.defaultdict(collections.Counter)
    for o in objs:
        p='<s>'
        for a in o['seq']: c=cls(a); cb[p][c]+=1; em[c][a]+=1; p=c
    def gen(L,r):
        out=[];p='<s>'
        for _ in range(L):
            ks,ws=zip(*cb[p].items()); c=r.choices(ks,ws)[0]; ks2,ws2=zip(*em[c].items()); a=r.choices(ks2,ws2)[0]; out.append(a); p=c
        return out
    Ls=[len(o['seq']) for o in objs]
    closed=[o for o in objs if 'CLOSER' in o['lab']]
    obs_full=collections.Counter(widths(o) for o in closed)
    def summ(cl):
        full=collections.Counter(widths(o) for o in cl)
        tw=collections.Counter(widths(o)[4] for o in cl); cw=collections.Counter(widths(o)[3] for o in cl)
        n=len(cl)
        return dict(Hsig=H(full),ndist=len(full),top1=full.most_common(1)[0][1]/n,
                    title1=tw[1]/n,title0=tw[0]/n,title2=sum(v for k,v in tw.items() if k>=2)/n,
                    count1=cw[1]/n,count0=cw[0]/n,count2=sum(v for k,v in cw.items() if k>=2)/n,
                    title_fixed=(tw[0]+tw[1])/n, nclosed=n)
    obs=summ(closed)
    r=random.Random(11); null=collections.defaultdict(list)
    for _ in range(nrep):
        fake=[]
        for L in Ls:
            s=gen(L,r); fake.append(dict(seq=s,lab=parse(s)))
        fc=[o for o in fake if 'CLOSER' in o['lab']]
        if len(fc)<20: continue
        for k,v in summ(fc).items(): null[k].append(v)
    print(f'\n   generative null (slot-class bigram, {nrep} corpora): statistic, observed, null median [5%,95%], P')
    for k in ('nclosed','Hsig','ndist','top1','title0','title1','title2','title_fixed','count0','count1','count2'):
        nl=null[k]; side='lo' if k in('Hsig','ndist','title2','count2') else 'hi'
        print(f'     {k:12s} obs {obs[k]:8.3f}  null {q(nl,.5):8.3f} [{q(nl,.05):.3f},{q(nl,.95):.3f}]  P_{side} {pval(obs[k],nl,side):.3f}')
    return obs

# =========================== (d) hole / placeholder test ===========================
def hole_test(objs,nperm,label,level):
    """For each field S (OPENER, MARKER, TITLE, COUNT, SUFFIX, NAME) and each sign x whose majority slot is not S:
    enrichment of x in texts where S is empty vs filled; null = permute 'S empty' labels within (site, type, length) strata."""
    fields=['OPENER','MARKER','TITLE','COUNT','SUFFIX','NAME']
    closed=[o for o in objs if 'CLOSER' in o['lab']]   # frame-bearing texts only, so 'empty' is meaningful
    print(f'\n-- (d) HOLE TEST [{label}]: {len(closed)} closer-bearing texts; placeholder = sign enriched when a field is EMPTY, with position')
    strata=collections.defaultdict(list)
    for k,o in enumerate(closed): strata[(o['site'],o['ot'],min(len(o['seq']),8))].append(k)
    toks=collections.Counter(a for o in closed for a in set(o['seq']))
    cand=[a for a,n in toks.items() if n>=20]
    narrows=0; hits=[]
    for S in fields:
        empty=[('%s'%S not in o['lab']) for o in closed]
        ne=sum(empty); nf=len(closed)-ne
        if ne<30 or nf<30: print(f'   {S}: empty {ne} filled {nf} -- skipped'); continue
        signs=[a for a in cand if SLOT.get(a)!=S]
        def enr(emp):
            ce=collections.Counter(); cf=collections.Counter()
            for o,e in zip(closed,emp):
                for a in set(o['seq']):
                    if e: ce[a]+=1
                    else: cf[a]+=1
            return {a:((ce[a]+0.5)/(sum(emp)+1))/((cf[a]+0.5)/(len(emp)-sum(emp)+1)) for a in signs},ce,cf
        obs,ce,cf=enr(empty)
        r=random.Random(13); nullmax=[]; nulls=collections.defaultdict(list)
        for _ in range(nperm):
            pe=empty[:]
            for ks in strata.values():
                vals=[empty[k] for k in ks]; r.shuffle(vals)
                for k,v in zip(ks,vals): pe[k]=v
            e2,_,_=enr(pe); nullmax.append(max(math.log(v) for v in e2.values()))
            for a in signs: nulls[a].append(math.log(e2[a]))
        narrows+=len(signs)
        top=sorted(signs,key=lambda a:-obs[a])[:6]
        print(f'   field {S}: empty {ne} / filled {nf}; {len(signs)} candidate signs; max-null 95th log-OR {q(nullmax,.95):.2f}')
        for a in top:
            lo=math.log(obs[a]); pmax=pval(lo,nullmax,'hi')
            # position of x in empty texts: offset from end, and whether it sits where S would sit
            pos=collections.Counter()
            for o,e in zip(closed,empty):
                if e and a in o['seq']: pos[o['seq'].index(a)-len(o['seq'])]+=1   # negative offset from end
            top_pos=pos.most_common(2)
            flag='***' if pmax<0.05 else ''
            print(f'     {M(a):>22s} slot {SLOT.get(a,"-"):7s} in-empty {ce[a]:3d}/{ne} in-filled {cf[a]:3d}/{nf}  log-OR {lo:+.2f}  P(max-null) {pmax:.3f} {flag}  offsets-from-end {top_pos}')
            if pmax<0.05: hits.append((S,a,lo,pmax,ce[a],cf[a],top_pos))
    print(f'   arrows fired in (d): {narrows} sign x field tests, controlled by the per-field max-statistic null (family-wise within field); fields tested {len(set(h[0] for h in hits))} with hits')
    return hits

# =========================== main ===========================
if CY==1:
    rows,s=anchoring(OBJ,'Indus',NP)
    # by object type
    for ot in ('seal','tablet'):
        sub=[o for o in OBJ if o['ot']==ot]; anchoring(sub,f'Indus {ot}',max(50,NP//3),top=8)
    # held out
    anchoring([o for o in OBJ if not o['big']],'Indus held-out sites (not MD/HP)',max(50,NP//3),minn=15,top=8)
    ur=load_ur3(); lb=load_linb()
    anchoring(ur,'Ur III legends (syllables)',max(50,NP//3),top=10)
    anchoring(lb,'Linear B lines (words/ideograms/numbers)',max(50,NP//3),top=10)
elif CY==2:
    eval_models(OBJ,'Indus',SLOT); per_sign_gain(OBJ,SLOT,'Indus')
    for ot in ('seal','tablet'):
        sub=[o for o in OBJ if o['ot']==ot]; eval_models(sub,f'Indus {ot}',SLOT,folds=5)
    ur=load_ur3(); lb=load_linb()
    # controls: slot classes unknown -> use first/last-position-majority class as a crude 'slot' (3 classes) so the slot model is defined
    def crude_slot(objs):
        pos=collections.defaultdict(collections.Counter)
        for o in objs:
            L=len(o['seq'])
            for k,a in enumerate(o['seq']): pos[a]['first' if k==0 else 'last' if k==L-1 else 'mid']+=1
        return {a:c.most_common(1)[0][0] for a,c in pos.items()}
    eval_models(ur,'Ur III legends',crude_slot(ur)); eval_models(lb,'Linear B lines',crude_slot(lb))
    print('\n   NOTE: for Indus the slot-class model uses the S310 frame slots; for the controls a crude first/mid/last majority class.')
    eval_models(OBJ,'Indus (crude first/mid/last classes, for comparability)',crude_slot(OBJ))
elif CY==3:
    closed,full=sig_table(OBJ,'Indus'); generative_null(OBJ,NP,'Indus')
    for ot in ('seal','tablet'):
        sub=[o for o in OBJ if o['ot']==ot]; sig_table(sub,f'Indus {ot}')
    sig_table([o for o in OBJ if not o['big']],'Indus held-out sites')
    hits=hole_test(OBJ,NP,'Indus',LV)
    # held-out replication of hits
    ho=[o for o in OBJ if not o['big']]
    if hits:
        print('\n   held-out (non-MD/HP) check of hits: sign, field, in-empty/in-filled, log-OR')
        cl=[o for o in ho if 'CLOSER' in o['lab']]
        for S,a,lo,pm,_,_,_ in hits:
            ce=sum(1 for o in cl if S not in o['lab'] and a in o['seq']); ne=sum(1 for o in cl if S not in o['lab'])
            cf=sum(1 for o in cl if S in o['lab'] and a in o['seq']); nf=len(cl)-ne
            lo2=math.log(((ce+0.5)/(ne+1))/((cf+0.5)/(nf+1)))
            print(f'     {M(a):>22s} {S:7s} empty {ce}/{ne} filled {cf}/{nf}  log-OR {lo2:+.2f} (fit-set {lo:+.2f})')
