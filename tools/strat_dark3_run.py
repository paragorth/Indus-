"""Driver for tools/strat_dark3.py. Modes:
 screen  : every transformation x every test, 20 random parameterisations, train split, seq_raw, P null draws.
 confirm : arrows named in argv (or all with screen |z|>=3) on train + held-out, seq_raw/seq_strong/seq_all, P=400.
 ur3     : same transformations on Ur III owner names (syllables) and legends (words): do names stay name-like?
 control : screen on the bigram-generated Indus corpus (data/codelib/indus_bigram.jsonl): whole-machine control.
 pipelines: random compositions of two transformations (new arrow set), screen + confirm.
Usage: python3 tools/strat_dark3_run.py MODE [P] [seed]"""
import sys,os,random,json,statistics as st,time,collections
sys.path.insert(0,'tools'); import strat_dark3 as D
mode=sys.argv[1]; P=int(sys.argv[2]) if len(sys.argv)>2 else 40; seed=int(sys.argv[3]) if len(sys.argv)>3 else 3
NTH=20
TR=[t for t in D.TRANS]
def run_split(split,var,P,rng,tr_names,tests,log,data=None):
    X=data if data is not None else D.texts(split,var)
    seqs=[t['seq'] for t in X]; signs=sorted(set(s for t in seqs for s in t))
    res={}
    for tn in tr_names:
        thetas=[D.draw_theta(tn,rng,signs) for _ in range(NTH)]
        t0=time.time(); RR=D.arrow_multi(tn,tests,seqs,X,thetas,P,rng)
        for sn in tests:
            r=RR[sn]
            if r is None: log(f'{split} {var} {tn:11s} {sn:9s} n/a'); continue
            Dm,null,vals,base=r; p=D.pval(Dm,null); z=D.zval(Dm,null)
            res[(tn,sn)]={'D':Dm,'p':p,'z':z,'base':base,'val_med':st.median(vals),'val_min':min(vals),'val_max':max(vals),'null_mean':st.mean(null) if null else None,'null_sd':st.pstdev(null) if len(null)>1 else None,'n':len(seqs)}
            log(f'{split} {var} {tn:11s} {sn:9s} base {base:.3f} -> {st.median(vals):.3f} [{min(vals):.3f},{max(vals):.3f}] D {Dm:+.3f} null {(st.mean(null) if null else float("nan")):+.3f}±{(st.pstdev(null) if len(null)>1 else float("nan")):.3f} z {z:+.1f} p {p:.3f} ({time.time()-t0:.0f}s)')
    return res
def main():
    rng=random.Random(seed)
    if mode=='screen':
        f=open(D.OUT+'loop3_screen_raw.txt','w')
        def log(s): print(s,flush=True); f.write(s+'\n'); f.flush()
        res=run_split('train','seq_raw',P,rng,TR,list(D.TESTS),log)
        json.dump({f'{k[0]}|{k[1]}':v for k,v in res.items()},open(D.OUT+'loop3_screen_raw.json','w'),indent=1)
    elif mode=='control':
        X=[{'seq':[int(s[1:]) for s in r['seq']],'emblem':'NONE','type':'X'} for r in map(json.loads,open('data/codelib/indus_bigram.jsonl'))]
        f=open(D.OUT+'loop3_control_bigram.txt','w')
        def log(s): print(s,flush=True); f.write(s+'\n'); f.flush()
        res=run_split('all','bigram',P,rng,TR,list(D.TESTS),log,data=X)
        json.dump({f'{k[0]}|{k[1]}':v for k,v in res.items()},open(D.OUT+'loop3_control_bigram.json','w'),indent=1)
    elif mode=='confirm':
        names=sys.argv[4:] if len(sys.argv)>4 else None
        if names is None:
            scr=json.load(open(D.OUT+'loop3_screen_raw.json'))
            names=[k for k,v in scr.items() if abs(v['z'])>=3]
        pairs=[tuple(n.split('|')) for n in names]
        f=open(D.OUT+'loop3_confirm.txt','w')
        def log(s): print(s,flush=True); f.write(s+'\n'); f.flush()
        allres={}
        for split in ('train','heldout'):
            for var in ('seq_raw','seq_strong','seq_all'):
                for tn,sn in pairs:
                    r=run_split(split,var,P,rng,[tn],[sn],log)
                    for k,v in r.items(): allres[f'{split}|{var}|{k[0]}|{k[1]}']=v
        json.dump(allres,open(D.OUT+'loop3_confirm.json','w'),indent=1)
    elif mode=='confirm1':
        # python3 tools/strat_dark3_run.py confirm1 P seed SPLIT trans:test1,test2 ...
        split=sys.argv[4]; specs=[a.split(':') for a in sys.argv[5:]]
        f=open(D.OUT+f'loop3_confirm_{split}.txt','w')
        def log(s): print(s,flush=True); f.write(s+'\n'); f.flush()
        allres={}
        for var in ('seq_raw','seq_strong','seq_all'):
            for tn,tests in specs:
                r=run_split(split,var,P,rng,[tn],tests.split(','),log)
                for k,v in r.items(): allres[f'{split}|{var}|{k[0]}|{k[1]}']=v
                json.dump(allres,open(D.OUT+f'loop3_confirm_{split}.json','w'),indent=1)
    elif mode=='ur3':
        R=D.refs(); f=open(D.OUT+'loop3_ur3_control.txt','w')
        def log(s): print(s,flush=True); f.write(s+'\n'); f.flush()
        for key in ('ur3_names_syll','ur3_legends_words'):
            # map tokens to ints so numeral/frame/family transforms run (they will mostly be no-ops: no W-structure)
            voc={}; X=[]
            for s in R[key]:
                X.append({'seq':[voc.setdefault(w,1000+len(voc)) for w in s],'emblem':'NONE','type':'X'})
            tr=[t for t in TR if t not in ('numonly','nonnum','splitnum','emblem','objtype','family','dropframe')]
            res=run_split(key,'tokens',P,rng,tr,['namecalib','repeat','slot'],log,data=X)
            json.dump({f'{k[0]}|{k[1]}':v for k,v in res.items()},open(D.OUT+f'loop3_ur3_{key}.json','w'),indent=1)
    elif mode=='pipelines':
        f=open(D.OUT+'loop3_pipelines.txt','w')
        def log(s): print(s,flush=True); f.write(s+'\n'); f.flush()
        base_tr=[t for t in TR if t not in ('identity','relabel')]
        pipes=[]
        while len(pipes)<20:
            a,b=rng.sample(base_tr,2)
            if (a,b) not in pipes: pipes.append((a,b))
        for a,b in pipes:
            name=f'{a}>{b}'
            def comp(T,M,rng_,th,a=a,b=b):
                T2=D.TRANS[a](T,M,rng_,th['a'])
                M2=M if len(T2)==len(M) else [{'emblem':'NONE','type':'X'}]*len(T2)
                return D.TRANS[b](T2,M2,rng_,th['b'])
            D.TRANS[name]=comp
        old=D.draw_theta
        def draw(name,rng_,signs):
            if '>' in name:
                a,b=name.split('>'); return {'a':old(a,rng_,signs),'b':old(b,rng_,signs)}
            return old(name,rng_,signs)
        D.draw_theta=draw
        names=[f'{a}>{b}' for a,b in pipes]
        res=run_split('train','seq_raw',P,rng,names,list(D.TESTS),log)
        json.dump({f'{k[0]}|{k[1]}':v for k,v in res.items()},open(D.OUT+'loop3_pipelines.json','w'),indent=1)
        surv=[k for k,v in res.items() if abs(v['z'])>=3]
        log(f'pipeline survivors |z|>=3 on train: {len(surv)} of {len(res)}')
        if os.environ.get('NOCONFIRM'): return
        allres={}
        for split in ('train','heldout'):
            for var in ('seq_raw','seq_strong','seq_all'):
                for tn,sn in surv:
                    r=run_split(split,var,60,rng,[tn],[sn],log)
                    for k,v in r.items(): allres[f'{split}|{var}|{k[0]}|{k[1]}']=v
        json.dump(allres,open(D.OUT+'loop3_pipelines_confirm.json','w'),indent=1)
main()
