"""Loop 36 cycle 3: score the out-of-corpus candidate texts (data/derived/dark/loop36_newfinds.csv) against the frozen
predictions in data/derived/dark/loop36_predictions.txt. Two sets: STRICT (signs at confidence >= B only, '?' elsewhere)
and LENIENT (every read sign, C included). Nulls: within-text shuffle (2,000 reps) and a home-corpus bigram model
(length-matched synthetic texts, 2,000 reps) as in S-DARK-21.1. Zero counts are reported as < 3/n.
Usage: python3 tools/dark_loop36_score.py > data/derived/dark/loop36_cycle3.txt"""
import csv,json,random,collections,math,sys
ROOT='/home/user/Indus-/'
random.seed(36)
C=json.load(open(ROOT+'data/derived/merged-corpus-canonical.json'))
HOME=[r['seq_raw'] for r in C if r['seq_raw']]
HOME_TYPES=set(s for t in HOME for s in t)
OPENERS={817,861,820,920,692}; CONN={2,60}; CLOSERS={740,520,154,156,158,527,151,236,595,700,400}
JAR=740
CHAIN=[320,920,60,741,2,803,32,350,798,415,220,233,705,255,435,690,740,400]
CHAINPOS={s:i for i,s in enumerate(CHAIN)}
FISH=[235,240,233,231]
GOODS={390,405,407,520,900,845,923,550}; SHORTNUM={31,2,32}; NUM3_8={3,4,5,16,17,18,33,34,35,36}
FROZEN=[(590,390),(590,405),(435,690),(255,435),(840,32),(17,585),(3,156),(33,520),(32,226)]
def parse(row,strict):
    signs=row['signs_W'].split('-'); conf=(row['confidence_per_sign'] or '').split('-')
    out=[]
    for i,s in enumerate(signs):
        c=conf[i] if i<len(conf) else '?'
        if s in ('?','') or '/' in s: out.append(None); continue
        if strict and c not in ('A','B'): out.append(None); continue
        out.append(int(s))
    return out
def load(strict):
    rows=list(csv.DictReader(open(ROOT+'data/derived/dark/loop36_newfinds.csv')))
    texts=[]
    for r in rows:
        if r['tier'].startswith('A') and 'SALUT' in r['id']: continue   # duplicate of corpus M-1367
        if 'not transcribed' in r['signs_W'] or not r['signs_W']: continue
        t=parse(r,strict)
        if sum(x is not None for x in t)>=2: texts.append((r['id'],r['tier'][0],t))
    return texts
def stats(texts):
    S={}
    conc=[[x for x in t if x is not None] for _,_,t in texts]
    # P1 opener initial
    op_tok=sum(1 for t in conc for x in t if x in OPENERS); op_init=sum(1 for t in conc if t and t[0] in OPENERS)
    S['P1_opener_tokens']=op_tok; S['P1_opener_initial']=op_init
    # P2 connective initial
    S['P2_conn_tokens']=sum(1 for t in conc for x in t if x in CONN); S['P2_conn_initial']=sum(1 for t in conc if t and t[0] in CONN)
    # P4 jar final / closer last
    jar_texts=[t for t in conc if JAR in t]; S['P4_jar_texts']=len(jar_texts); S['P4_jar_final']=sum(1 for t in jar_texts if t[-1]==JAR)
    comb=[t for t in conc if 400 in t]; S['P4_comb_texts']=len(comb); S['P4_comb_final']=sum(1 for t in comb if t[-1]==400)
    S['P4_closer_last']=sum(1 for t in conc if t and t[-1] in CLOSERS); S['n_texts']=len(conc)
    # P5 two closers
    S['P5_two_closers']=sum(1 for t in conc if len(set(t)&(CLOSERS-{700,400}))>=2)
    # P6 repeats
    rep=[t for t in conc if len(t)>=3]; S['P6_texts3']=len(rep); S['P6_repeat']=sum(1 for t in rep if len(set(t))<len(t))
    # P7 minimum lot
    pairs=[(a,b) for _,_,t in texts for a,b in zip(t,t[1:]) if a is not None and b is not None and b in GOODS and (a in SHORTNUM or a in NUM3_8 or a in (1,))]
    S['P7_pairs']=len(pairs); S['P7_short12']=sum(1 for a,b in pairs if a in SHORTNUM); S['P7_tall1']=sum(1 for a,b in pairs if a==1)
    # P8 chain order
    ok=bad=0
    for t in conc:
        pos=[(CHAINPOS[x],i) for i,x in enumerate(t) if x in CHAINPOS]
        for a in range(len(pos)):
            for b in range(a+1,len(pos)):
                if pos[a][0]==pos[b][0]: continue
                if (pos[a][0]<pos[b][0])==(pos[a][1]<pos[b][1]): ok+=1
                else: bad+=1
    S['P8_chain_ok']=ok; S['P8_chain_bad']=bad
    # P9 fish order
    fo=fb=0
    for t in conc:
        f=[(FISH.index(x),i) for i,x in enumerate(t) if x in FISH]
        for a in range(len(f)):
            for b in range(a+1,len(f)):
                if (f[a][0]<f[b][0])==(f[a][1]<f[b][1]): fo+=1
                else: fb+=1
    S['P9_fish_ok']=fo; S['P9_fish_bad']=fb
    # P11 frozen pairs
    fk=fn=0
    for t in conc:
        for a,b in FROZEN:
            if a in t and b in t:
                if any(x==a and y==b for x,y in zip(t,t[1:])): fk+=1
                else: fn+=1
    S['P11_frozen_kept']=fk; S['P11_frozen_broken']=fn
    # P12 seated person
    S['P12_176_then_100']=sum(1 for t in conc if 176 in t and 100 in t and t.index(176)<t.index(100))
    S['P12_100_then_176']=sum(1 for t in conc if 176 in t and 100 in t and t.index(176)>t.index(100))
    # P13 >=6-sign texts recurring
    six=[t for t in conc if len(t)>=6]; S['P13_texts6']=len(six)
    homeset=set(tuple(t) for t in HOME); S['P13_recur']=sum(1 for t in six if tuple(t) in homeset or tuple(t[::-1]) in homeset)
    # P14 unseen signs
    toks=[x for t in conc for x in t]; S['P14_tokens']=len(toks); S['P14_unseen_tokens']=sum(1 for x in toks if x not in HOME_TYPES)
    S['P14_types']=len(set(toks)); S['P14_unseen_types']=len(set(x for x in toks if x not in HOME_TYPES))
    # P10 nesting / reuse: >=3-sign runs attested in home corpus
    runs=set()
    for t in HOME:
        for i in range(len(t)-2): runs.add(tuple(t[i:i+3]))
    def reuse(t): return any(tuple(t[i:i+3]) in runs for i in range(len(t)-2))
    r3=[t for t in conc if len(t)>=3]; S['P10_texts3']=len(r3); S['P10_reuse']=sum(1 for t in r3 if reuse(t))
    short=[t for t in conc if 3<=len(t)<=5]
    def nested(t):
        n=len(t)
        for h in HOME:
            if len(h)>n and any(h[i:i+n]==t for i in range(len(h)-n+1)): return True
        return False
    S['P10_short']=len(short); S['P10_nested']=sum(1 for t in short if nested(t))
    return S
def shuffled(texts):
    out=[]
    for id_,tier,t in texts:
        conc=[x for x in t if x is not None]; random.shuffle(conc); it=iter(conc)
        out.append((id_,tier,[next(it) if x is not None else None for x in t]))
    return out
def bigram_model():
    uni=collections.Counter(); big=collections.defaultdict(collections.Counter); start=collections.Counter()
    for t in HOME:
        start[t[0]]+=1
        for a,b in zip(t,t[1:]): big[a][b]+=1
        for x in t: uni[x]+=1
    return start,big,uni
def synth(texts,model):
    start,big,uni=model; out=[]
    def draw(c):
        items=list(c.items()); r=random.random()*sum(v for _,v in items); acc=0
        for k,v in items:
            acc+=v
            if acc>=r: return k
        return items[-1][0]
    for id_,tier,t in texts:
        n=sum(x is not None for x in t); s=[draw(start)]
        while len(s)<n: s.append(draw(big[s[-1]]) if big[s[-1]] else draw(uni))
        out.append((id_,tier,s))
    return out
def ub(n): return f'< {3/n:.2f}' if n>0 else 'n=0'
def report(name,texts):
    print(f'\n===== {name}: {len(texts)} texts =====')
    for id_,tier,t in texts: print('  ',tier,id_,'-'.join('?' if x is None else str(x) for x in t))
    obs=stats(texts); REPS=500
    sh=[stats(shuffled(texts)) for _ in range(REPS)]
    model=bigram_model(); bg=[stats(synth(texts,model)) for _ in range(REPS)]
    def null(key,pool): v=[s[key] for s in pool]; return sum(v)/len(v), sum(1 for x in v if x>=obs[key])/len(v), sum(1 for x in v if x<=obs[key])/len(v)
    print('n_texts',obs['n_texts'])
    for key in ['P1_opener_initial','P2_conn_initial','P4_jar_final','P4_comb_final','P4_closer_last','P5_two_closers','P6_repeat','P7_short12','P8_chain_ok','P8_chain_bad','P9_fish_ok','P11_frozen_kept','P11_frozen_broken','P12_100_then_176','P13_recur','P14_unseen_tokens','P10_reuse','P10_nested']:
        m,pge,ple=null(key,sh); mb,pgeb,pleb=null(key,bg)
        print(f'{key:22s} obs {obs[key]:3d}  shuffle-mean {m:6.2f} (P>= {pge:.3f}, P<= {ple:.3f})  bigram-mean {mb:6.2f} (P>= {pgeb:.3f}, P<= {pleb:.3f})')
    print('denominators:',{k:v for k,v in obs.items() if k.endswith(('tokens','texts','pairs','texts3','texts6','short','types','jar_texts','comb_texts'))})
    print('upper bounds for zero counts: P2 connective-initial',ub(obs['P2_conn_tokens']),'; P12 100-before-176',ub(obs['P12_176_then_100']+obs['P12_100_then_176']),'; P13 recurrence of >=6-sign texts',ub(obs['P13_texts6']),'; P14 unseen tokens',ub(obs['P14_tokens']))
    return obs
if __name__=='__main__':
    for strict,name in [(True,'STRICT (A/B signs only)'),(False,'LENIENT (all read signs)')]:
        T=load(strict)
        report(name+' all tiers',T)
        for tier,label in [('A','tier A excavated'),('B','tier B museum'),('C','tier C copper plates')]:
            sub=[x for x in T if x[1]==tier]
            if sub: report(name+' '+label,sub)
