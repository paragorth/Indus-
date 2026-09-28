"""Run the Laursen 2010 Gulf Type seals (data/gulf_seals.csv) through the frame grammar as a
'Gulf/western' register. Frozen predictions are only read, never changed."""
import csv,json,collections,random,math
from scipy.stats import fisher_exact
m=json.load(open('data/derived/merged-corpus-canonical.json'))
OPEN={817,861,820}; JAR=740; PERSON={90,91,93}; TWIN={91}
G=[g for g in csv.DictReader(open('data/gulf_seals.csv')) if g['sequence_glyph_coding']]
def seq(g): return [int(x) for x in g['sequence_glyph_coding'].split('-')]
abroad=[g for g in G if g['found_abroad']=='True']; homegulf=[g for g in G if g['found_abroad']=='False']
HOME={'Harappa','Mohenjo-daro','Dholavira','Lothal','Kalibangan','Chanhu-daro','Banawali','Rakhigarhi'}
seen=set(); H=[]
for r in m:
    s=r.get('seq_raw')
    if r['site'] in HOME and r['type']=='SEAL:S' and s and (r['site'],tuple(s)) not in seen:
        seen.add((r['site'],tuple(s))); H.append(s)
def feats(s):
    return dict(open_initial=s[0] in OPEN, open_any=any(x in OPEN for x in s), jar_final=s[-1]==JAR,
                jar_person_end=len(s)>=2 and s[-2]==JAR and s[-1] in PERSON, person_initial=s[0] in PERSON,
                twin=any(x in TWIN for x in s) or any(a in (121,99) and b in (121,99) for a,b in zip(s,s[1:])),
                frame=s[0] in OPEN or s[-1]==JAR or (len(s)>=2 and s[-2]==JAR))
def rate(S,k): return sum(feats(s)[k] for s in S)/len(S)
out={}
for lab,S in (('Gulf abroad',[seq(g) for g in abroad]),('Gulf type found in Indus valley',[seq(g) for g in homegulf]),('Indus square seals',H)):
    out[lab]={k:round(rate(S,k),3) for k in feats([1,2])}; out[lab]['n']=len(S)
for k,v in out.items(): print(k,v)
# length-matched permutation: how often does a random set of home square seals with the same lengths show rates this extreme?
A=[seq(g) for g in abroad]; bylen=collections.defaultdict(list)
for s in H: bylen[min(len(s),8)].append(s)
random.seed(4); R=10000
def samp(): return [random.choice(bylen[min(len(s),8)]) for s in A]
null={k:[] for k in ('open_initial','frame','person_initial','jar_person_end')}
for _ in range(R):
    X=samp()
    for k in null: null[k].append(rate(X,k))
for k in null:
    o=rate(A,k); lo=sum(v<=o for v in null[k])/R; hi=sum(v>=o for v in null[k])/R
    print(f'{k}: abroad {o:.3f} | length-matched home mean {sum(null[k])/R:.3f} | P(home <= abroad) {lo:.4f}, P(home >= abroad) {hi:.4f}')
# twins position
for g in abroad:
    s=seq(g)
    for i,x in enumerate(s):
        if x in TWIN: print('  twins in seal',g['seal_no'],'position',i+1,'of',len(s),s)
    for i,(a,b) in enumerate(zip(s,s[1:])):
        if a in (121,99) and b in (121,99): print('  two-man pair (coded 121-99) in seal',g['seal_no'],'position',i+1,'of',len(s),s)
# bigram fit (held out), length-matched
def train(T):
    c=collections.Counter(); u=collections.Counter()
    for s in T:
        t=['^']+s+['$']
        for a,b in zip(t,t[1:]): c[(a,b)]+=1; u[a]+=1
    V=len({x for s in T for x in s})+2
    return lambda a,b: math.log((c[(a,b)]+0.5)/(u[a]+0.5*V))
def score(lp,s):
    t=['^']+s+['$']; return sum(lp(a,b) for a,b in zip(t,t[1:]))/(len(t)-1)
random.seed(5); Hs=H[:]; random.shuffle(Hs); half=len(Hs)//2; train_set,test_set=Hs[:half],Hs[half:]
lp=train(train_set)
tb=collections.defaultdict(list)
for s in test_set: tb[min(len(s),8)].append(s)
ga=sum(score(lp,s) for s in A)/len(A)
nullfit=[]
for _ in range(R):
    X=[random.choice(tb[min(len(s),8)]) for s in A]; nullfit.append(sum(score(lp,s) for s in X)/len(X))
print(f'bigram fit (mean log-prob per transition, model trained on half the home square seals): Gulf abroad {ga:.3f} | held-out home, length-matched mean {sum(nullfit)/R:.3f}, 2.5-97.5% [{sorted(nullfit)[250]:.3f}, {sorted(nullfit)[9750]:.3f}] | P(home <= Gulf) {sum(v<=ga for v in nullfit)/R:.4f}')
hg=[seq(g) for g in homegulf]; print(f'  Gulf type found in Indus valley: {sum(score(lp,s) for s in hg)/len(hg):.3f} (n={len(hg)})')
# motifs
mot=collections.Counter((g['motif'] or 'none').split(':')[0] for g in abroad); print('motifs abroad:',mot)
hm=collections.Counter()
seen=set()
for r in m:
    if r['site'] in HOME and r['type']=='SEAL:S' and r.get('seq_raw') and (r['site'],tuple(r['seq_raw'])) not in seen:
        seen.add((r['site'],tuple(r['seq_raw']))); hm[str(r.get('symbol')).split(':')[0]]+=1
tot=sum(hm.values()); print('home square seal motifs:',[(k,v,round(v/tot,3)) for k,v in hm.most_common(6)])
json.dump(out,open('results/gulf_seals_rates.json','w'),indent=1)
# control: home SQUARE seals with the gaur emblem (animal held, format home) vs unicorn square seals
seen=set(); gaur=[]; uni=[]
for r in m:
    s=r.get('seq_raw')
    if r['site'] in HOME and r['type']=='SEAL:S' and s and (r['site'],tuple(s)) not in seen:
        seen.add((r['site'],tuple(s))); sym=str(r.get('symbol')).split(':')[0]
        if sym=='Gaur': gaur.append(s)
        elif sym=='Bull1': uni.append(s)
for lab,S in (('home square gaur',gaur),('home square unicorn',uni)):
    print(f'{lab}: n={len(S)} frame {rate(S,"frame"):.3f} open_initial {rate(S,"open_initial"):.3f} person_initial {rate(S,"person_initial"):.3f} twins {rate(S,"twin"):.3f} bigram fit {sum(score(lp,s) for s in S)/len(S):.3f}')
