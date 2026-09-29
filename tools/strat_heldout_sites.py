"""S313: true held-out check (S312). Results found mainly on Mohenjo-daro + Harappa, re-tested ONLY on texts from
all other sites (Dholavira, Lothal, Kalibangan, Chanhu-daro, Banawali, Rakhigarhi, Farmana, abroad ...).
(1) S311 name elements: initial set {692,575,125,416,413,920,495} vs final set {840,460,70,35,440,435,690,717},
    first-position share, names of 2-4 signs from the S310 parse; control: shuffle within names (2,000x).
(2) S297 fish order: ordered adjacent pairs among 235<240<233<231; count in predicted vs reverse order.
(3) S289 closer paradigm: final-heavy + jar-avoiding signs among the predicted set.
(4) S300 arrow phrase: fish directly before W520."""
import json,collections,random
from scipy.stats import binomtest
P=json.load(open('data/derived/parsed_texts.json'))
BIG={'Mohenjo-daro','Harappa'}
H=[o for o in P if o['site'] not in BIG]
print('held-out texts',len(H),'from',len({o['site'] for o in H}),'sites')
INI={692,575,125,416,413,920,495}; FIN={840,460,70,35,440,435,690,717}
names=[]
for o in H:
    idx=[k for k,l in enumerate(o['slots']) if l=='NAME']
    if 2<=len(idx)<=4 and idx==list(range(idx[0],idx[-1]+1)): names.append([o['seq'][k] for k in idx])
def stat(N):
    fi=[0,0]; ff=[0,0]
    for s in N:
        for i,a in enumerate(s):
            if a in INI: fi[1]+=1; fi[0]+= i==0
            if a in FIN: ff[1]+=1; ff[0]+= i==0
    return fi[0]/max(1,fi[1])-ff[0]/max(1,ff[1]),fi,ff
obs,fi,ff=stat(names); rnd=random.Random(23); null=sorted(stat([rnd.sample(s,len(s)) for s in names])[0] for _ in range(2000))
print(f'(1) names {len(names)}: initial-set first {fi[0]}/{fi[1]}, final-set first {ff[0]}/{ff[1]}; diff {obs:.3f}; null median {null[1000]:.3f}, max {null[-1]:.3f}; P={(sum(x>=obs for x in null)+1)/2001:.4f}')
ORD=[235,240,233,231]; rank={a:i for i,a in enumerate(ORD)}; fw=bw=0
for o in H:
    s=o['seq']
    for a,b in zip(s,s[1:]):
        if a in rank and b in rank and a!=b: fw+= rank[a]<rank[b]; bw+= rank[a]>rank[b]
print(f'(2) fish pairs in predicted order {fw}, reverse {bw}; p={binomtest(fw,fw+bw).pvalue if fw+bw else 1:.3g}')
fa=sum(1 for o in H for a,b in zip(o['seq'],o['seq'][1:]) if b==520 and a in {220,240,235,233,231})
n520=sum(1 for o in H for i,a in enumerate(o['seq']) if a==520 and i>0)
print(f'(4) W520 with a fish directly before: {fa}/{n520}')
