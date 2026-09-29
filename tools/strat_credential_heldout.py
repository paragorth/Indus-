"""S349: out-of-sample test of the credential model (S347) on sites excavated after the 1977 concordance and absent
from IM77 (held out): Dholavira, Rakhigarhi, Farmana, Kanmer, Bagasra, Shikarpur, Kotada Bhadli, Khirsara, Pabumath,
Juna Khatiya, Binjor, Tigrana, Bhirrana, Salut and others not in IM77. Predictions frozen in
prereg/preregistration-4-credential.json before running:
 P1 nesting: share of held-out texts (3-5 signs) that occur whole inside a longer text anywhere > 2x shuffled.
 P2 reuse: share of held-out texts (>= 3 signs) that contain a >= 3-sign run known from Mohenjo-daro/Harappa > shuffled.
 P3 no name stock: held-out middles are not more repetitive than the bigram null (ratio >= 0.9)."""
import json,csv,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
im=set(r['site'] for r in csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
imn={s.lower().replace('-','').replace(' ','') for s in im}
def inim(site): return site.lower().replace('-','').replace(' ','') in imn or site in ('Mohenjo-daro','Harappa','Chanhu-daro','Lothal','Kalibangan')
seen=set(); A=[]; H=[]
for r in C:
    s=r['seq_raw']
    if not s or len(s)<3 or (r['site'],tuple(s)) in seen: continue
    seen.add((r['site'],tuple(s)))
    (H if not inim(r['site']) else A).append((r['site'],tuple(s)))
print('held-out texts',len(H),'sites',sorted({x for x,_ in H})[:25])
big={s for x,s in A if x in ('Mohenjo-daro','Harappa')}
def runs(S):
    out=set()
    for s in S:
        for n in (3,4,5):
            for i in range(len(s)-n+1): out.add(s[i:i+n])
    return out
R=runs(big)
def p2(texts): return sum(any(s[i:i+3] in R for i in range(len(s)-2)) for s in texts)/len(texts)
def p1(texts,allt):
    L=[t for t in allt if len(t)>=4]; sub=set()
    for t in L:
        for n in (3,4,5):
            for i in range(len(t)-n+1):
                if n<len(t): sub.add(t[i:i+n])
    sm=[t for t in texts if 3<=len(t)<=5]; return sum(t in sub for t in sm)/max(1,len(sm))
ht=[s for _,s in H]; at=[s for _,s in A]+ht
o1=p1(ht,at); o2=p2(ht); rnd=random.Random(39); n1=[];n2=[]
for _ in range(50):
    sh=[tuple(rnd.sample(s,len(s))) for s in ht]; sa=[tuple(rnd.sample(s,len(s))) for s in at]
    n1.append(p1(sh,sa)); n2.append(p2(sh))
print(f'P1 nesting held-out {o1:.3f} vs shuffled median {sorted(n1)[25]:.3f} (needs > 2x)')
print(f'P2 reuse of big-city runs {o2:.3f} vs shuffled median {sorted(n2)[25]:.3f}')
