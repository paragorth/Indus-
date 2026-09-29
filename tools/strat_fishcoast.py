"""S315 (intuitive gloss test): if the fish signs mean something maritime (fish trade, boatmen, seafarers), texts from
coastal / port sites should use them more than inland sites. Coastal: Lothal, Dholavira, Chanhu-daro? (river port,
excluded as ambiguous), Kuntasi, Bagasra, Nageshwar, Sutkagen-dor, Sotka-koh, Balakot, Kanmer, Shikarpur, Desalpur,
Surkotada, Gola Dhoro, Juna Khatiya, Kotada Bhadli, plus the Gulf. Inland: Harappa, Kalibangan, Banawali,
Rakhigarhi, Farmana, Bhirrana, Mitathal, Rupar, Alamgirpur, Hulas. Mohenjo-daro (river city) reported separately.
Statistic: share of texts with any fish sign; control: coast/inland labels permuted among SITES (site-level, 5,000x)."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
FISH={220,240,235,233,231,226}
COAST={'Lothal','Dholavira','Kuntasi','Bagasra','Nageshwar','Sutkagen-dor','Sotka-koh','Balakot','Kanmer','Shikarpur','Desalpur','Surkotada','Gola Dhoro','Juna Khatiya','Kotada Bhadli','Failaka',"Qala'at al-Bahrain",'Saar','Karzakan','Janabiyah','Dilmun',"Ra's al-Junayz",'Salut','Kalba','Hajar'}
INLAND={'Harappa','Kalibangan','Banawali','Rakhigarhi','Farmana','Bhirrana','Mitathal','Rupar','Alamgirpur','Hulas'}
seen=set(); by=collections.defaultdict(lambda:[0,0])
for r in C:
    s=r['seq_raw']
    if not s or len(s)<2 or (r['site'],tuple(s)) in seen: continue
    seen.add((r['site'],tuple(s))); by[r['site']][0]+=1; by[r['site']][1]+=any(a in FISH for a in s)
def rate(sites): n=sum(by[x][0] for x in sites); return sum(by[x][1] for x in sites)/max(1,n),n
c=[x for x in COAST if by[x][0]]; i=[x for x in INLAND if by[x][0]]
rc,nc=rate(c); ri,ni=rate(i); md=rate(['Mohenjo-daro'])
print(f'coastal/port {rc:.3f} (n={nc}, {len(c)} sites) | inland {ri:.3f} (n={ni}, {len(i)} sites) | Mohenjo-daro {md[0]:.3f} (n={md[1]})')
for x in sorted(c+i,key=lambda x:-by[x][0])[:14]: print('   ',x,by[x][0],round(by[x][1]/by[x][0],3))
allsites=c+i; rnd=random.Random(25); obs=rc-ri; ge=0
for _ in range(5000):
    p=allsites[:]; rnd.shuffle(p); a=p[:len(c)]; b=p[len(c):]
    ge+= (rate(a)[0]-rate(b)[0])>=obs
print(f'difference {obs:.3f}; site-level permutation P={(ge+1)/5001:.4f}')
