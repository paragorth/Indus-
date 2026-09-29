"""S328: are texts found abroad 'a complete home text + an added prefix', like Kish (S327)? For each abroad text,
the longest ending (>= 3 signs) that equals a COMPLETE home text, and the longest beginning that does. Control: the
same search for home texts from small sites (not Mohenjo-daro/Harappa) of the same lengths, against Mohenjo-daro +
Harappa texts only. Also where the Kish prefix 416-840(-60) occurs at home."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
ABROAD={'Failaka','Hajar','Kish','Luristan','Nippur',"Qala'at al-Bahrain","Ra's al-Junayz",'Susa','Tell Umma','Tello','Tepe Yahya','Ur','Gonur Depe','Salut','Girsu','Karzakan','Saar','Janabiyah','Kalba','Dilmun','Altyn Depe','Shortughai','Miri Qalat'}
big={tuple(r['seq_raw']) for r in C if r['site'] in ('Mohenjo-daro','Harappa') and r['seq_raw'] and len(r['seq_raw'])>=3}
def test(texts):
    hit=[]
    for site,s in texts:
        for k in range(len(s)-1,2,-1):
            if tuple(s[-k:]) in big and k<len(s): hit.append((site,s,'ends with home text',s[-k:])); break
            if tuple(s[:k]) in big and k<len(s): hit.append((site,s,'starts with home text',s[:k])); break
    return hit
ab=[(r['site'],r['seq_raw']) for r in C if r['site'] in ABROAD and r['seq_raw'] and len(r['seq_raw'])>=4]
H=test(ab); print(f'abroad texts >=4 signs: {len(ab)}; contain a complete Mohenjo-daro/Harappa text + extra: {len(H)}')
for h in H: print('   ',h)
small=[(r['site'],r['seq_raw']) for r in C if r['site'] not in ABROAD|{'Mohenjo-daro','Harappa'} and r['seq_raw'] and len(r['seq_raw'])>=4]
rnd=random.Random(34); rates=[]
L=collections.Counter(len(s) for _,s in ab); bylen=collections.defaultdict(list)
for x in small: bylen[len(x[1])].append(x)
for _ in range(2000):
    smp=[rnd.choice(bylen[l] or bylen[4]) for l,n in L.items() for _ in range(n)]
    rates.append(len(test(smp)))
rates.sort(); print(f'small home sites, length-matched: median {rates[1000]}, 95% {rates[1900]}; P={sum(x>=len(H) for x in rates)/2000:.3f}')
for r in C:
    s=r['seq_raw'] or []
    if any(tuple(s[i:i+2])==(416,840) for i in range(len(s)-1)): print('  416-840 at',r['site'],r['type'],s)
