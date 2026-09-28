"""S241: Do Indus texts found abroad (Mesopotamia, Gulf, Iran, Central Asia) end with the common home endings (last two signs among the
20 commonest home final bigrams), as the new Umma seal does (...803-740)? Control: length-matched random home seal texts 10000x."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
FOR={'Ur','Kish','Susa','Nippur','Tell Umma','Tello',"Ra's al-Junayz",'Failaka',"Qala'at al-Bahrain",'Salut','Tepe Yahya','Altyn Depe','Luristan','Hajar','Karzakan','Shortughai','Miri Qalat'}
home=[x for x in d if x['type'].startswith('SEAL') and x['site'] not in FOR and len(x['seq'])>=2]
foreign=[x for x in d if x['site'] in FOR and len(x['seq'])>=2]
fin=C.Counter(tuple(x['seq'][-2:]) for x in home); top=set(k for k,v in fin.most_common(20))
obs=sum(tuple(x['seq'][-2:]) in top for x in foreign)
print('foreign texts',len(foreign),C.Counter(x['site'] for x in foreign).most_common(8))
print('with common home ending',obs)
for x in foreign: print(' ',x['site'][:10],x['type'],x['seq'],'*' if tuple(x['seq'][-2:]) in top else '')
bylen=C.defaultdict(list)
for x in home: bylen[min(len(x['seq']),7)].append(x)
random.seed(127); nl=[]
for _ in range(10000):
    nl.append(sum(tuple(random.choice(bylen[min(len(x['seq']),7)])['seq'][-2:]) in top for x in foreign))
print('null mean %.2f P(<=obs)=%.4f'%(sum(nl)/10000,sum(v<=obs for v in nl)/10000))
