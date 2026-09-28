"""S251: Animals facing the unusual way (Konasukawa 2013: only ~1% of unicorns face the 'wrong' way). IM77 field codes: unicorn to R (11,13)
vs to L (12,14). Do the rare-direction seals carry different texts (opener, jar-final, length, direction of writing)? Control: permutation 5000x."""
import csv,collections as C,random
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
T=C.defaultdict(list); fs={}; dirn={}
for r in rows:
    if r['object_type']!='seal': continue
    T[r['text_no']]+=[int(t) for t in r['signs_clean'].split() if t.isdigit() and t!='0']; fs[r['text_no']]=r['fs80']; dirn.setdefault(r['text_no'],r['direction'])
uni=[(t,'R' if fs[t] in ('11','13') else 'L') for t in T if fs[t] in ('11','12','13','14') and T[t]]
print(C.Counter(l for t,l in uni))
rare=min(C.Counter(l for t,l in uni).items(),key=lambda kv:kv[1])[0]
feats={'opener':lambda t:bool(set(T[t])&{267,391}),'jar-final':lambda t:T[t][-1]==342,'length>=5':lambda t:len(T[t])>=5,'left-to-right writing':lambda t:dirn[t]=='left-to-right'}
random.seed(139)
for nm,f in feats.items():
    lab=[l for t,l in uni]; vals=[f(t) for t,l in uni]
    def diff(lab):
        a=[v for v,l in zip(vals,lab) if l==rare]; b=[v for v,l in zip(vals,lab) if l!=rare]; return sum(a)/len(a)-sum(b)/len(b),sum(a),len(a),sum(b),len(b)
    o=diff(lab); nl=[]
    for _ in range(5000): random.shuffle(lab); nl.append(diff(lab)[0])
    print('%-22s rare(%s) %d/%d vs common %d/%d  P=%.4f'%(nm,rare,o[1],o[2],o[3],o[4],sum(abs(v)>=abs(o[0]) for v in nl)/5000))
