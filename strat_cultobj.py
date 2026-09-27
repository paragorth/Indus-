"""S237: Test of K. V. Pendor's Gondi reading (Facebook group 'Deciphering Indus Script', images supplied by the user): the object in front
of the unicorn is 'always used' and marks the farming activity, the text above describing its method. IM77 field-symbol codes 11/12
(unicorn, no cult object) vs 13/14 (unicorn facing cult object). (a) How often is the object absent? (b) Do texts differ with vs without it
(frame, length, closer)? Control: permute with/without labels 5000x."""
import csv,collections as C,random
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
T=C.defaultdict(list); fs={}
for r in rows:
    if r['object_type']!='seal': continue
    T[r['text_no']]+= [int(t) for t in r['signs_clean'].split() if t.isdigit() and t!='0']
    fs[r['text_no']]=r['fs80']
uni=[(t,'with' if fs[t] in ('13','14') else 'without') for t in T if fs[t] in ('11','12','13','14') and T[t]]
print(C.Counter(l for t,l in uni))
feat=lambda s:(s[-1]==342, bool(set(s)&{267,391}), len(s))
random.seed(113)
for j,nm in enumerate(['jar-final','opener','length']):
    def diff(lab):
        a=[feat(T[t])[j] for (t,_),l in zip(uni,lab) if l=='with']; b=[feat(T[t])[j] for (t,_),l in zip(uni,lab) if l=='without']
        return sum(a)/len(a)-sum(b)/len(b),sum(a)/len(a),sum(b)/len(b)
    lab=[l for t,l in uni]; o=diff(lab); nl=[]
    for _ in range(5000):
        random.shuffle(lab); nl.append(diff(lab)[0])
    print('%-10s with %.2f without %.2f P=%.4f'%(nm,o[1],o[2],sum(abs(v)>=abs(o[0]) for v in nl)/5000))
