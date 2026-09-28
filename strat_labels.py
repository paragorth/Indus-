"""S254: Generalise S253. Signs written alone on a separate line of a seal (IM77, line>=2 with one sign): does any of them go with a
particular field symbol beyond chance? All (sign, motif) pairs with the sign on 3+ seals; Fisher exact, Benjamini-Hochberg over all pairs.
Also: are stand-alone label signs more motif-specific than the same signs inside texts?"""
import csv,collections as C
from scipy.stats import fisher_exact
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
fs={}; lab=C.defaultdict(set); allsig=C.defaultdict(set)
for r in rows:
    if r['object_type']!='seal': continue
    fs[r['text_no']]=r['fs80']
    s=[int(t) for t in r['signs_clean'].split() if t.isdigit() and t!='0']
    for w in s: allsig[r['text_no']].add(w)
    if int(r['line'] or 1)>=2 and len(s)==1: lab[r['text_no']].add(s[0])
motif=lambda c:{'11':'unicorn','12':'unicorn','13':'unicorn','14':'unicorn','31':'zebu','32':'zebu','45':'short-horned bull','46':'short-horned bull','65':'buffalo','66':'buffalo','71':'elephant','72':'elephant','91':'tiger','92':'tiger','111':'rhino','112':'rhino'}.get(c,'other')
T=list(fs); M={t:motif(fs[t]) for t in T}
labcount=C.Counter(w for t in T for w in lab[t])
print('stand-alone label signs (3+ seals):',[(w,n) for w,n in labcount.most_common() if n>=3])
tests=[]
for w,n in labcount.items():
    if n<3: continue
    for m in set(M.values()):
        a=sum(1 for t in T if w in lab[t] and M[t]==m); b=sum(1 for t in T if w in lab[t] and M[t]!=m)
        c=sum(1 for t in T if w not in lab[t] and M[t]==m); d=sum(1 for t in T if w not in lab[t] and M[t]!=m)
        p=fisher_exact([[a,b],[c,d]],alternative='greater')[1]; tests.append((p,w,m,a,a+b,c+a))
tests.sort(); k=len(tests)
for i,(p,w,m,a,n,tot) in enumerate(tests[:10]):
    print('M%d as label on %s: %d/%d label seals; motif seals %d; p=%.2g; BH q=%.3g'%(w,m,a,n,tot,p,p*k/(i+1)))
