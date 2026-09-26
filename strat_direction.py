"""S161: Do left-to-right (L/R) texts obey the same frame once read in the recorded reading order?
If the direction label is right, L/R texts should end with the jar (W740) and begin with the opener as often as R/L texts.
If some L/R objects are mirrored stamps or mislabelled, the frame appears reversed (jar-first, opener-last).
Control: the same rates in R/L texts of the same object type."""
import json,collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
OP={817,861,820}
def f(s): return dict(jar_last=s[-1]==740, jar_first=s[0]==740, op_first=s[0] in OP, op_last=s[-1] in OP,
                      two_second=len(s)>1 and s[1]==2, two_penult=len(s)>1 and s[-2]==2)
out=C.defaultdict(lambda: C.defaultdict(C.Counter))
for x in d:
    s=x['seq']; dr=x['dir.'].strip().upper()
    if len(s)<3 or dr not in('R/L','L/R'): continue
    g='seal' if x['type'].startswith('SEAL') else ('tablet' if x['type'].startswith('TAB') else 'other')
    out[g][dr]['n']+=1
    for k,v in f(s).items(): out[g][dr][k]+=v
for g in out:
    for dr in ('R/L','L/R'):
        c=out[g][dr]; n=c['n']
        print(g,dr,n,' '.join(f"{k}={c[k]}({c[k]/n:.0%})" for k in ('jar_last','jar_first','op_first','op_last','two_second','two_penult')))
