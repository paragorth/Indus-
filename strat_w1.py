"""S234: Is the single stroke W1 (M97/98) a numeral 'one' or a marker? Compare W1 with W3/W4 (clear numerals): share of tokens followed
by another numeral, by a counted sign (S198), text-final, text-initial. Control: permutation of the W1/W3+W4 label among these tokens 5000x.
IM77 check with M97/M98 vs M102-M105."""
import json,collections as C,random,csv
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
NUM=(set(range(1,8))|set(range(12,21))|set(range(25,30))|set(range(32,38))|{39,55,56})-{2}
CT={740,220,700,390,405,900}
def feats(s,i):
    nx=s[i+1] if i+1<len(s) else None
    return (nx in NUM, nx in CT, nx is None, i==0)
seen=set(); tok=[]
for x in d:
    s=tuple(x['seq'])
    if (s,x['site']) in seen: continue
    seen.add((s,x['site']))
    for i,w in enumerate(s):
        if w in (1,3,4): tok.append(('W1' if w==1 else 'W34',)+feats(s,i))
names=['next numeral','next counted sign','text-final','text-initial']; random.seed(107)
for j,nm in enumerate(names,1):
    f=lambda T:(sum(t[j] for t in T if t[0]=='W1')/sum(1 for t in T if t[0]=='W1'))-(sum(t[j] for t in T if t[0]!='W1')/sum(1 for t in T if t[0]!='W1'))
    o=f(tok); lab=[t[0] for t in tok]; nl=[]
    for _ in range(3000):
        random.shuffle(lab); nl.append(f([(l,)+t[1:] for l,t in zip(lab,tok)]))
    a=[t[j] for t in tok if t[0]=='W1']; b=[t[j] for t in tok if t[0]!='W1']
    print('%-18s W1 %d/%d  W3+W4 %d/%d  diff %+.3f P=%.4f'%(nm,sum(a),len(a),sum(b),len(b),o,sum(abs(v)>=abs(o) for v in nl)/3000))
print('what follows W1:',C.Counter(s[i+1] if i+1<len(s) else 'END' for s,_ in [(tuple(x['seq']),0) for x in d] for i,w in enumerate(s) if w==1).most_common(10))
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv'))); seen=set(); c=C.Counter()
for r in rows:
    s=tuple(int(t) for t in r['signs_clean'].split() if t.isdigit() and t!='0')
    if s in seen: continue
    seen.add(s)
    for i,w in enumerate(s):
        if w in (97,98,102,103,104,105):
            nx=s[i+1] if i+1<len(s) else None
            c[('one' if w in (97,98) else 'three/four', 'next numeral' if nx is not None and 87<=nx<=121 and nx!=99 else 'final' if nx is None else 'other')]+=1
print('IM77',dict(c))
