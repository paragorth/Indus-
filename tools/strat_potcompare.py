"""S354: pots against every other object class, on the business-frame features.
Per class: n objects, mean length, % with an opener at the start (817/861/820/920/692), % ending in the jar W740,
% with a counted tree (numeral + 390/405/407), % whose whole text also occurs on a seal (other object), % single-sign.
Control: 95% bootstrap intervals (1000x) for the pot classes vs the pooled rest."""
import json,collections,random,sys
C=json.load(open('data/derived/merged-corpus-canonical.json'))
NUM={1,3,4,5,16,17,18,31,32,33,34}; TREE={390,405,407}; OPEN={817,861,820,920,692}
def cls(t):
    if t.startswith('POT:T:g'): return 'pot graffito (after firing)'
    if t.startswith('POT:T:s'): return 'pot stamped with a seal'
    if t.startswith('POT:T:p') or t.startswith('POT:M'): return 'pot marked before firing'
    return {'SEAL':'seal','TAB':'tablet','TAG':'tag (clay sealing)','BNGL':'bangle','ROD':'rod','IMPL':'tool'}.get(t.split(':')[0])
key=sys.argv[1] if len(sys.argv)>1 else 'seq_raw'
R=[(cls(r['type']),tuple(r[key])) for r in C if r.get(key) and cls(r['type'])]
sealtexts=collections.Counter(s for c,s in R if c=='seal')
def feats(c,s):
    on_seal=sealtexts[s]-(1 if c=='seal' else 0)>0
    return [len(s), s[0] in OPEN, s[-1]==740, any(s[i] in TREE and s[i-1] in NUM for i in range(1,len(s))), on_seal, len(s)==1]
names=['mean signs','opener %','ends in jar %','counted tree %','same text on a seal %','one sign only %']
by=collections.defaultdict(list)
for c,s in R: by[c].append(feats(c,s))
def summ(rows): 
    n=len(rows); return [sum(r[0] for r in rows)/n]+[100*sum(r[i] for r in rows)/n for i in range(1,6)]
print(f'[{key}]','class'.ljust(30),'n'.rjust(5),*[x.rjust(10) for x in ['signs','opener','jar end','cnt tree','on seal','1 sign']])
for c in sorted(by,key=lambda c:-len(by[c])):
    v=summ(by[c]); print(' '*len(key)+'  '+c.ljust(30),str(len(by[c])).rjust(5),*[f'{x:10.1f}' if i==0 else f'{x:9.0f}%' for i,x in enumerate(v)])
random.seed(4)
for c in [k for k in by if k.startswith('pot')]:
    rows=by[c]; lo=[];hi=[]
    boots=[summ(random.choices(rows,k=len(rows))) for _ in range(1000)]
    ci=[(sorted(b[i] for b in boots)[25],sorted(b[i] for b in boots)[974]) for i in range(6)]
    print('  95% CI',c,' '.join(f'{names[i]}: {a:.1f}-{b:.1f}' for i,(a,b) in enumerate(ci)))
