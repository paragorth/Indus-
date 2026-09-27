"""S185: Are opener seals physically larger (authority/rank) once text length is controlled? Real dimensions from
data/raw/inscriptions.csv (horizontal/vertical mm), joined by CISI id to reading-order texts. Square SEAL:S only.
Control: permutation of the opener label within (site x text-length) strata; statistic = mean max side of opener seals."""
import csv,json,random,collections as C,statistics as st
raw={x['cisi']:x for x in csv.DictReader(open('data/raw/inscriptions.csv')) if x['cisi'] not in ('-','')}
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
OP={817,861,820}
def num(v):
    try: return float(v)
    except: return 0.0
rows=[]
for x in d:
    r=raw.get(x['cisi'])
    if not r or x['type']!='SEAL:S' or len(x['seq'])<2: continue
    h,v=num(r['horizontal(mm)']),num(r['vertical(mm)'])
    if h<=0 or v<=0: continue
    rows.append(dict(size=max(h,v),op=x['seq'][0] in OP and x['seq'][1]==2,L=min(len(x['seq']),7),site=x['site'],jar=x['seq'][-1]==740,animal=x['symbol'].split(':')[0]))
print('seals',len(rows),'opener',sum(r['op'] for r in rows))
o=[r['size'] for r in rows if r['op']]; n=[r['size'] for r in rows if not r['op']]
print('raw mean size opener %.1f vs other %.1f'%(st.mean(o),st.mean(n)))
S=C.defaultdict(list)
for r in rows: S[(r['site'],r['L'])].append(r)
obs=st.mean(o); random.seed(0); ge=0; tot=0
for _ in range(5000):
    vals=[]
    for rs in S.values():
        k=sum(r['op'] for r in rs)
        if k: vals+= [r['size'] for r in random.sample(rs,k)]
    m=st.mean(vals); tot+=m; ge+=m>=obs
print('length+site controlled: expected %.2f, observed %.2f, P=%.4f'%(tot/5000,obs,ge/5000))
print('size by length',[(L,round(st.mean([r['size'] for r in rows if r['L']==L]),1)) for L in range(2,8)])
u=[r for r in rows if r['animal']=='Bull1']; print('unicorn only: opener %.1f vs other %.1f'%(st.mean([r['size'] for r in u if r['op']]),st.mean([r['size'] for r in u if not r['op']])))
