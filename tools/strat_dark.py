"""S362: ARROW-IN-THE-DARK machine. Random hypotheses = (random READING of a text) x (random OUTSIDE FACT) x (relation =
mutual information). Train: Mohenjo-daro + Harappa; survivors (perm p < 0.001) are re-tested on HELD-OUT sites (perm
p < 0.01) and must hold on seq_raw, seq_strong and seq_all. Whole-machine control: the same run with facts permuted
within site (expected survivors ~0). Usage: python3 tools/strat_dark.py N_HYP [control]"""
import json,csv,random,math,sys,collections
import numpy as np
NH=int(sys.argv[1]) if len(sys.argv)>1 else 3000; CONTROL=len(sys.argv)>2
C=json.load(open('data/derived/merged-corpus-canonical.json'))
raw={r['cisi']:r for r in csv.DictReader(open('data/raw/inscriptions.csv')) if r['cisi']}
NUM={1:1,3:3,4:4,5:5,16:6,17:7,18:8,31:1,32:2,33:3,34:4}
def fam(s): return s//100 if s>=100 else 0
# ---------- objects with facts ----------
def fbin(x,q):
    try: v=float(x)
    except: return None
    return None if v<=0 else v
OBJ=[]
for r in C:
    if not r.get('seq_raw') or len(r['seq_raw'])<2: continue
    x=raw.get(r['cisi'],{})
    f={'site':r['site'],'region':x.get('region'),'type':r['type'].split(':')[0],'type2':r['type'],
       'emblem':(r.get('symbol') or '').split(':')[0] or None,'cult':x.get('cult'),'material':x.get('material'),
       'color':x.get('color'),'shape':x.get('shape'),'xsec':x.get('cross-section'),'boss':x.get('boss'),'dir':x.get('dir.'),
       'sides':x.get('sides'),'area':r.get('area-section'),'time':x.get('time') or r.get('period'),
       'h':fbin(x.get('horizontal(mm)'),4),'v':fbin(x.get('vertical(mm)'),4),'th':fbin(x.get('thickness(mm)'),4)}
    for k,v in list(f.items()):
        if v in ('','-','--','None','?'): f[k]=None
    OBJ.append({'seq_raw':r['seq_raw'],'seq_strong':r['seq_strong'],'seq_all':r['seq_all'],'f':f})
TRAIN=[o for o in OBJ if o['f']['site'] in ('Mohenjo-daro','Harappa')]; TEST=[o for o in OBJ if o['f']['site'] not in ('Mohenjo-daro','Harappa')]
print('objects',len(OBJ),'train',len(TRAIN),'held-out',len(TEST))
rng=random.Random(7)
if CONTROL:  # permute facts within site
    for key in ('type','type2','emblem','cult','material','color','shape','xsec','boss','dir','sides','area','time','h','v','th','region'):
        bysite=collections.defaultdict(list)
        for o in OBJ: bysite[o['f']['site']].append(o)
        for os_ in bysite.values():
            vals=[o['f'][key] for o in os_]; rng.shuffle(vals)
            for o,v in zip(os_,vals): o['f'][key]=v
FACTS=['site','region','type','type2','emblem','cult','material','color','shape','xsec','boss','dir','sides','area','time','h','v','th']
# ---------- random readings ----------
def make_reading(rng):
    kind=rng.choice(['pos','pair','mod','even','odd','famfl','nfam','rep','nnum','sumnum','maxw','lenpar','run','fampos','dist','first2','last2','contains','posrel'])
    p={}
    if kind=='pos': p['i']=rng.choice([0,1,2,-1,-2,-3])
    if kind=='pair': p['d']=rng.randint(1,4); p['i']=rng.choice([0,1,-2,'any'])
    if kind=='mod': p['base']=rng.randint(2,13); p['m']=rng.randint(2,12); p['digit']=rng.choice(['w','fam','num'])
    if kind=='fampos': p['i']=rng.choice([0,1,-1,-2])
    if kind=='dist': p['a']=rng.choice([740,2,390,220,520,817,861,820,700,400,90,60,176,100,415,590,235,240,233,231,226,900,405,407]); p['b']=rng.choice([740,2,390,220,520,817,861,820,700,400,90,60,176,100,415,590,235,240,233,231,226,900,405,407])
    if kind=='contains': p['s']=rng.randint(1,999)
    if kind=='posrel': p['s']=rng.choice([740,2,390,220,520,700,400,90,60,176,100,415,590,900])
    return kind,p
def read(kind,p,s):
    L=len(s)
    if kind=='pos': i=p['i']; return s[i] if -L<=i<L else None
    if kind=='pair':
        d=p['d']
        if p['i']=='any': return tuple(sorted((s[j],s[j+d]) for j in range(L-d)))[:1] and min((s[j],s[j+d]) for j in range(L-d)) if L>d else None
        i=p['i']; j=i+d if i>=0 else i-d
        return (s[i],s[j]) if -L<=i<L and -L<=j<L else None
    if kind=='mod':
        dig={'w':lambda x:x,'fam':fam,'num':lambda x:NUM.get(x,0)}[p['digit']]
        v=0
        for x in s: v=(v*p['base']+dig(x))%p['m']
        return v
    if kind=='even': return tuple(s[0::2])
    if kind=='odd': return tuple(s[1::2]) or None
    if kind=='famfl': return (fam(s[0]),fam(s[-1]))
    if kind=='nfam': return len({fam(x) for x in s})
    if kind=='rep': return L-len(set(s))
    if kind=='nnum': return sum(x in NUM for x in s)
    if kind=='sumnum': return min(sum(NUM.get(x,0) for x in s),12)
    if kind=='maxw': return max(s)//100
    if kind=='lenpar': return L%2
    if kind=='run':
        best=cur=1
        for a,b in zip(s,s[1:]): cur=cur+1 if fam(a)==fam(b) else 1; best=max(best,cur)
        return best
    if kind=='fampos': i=p['i']; return fam(s[i]) if -L<=i<L else None
    if kind=='dist':
        a,b=p['a'],p['b']
        if a in s and b in s: return max(-3,min(3,s.index(b)-s.index(a)))
        return 'absent'
    if kind=='first2': return tuple(s[:2])
    if kind=='last2': return tuple(s[-2:])
    if kind=='contains': return p['s'] in s
    if kind=='posrel':
        if p['s'] not in s: return 'absent'
        return round(s.index(p['s'])/(L-1),1) if L>1 else 0
# ---------- MI with permutation ----------
def mi_p(xs,ys,nperm,rng):
    xs=np.asarray(xs); ys=np.asarray(ys); n=len(xs)
    _,xi=np.unique(xs,return_inverse=True); _,yi=np.unique(ys,return_inverse=True)
    nx=xi.max()+1; ny=yi.max()+1
    if nx<2 or ny<2 or n<30: return None
    def mi(a,b):
        t=np.bincount(a*ny+b,minlength=nx*ny).reshape(nx,ny)/n
        px=t.sum(1,keepdims=True); py=t.sum(0,keepdims=True)
        nz=t>0; return float((t[nz]*np.log(t[nz]/(px@py)[nz])).sum())
    o=mi(xi,yi); ge=0; r=np.random.default_rng(rng.randint(0,10**9)); yy=yi.copy()
    for k in range(nperm):
        r.shuffle(yy); ge+=mi(xi,yy)>=o
        if ge>=5 and k>=20: return o,(ge+1)/(k+2)   # early stop
    return o,(ge+1)/(nperm+1)
def evaluate(kind,p,fact,objs,level,nperm,rng,minn=40):
    xs=[];ys=[]
    for o in objs:
        y=o['f'][fact]
        if y is None: continue
        x=read(kind,p,o[level])
        if x is None: continue
        xs.append(str(x)); ys.append(str(y))
    if len(xs)<minn: return None
    # collapse rare x and y values (<5) to 'other' to avoid MI inflation
    cx=collections.Counter(xs); cy=collections.Counter(ys)
    xs=[x if cx[x]>=5 else 'other' for x in xs]; ys=[y if cy[y]>=5 else 'other' for y in ys]
    r=mi_p(xs,ys,nperm,rng)
    return (r[0],r[1],len(xs)) if r else None
# ---------- run ----------
surv=[]; tried=0
for h in range(NH):
    kind,p=make_reading(rng); fact=rng.choice(FACTS)
    if fact in ('site','region','area') and kind in ('contains',): pass
    r=evaluate(kind,p,fact,TRAIN,"seq_raw",500,rng)
    tried+=1
    if not r or r[1]>0.0021: continue
    # held-out replication
    r2=evaluate(kind,p,fact,TEST,'seq_raw',1000,rng,minn=60) if fact not in ('site','region') else None
    ok_held = (r2 is not None and r2[1]<0.01)
    # level robustness on train
    r3=[evaluate(kind,p,fact,TRAIN,lv,300,rng) for lv in ('seq_strong','seq_all')]
    ok_lv=all(x and x[1]<0.01 for x in r3)
    surv.append(dict(kind=kind,p=p,fact=fact,train=r,held=r2,ok_held=ok_held,levels=[x[:2] if x else None for x in r3],ok_levels=ok_lv))
print(f'hypotheses tried {tried}; train survivors {len(surv)}; held-out+levels survivors {sum(s["ok_held"] and s["ok_levels"] for s in surv)}; held-out n/a (site facts) {sum(s["held"] is None for s in surv)}')
surv.sort(key=lambda s:-(s['train'][0]))
for s in surv:
    if s['ok_held'] and s['ok_levels']:
        print('SURVIVOR',s['kind'],s['p'],'->',s['fact'],'train MI %.3f p %.4f n %d'%s['train'],'| held MI %.3f p %.4f n %d'%s['held'])
print('--- train-only survivors (site/region facts or failed held-out) top 25:')
for s in surv[:25]:
    if not (s['ok_held'] and s['ok_levels']): print('  ',s['kind'],s['p'],'->',s['fact'],'train MI %.3f p %.4f n %d'%s['train'],'held',s['held'][:2] if s['held'] else None,'levels ok',s['ok_levels'])
json.dump(surv,open('data/derived/strat_dark%s.json'%('_control' if CONTROL else ''),'w'),default=str,indent=0)
