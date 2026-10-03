"""S-DARK-29: THE SAME-HOLDER TEST. Does textual kinship between seals predict physical proximity of find spots?
Credential model: holder in the middle, grade in the closer -> seals sharing a middle but differing in closer were held by
one person/firm and should lie closer together (same area, same room, similar depth) than frequency-matched random pairs.
Stock-text/workshop model: kinship follows production (material, emblem), not deposition.
  cycle 1: kinship classes (identical text; same middle + different closer; same closer unit + different middle;
           one-sign substitution in the same slot; shared emblem only, no shared sign) at Mohenjo-daro and Harappa.
           Share of pairs in the same area-section / same room (area+block+room-grid) / same recorded period and mean
           |depth difference|, against NP draws of random same-site pairs matched on object type x text length.
  cycle 2: reverse: seals found in the same room -- is their textual similarity (Jaccard, same middle, same closer unit,
           same opener, identical text) higher than for rooms made by permuting room labels within site x object type?
           Seals vs sealings/tablets separately.
  cycle 3: production control: the cycle-1 null additionally matched on material x coarse emblem; and do kin pairs share
           material / emblem more than the length-matched null (production signal)?
  cycle 4: replication at Kalibangan, Chanhu-daro (room-grid) and Lothal (area); direct test 'one holder, several grades':
           same-room pairs with the same middle and a different closer vs the room-permutation null, every site.
Spatial fields come from data/derived/merged-corpus-canonical.json (area-section, block-house, room-grid, time);
depth is joined from data/raw/inscriptions.csv by CISI number (first face; feet, metres x 3.281).
Usage: python3 tools/dark_loop29.py <cycle 1|2|3|4> <seq_raw|seq_strong|seq_all> [nperm]
"""
import json,sys,random,collections,math,csv,re,itertools
C=json.load(open('data/derived/merged-corpus-canonical.json'))
BR=json.load(open('data/derived/bridge_extended.json'))
CY=int(sys.argv[1]); LV=sys.argv[2]; NP=int(sys.argv[3]) if len(sys.argv)>3 else 1000
rnd=random.Random(29)
def bad(v): return v is None or v.strip() in ('-','--','- -','')
# depth from CSV by cisi
DEPTH={}
for r in csv.DictReader(open('data/raw/inscriptions.csv')):
    c=r['cisi']
    if bad(c) or c in DEPTH: continue
    d=r['depth'].strip()
    m=re.match(r'^-?\s*(-?\d+(?:\.\d+)?)\s*(ft|m)$',d)
    if m:
        v=abs(float(m.group(1))); DEPTH[c]=v*3.281 if m.group(2)=='m' else v
def oclass(t):
    t=t.split(':')[0]
    return {'SEAL':'seal','TAB':'tablet','TAG':'tablet'}.get(t,'other')
def emb(s):
    s=(s or '').strip()
    if s in ('','-','None'): return None
    return s.split(':')[0]
OBJ=[]
for r in C:
    s=r[LV]
    if not s or len(s)<2 or r['complete']!='Y': continue
    oc=oclass(r['type'])
    if oc=='other': continue
    area=None if bad(r['area-section']) else r['area-section'].strip()
    room=None
    block=None if (bad(r['block-house']) or area is None) else (area,r['block-house'].strip())
    if not bad(r['room-grid']):
        room=(area or '', (r['block-house'] or '').strip(), r['room-grid'].strip())
    OBJ.append(dict(cisi=r['cisi'],site=r['site'],oc=oc,typ=r['type'],seq=tuple(s),area=area,room=room,block=block,
                    depth=DEPTH.get(r['cisi']),time=None if bad(r['time']) else r['time'].strip(),
                    mat=None if bad(r['material']) else r['material'].strip().capitalize(),emb=emb(r['symbol'])))
# ---------------- frame parser (S310 parse_all.py + S331 openers, as in dark_loop19/22/26) ----------------
OPEN={817,861,820,920,692}; MARK={2,60}; MJAR={741,742,745}; SUF={400,90}; CL=[740,520,151,156,527,226,617,154,158,236,700]
FISH={235,240,233,231,220}; NUM={1,3,4,5,16,17,18,31,32,33,34,55,56}
left=collections.defaultdict(collections.Counter)
for o in OBJ:
    s=list(o['seq'])
    while len(s)>1 and s[-1] in SUF: s.pop()
    if len(s)>=2 and s[-1] in CL: left[s[-1]][s[-2]]+=1
QUAL={}
for c,cnt in left.items():
    tot=sum(cnt.values()); acc=0; q=set()
    for a,n in cnt.most_common():
        if acc/tot>=0.6: break
        q.add(a); acc+=n
    QUAL[c]=q
def parse(s):
    lab=['NAME']*len(s); i=0; j=len(s)
    if s[0] in OPEN:
        lab[0]='OPENER'; i=1
        if len(s)>1 and s[1] in MARK:
            lab[1]='MARKER'; i=2
            if s[0]==920 and len(s)>2 and s[2] in MJAR: lab[2]='MARKER'; i=3
    while j-1>i and s[j-1] in SUF and j>=2 and (s[j-2] in CL or s[j-2] in SUF): lab[j-1]='SUFFIX'; j-=1
    if j-1>=i and s[j-1] in CL:
        c=s[j-1]; lab[j-1]='CLOSER'; j-=1
        if c==520:
            if j-2>=i and s[j-1]==33 and s[j-2] in (705,706): lab[j-1]=lab[j-2]='TITLE'; j-=2
            while j-1>=i and s[j-1] in FISH: lab[j-1]='TITLE'; j-=1
        elif c==740:
            if j-1>=i and s[j-1]==100: lab[j-1]='TITLE'; j-=1
            if j-1>=i and s[j-1] in QUAL.get(c,()): lab[j-1]='TITLE'; j-=1
        elif j-1>=i and s[j-1] in QUAL.get(c,()): lab[j-1]='TITLE'; j-=1
        if j-1>=i and s[j-1] in NUM and lab[j]=='TITLE': lab[j-1]='TITLE'; j-=1
    for k in range(i,j-1):
        if s[k] in NUM and lab[k]=='NAME' and lab[k+1]=='NAME': lab[k]=lab[k+1]='COUNT'
    for k in range(i,j):
        if s[k] in NUM and lab[k]=='NAME': lab[k]='COUNT'
    return lab
for o in OBJ:
    lab=parse(o['seq']); o['lab']=lab
    o['mid']=tuple(a for a,l in zip(o['seq'],lab) if l in('NAME','COUNT'))
    o['clo']=tuple(a for a,l in zip(o['seq'],lab) if l in('TITLE','CLOSER'))   # closer unit incl. frozen title
    o['opener']=lab[0]=='OPENER'
    o['set']=frozenset(o['seq'])
    o['L']=len(o['seq'])
def fmt(seq): return '-'.join(str(a) for a in seq)
print(f'== S-DARK-29 cycle {CY} level {LV} nperm {NP}; objects {len(OBJ)} (complete, >=2 signs, seals+tablets/sealings); depth joined for {sum(o["depth"] is not None for o in OBJ)}')

# ---------------- proximity statistics for a list of pairs ----------------
def prox(pairs):
    sa=[a['area']==b['area'] for a,b in pairs if a['area'] and b['area']]
    sr=[a['room']==b['room'] for a,b in pairs if a['room'] and b['room']]
    sb=[a['block']==b['block'] for a,b in pairs if a['block'] and b['block']]
    dd=[abs(a['depth']-b['depth']) for a,b in pairs if a['depth'] is not None and b['depth'] is not None]
    st=[a['time']==b['time'] for a,b in pairs if a['time'] and b['time']]
    f=lambda x:(sum(x)/len(x) if x else float('nan'))
    return dict(n=len(pairs),area=f(sa),n_area=len(sa),k_area=sum(sa),room=f(sr),n_room=len(sr),k_room=sum(sr),block=f(sb),n_block=len(sb),k_block=sum(sb),
                depth=f(dd),n_depth=len(dd),time=f(st),n_time=len(st),
                mat=f([a['mat']==b['mat'] for a,b in pairs if a['mat'] and b['mat']]),
                emb=f([a['emb']==b['emb'] for a,b in pairs if a['emb'] and b['emb']]))
def matched_null(pairs,strata_key,pool,NP):
    """for each kin pair (a,b) draw a' from a's stratum and b' from b's stratum (a'!=b', different cisi)."""
    S=collections.defaultdict(list)
    for o in pool: S[strata_key(o)].append(o)
    out=collections.defaultdict(list)
    for _ in range(NP):
        rp=[]
        for a,b in pairs:
            A=S[strata_key(a)]; B=S[strata_key(b)]
            for _t in range(20):
                x=rnd.choice(A); y=rnd.choice(B)
                if x is not y and x['cisi']!=y['cisi']: break
            else: continue
            rp.append((x,y))
        p=prox(rp)
        for k in ('area','block','room','depth','time','mat','emb'): out[k].append(p[k])
    return out
def pval(obs,null,hi=True):
    v=[x for x in null if x==x]
    if not v or obs!=obs: return float('nan')
    return (sum(1 for x in v if (x>=obs if hi else x<=obs))+1)/(len(v)+1)
def mean(v):
    v=[x for x in v if x==x]; return sum(v)/len(v) if v else float('nan')
def ub(k,n): return f'{k}/{n}' if k>0 else f'0/{n} (<{3/n:.3f})' if n else '0/0'

def kin_classes(objs):
    K=collections.defaultdict(list)
    n=len(objs)
    for i in range(n):
        a=objs[i]
        for j in range(i+1,n):
            b=objs[j]
            if a['cisi']==b['cisi'] and a['cisi']!='-': continue
            if a['seq']==b['seq']: K['identical'].append((a,b)); continue
            if len(a['mid'])>=2 and a['mid']==b['mid'] and a['clo']!=b['clo']: K['same_mid_diff_closer'].append((a,b))
            if len(a['mid'])==1 and a['mid']==b['mid'] and a['clo']!=b['clo']: K['same_mid1_diff_closer'].append((a,b))
            if a['clo'] and a['clo']==b['clo'] and a['mid'] and b['mid'] and a['mid']!=b['mid']: K['same_closer_diff_mid'].append((a,b))
            if a['L']==b['L'] and a['L']>=3 and sum(x!=y for x,y in zip(a['seq'],b['seq']))==1: K['minimal_pair'].append((a,b))
            if a['emb'] and a['emb']==b['emb'] and not (a['set']&b['set']): K['emblem_only'].append((a,b))
    return K
ORDER=['identical','same_mid_diff_closer','same_mid1_diff_closer','same_closer_diff_mid','minimal_pair','emblem_only']

def report_classes(site,oc,strata_key,label):
    pool=[o for o in OBJ if o['site']==site and o['oc']==oc]
    K=kin_classes(pool)
    print(f'\n--- {site} {oc}s: {len(pool)} objects; strata = {label}')
    print(f'{"class":24s} {"pairs":>6s} | same-area obs/null P (n) | same-block obs/null P | same-room obs/null P (n) | mean|dDepth| ft obs/null P (n) | same-period obs/null P (n) | same-mat obs/null | same-emb obs/null')
    res={}
    for k in ORDER:
        pairs=K.get(k,[])
        if not pairs: print(f'{k:24s} {0:6d}'); continue
        if len(pairs)>3000: pairs=rnd.sample(pairs,3000)
        ob=prox(pairs); nu=matched_null(pairs,strata_key,pool,NP)
        res[k]=(ob,{x:mean(nu[x]) for x in nu},{x:pval(ob[x],nu[x],hi=(x!='depth')) for x in nu})
        o,m,p=res[k]
        print(f'{k:24s} {len(pairs):6d} | {ub(o["k_area"],o["n_area"])} {o["area"]:.3f}/{m["area"]:.3f} P={p["area"]:.3f} | {ub(o["k_block"],o["n_block"])} {o["block"]:.3f}/{m["block"]:.3f} P={p["block"]:.3f} | {ub(o["k_room"],o["n_room"])} {o["room"]:.3f}/{m["room"]:.3f} P={p["room"]:.3f} | {o["depth"]:.2f}/{m["depth"]:.2f} P={p["depth"]:.3f} (n={o["n_depth"]}) | {o["time"]:.3f}/{m["time"]:.3f} P={p["time"]:.3f} (n={o["n_time"]}) | {o["mat"]:.2f}/{m["mat"]:.2f} | {o["emb"]:.2f}/{m["emb"]:.2f}')
    return K,res

def list_examples(K,k,site,maxn=12):
    print(f'  examples {k} at {site}:')
    for a,b in K.get(k,[])[:maxn]:
        print(f'    {a["cisi"]:8s} {fmt(a["seq"]):28s} mid={fmt(a["mid"])} clo={fmt(a["clo"])} area={a["area"]} room={a["room"] and a["room"][1:]} d={a["depth"]} | {b["cisi"]:8s} {fmt(b["seq"]):28s} clo={fmt(b["clo"])} area={b["area"]} room={b["room"] and b["room"][1:]} d={b["depth"]} -> {"SAME ROOM" if a["room"] and a["room"]==b["room"] else ("same area" if a["area"] and a["area"]==b["area"] else "")}')

if CY==1:
    key=lambda o:(o['typ'],o['L'])
    for site in ('Mohenjo-daro','Harappa'):
        for oc in ('seal','tablet'):
            K,res=report_classes(site,oc,key,'object type x text length')
            if oc=='seal':
                list_examples(K,'same_mid_diff_closer',site)
                list_examples(K,'minimal_pair',site,8)
    # per-class counts of same-room pairs listed
    print('\nNOTE: "room" = area-section + block-house + room-grid; Harappa room-grid is a trench grid square, not a room.')

if CY==3:
    key=lambda o:(o['typ'],o['L'],o['mat'],o['emb'])
    for site in ('Mohenjo-daro','Harappa'):
        for oc in ('seal','tablet'):
            report_classes(site,oc,key,'object type x text length x material x coarse emblem')

# ---------------- cycle 2 / 4: rooms ----------------
def pair_sim(a,b):
    j=len(a['set']&b['set'])/len(a['set']|b['set'])
    return dict(jac=j,mid=float(a['mid']==b['mid'] and len(a['mid'])>=1),mid2=float(a['mid']==b['mid'] and len(a['mid'])>=2),
                clo=float(a['clo']==b['clo'] and bool(a['clo'])),opn=float(a['opener']==b['opener']),ident=float(a['seq']==b['seq']),
                holder=float(a['mid']==b['mid'] and len(a['mid'])>=2 and a['clo']!=b['clo']),
                holder1=float(a['mid']==b['mid'] and len(a['mid'])>=1 and a['clo']!=b['clo'] and a['seq']!=b['seq']),
                minpair=float(a['L']==b['L'] and a['L']>=3 and sum(x!=y for x,y in zip(a['seq'],b['seq']))==1))
STATS=['jac','mid','mid2','clo','opn','ident','holder','holder1','minpair']
def room_stats(groups):
    acc=collections.defaultdict(float); n=0
    for g in groups:
        for a,b in itertools.combinations(g,2):
            if a['cisi']==b['cisi'] and a['cisi']!='-': continue
            s=pair_sim(a,b); n+=1
            for k in STATS: acc[k]+=s[k]
    return {k:(acc[k]/n if n else float('nan')) for k in STATS},n
def room_test(site,oc,keyname='room',strat='typ'):
    pool=[o for o in OBJ if o['site']==site and o['oc']==oc and o[keyname]]
    G=collections.defaultdict(list)
    for o in pool: G[o[keyname]].append(o)
    groups=[g for g in G.values() if len(g)>=2]
    if not groups: print(f'\n--- {site} {oc}s by {keyname}: no groups'); return None
    obs,npairs=room_stats(groups)
    # null: permute labels within site x object type (optionally x length bin)
    S=collections.defaultdict(list)
    sk=(lambda o:o['typ']) if strat=='typ' else (lambda o:(o['typ'],min(o['L'],6)))
    for o in pool: S[sk(o)].append(o)
    null=collections.defaultdict(list)
    for _ in range(NP):
        lab={}
        for k,os_ in S.items():
            labs=[o[keyname] for o in os_]; rnd.shuffle(labs)
            for o,l in zip(os_,labs): lab[id(o)]=l
        G2=collections.defaultdict(list)
        for o in pool: G2[lab[id(o)]].append(o)
        st,_=room_stats([g for g in G2.values() if len(g)>=2])
        for k in STATS: null[k].append(st[k])
    print(f'\n--- {site} {oc}s grouped by {keyname}: {len(pool)} objects, {len(groups)} groups with >=2, {npairs} within-group pairs; null = labels permuted within {strat} ({NP}x)')
    for k in STATS:
        kcount=round(obs[k]*npairs)
        print(f'  {k:8s} obs {obs[k]:.4f} ({ub(kcount,npairs) if k!="jac" else ""}) null {mean(null[k]):.4f} ratio {obs[k]/mean(null[k]) if mean(null[k])>0 else float("nan"):.2f} P={pval(obs[k],null[k]):.3f}')
    return obs,null,npairs

if CY==2:
    for site in ('Mohenjo-daro','Harappa'):
        for oc in ('seal','tablet'):
            room_test(site,oc,'room','typ')
            room_test(site,oc,'room','typL')
            room_test(site,oc,'area','typ')
    # mixed: seals + tablets in one room (do sealings sit with the seals that made them?)
    for site in ('Mohenjo-daro','Harappa'):
        pool=[o for o in OBJ if o['site']==site and o['room']]
        G=collections.defaultdict(list)
        for o in pool: G[o['room']].append(o)
        cross=[(a,b) for g in G.values() for a,b in itertools.combinations(g,2) if a['oc']!=b['oc']]
        if cross:
            s=[pair_sim(a,b) for a,b in cross]
            print(f'\n--- {site} seal x tablet pairs in the same room: {len(cross)}; identical text {ub(sum(int(x["ident"]) for x in s),len(cross))}; same middle {sum(int(x["mid"]) for x in s)}; Jaccard {mean([x["jac"] for x in s]):.3f}')
            allc=[(a,b) for a in pool for b in pool if a['oc']=='seal' and b['oc']=='tablet']
            if allc:
                sub=rnd.sample(allc,min(20000,len(allc))); s2=[pair_sim(a,b) for a,b in sub]
                print(f'    all same-site seal x tablet pairs (sample {len(sub)}): identical {mean([x["ident"] for x in s2]):.4f}; same middle {mean([x["mid"] for x in s2]):.4f}; Jaccard {mean([x["jac"] for x in s2]):.3f}')

if CY==4:
    for site in ('Kalibangan','Chanhu-daro','Lothal','Dholavira'):
        for oc in ('seal','tablet'):
            pool=[o for o in OBJ if o['site']==site and o['oc']==oc]
            if len(pool)<5: continue
            key=lambda o:(o['typ'],o['L'])
            K,res=report_classes(site,oc,key,'object type x text length')
            if oc=='seal': list_examples(K,'same_mid_diff_closer',site,6); list_examples(K,'minimal_pair',site,6); list_examples(K,'identical',site,6)
            room_test(site,oc,'room','typ')
            room_test(site,oc,'area','typ')
    print('\n=== direct test, all sites: same-room seal pairs with the same middle (>=2 signs) and a different closer unit (holder) vs room-label permutation')
    for site in ('Mohenjo-daro','Harappa','Kalibangan','Chanhu-daro','Lothal','Dholavira'):
        r=room_test(site,'seal','room','typ')
        if r:
            pool=[o for o in OBJ if o['site']==site and o['oc']=='seal' and o['room']]
            G=collections.defaultdict(list)
            for o in pool: G[o['room']].append(o)
            for g in G.values():
                for a,b in itertools.combinations(g,2):
                    if pair_sim(a,b)['holder1']: print(f'    HOLDER-PAIR {site} room={a["room"]}: {a["cisi"]} {fmt(a["seq"])} | {b["cisi"]} {fmt(b["seq"])}  (mid {fmt(a["mid"])}; closers {fmt(a["clo"])} vs {fmt(b["clo"])})')
