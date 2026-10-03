"""S-DARK-37.2: arrow B (sides as paraphrases / abbreviations / grade-swaps of one message).
Usage: python3 tools/dark_loop37_c2.py <seq_raw|seq_strong|seq_all> [nperm]"""
import sys,collections,random,itertools,json
sys.path.insert(0,'/home/user/Indus-/tools')
from dark_loop37 import *
LV=sys.argv[1] if len(sys.argv)>1 else 'seq_raw'; NP=int(sys.argv[2]) if len(sys.argv)>2 else 1000
rnd=random.Random(372)
BR=json.load(open(BRIDGE))
objs=load_faces(LV)
allseqs=[f['seq'] for o in objs.values() for f in o['faces'] if f['seq']]
parse=make_parser(learn_qual(allseqs))
M,_=collapse(multi(objs,1))
print(f'== S-DARK-37 cycle 2 level {LV} nperm {NP}: {len(M)} collapsed multi-sided objects')
def middle(s):
    lab=parse(s); return tuple(w for w,l in zip(s,lab) if l in ('NAME','COUNT','TITLE'))
def closer_of(s):
    lab=parse(s); c=[w for w,l in zip(s,lab) if l=='CLOSER']; return c[0] if c else None
def jacc(a,b):
    A,B=set(a),set(b); return len(A&B)/len(A|B) if A|B else 0
def lev(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nd=[i]
        for j,y in enumerate(b,1): nd.append(min(d[j]+1,nd[j-1]+1,d[j-1]+(x!=y)))
        d=nd
    return d[-1]
def subseq(a,b):
    """is b a subsequence of a"""
    it=iter(a); return all(x in it for x in b)
def pairstats(a,b):
    ma,mb=middle(a),middle(b)
    return dict(shared=len(set(a)&set(b))>0, jacc=jacc(a,b), shared_mid=bool(set(ma)&set(mb)) if ma and mb else False,
                same_mid=(sorted(ma)==sorted(mb) and len(ma)>=1), ned=lev(a,b)/max(len(a),len(b)), ident=(a==b),
                abbrev=(len(b)>=2 and len(a)>len(b) and set(b)<=set(a)) or (len(a)>=2 and len(b)>len(a) and set(a)<=set(b)),
                abbrev_seq=(len(b)>=2 and len(a)>len(b) and subseq(a,b)) or (len(a)>=2 and len(b)>len(a) and subseq(b,a)),
                grade=(sorted(ma)==sorted(mb) and len(ma)>=1 and closer_of(a) is not None and closer_of(b) is not None and closer_of(a)!=closer_of(b)),
                samecl_diffmid=(closer_of(a) is not None and closer_of(a)==closer_of(b) and sorted(ma)!=sorted(mb)))
KEYS=['shared','jacc','shared_mid','same_mid','ned','ident','abbrev','abbrev_seq','grade','samecl_diffmid']
# pool of faces for the null: by site x type x length
pool=collections.defaultdict(list)
for o in objs.values():
    for f in o['faces']:
        if f['seq']: pool[(o['site'],o['type'],len(f['seq']))].append((o['oid'],f['seq']))
def draw(o,f):
    k=(o['site'],o['type'],len(f['seq'])); cand=[s for oid,s in pool[k] if oid!=o['oid']]
    if len(cand)<3:
        cand=[s for L in (len(f['seq'])-1,len(f['seq'])+1) for oid,s in pool[(o['site'],o['type'],L)] if oid!=o['oid']] or cand
    return rnd.choice(cand) if cand else None
SUBS=[('all',M),('no count faces',[dict(o,faces=[f for f in o['faces'] if not is_count_face(f['seq'])]) for o in M if sum(not is_count_face(f['seq']) for f in o['faces'])>=2]),
      ('Harappa tablets (no count faces)',[dict(o,faces=[f for f in o['faces'] if not is_count_face(f['seq'])]) for o in M if o['site']=='Harappa' and o['ot']=='tablet' and sum(not is_count_face(f['seq']) for f in o['faces'])>=2]),
      ('Mohenjo-daro all',[o for o in M if o['site']=='Mohenjo-daro']),
      ('Mohenjo-daro >=3 faces',[o for o in M if o['site']=='Mohenjo-daro' and len(o['faces'])>=3]),
      ('seals',[o for o in M if o['ot']=='seal']),('sealings',[o for o in M if o['ot']=='sealing']),
      ('non-MD-Harappa',[o for o in M if not o['big']]),
      ('faces >=2 signs both, complete',[dict(o,faces=[f for f in o['faces'] if f['complete'] and len(f['seq'])>=2]) for o in M if sum(f['complete'] and len(f['seq'])>=2 for f in o['faces'])>=2])]
examples=collections.defaultdict(list)
for name,S in SUBS:
    pairs=[(o,a,b) for o in S for a,b in itertools.combinations(o['faces'],2)]
    if len(pairs)<5: print(f'\n-- {name}: {len(pairs)} face pairs, too few'); continue
    obs=collections.Counter()
    for o,a,b in pairs:
        st=pairstats(a['seq'],b['seq'])
        for k in KEYS: obs[k]+=st[k]
        for k in ('abbrev_seq','grade','ident','same_mid'):
            if st[k] and name=='all': examples[k].append((o['cisi'],o['type'],'-'.join(map(str,a['seq'])),'-'.join(map(str,b['seq']))))
    null=collections.defaultdict(list)
    for _ in range(NP):
        c=collections.Counter()
        for o,a,b in pairs:
            r=draw(o,b)
            if r is None: continue
            st=pairstats(a['seq'],r)
            for k in KEYS: c[k]+=st[k]
        for k in KEYS: null[k].append(c[k])
    n=len(pairs)
    print(f'\n-- {name}: {len(S)} objects, {n} face pairs (null: face 2 replaced by a face of another object, same site x type x length)')
    for k in KEYS:
        o_=obs[k]; nl=null[k]; mu=sum(nl)/NP
        if k in ('jacc','ned'): print(f'   mean {k:14s}: {o_/n:.3f} vs null {mu/n:.3f}  P_hi={pval(o_,nl):.3f} P_lo={pval(o_,nl,"lo"):.3f}')
        else: print(f'   {k:19s}: {o_:4d}/{n} = {o_/n:.3f} vs null {mu/n:.3f} ({mu:.1f})  P_hi={pval(o_,nl):.3f} P_lo={pval(o_,nl,"lo"):.3f}')
print('\n-- examples (all, recorded face order):')
for k in ('ident','same_mid','grade','abbrev_seq'):
    ex=examples[k]; print(f'  {k}: {len(ex)}')
    for e in ex[:30]: print('     ',*e)
# which signs are shared across faces more than chance? (sign-level, all pairs, no count faces)
S=[dict(o,faces=[f for f in o['faces'] if not is_count_face(f['seq'])]) for o in M if sum(not is_count_face(f['seq']) for f in o['faces'])>=2]
pairs=[(o,a,b) for o in S for a,b in itertools.combinations(o['faces'],2)]
sh=collections.Counter(); shn=collections.Counter()
for o,a,b in pairs:
    for w in set(a['seq'])&set(b['seq']): sh[w]+=1
for _ in range(200):
    for o,a,b in pairs:
        r=draw(o,b)
        if r:
            for w in set(a['seq'])&set(r): shn[w]+=1
print('\n-- signs shared between two faces of one object (no count faces), observed vs null/200:')
for w,c in sh.most_common(20): print(f'   {Mname(w,BR):16s} {c:3d} vs {shn[w]/200:.1f}')
