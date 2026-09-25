"""Part 3 engine. All numbers computed from the frozen corpus; no hand-copied figures.
Outputs: lexicon.json (features + functional type + semantic survivors per sign), readings.json, and prints."""
from p3lib import *
from collections import Counter, defaultdict
import math, json, random, itertools

HOMEt = texts()                    # dedup home corpus, len>=2
MDS   = texts('MD_SEAL')
ALLT  = [x['text'] for x in HOMEt]
NUMV  = {1:1,2:2,3:3,4:4,5:5,16:6,17:7,18:8,31:'I',32:'II',33:'III'}
CNUM  = {3,4,5,16,17,18}

# ---------- object type proxy
def otype(x):
    if x['material']=='Copper': return 'copper'
    if x['icon'] and x['material'] in ('Steatite','NULL','steatite'): return 'seal'
    if x['material'] in ('Faience','Clay','Terracotta','Paste'): return 'faience_clay'
    if not x['icon']: return 'tablet'
    return 'other'
for x in HOMEt: x['ot']=otype(x)

# ---------- per-sign features
tok=Counter(s for t in ALLT for s in t)
fin=Counter(t[-1] for t in ALLT); ini=Counter(t[0] for t in ALLT)
left=defaultdict(Counter); right=defaultdict(Counter)
for t in ALLT:
    for a,b in zip(t,t[1:]): right[a][b]+=1; left[b][a]+=1
byot=defaultdict(Counter); totot=Counter()
for x in HOMEt:
    for s in x['text']: byot[x['ot']][s]+=1; totot[x['ot']]+=1
big=Counter()
for t in ALLT:
    for p in set(zip(t,t[1:])): big[p]+=1
md=set(x['text'] for x in TEXTS if x['register']=='MD_SEAL' and len(x['text'])>=2)
ha=set(x['text'] for x in TEXTS if x['site']=='Harappa' and len(x['text'])>=2)
shared=md&ha
sharedsigns=Counter(s for t in shared for s in t)

def H(c):
    n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values()) if n else 0.0

def numprofile(s):
    c=Counter()
    for t in ALLT:
        for a,b in zip(t,t[1:]):
            if a in NUMV and b==s: c[NUMV[a]]+=1
    return c

def feat(s):
    n=tok[s]; np_=numprofile(s); np_.pop(2,None); nn=sum(np_.values())  # glyph 2 is the connective; excluded from numeral profiling
    short={k:v for k,v in np_.items() if isinstance(k,int) and k>=3}; ns=sum(short.values())
    if ns>=10 and len(short)>=3 and max(short.values())/ns<=0.6 and nn/n>=0.15: numkind='COUNT-PARADIGM'
    elif nn>=8 and max(np_.values())/nn>=0.8: numkind='FIXED-COMPOUND(%s)'%max(np_,key=np_.get)
    elif nn>=8: numkind='mixed'
    else: numkind='-'
    sealshare=byot['seal'][s]/n; base=totot['seal']/sum(totot.values())
    enr=math.log2((sealshare+.005)/(base+.005))-math.log2((1-sealshare+.005)/(1-base+.005))
    rec=0; tot=0
    for t in ALLT:
        for i,x in enumerate(t):
            if x!=s: continue
            tot+=1
            if (i>0 and big[(t[i-1],x)]>=2) or (i<len(t)-1 and big[(x,t[i+1])]>=2): rec+=1
    d=BRIDGE.get(str(s),{})
    desc=d.get('desc','?'); mah=','.join(d.get('mahadevan',[]))
    return dict(glyph=s, n=n, mahadevan=mah, shape=desc[:70], cls=cls(s),
                final_pct=round(100*fin[s]/n), initial_pct=round(100*ini[s]/n),
                numeral_before_pct=round(100*nn/n), numeral_profile={str(k):v for k,v in sorted(np_.items(),key=lambda kv:str(kv[0]))},
                numeral_kind=numkind, seal_enrichment=round(enr,2),
                tablet_pct=round(100*(byot['tablet'][s]+byot['copper'][s]+byot['faience_clay'][s])/n),
                context_recurrence_pct=round(100*rec/tot), cross_site_formula_tokens=sharedsigns[s],
                left_entropy=round(H(left[s]),2), right_entropy=round(H(right[s]),2),
                left_types=len(left[s]), right_types=len(right[s]),
                is_person=('person' in desc.lower()), is_fish=('fish' in desc.lower()), is_tree=('tree' in desc.lower()),
                is_strokes=('stroke' in desc.lower() and s in NUMV))

# ---------- functional typing (Tier-1 layer). Rules are explicit; every rule is a computable feature.
def functional_type(f):
    s=f['glyph']
    if s in NUMV and s not in (2,): return 'NUMERAL/STROKE'
    if s==2: return 'CONNECTIVE (also stroke-2 in fixed compounds)'
    if f['cls']=='I': return 'OPENER (initial paradigm)'
    if f['cls']=='C': return 'CONNECTIVE'
    if f['cls']=='T':
        return 'ENDING: register-neutral (grammatical-like)' if abs(f['seal_enrichment'])<0.4 else 'ENDING: register-specific (lexical-like)'
    if f['cls']=='P': return 'PRE-ENDING SELECTOR (title/rank-like)'
    if f['numeral_kind']=='COUNT-PARADIGM': return 'COUNTED NOUN (takes a variable quantity)'
    if f['numeral_kind'].startswith('FIXED'): return 'COMPOUND ELEMENT (fixed stroke+sign word)'
    if f['context_recurrence_pct']>=75 and f['n']>=40: return 'FORMULAIC CORE ELEMENT'
    return 'OPEN-CLASS CORE ELEMENT (name-like)'

# ---------- semantic candidate scoring: the degeneracy game (per sign, joint constraints applied after)
CANDS=['personal-name element','lineage/clan name','title/office','deity/astral name element','commodity/material',
       'measure/unit','animal','person/occupation','place/institution','document-function marker','grammatical suffix',
       'phonetic-only (rebus)','no stable meaning']
def semantic_survivors(f, hyp):
    """Return candidates NOT excluded by computable constraints. hyp in {'LING','ADMIN'}.
    Each exclusion is a rule with a reason; rules only exclude, never rank."""
    out={}
    for c in CANDS:
        why=[]
        cl=f['cls']; nk=f['numeral_kind']
        # C1 grammar class
        if cl=='T' and c in ('commodity/material','measure/unit','animal') and nk!='COUNT-PARADIGM': why.append('C1: ending never counted')
        if cl=='C' and c not in ('grammatical suffix','phonetic-only (rebus)','no stable meaning','document-function marker'): why.append('C1: connective is functional')
        if cl=='I' and c in ('grammatical suffix','measure/unit'): why.append('C1: opener position')
        if cl=='P' and c in ('grammatical suffix','commodity/material','measure/unit'): why.append('C1: pre-ending selector')
        # C2 countability
        if nk=='COUNT-PARADIGM' and c in ('grammatical suffix','document-function marker','title/office','lineage/clan name','no stable meaning'): why.append('C2: takes variable count')
        if nk.startswith('FIXED') and c in ('commodity/material','measure/unit','grammatical suffix'): why.append('C2: fixed stroke compound, not counted')
        # C3 pictorial (logographic reading) - only bites under ADMIN (logographic) hypothesis
        if hyp=='ADMIN':
            if f['is_person'] and c in ('commodity/material','measure/unit','animal'): why.append('C3: person pictogram')
            if f['is_fish'] and c in ('person/occupation','measure/unit'): why.append('C3: fish pictogram')
            if f['is_tree'] and c in ('person/occupation','animal'): why.append('C3: tree pictogram')
            if c=='phonetic-only (rebus)': why.append('C3: ADMIN is logographic')
        if hyp=='LING':
            if c=='document-function marker': why.append('H-LING: seals carry names/titles, not document types')
        # C4 register
        if f['seal_enrichment']<-1.5 and c in ('personal-name element','lineage/clan name'): why.append('C4: tablet-concentrated, names sit on seals')
        if abs(f['seal_enrichment'])<0.3 and f['n']>=200 and c=='document-function marker': why.append('C4: register-neutral, a document marker would be register-specific')
        # C6 cross-site formulaic
        if f['cross_site_formula_tokens']>=5 and c=='personal-name element': why.append('C6: recurs verbatim at both cities')
        # open-class evidence
        if f['cls']=='X' and f['context_recurrence_pct']<60 and c in ('grammatical suffix','document-function marker'): why.append('open-class distribution')
        out[c]=why
    return {c:w for c,w in out.items()}

TOP=[s for s,_ in Counter(s for t in ALLT for s in t).most_common(60)]
LEX=[]
for s in TOP:
    f=feat(s); f['functional_type']=functional_type(f)
    f['survivors_LING']=[c for c,w in semantic_survivors(f,'LING').items() if not w]
    f['survivors_ADMIN']=[c for c,w in semantic_survivors(f,'ADMIN').items() if not w]
    f['excluded_LING']={c:w for c,w in semantic_survivors(f,'LING').items() if w}
    f['excluded_ADMIN']={c:w for c,w in semantic_survivors(f,'ADMIN').items() if w}
    both=set(f['survivors_LING'])&set(f['survivors_ADMIN'])
    f['survivors_both']=sorted(both)
    f['semantic_tier']= 'Tier1(function fixed; meaning open)' if len(both)>=3 else ('Tier2(narrow)' if len(both)==2 else 'Tier1(unique)')
    LEX.append(f)
json.dump(LEX, open('/home/claude/p3/lexicon.json','w'), indent=1)

# ---------- joint constraint J1: substitution paradigms share a category set
pos_index=defaultdict(set)
for t in ALLT:
    for i in range(len(t)): pos_index[t[:i]+('*',)+t[i+1:]].add(t[i])
pair=Counter()
for key,f_ in pos_index.items():
    if 2<=len(f_)<=8:
        for a,b in itertools.combinations(sorted(f_),2): pair[(a,b)]+=1
adj=defaultdict(set)
for (a,b),c in pair.items():
    if c>=3: adj[a].add(b); adj[b].add(a)
BY={f['glyph']:f for f in LEX}
for f in LEX:
    partners=[p for p in adj[f['glyph']] if p in BY]
    f['paradigm_partners']=partners
    inter=set(f['survivors_both'])
    for p in partners: inter&=set(BY[p]['survivors_both'])
    f['survivors_after_paradigm']=sorted(inter) if partners else f['survivors_both']
json.dump(LEX, open('/home/claude/p3/lexicon.json','w'), indent=1)

# ---------- degeneracy summary
print("== SEMANTIC DEGENERACY (top 60 signs): number of surviving meaning-categories per sign")
print("sign  n    class  functional-type                                   |LING| |ADMIN| |both|")
for f in LEX:
    print(f"{f['glyph']:4d} {f['n']:4d}  {f['cls']}  {f['functional_type'][:48]:48s}  {len(f['survivors_LING']):3d}   {len(f['survivors_ADMIN']):3d}   {len(f['survivors_both']):3d}   {f['shape'][:30]}")
avg=sum(len(f['survivors_both']) for f in LEX)/len(LEX)
avg2=sum(len(f['survivors_after_paradigm']) for f in LEX)/len(LEX)
print(f"mean surviving categories per sign (both hypotheses): {avg:.1f} of {len(CANDS)}; after paradigm intersection: {avg2:.1f}")
for f in LEX:
    if f['paradigm_partners']: print(f"   {f['glyph']:4d} partners={f['paradigm_partners']} -> {f['survivors_after_paradigm']}")
print("signs with <=2 survivors:", [(f['glyph'],f['survivors_both']) for f in LEX if len(f['survivors_both'])<=2])

# ---------- reading generator
def parse(t):
    L=len(t); out=[]
    for i,s in enumerate(t):
        c=cls(s); tag=c
        if s in NUMV and s!=2: tag='N'
        # count construction / fixed compound
        note=''
        if i<L-1 and s in NUMV and s!=2:
            nxt=t[i+1]; f=next((g for g in LEX if g['glyph']==nxt),None)
            fk=f['numeral_kind'] if f else '-'
            if fk=='COUNT-PARADIGM': note=f'quantity {NUMV[s]} of next'
            elif fk.startswith('FIXED'): note='fixed compound with next'
            else: note='stroke sign, role unclear'
        out.append((s,tag,note))
    return out
def describe(x):
    t=x['text']; p=parse(t); L=len(t)
    hascount=any(a in CNUM and b in {390,405,407,900,220} for a,b in zip(t,t[1:]))
    hasfixed=any(a in NUMV and next((g for g in LEX if g['glyph']==b),{}).get('numeral_kind','').startswith('FIXED') for a,b in zip(t,t[1:]))
    endT = t[-1] in TERM
    opener = t[0] in INIT
    stack = L>=2 and t[-1] in TERM and t[-2] in TERM
    # text-type odds from corpus base rates on MD seals (empirical, not asserted)
    kind=[]
    if opener: kind.append('opens with the opener formula (institution/locus/document-type slot)')
    if hascount: kind.append('contains a quantity construction (N of counted-noun)')
    if hasfixed: kind.append('contains a fixed stroke+sign compound (a lexical unit, e.g. 6-fish / 7-X)')
    if endT: kind.append('closes with an ending-paradigm sign' + (' (stacked)' if stack else ''))
    if not (opener or hascount or hasfixed or endT): kind.append('bare core sequence (no opener, no quantity, no ending) - name-like or fragment')
    return dict(id=x['id'], cisi=x['cisi'], site=x['site'], material=x['material'], icon=x['icon'], text=list(t),
                parse=[dict(glyph=s,tag=tag,shape=shape(s)[:45],note=n) for s,tag,n in p], summary=kind)

random.seed(2026)
known=[x for x in HOMEt if x['cisi'] in ('M-14','M-15')]
pool=[x for x in HOMEt if x['register']=='MD_SEAL' and x['cisi'].startswith('M-') and x['cisi'] not in ('M-14','M-15') and len(x['text'])>=4]
sample=random.sample(pool,10)
READ=[describe(x) for x in known+sample]
json.dump(READ, open('/home/claude/p3/readings.json','w'), indent=1)
print("\n== READINGS")
for r in READ:
    print(f"\n{r['cisi']} ({r['material']}, {r['icon']}): text {r['text']}")
    for p in r['parse']: print(f"   {p['glyph']:4d} [{p['tag']}] {p['shape']:45s} {p['note']}")
    for k in r['summary']: print("   ->", k)
