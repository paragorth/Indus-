"""S-DARK-37: multi-sided objects as ONE TEXT (A), PARAPHRASES (B) or INDEPENDENT ENTRIES (C).
Shared loader + frame parser for the loop-37 cycles.
Faces are the rows of data/raw/inscriptions.csv keyed id = n.k (k = recorded face order, S186/S215);
`sides` is the number of inscribed sides. Signs are reversed to reading order and 000 dropped, as in
merged-corpus-canonical.json; seq_strong / seq_all are produced by the per-sign maps read off that file
(S268 merges), so the three levels here are the canonical ones applied to the current csv (S-DARK-23 caution).
Usage (cycles): python3 tools/dark_loop37_c<N>.py <seq_raw|seq_strong|seq_all> [nperm]
"""
import csv,json,re,collections,random,itertools,math
ROOT='/home/user/Indus-'
RAW=ROOT+'/data/raw/inscriptions.csv'
CANON=ROOT+'/data/derived/merged-corpus-canonical.json'
BRIDGE=ROOT+'/data/derived/bridge_extended.json'
IM77=ROOT+'/data/im77/im77_corpus_lines.csv'

def merge_maps():
    C=json.load(open(CANON))
    ms=collections.defaultdict(collections.Counter); ma=collections.defaultdict(collections.Counter)
    for r in C:
        if len(r['seq_raw'])==len(r['seq_strong'])==len(r['seq_all']):
            for a,b,c in zip(r['seq_raw'],r['seq_strong'],r['seq_all']): ms[a][b]+=1; ma[a][c]+=1
    MS={a:c.most_common(1)[0][0] for a,c in ms.items()}; MA={a:c.most_common(1)[0][0] for a,c in ma.items()}
    return {'seq_raw':{},'seq_strong':MS,'seq_all':MA}

def otype(t):
    t0=t.split(':')[0]
    return {'SEAL':'seal','TAB':'tablet','POT':'pot','TAG':'sealing','BNGL':'bangle'}.get(t0,'other')

def load_faces(level='seq_raw'):
    """returns dict oid -> dict(site,type,ot,sides,faces=[dict(k,seq,complete,dir,text)])"""
    mp=merge_maps()[level]
    objs=collections.OrderedDict()
    for r in csv.DictReader(open(RAW)):
        oid,_,k=r['id'].partition('.')
        toks=[int(t) for t in re.findall(r'\d{3}',r['text'])][::-1]
        seq=[mp.get(t,t) for t in toks if t not in (0,999)]
        o=objs.setdefault(oid,dict(oid=oid,cisi=r['cisi'],site=r['site'],type=r['type'],ot=otype(r['type']),
                                   sides=r['sides'],faces=[],big=r['site'] in ('Mohenjo-daro','Harappa')))
        o['faces'].append(dict(k=int(k or 1),seq=seq,complete=r['complete']=='Y',dir=r['dir.'].strip(),
                               text=r['text'],has0=(0 in toks),nraw=len(toks)))
    for o in objs.values(): o['faces'].sort(key=lambda f:f['k'])
    return objs

def multi(objs,min_signs=1,complete_only=False):
    """objects with >= 2 inscribed faces (faces with >= min_signs signs)"""
    out=[]
    for o in objs.values():
        fs=[f for f in o['faces'] if len(f['seq'])>=min_signs and (f['complete'] or not complete_only)]
        if len(fs)>=2:
            o2=dict(o); o2['faces']=fs; out.append(o2)
    return out

def collapse(mobjs,moulded_types=('TAB:B','TAB:C','TAG','MDLN')):
    """S-DARK-13: one die per identical set of face texts for moulded / impressed types, within site"""
    seen=set(); out=[]; dropped=0
    for o in mobjs:
        if o['type'].split(':')[0] in ('TAG',) or o['type'] in moulded_types:
            key=(o['site'],o['type'],tuple(sorted(tuple(f['seq']) for f in o['faces'])))
            if key in seen: dropped+=1; continue
            seen.add(key)
        out.append(o)
    return out,dropped

# ---------------- frame parser (tools/dark_loop19.py, S310/S331) ----------------
OPEN={817,861,820,920,692}; MARK={2,60}; MJAR={741,742,745}; SUF={400,90}; CL=[740,520,151,156,527,226,617,154,158,236,700]
FISH={235,240,233,231,220}; NUM={1,3,4,5,16,17,18,31,32,33,34,55,56}
def learn_qual(seqs):
    left=collections.defaultdict(collections.Counter)
    for s in seqs:
        s=list(s)
        while len(s)>1 and s[-1] in SUF: s.pop()
        if len(s)>=2 and s[-1] in CL: left[s[-1]][s[-2]]+=1
    Q={}
    for c,cnt in left.items():
        tot=sum(cnt.values()); acc=0; q=set()
        for a,n in cnt.most_common():
            if acc/tot>=0.6: break
            q.add(a); acc+=n
        Q[c]=q
    return Q
def make_parser(QUAL):
    def parse(s):
        if not s: return []
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
    return parse
SLOTRANK={'OPENER':0,'MARKER':1,'NAME':2,'COUNT':2,'TITLE':3,'CLOSER':4,'SUFFIX':5}

def is_opener_first(s): return bool(s) and s[0] in OPEN
def closer_last(s):
    s=list(s)
    while len(s)>1 and s[-1] in SUF: s.pop()
    return bool(s) and s[-1] in CL
def is_count_face(s):
    """Harappa voucher count face: numerals + W700 only (S93/S215)"""
    return bool(s) and all(w in NUM or w==700 for w in s) and any(w in NUM for w in s)

def pval(obs,null,side='hi'):
    n=len(null)
    if side=='hi': return (sum(1 for v in null if v>=obs)+1)/(n+1)
    return (sum(1 for v in null if v<=obs)+1)/(n+1)

def Mname(w,BR=None):
    BR=BR or json.load(open(BRIDGE))
    m=BR.get(str(w)); return f'W{w}(M{"/".join(map(str,m))})' if m else f'W{w}'

# ---------------- IM77 loader ----------------
def load_im77():
    """text_no -> dict(site, otype, sides=[(side, [signs in reading order, lines concatenated])])"""
    objs=collections.OrderedDict()
    for r in csv.DictReader(open(IM77)):
        o=objs.setdefault(r['text_no'],dict(oid=r['text_no'],site=r['site'],type=r['object_type'],sides=collections.OrderedDict()))
        if r['line']=='9' or not r['signs_clean'].strip(): continue
        signs=[int(x) for x in r['signs_clean'].split() if x!='0']
        o['sides'].setdefault(int(r['side']),[]).extend(signs)
    for o in objs.values():
        o['faces']=[dict(k=k,seq=v) for k,v in sorted(o['sides'].items()) if v]
    return objs
