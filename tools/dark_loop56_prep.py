"""S-DARK-56 prep: one-person-per-entry NAME LISTS and non-name poles for the 'not names' comparison.
Writes data/derived/dark/loop56_corpora/<name>.jsonl, one JSON per line: {"seq":[...], "src":...}.
Every name list is DEDUPLICATED to one copy per distinct name string (so uniqueness is 1 by construction and the
comparison must rest on element-level statistics).
Sources: CDLI ATF dump (scratchpad cdli.atf + cdli_cat.csv; @seal sections, line 1 = owner name) for Ur III, Old
Babylonian (+ Early OB) and Old Assyrian; DAMOS Linear B items (other-scripts/linear-a/data/damos_items.jsonl) for
personnel series; EDH Latin (loop32_corpora/latin_edh.jsonl) for funerary names; loop32 icd10 / hts / proto_elamite.
Indus middles are built by the analysis script (S-DARK-26 parser), not here.
Usage: python3 tools/dark_loop56_prep.py
"""
import json,re,os,csv,collections,unicodedata
OUT='data/derived/dark/loop56_corpora/'; os.makedirs(OUT,exist_ok=True)
S='/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
CORP='data/derived/dark/loop32_corpora/'
LOG=[]
def P(*a):
    s=' '.join(str(x) for x in a); print(s); LOG.append(s)
def dump(name,seqs,src):
    seqs=[tuple(s) for s in seqs if s]
    dist=sorted(set(seqs))
    with open(OUT+name+'.jsonl','w') as f:
        for s in dist: f.write(json.dumps({'seq':list(s),'src':src})+'\n')
    el=collections.Counter(a for s in dist for a in s)
    P(f'{name}: raw {len(seqs)} -> distinct {len(dist)}; element types {len(el)}; mean len {sum(map(len,dist))/max(1,len(dist)):.2f}; {src}')
    return dist

# ---------------- cuneiform seal legends ----------------
TITLE={'dumu','dub-sar','lugal','arad2','arad2-zu','arad','arad-zu','lu2','ensi2','ugula','nu-banda3','gudu4','dam-gar3','sanga','sukkal','szabra','agrig','sipa','lunga','simug','aszgab','nagar','kuruszda','szagina','ra2-gab','gal5-la2','muhaldim','kiszib3','dam','szesz','nin','ama','ab-ba','i3-du8','sagi','gal','nar','azlag2','ma2-lah5','szu-i','ad-kup4','asz-gab','bahar2','szitim','engar','mu','na4-kiszib','kiszib','ir3','ir11','ir3-zu','ir11-zu','geme2','munus','an','dingir'}
def clean_cune(line):
    t=line.split('.',1)[1].strip()
    t=re.sub(r'[#!?\[\]<>]','',t)          # damage / editorial marks
    t=t.replace('_','')                      # logogram markers in Akkadian texts
    if '...' in t or re.search(r'(^|[-\s])x([-\s]|$)',t): return None
    return t
def cune_tokens(name):
    toks=[x for x in re.split(r'[-\s]+',name) if x]
    toks=[t.lower() for t in toks]
    return tuple(toks)
def cdli_names():
    csv.field_size_limit(10**9)
    cat={}
    for r in csv.DictReader(open(S+'cdli_cat.csv',encoding='utf-8',errors='replace')):
        cat[r['id_text'].lstrip('P').lstrip('0')]=r['period']
    out=collections.defaultdict(list); pid=None; inseal=False; want=False
    for line in open(S+'cdli.atf',encoding='utf-8',errors='replace'):
        if line.startswith('&P'):
            pid=line[2:8].lstrip('0'); inseal=False
        elif line.startswith('@seal'):
            inseal=True; want=True
        elif line.startswith('@'):
            inseal=False
        elif inseal and want and re.match(r'^1\.\s',line):
            want=False
            per=cat.get(pid,'?')
            grp=('ur3' if per.startswith('Ur III') else 'ob' if 'Old Babylonian' in per else 'oa' if per.startswith('Old Assyrian') else None)
            if not grp: continue
            t=clean_cune(line)
            if not t: continue
            toks=cune_tokens(t)
            if not toks or len(toks)>8: continue
            # drop lines that are a title/kin word alone or that start with one (not a name line)
            first=t.split()[0].lower()
            if first in TITLE or toks[0] in TITLE: continue
            if any(tok in('dumu','arad2','arad','ir11','ir3') for tok in toks): continue   # legend compressed onto line 1
            out[grp].append(toks)
    return out
cn=cdli_names()
UR3=dump('ur3_names_dedup',cn['ur3'],'CDLI Ur III @seal line 1, sign tokens (split on -), one per distinct name')
OB=dump('ob_names_dedup',cn['ob'],'CDLI Old Babylonian + Early OB @seal line 1, sign tokens, one per distinct name')
OA=dump('oa_names_dedup',cn['oa'],'CDLI Old Assyrian @seal line 1, sign tokens, one per distinct name (SMALL)')
# Ur III element level: split names into Sumerian 'elements' using a greedy lexicon of frequent whole-name components.
# Elements are the hyphen-joined chunks that recur as whole names elsewhere (e.g. ur, lu2, {d}nanna, kal-la), a
# data-driven approximation; a sign chunk is an element if it occurs as a complete name or as the residue after
# removing another element in >= 5 names.
def ur3_elements(names):
    whole=collections.Counter(tuple(n) for n in names)
    chunks=collections.Counter()
    for n in names:
        for i in range(len(n)):
            for j in range(i+1,min(len(n),i+3)+1): chunks[n[i:j]]+=1
    lex={c for c,v in chunks.items() if v>=5 and (len(c)==1 or c in whole or c[0].startswith('{d}'))}
    out=[]
    for n in names:
        res=[];i=0
        while i<len(n):
            best=None
            for j in range(min(len(n),i+3),i,-1):
                if n[i:j] in lex: best=j;break
            if best is None: best=i+1
            res.append('-'.join(n[i:best])); i=best
        out.append(tuple(res))
    return out
dump('ur3_names_elem',ur3_elements(UR3),'Ur III distinct names re-tokenised into greedy frequent-chunk elements (<=3 signs)')

# ---------------- Linear B personnel lists ----------------
BAD=set('[]?*<>⟦⟧|/')
def lb_ok(w):
    if not re.fullmatch(r'[a-z0-9-]+',w): return False
    if '-' not in w and len(w)<2: return False
    if w in('vac','vacat','vest','lat','inf','sup','mut','deest'): return False
    return True
def strip_diac(w):
    return ''.join(c for c in unicodedata.normalize('NFD',w) if unicodedata.category(c)!='Mn')
FIRST_WORD={'Da','Db','Dc','Dd','De','Df','Dg','Dh','Dk','Dl','Dm','Dn','Dp','Dq','Dv','D','Cn','Ea','Eb','En','Eo','Ep','Es','Sc','Jn','Nn','Ma','Na','Nc','Ne','Ng'}
LIST={'As','B','An','Ap','Ai','Ak','Ad','Ae','V','Vc','Vd','Xd','Ce'}   # Vc/Xd Knossos one-name tablets
OCC={'do-e-ro','do-e-ra','ka-ke-u','te-ko-to','ku-wa','ko-wa','ko-wo','pe-di-ra','ka-ko','ta-ra-si-ja','ko-to-na','ki-ti-me-na','ke-ke-me-na','o-na-to','e-ke','to-so','to-sa','pa-ro','o-pe-ro','a-pu-do-si','me-no','e-ke-qe','e-ke-si','o-na-te-re','te-re-ta','ka-ma','e-ke-qe','ki-ti-je-si','pe-ma','o-pe-ro-sa','to-so-de','a-ke-ro','ra-wa-ke-ta','e-qe-ta','i-je-re-ja','i-je-re-u','ku-ru-so'}
def linb_names():
    items=[json.loads(l) for l in open('other-scripts/linear-a/data/damos_items.jsonl')]
    names=[];src=collections.Counter()
    for it in items:
        h=it.get('heading','')
        m=re.match(r'(KN|PY|TH|MY|TI)\s+([A-Z][a-z]?)(?:\(\d+\))?\s',h)
        if not m: continue
        site,ser=m.group(1),m.group(2)
        if ser not in FIRST_WORD and ser not in LIST: continue
        for ln in it['content'].split('\n'):
            ln=strip_diac(ln)
            if not re.match(r'^\.?[0-9AaBb]',ln.strip()) and not ln.startswith(' '): continue
            toks=ln.split()
            toks=[t for t in toks if not re.match(r'^\.?[0-9AaBbv]+[ab]?$',t) and t not in(',','/',"'",'[',']')]
            words=[t.strip(",'/") for t in toks]
            words=[w for w in words if lb_ok(w) and w not in OCC and not re.fullmatch(r'[0-9]+',w)]
            if not words: continue
            if ser in FIRST_WORD: words=words[:1]
            for w in words:
                if '-' in w and w.count('-')>=1 and len(w.split('-'))<=7:
                    names.append((site,tuple(w.split('-')))); src[(site,ser)]+=1
    return names,src
LBN,src=linb_names()
P('Linear B personnel words by site/series:',src.most_common(40))
LB=dump('linb_personnel_dedup',[n for s,n in LBN],'DAMOS Linear B personnel/landholder/flock series (KN+PY+TH+MY): names as syllabogram tuples, one per distinct name')
dump('linb_personnel_KN',[n for s,n in LBN if s=='KN'],'same, Knossos only')
dump('linb_personnel_PY',[n for s,n in LBN if s=='PY'],'same, Pylos only')

# ---------------- Latin funerary names (EDH) ----------------
LAT_STOP={'hic','situs','est','et','sacrum','qui','quae','vixit','filio','filiae','filius','filia','coniugi','uxori','patri','matri','fratri','sorori','libertae','liberto','carissimae','carissimo','piissimae','piissimo','bene','merenti','dulcissimae','dulcissimo','sanctissimae','optimae','optimo','annis','annorum','vix','sibi','suis','posterisque','libertis','libertabusque','in','ex','pro','dis','manibus','memoriae','infelicissimae','infelicissimo','pientissimae','pientissimo','incomparabili','rarissimae','rarissimo','sanctae','sancto','posuit','fecit','fecerunt','parentes','parentibus','mater','pater','frater','soror','marito','coniunx','uxor','filii','liberti','liberta','libertus','sacrum','vivus','viva','sibi','de','suo','sua','fecerunt','ossa','vixit','mensibus','diebus','servus','serva','verna','alumno','alumnae','contubernali','patrono','patronae','amico','amicae','benemerenti'}
def latin_names():
    out=[]
    for l in open(CORP+'latin_edh.jsonl'):
        s=json.loads(l)['seq']
        if len(s)>=3 and s[0]=='dis' and s[1]=='manibus':
            nm=[]
            for w in s[2:7]:
                if w in LAT_STOP or len(w)<3 or not w.isalpha(): break
                nm.append(w)
                if len(nm)==3: break
            if 1<=len(nm)<=3: out.append(tuple(nm))
    return out
LAT=dump('latin_names_dedup',latin_names(),'EDH Italy dis manibus formula: 1-3 name words after the formula, one per distinct name')
# letter level as a crude 'sign' analogue (phonemic letters, flagged)
dump('latin_names_letters',[tuple(ch for w in n for ch in w) for n in LAT],'same names as letter strings (phonemic letters, not signs; flagged)')

# ---------------- non-name poles ----------------
def jl(name): return [tuple(json.loads(l)['seq']) for l in open(CORP+name+'.jsonl')]
dump('icd10',jl('icd10'),'ICD-10-CM codes, characters (loop32)')
dump('hts',jl('hts'),'HTS tariff numbers, 2-digit groups (loop32)')
pe=[]
for s in jl('proto_elamite'):
    m=tuple(t for t in s if not (t[0]=='N' and len(t)>1 and t[1].isdigit()))
    if m: pe.append(m)
dump('proto_elamite_mid',pe,'Proto-Elamite entry lines minus numerals, one per distinct string (loop32)')
open(OUT+'SOURCES.txt','w').write('\n'.join(LOG)+'\n')
