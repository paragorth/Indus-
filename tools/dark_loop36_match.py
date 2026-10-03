"""Loop 36: check a candidate Wells-number sequence against the whole corpus (inscriptions.csv, 5,680 rows, every site)
and against IM77 through the bridge. Match classes: exact, reverse, edit-distance-1 (any site), subsequence (candidate inside a
corpus text), and site-restricted variants. Usage:
  python3 tools/dark_loop36_match.py 820 2 415 740            # one candidate
  python3 tools/dark_loop36_match.py --csv data/derived/dark/loop36_newfinds.csv   # all rows (column 'signs_W', '-' separated, ? = unread)
"""
import csv,json,sys,re,collections
ROOT='/home/user/Indus-/'
def load_wells():
    rows=[]
    for r in csv.DictReader(open(ROOT+'data/raw/inscriptions.csv')):
        t=r['text'].strip('+')
        if not t: continue
        # lines separated by '/', signs by '-'; '000' = lost sign
        signs=[]
        for part in re.split(r'[/]',t):
            for s in part.split('-'):
                s=s.strip()
                if s.isdigit(): signs.append(int(s))
        if signs: rows.append((r['id'],r['site'],r['type'],signs,r['dir.']))
    return rows
def load_im77():
    B=json.load(open(ROOT+'data/derived/bridge_extended.json'))
    P=json.load(open(ROOT+'data/derived/dark/bridge_proposals.json'))
    w2m={int(k):set(v) for k,v in B.items()}
    for p in P['proposals']: w2m.setdefault(p['W'],set()).add(p['M'])
    texts=collections.defaultdict(list)
    for r in csv.DictReader(open(ROOT+'data/im77/im77_corpus_lines.csv')):
        key=(r['text_no'],r['site'],r['side'])
        texts[key].append((int(r['line']),[int(x) for x in r['signs_clean'].split() if x.isdigit()]))
    out=[]
    for k,lines in texts.items():
        lines.sort(); seq=[s for _,l in lines for s in l]
        if seq: out.append((k[0],k[1],seq))
    return w2m,out
def ed1(a,b):
    """edit distance <= 1 with wildcards (None)"""
    if abs(len(a)-len(b))>1: return False
    def eq(x,y): return x is None or y is None or x==y
    if len(a)==len(b):
        return sum(not eq(x,y) for x,y in zip(a,b))<=1
    if len(a)>len(b): a,b=b,a
    for i in range(len(b)):
        if all(eq(x,y) for x,y in zip(a,b[:i]+b[i+1:])): return True
    return False
def subseq(a,b):
    if len(a)>len(b): return False
    for i in range(len(b)-len(a)+1):
        if all(x is None or x==y for x,y in zip(a,b[i:i+len(a)])): return True
    return False
def check(cand,W,w2m,IM):
    c=[None if (x in ('?','',None) or not x.strip().isdigit()) else int(x) for x in cand]
    conc=[x for x in c if x is not None]
    res={'exact':[],'reverse':[],'ed1':[],'subseq':[],'im77_exact':[],'im77_ed1':[]}
    # S-DARK-21.1 wildcard cap: at most 2 wildcards and >= 3 concrete positions for an exact/near verdict;
    # texts of <= 3 signs allow no wildcard
    nwild=len(c)-len(conc)
    res['indeterminate']= (nwild>2) or (len(conc)<3) or (len(c)<=3 and nwild>0)
    if len(conc)==0: return res
    for id_,site,typ,s,d in W:
        if len(s)==len(c) and all(x is None or x==y for x,y in zip(c,s)): res['exact'].append((id_,site,s))
        elif len(s)==len(c) and all(x is None or x==y for x,y in zip(c,s[::-1])): res['reverse'].append((id_,site,s))
        elif ed1(c,s) or ed1(c[::-1],s): res['ed1'].append((id_,site,s))
        elif len(conc)>=3 and (subseq(c,s) or subseq(c[::-1],s)): res['subseq'].append((id_,site,s))
    # IM77: candidate W -> set of M; match if every position compatible
    def comp(w,m): return w is None or (m in w2m.get(w,set()))
    for tno,site,s in IM:
        if len(s)==len(c) and all(comp(x,y) for x,y in zip(c,s)): res['im77_exact'].append((tno,site,s))
        elif len(s)==len(c) and all(comp(x,y) for x,y in zip(c,s[::-1])): res['im77_exact'].append((tno,site,s[::-1]))
        elif abs(len(s)-len(c))<=1 and len(conc)>=3:
            # ed1 in M space
            cm=[None if x is None else w2m.get(x,set()) for x in c]
            def eqm(x,y): return x is None or (y in x)
            ok=False
            if len(s)==len(c): ok=sum(not eqm(x,y) for x,y in zip(cm,s))<=1
            else:
                a,b=(cm,s) if len(cm)<len(s) else (s,cm)
                for i in range(len(b)):
                    bb=b[:i]+b[i+1:]
                    if len(cm)<len(s): ok=ok or all(eqm(x,y) for x,y in zip(a,bb))
                    else: ok=ok or all(eqm(x,y) for x,y in zip(bb,a))
            if ok: res['im77_ed1'].append((tno,site,s))
    return res
def verdict(res):
    if res.get('indeterminate'):
        if res['exact'] or res['im77_exact'] or res['reverse']: return 'INDETERMINATE(wildcard match)'
        return 'INDETERMINATE(too few concrete signs)'
    if res['exact'] or res['im77_exact']: return 'IN-CORPUS'
    if res['reverse']: return 'IN-CORPUS(reverse)'
    if res['ed1'] or res['im77_ed1']: return 'NEAR'
    return 'NEW'
if __name__=='__main__':
    W=load_wells(); w2m,IM=load_im77()
    if sys.argv[1]=='--csv':
        rows=list(csv.DictReader(open(sys.argv[2])))
        for r in rows:
            cand=r['signs_W'].split('-')
            res=check(cand,W,w2m,IM); v=verdict(res)
            print(r['id'],r['site'],r['signs_W'],v, {k:[(a,b,'-'.join(map(str,c))) for a,b,c in v2[:3]] for k,v2 in res.items() if isinstance(v2,list) and v2})
    else:
        cand=sys.argv[1:]
        res=check(cand,W,w2m,IM); print(verdict(res))
        for k,v in res.items():
            for x in v[:10]: print(k,x)
