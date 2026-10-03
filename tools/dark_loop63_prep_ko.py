"""S-DARK-63 cycle 4 prep: KOREAN personal names (Wikidata, CC0) as a fourth logographic-origin comparator.
No openly licensed hanja given-name list exists (Wikidata has hanja native names for only ~100 South Korean people), so the
element is the Hangul syllable, i.e. the Sino-Korean morpheme with homophones merged - the same treatment as Vietnamese.
Source query (Wikidata Query Service, CC0):
  SELECT ?p ?n ?g WHERE { VALUES ?c { wd:Q884 wd:Q423 } ?p wdt:P27 ?c; wdt:P31 wd:Q5; rdfs:label ?n .
                          FILTER(LANG(?n)="ko") OPTIONAL { ?p wdt:P21 ?g } }
Usage: python3 tools/dark_loop63_prep_ko.py <ko_wd.csv>   (raw csv is not kept; ~5 MB)
Writes data/derived/dark/loop63_corpora/ko_given.jsonl, ko_full.jsonl and appends to SOURCES.txt.
"""
import sys,csv,re,json,collections
D='data/derived/dark/loop63_corpora/'
HANGUL=re.compile(r'^[가-힣]+$')
COMP={'남궁','황보','제갈','선우','독고','사공','서문','동방','망절','어금','소봉','장곡'}
per=collections.defaultdict(lambda:[None,set()])
for r in csv.DictReader(open(sys.argv[1],encoding='utf8')):
    nm=r['n'].strip()
    g={'http://www.wikidata.org/entity/Q6581097':'M','http://www.wikidata.org/entity/Q6581072':'F'}.get(r.get('g',''),'U')
    per[r['p']][0]=nm; per[r['p']][1].add(g)
rows_g=[]; rows_f=[]; st=collections.Counter()
for p,(nm,gs) in per.items():
    if not nm or not HANGUL.match(nm): st['nonhangul']+=1; continue
    g=list(gs)[0] if len(gs)==1 else 'U'
    if len(nm)>=3 and nm[:2] in COMP: s=nm[:2]
    else: s=nm[0]
    given=nm[len(s):]
    if not 1<=len(given)<=3 or len(nm)>5: st['len']+=1; continue
    st['ok']+=1
    rows_g.append((list(given),g,s)); rows_f.append((list(nm),g,s))
print(st)
SRC=[]
def dump(name,rows,desc):
    seen=set(); out=[]
    for seq,g,sur in rows:
        if tuple(seq) in seen: continue
        seen.add(tuple(seq)); out.append(dict(seq=seq,g=g,sur=sur))
    with open(D+name+'.jsonl','w') as f:
        for o in out: f.write(json.dumps(o,ensure_ascii=False)+'\n')
    el=collections.Counter(a for o in out for a in o['seq']); L=collections.Counter(len(o['seq']) for o in out)
    gc=collections.Counter(o['g'] for o in out)
    s=f'{name}: raw {len(rows)} -> distinct {len(out)}; element types {len(el)}; mean len {sum(len(o["seq"]) for o in out)/len(out):.2f}; lengths {sorted(L.items())}; gender {dict(gc)}; {desc}'
    print(s); SRC.append(s)
dump('ko_given',rows_g,'Korean given names of people with South/North Korean citizenship (Wikidata ko labels, CC0), surname stripped (first syllable, or a 2-syllable compound surname), element = Hangul syllable (Sino-Korean morpheme, homophones merged)')
dump('ko_full',rows_f,'the same with the surname first (head FIRST)')
src=open(D+'SOURCES.txt').read().replace("Korean is NOT in the battery.","Korean is NOT in cycles 1-3; cycle 4 adds Korean names from Wikidata (CC0) with element = Hangul syllable (tools/dark_loop63_prep_ko.py).")
open(D+'SOURCES.txt','w').write(src.rstrip('\n')+'\n'+'\n'.join(SRC)+'\n')
