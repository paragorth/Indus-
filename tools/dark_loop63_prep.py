"""S-DARK-63 prep: LOGOGRAPHICALLY WRITTEN personal names as the comparator for the Indus middle.
Builds one-per-distinct-name element lists (element = one logogram / one Sino-Vietnamese syllable) from open corpora:
  cn_given   modern Chinese given names (surname stripped with the repo's 1k surname list; 1-3 characters), gender tag kept
  cn_full    the same names with the surname (surname + given = 'head + middle' analogue, head FIRST)
  cn_ancient historical Chinese names (full name strings, 2-6 characters)
  jp_given   JMnedict given names tagged fem / masc / given, all-kanji only, 1-4 characters, gender kept
  jp_person  JMnedict 'person' entries (full names of real people), all-kanji; surname split by longest JMnedict surname prefix
  jp_surname JMnedict surnames (the closed 'head' set)
  vi_given   UIT-ViNames given names (full name minus first token), syllables lowercased, gender kept
  vi_full    UIT-ViNames full names (surname first)
Output: data/derived/dark/loop63_corpora/<name>.jsonl, one {"seq":[...],"g":gender,"sur":surname} per distinct name; SOURCES.txt.
"""
import re,json,gzip,collections,zipfile,csv,os,random
D='data/derived/dark/loop63_corpora/'
os.makedirs(D,exist_ok=True)
SRC=[]
def dump(name,rows,desc):
    seen=set(); out=[]
    for seq,g,sur in rows:
        if not seq or tuple(seq) in seen: continue
        seen.add(tuple(seq)); out.append(dict(seq=list(seq),g=g,sur=sur))
    with open(D+name+'.jsonl','w') as f:
        for o in out: f.write(json.dumps(o,ensure_ascii=False)+'\n')
    el=collections.Counter(a for o in out for a in o['seq'])
    L=collections.Counter(len(o['seq']) for o in out)
    s=f'{name}: raw {len(rows)} -> distinct {len(out)}; element types {len(el)}; mean len {sum(len(o["seq"]) for o in out)/len(out):.2f}; lengths {sorted(L.items())}; {desc}'
    print(s); SRC.append(s)
CJK=re.compile(r'^[一-鿿㐀-䶿\U00020000-\U0002a6df々]+$')
# ---- Chinese surnames from the repo xlsx (parsed without openpyxl) ----
def xlsx_strings(path):
    z=zipfile.ZipFile(path); ss=z.read('xl/sharedStrings.xml').decode('utf8')
    return re.findall(r'<t[^>]*>(.*?)</t>',ss,re.S)
sur=set(s.strip() for s in xlsx_strings(D+'cn_surnames.xlsx') if CJK.match(s.strip()))
sur2={s for s in sur if len(s)==2}; sur1={s for s in sur if len(s)==1}
print('Chinese surnames',len(sur),'two-char',len(sur2))
# ---- modern Chinese names with gender ----
rows_g=[]; rows_f=[]; nsplit=collections.Counter()
for ln in open(D+'cn_gender.txt',encoding='utf8'):
    p=ln.strip().split(',')
    if len(p)!=2 or not CJK.match(p[0]): continue
    nm,g=p; g={'男':'M','女':'F'}.get(g,'U')
    if len(nm)>=4 and nm[:2] in sur2: s,given=nm[:2],nm[2:]
    elif len(nm)>=2 and nm[0] in sur1: s,given=nm[0],nm[1:]
    else: nsplit['nosur']+=1; continue
    if not 1<=len(given)<=3: nsplit['len']+=1; continue
    nsplit['ok']+=1
    rows_g.append((list(given),g,s)); rows_f.append((list(nm),g,s))
print('cn split',nsplit)
dump('cn_given',rows_g,'modern Chinese given names, element = character; wainshine/Chinese-Names-Corpus Gender(120W) file, Apache-2.0')
dump('cn_full',rows_f,'the same with surname prefixed (head FIRST)')
rows=[]
for ln in open(D+'cn_ancient.txt',encoding='utf8'):
    nm=ln.strip()
    if CJK.match(nm) and 2<=len(nm)<=6: rows.append((list(nm),'U',nm[0]))
dump('cn_ancient',rows,'historical Chinese full names, element = character; wainshine Ancient_Names_Corpus(25W), Apache-2.0')
# ---- JMnedict ----
txt=gzip.open(D+'JMnedict.xml.gz','rt',encoding='utf8').read()
ents=re.findall(r'<entry>(.*?)</entry>',txt,re.S)
given=[]; surn=set(); person=[]
for e in ents:
    types=set(re.findall(r'<name_type>&(\w+);</name_type>',e))
    kebs=[k for k in re.findall(r'<keb>(.*?)</keb>',e) if CJK.match(k)]
    if not kebs: continue
    if types&{'fem','masc','given'}:
        g='F' if 'fem' in types and 'masc' not in types else 'M' if 'masc' in types and 'fem' not in types else 'U'
        for k in kebs:
            if 1<=len(k)<=4: given.append((list(k),g,None))
    if 'surname' in types:
        for k in kebs: surn.add(k)
    if 'person' in types:
        for k in kebs:
            if 2<=len(k)<=7: person.append(k)
dump('jp_given',given,'JMnedict given names (fem/masc/given), all-kanji, element = kanji; EDRDG, CC BY-SA 4.0')
dump('jp_surname',[(list(s),'U',None) for s in surn if len(s)<=4],'JMnedict surnames, all-kanji')
pr=[]; nsp=collections.Counter()
for k in person:
    sp=None
    for L in (3,2,1):
        if len(k)>L and k[:L] in surn: sp=L; break
    if sp is None: nsp['nosur']+=1; continue
    g=k[sp:]
    if len(g)>4: nsp['long']+=1; continue
    nsp['ok']+=1; pr.append((list(k),'U',k[:sp]))
print('jp person split',nsp)
dump('jp_person',pr,'JMnedict full personal names (surname + given), surname = longest JMnedict-surname prefix, head FIRST')
dump('jp_person_given',[(list(''.join(s)[len(su):]),'U',su) for s,_,su in pr],'given-name part of jp_person')
# ---- Vietnamese ----
rows=[]; rows_f=[]
for r in csv.DictReader(open(D+'vinames/UIT-ViNames/UIT-ViNames - Full.csv',encoding='utf-8-sig')):
    toks=[t.lower() for t in r['Full_Names'].split() if t.strip()]
    if len(toks)<2 or len(toks)>5: continue
    g={'0':'F','1':'M'}.get(r['Gender'].strip(),'U')
    rows.append((toks[1:],g,toks[0])); rows_f.append((toks,g,toks[0]))
dump('vi_given',rows,'UIT-ViNames (Vietnamese full names, 26,850, gender), given name = tokens after the surname, element = syllable (Sino-Vietnamese morpheme), lowercased; GitHub JkUndead/UIT-ViNames-Dataset, academic dataset (To et al. 2020), NO licence file: raw file not redistributed, only derived element lists kept here')
dump('vi_full',rows_f,'the same with the surname first')
open(D+'SOURCES.txt','w').write('\n'.join([
 'S-DARK-63 corpora (logographically written personal names), prepared by tools/dark_loop63_prep.py',
 'cn_gender.txt / cn_ancient.txt / cn_surnames.xlsx: https://github.com/wainshine/Chinese-Names-Corpus (Apache-2.0; files Chinese_Names_Corpus_Gender（120W）.txt, Ancient_Names_Corpus（25W）.txt, Chinese_Family_Name（1k）.xlsx)',
 'JMnedict.xml.gz: http://ftp.edrdg.org/pub/Nihongo/JMnedict.xml.gz (Electronic Dictionary Research and Development Group, CC BY-SA 4.0; http://www.edrdg.org/edrdg/licence.html)',
 'UIT-ViNames-v1.zip: https://github.com/JkUndead/UIT-ViNames-Dataset (To, Nguyen, Nguyen 2020, UIT-ViNames; no licence file in the repo; the raw zip is deleted after prep, only the derived jsonl is kept)',
 'Korean hanja given-name list: none found with an open licence (rutopio/Korean-Name-Hanja-Charset is a character inventory, not names); Korean is NOT in the battery.',
 '']+SRC)+'\n')
