# Needs the CDLI bulk dump in data/raw/ (cdli_cat.csv, cdli.atf from https://github.com/cdli-gh/data via Git LFS).
import csv,re,json
from collections import Counter,defaultdict
csv.field_size_limit(10**9)
cat={}
with open('data/raw/cdli_cat.csv',newline='',encoding='utf-8',errors='replace') as f:
    for x in csv.DictReader(f):
        if x['id_text'].isdigit(): cat['P%06d'%int(x['id_text'])]=(x['period'],x['provenience'],x['designation'],x['genre'])
texts={};cur=None
for l in open('data/raw/cdli.atf',encoding='utf-8',errors='replace'):
    if l.startswith('&P'): cur=l[1:8]; texts[cur]=[]
    elif cur and l[:1].isdigit(): texts[cur].append(re.sub(r'^\S+\s+','',l.strip()))
clean=lambda w: re.sub(r'[#!?\[\]<>⸢⸣]','',w)
# global word frequencies (for rarity)
wf=Counter(clean(w) for L in texts.values() for l in L for w in l.split())
trade=re.compile(r'(dilmun|tilmun|ga-esz8 a-ab-ba|a-ab-ba-ta|me-luh|ma2 dilmun|a-li-ik|a-lik ti-il-mu-un)',re.I)
sel=[p for p,L in texts.items() if re.match(r'(Ur III|Old Babylonian|Early Old|Old Akkadian|Lagash II)',cat.get(p,('',))[0]) and any(trade.search(l) for l in L)]
print('trade tablets',len(sel))
# personal-name slots: word after ki / kiszib3 / giri3 / szu-i / before dumu; and word(s) before 'lu2 me-luh-ha' / 'dumu me-luh-ha'
slot=re.compile(r'(?:^|\s)(?:ki|kiszib3|giri3|mu-DU|sza3-ba|dumu|dam-gar3|ga-esz8)\s+(\S+)')
names=Counter(); where=defaultdict(set)
for p in sel:
    for l in texts[p]:
        for m in slot.finditer(l):
            w=clean(m.group(1))
            if re.search(r'[a-z]',w) and not re.match(r'^(\d|lugal|e2|ensi2|sza3|lu2|dumu|ku3|sze|ki|ma2|a-ab-ba|dilmun|tilmun|me-luh|gu2|ma-na|gin2)',w):
                names[w]+=1; where[w].add(p)
rare=[(w,c,wf[w]) for w,c in names.items() if wf[w]<=3]
print('distinct name-slot words',len(names),'| occurring <=3 times in all CDLI:',len(rare))
for w,c,g in sorted(rare,key=lambda x:x[2])[:60]:
    p=next(iter(where[w])); print(f"  {w:28s} in-trade {c}  CDLI-total {g}  {p} {cat[p][0][:14]} {cat[p][1][:18]} {cat[p][2][:26]}")
# numbered affiliations: a number immediately followed by lu2/dumu/erin2 + dilmun|me-luh
num=[ (p,l) for p in sel for l in texts[p] if re.search(r'\d\((disz|u|asz)\)\s+(lu2|dumu|erin2|gurusz)\s+(me-luh|dilmun|tilmun)',l)]
print('numbered groups of Dilmun/Meluhha men:',len(num)); [print('  ',p,cat[p][2][:25],'|',l) for p,l in num[:15]]
