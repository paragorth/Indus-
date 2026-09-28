"""S256: which line of a two-line side is the start of the text? Count frame signs
(opener M267/391 first, marker M99 second, jar M342 last) per line. Control: one-line
texts of the same corpus give the base rates for a start line and an end line."""
import csv,collections
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
OPEN={'267','391'}
C=collections.Counter(); one=collections.Counter(); n1=0; n2=0
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); full=[l for l in ls if int(l['n_signs'] or 0)>=2]
    if len(full)==1 and len(ls)==1:
        s=full[0]['signs_clean'].split(); n1+=1
        one['open_first']+=s[0] in OPEN; one['jar_last']+=s[-1]=='342'; one['jar_any']+=('342' in s)
    if len(full)==2:
        n2+=1
        for i,l in enumerate(full):
            s=l['signs_clean'].split()
            C[f'L{i+1}_open_first']+=s[0] in OPEN; C[f'L{i+1}_jar_last']+=s[-1]=='342'
            C[f'L{i+1}_open_last']+=s[-1] in OPEN; C[f'L{i+1}_jar_first']+=s[0]=='342'
            C[f'L{i+1}_jar_any']+=('342' in s)
print('one-line texts',n1,{k:round(v/n1,3) for k,v in one.items()})
print('two-line sides',n2); [print(' ',k,v) for k,v in sorted(C.items())]
