"""S300: the arrow title complex. Left partners of the arrow closer W520 (M211): a fish word, or U-with-stroke
+ tall 3 (705/706-33 = M336-M89). Is 705/706-33 an optional insert between the fish words and the arrow?
Predictions: texts with ...705/706-33-520 carry a fish EARLIER far above the base rate; texts with fish-520 almost
never carry 705/706. Binomial vs base rates over all texts (len >= 3); replicated in IM77."""
import json,csv,collections
from scipy.stats import binomtest
def analyse(T,F,U,THREE,ARR,label):
    base_f=sum(any(a in F for a in s) for s in T)/len(T); base_u=sum(any(a in U for a in s) for s in T)/len(T)
    fl=[];ul=[]
    for s in T:
        if ARR not in s: continue
        i=s.index(ARR)
        if i and s[i-1] in F: fl.append(any(a in U for a in s[:i]))
        if i>=2 and s[i-1]==THREE and s[i-2] in U: ul.append(any(a in F for a in s[:i-2]))
    print(f'{label}: texts {len(T)}; base fish {base_f:.2f}, base U {base_u:.3f}')
    print(f'  fish-ARROW texts {len(fl)}: with U anywhere before {sum(fl)} (expected {base_u*len(fl):.1f}); p_low={binomtest(sum(fl),len(fl),base_u,alternative="less").pvalue:.3g}')
    print(f'  U-3-ARROW texts {len(ul)}: with fish earlier {sum(ul)} ({sum(ul)/max(1,len(ul)):.2f}); p_high={binomtest(sum(ul),len(ul),base_f,alternative="greater").pvalue:.3g}')
C=json.load(open('data/derived/merged-corpus-canonical.json'))
for key in ['seq_raw','seq_all']:
    seen=set(); T=[]
    for r in C:
        s=r[key]
        if s and len(s)>=3 and (r['site'],tuple(s)) not in seen: seen.add((r['site'],tuple(s))); T.append(s)
    analyse(T,{220,240,235,233,231},{705,706},33,520,'Wells '+key)
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv'))); sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
seen=set(); T=[]
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    s=[int(x) for x in s if x.isdigit() and x not in ('0','000')]
    if len(s)>=3 and (ls[0]['site'],tuple(s)) not in seen: seen.add((ls[0]['site'],tuple(s))); T.append(s)
analyse(T,{59,67,65,72,70},{336},89,211,'IM77')
