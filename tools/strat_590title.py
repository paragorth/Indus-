"""S327: the title ending 590-390-740 (box-on-stand · tree · jar) found at Kish and Gonur Depe (S326). Who carries it at
home? For every text containing the run: site, object type, emblem, what precedes it. Emblem enrichment vs all seals
(Fisher), and the IM77 form of the run via the bridge (W590 = M249, W390 = M161-169, W740 = M342)."""
import json,collections,csv
from scipy.stats import fisher_exact
C=json.load(open('data/derived/merged-corpus-canonical.json'))
RUN=(590,390,740)
def has(s): return any(tuple(s[i:i+3])==RUN for i in range(len(s)-2))
hits=[r for r in C if r['seq_raw'] and has(r['seq_raw'])]
emb=lambda r:(r.get('symbol') or '').split(':')[0] or 'none'
print('texts with 590-390-740:',len(hits))
for r in hits: print('  ',r['cisi'],r['site'],r['type'],emb(r),r['seq_raw'])
seals=[r for r in C if r['type'].startswith('SEAL')]
eh=collections.Counter(emb(r) for r in hits if r['type'].startswith('SEAL')); ea=collections.Counter(emb(r) for r in seals)
n=sum(eh.values()); N=len(seals)
for e,k in eh.most_common():
    p=fisher_exact([[k,n-k],[ea[e]-k,N-n-(ea[e]-k)]],alternative='greater')[1]
    print(f'  emblem {e}: {k}/{n} seals with the run vs {ea[e]}/{N} of all seals; p={p:.3g}')
pre=collections.Counter()
for r in hits:
    s=r['seq_raw']; i=[j for j in range(len(s)-2) if tuple(s[j:j+3])==RUN][0]
    pre[tuple(s[max(0,i-2):i])]+=1
print('what comes before the run:',pre.most_common(10))
# IM77
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv'))); sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
im=collections.Counter()
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    for i in range(len(s)-2):
        if s[i]=='249' and s[i+1] in ('161','162','167','168','169') and s[i+2]=='342': im[(ls[0]['site'],ls[0]['object_type'])]+=1
print('IM77 M249-M161..169-M342:',dict(im))
