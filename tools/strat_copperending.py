"""S306: copper tablets (Mohenjo-daro, image on one face, text on the other) end differently? Endings: W407 final
and 845..407. Compared with other Mohenjo-daro objects (site control), distinct texts. Fisher one-sided.
Plus the title phrases on the image-captioned tablets (per motif)."""
import json,collections
from scipy.stats import fisher_exact
C=json.load(open('data/derived/merged-corpus-canonical.json')); seen=set(); rows=[]
for r in C:
    s=r['seq_raw']; k=(r['type'],tuple(s))
    if not s or r['site']!='Mohenjo-daro' or k in seen: continue
    seen.add(k); rows.append((r['type']=='TAB:C',s,r.get('symbol','')))
def has845(s): return 845 in s and 407 in s and s.index(845)<len(s)-1-s[::-1].index(407)
for name,f in [('407 final',lambda s:s[-1]==407),('845..407',has845),('jar final',lambda s:s[-1]==740)]:
    a=sum(f(s) for c,s,_ in rows if c); n=sum(1 for c,*_ in rows if c); b=sum(f(s) for c,s,_ in rows if not c); m=len(rows)-n
    print(f'{name}: copper {a}/{n} ({a/n:.2f}) vs other Mohenjo-daro {b}/{m} ({b/m:.3f}); p={fisher_exact([[a,n-a],[b,m-b]],alternative="greater")[1]:.2g}')
print('407-final copper texts:'); [print('  ',sym,s) for c,s,sym in rows if c and s[-1]==407]
print('407-final elsewhere at Mohenjo-daro:'); [print('  ',s) for c,s,sym in rows if not c and s[-1]==407]
