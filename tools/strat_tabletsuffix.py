"""S265: is W400 a suffix that tablets add to otherwise seal-like texts?
For tablets whose text ends in W400: is text-minus-400 attested as a complete SEAL text?
Control: tablets not ending in 400, test whether text-minus-last-sign is a seal text; and
tablets ending in 400, test whether the full text (with 400) is a seal text."""
import json,collections
m=json.load(open('data/derived/merged-corpus-reading-order.json'))
m=m if isinstance(m,list) else list(m.values())[0]
seal=collections.Counter(tuple(r['seq']) for r in m if r.get('seq') and str(r['type']).startswith('SEAL'))
tabs=[r for r in m if r.get('seq') and str(r['type']).startswith('TAB') and len(r['seq'])>=2]
# dedupe identical tablet texts (moulded duplicates) by (site,text)
def uniq(rs): return list({(r['site'],tuple(r['seq'])):r for r in rs}.values())
e400=uniq([r for r in tabs if r['seq'][-1]==400]); oth=uniq([r for r in tabs if r['seq'][-1]!=400])
a=sum(tuple(r['seq'][:-1]) in seal for r in e400); b=sum(tuple(r['seq']) in seal for r in e400)
c=sum(tuple(r['seq'][:-1]) in seal for r in oth); d=sum(tuple(r['seq']) in seal for r in oth)
print(f'tablets ending 400 (unique texts) n={len(e400)}: minus-400 is a seal text {a} ({a/len(e400):.1%}); full text is a seal text {b}')
print(f'other tablets n={len(oth)}: minus-last is a seal text {c} ({c/len(oth):.1%}); full text is a seal text {d} ({d/len(oth):.1%})')
for r in e400:
    if tuple(r['seq'][:-1]) in seal: print('  ',r['cisi'],r['site'],r['type'],r['seq'],'seal copies:',seal[tuple(r['seq'][:-1])])
