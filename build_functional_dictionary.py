"""Strategy 139: a FUNCTIONAL dictionary (what each common sign does, not what it sounds like), computed from data.
For the top 120 Wells signs: frequency, positional profile, counted/number behaviour, seal vs tablet leaning,
and a rule-based functional class. Output: FUNCTIONAL-DICTIONARY.md"""
import json, collections
d=json.load(open("data/derived/merged-corpus-reading-order.json"))
dos={x['glyph']:x for x in json.load(open('data/derived/sign-dossiers-top200.json'))}
NUM={1:1,2:2,3:3,4:4,5:5,16:6,17:7,18:8}; TALL={31:1,32:2,33:3,34:4}
U=list({(x['type'].split(':')[0],tuple(s for s in x['seq'] if s)) for x in d})
f=collections.Counter(); pos=collections.defaultdict(collections.Counter); cnt=collections.defaultdict(collections.Counter)
typ=collections.defaultdict(collections.Counter); left=collections.defaultdict(set)
for k,t in U:
    for i,s in enumerate(t):
        f[s]+=1; typ[s][k]+=1
        p='I' if i==0 else 'F' if i==len(t)-1 else 'preJar' if t[i+1]==740 else 'M'
        if len(t)==1: p='alone'
        pos[s][p]+=1
        if i: left[s].add(t[i-1])
        if i and t[i-1]!=2 and (t[i-1] in NUM or t[i-1] in TALL): cnt[s][(('T' if t[i-1] in TALL else '')+str(NUM.get(t[i-1],TALL.get(t[i-1]))))]+=1
tot_seal=sum(1 for k,_ in U if k=='SEAL'); tot_tab=sum(1 for k,_ in U if k=='TAB')
def cls(s):
    n=f[s]; P=pos[s]; c=sum(cnt[s].values())
    if s in NUM or s in TALL: return 'numeral' if s!=2 else 'marker (connective, "2"-shaped)'
    if s in (817,861,820): return 'opener (3-way alternation)'
    if s==60: return 'marker (connective)'
    if s==740: return 'closer (default, "jar")'
    if c/n>=0.25:
        vals=cnt[s]; top=vals.most_common(1)[0][1]/c
        return 'number-name (fixed number)' if top>=0.75 else 'counted item'
    if (P['F']+P['alone'])/n>=0.6: return 'closing element / title-end'
    if P['preJar']/n>=0.4: return 'title (before jar)'
    if P['I']/n>=0.4: return 'opening element'
    return 'core / name element'
rows=[]
for s,n in f.most_common(120):
    P=pos[s]; c=cnt[s]; seal=typ[s]['SEAL']/tot_seal; tab=typ[s]['TAB']/tot_tab
    lean='tablet' if tab>2*seal else 'seal' if seal>2*tab else '–'
    shape=dos[s]['shape'][:55] if s in dos and dos[s]['shape'] else ''
    mah=','.join(dos[s]['mahadevan'][:2]) if s in dos else ''
    top_counts=', '.join(f"{v}×{m}" for v,m in c.most_common(3))
    rows.append(f"| W{s} | {mah} | {shape} | {n} | {P['I']}/{P['M']}/{P['preJar']}/{P['F']}/{P['alone']} | {top_counts} | {lean} | **{cls(s)}** |")
hdr=("# Functional dictionary of the 120 commonest Indus signs (data-derived, 25 Sept 2026)\n\n"
"This is **not** a dictionary of sounds or meanings. For each sign it records **what the sign does** in the texts, computed from the merged Wells-coded corpus (distinct texts). It is the part of a dictionary that the corpus itself can support (see STRATEGIES S107–S138 for why sound values cannot be obtained from the corpus).\n\n"
"Columns:\n- positions = initial / medial / directly before the jar / final / alone (distinct texts);\n- counts = numbers written directly before the sign (T = tall strokes; the two-stroke sign W2 is excluded because it acts as a connective, not a number);\n- leaning = seal or tablet preference (>2× by share of texts);\n- class = rule-based function: numeral, marker, opener, closer, title (before jar), counted item, number-name (one fixed number), closing element, opening element, core/name element.\n\n"
"Shape descriptions come from the Wells/Yajnadevam sign list; Mahadevan numbers from the repository's bridge. Classes are functions, not translations.\n\n"
"| sign | Mahadevan | shape | n | positions I/M/preJar/F/alone | counts before | leaning | class |\n|---|---|---|---|---|---|---|---|\n")
open('FUNCTIONAL-DICTIONARY.md','w').write(hdr+'\n'.join(rows)+'\n')
print(collections.Counter(r.split('**')[1] for r in rows))
