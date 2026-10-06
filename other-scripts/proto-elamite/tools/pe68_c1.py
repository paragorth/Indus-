"""pe68 cycle 1: inventory of joins (PE, proto-cuneiform, Ur III) and what each PE fragment showed alone."""
import os, sys, csv, json, re
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe68_lib import *  # noqa

SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
csv.field_size_limit(10 ** 9)
inv = {'PE': [], 'PC': [], 'U3': []}
for r in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), errors='replace')):
    p = r['period']
    key = 'PE' if p.startswith('Proto-Elam') else 'PC' if p.startswith('Uruk') else 'U3' if p.startswith('Ur III') else None
    if not key:
        continue
    ji, mus, des = r['join_information'].strip(), r['museum_no'], r['designation']
    plus = (' + ' in mus) or (' + ' in des) or bool(re.search(r'\d\s*\+\s*\d', des))
    if ji or plus:
        inv[key].append({'P': 'P%06d' % int(r['id_text']) if r['id_text'].isdigit() else r['id_text'],
                         'des': des, 'mus': mus, 'join_info': ji[:200], 'plus': plus})
T = {t['id']: t for t in corpus('PE')}
rows = []
frag = {}
for pid in sorted(FRAGMAP):
    t = T[pid]
    F = resolve_fragments(pid, t['lines'])
    frag[pid] = F
    for lab, idx in F.items():
        V = [t['lines'][i] for i in idx]
        H = [l for i, l in enumerate(t['lines']) if i not in set(idx)]
        rows.append((pid, lab, len(idx), sum(1 for l in V if l['cls']), tablet_kind(V), sum(1 for l in H if l['cls']),
                     tablet_kind(t['lines']), bool([l for l in V if l['hdr']]), FRAGMAP[pid]['conf']))
json.dump({'inventory': inv, 'fragments': frag, 'fragmap': FRAGMAP, 'excluded': EXCLUDED,
           'sha_fragmap': sha({k: v for k, v in frag.items()})}, open(os.path.join(CK, 'c1_joins.json'), 'w'), indent=1)
n_plus_pe = sum(1 for x in inv['PE'] if x['plus'])
print('inventory PE %d (with +: %d), PC %d, U3 %d' % (len(inv['PE']), n_plus_pe, len(inv['PC']), len(inv['U3'])))
print('fragmap sha', sha({k: v for k, v in frag.items()})[:16])
for r in rows:
    print(r)
kinds = Counter((r[4], r[6]) for r in rows)
print(kinds)
