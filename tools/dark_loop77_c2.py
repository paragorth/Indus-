"""Loop 77 cycle 2: analogue (b): texts found together in one find-group (Harappa tablet series; also Mohenjo-daro tablets,
and Mohenjo-daro / Harappa seals as the S-DARK-64 reference) as 'entries on one tablet'. One designation per object (its
longest complete face; count faces drop out of DES/MID automatically). Groups at two grains: spot = area x block x
room/grid; area = area-section. Identical texts (stock copies) are counted separately, as in 73.2. Strata for the
permutation null: site x Wells type x designation length (5+ pooled). Three merge levels; IM77 locus x level.
Usage: python3 tools/dark_loop77_c2.py [nperm_spot] [nperm_area]
"""
import sys
NP1 = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
NP2 = int(sys.argv[2]) if len(sys.argv) > 2 else 200
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop77_common import *
R = {}
P(f'== Loop 77 cycle 2 (find-groups); nperm spot {NP1}, area {NP2}')

def textface(o):
    fs = [f for f in o['faces'] if f['complete'] and f['seq']]
    return max(fs, key=lambda f: (len(f['seq']), -f['k'])) if fs else None

def run(objs, LV, sets=('FULL', 'MID', 'DES')):
    for site, ot in [('Harappa', 'tablet'), ('Mohenjo-daro', 'tablet'), ('Mohenjo-daro', 'seal'), ('Harappa', 'seal')]:
        O = [o for o in objs.values() if o['site'] == site and o['ot'] == ot]
        for grain in ['spot', 'area']:
            if ot == 'seal' and grain == 'area': continue
            for ES in sets:
                if ot == 'seal' and ES == 'FULL': continue
                items = []
                for o in O:
                    f = textface(o)
                    if not f: continue
                    g = (o['area'], o['block'], o['room']) if grain == 'spot' else o['area']
                    if grain == 'spot' and not o['room']: continue
                    if grain == 'area' and not o['area']: continue
                    items.append((g, int(o['oid']), f[ES], (o['site'], o['type'], min(len(f[ES]), 5))))
                key = f'{LV}_{site}_{ot}_{grain}_{ES}'
                R[key] = within(f'{LV} {site} {ot}s by {grain}, {ES}', items, NP1 if grain == 'spot' else NP2, markov=True)
                if LV == 'seq_raw' and ES == 'DES' and R[key]:
                    P('     DES elements shared inside groups:', shared_elements(items).most_common(10))

for LV in ['seq_raw', 'seq_strong', 'seq_all']:
    P(f'\n##### {LV}')
    objs, QUAL, FR = load_objects(LV)
    run(objs, LV)

P('\n##### IM77: objects (longest side) grouped by locus x level (spot) and locus (area); bridged to W space')
T, pp = im77_sides()
P(f'  proposed-bridge token share {pp:.3f}')
for site, ots in [('Harappa', ('miniature tablet',)), ('Mohenjo-daro', ('miniature tablet', 'copper tablet')), ('Mohenjo-daro', ('seal',)), ('Harappa', ('seal',))]:
    for grain in ['spot', 'area']:
        for ES in ['MID', 'DES']:
            items = []; seen = set()
            for tn, L in T.items():
                s = max(L, key=lambda x: len(x['seq']))
                if s['site'] != site or s['ot'] not in ots or not s['seq']: continue
                if s['locus'] in ('', '0', '-'): continue
                if grain == 'spot' and s['level'] in ('', '0', '-'): continue
                g = (s['locus'], s['level']) if grain == 'spot' else s['locus']
                items.append((g, int(tn), s[ES], (site, s['ot'], min(len(s[ES]), 5))))
            R[f'IM77_{site}_{ots[0]}_{grain}_{ES}'] = within(f'IM77 {site} {ots[0]} by {grain}, {ES}', items, NP2, markov=True)
save('loop77_c2', R)
