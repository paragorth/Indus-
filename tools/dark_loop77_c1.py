"""Loop 77 cycle 1: calibration through this code path (PE tablets, Linear B tablets) and analogue (a): the faces of one
multi-sided Indus object as 'entries on one tablet'. Element sets FULL / MID / DES; three merge levels; IM77 sides.
Usage: python3 tools/dark_loop77_c1.py [nperm]
"""
import sys
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop77_common import *
import dark_loop73_common as L73
R = {}
P(f'== Loop 77 cycle 1; nperm {NPERM}')
P('\n##### calibration: PE entry middles on one tablet (S-DARK-73.2 reproduced through this code)')
a, b = pe_items()
R['PE_sys'] = within('PE tablets, null within numeral system (73.2 stratum)', a, NPERM)
R['PE_len'] = within('PE tablets, null within middle length (Indus-style stratum)', b, NPERM)
LB = L73.linb_names_tab()
R['LinB'] = within('Linear B personnel names on one tablet, null within series x length',
                   [(x['tablet'], i, x['name'], (x['series'], min(len(x['name']), 5))) for i, x in enumerate(LB)], NPERM)

P('\n##### (a) faces of one multi-sided object (complete faces, moulded identical face-sets collapsed)')
for LV in ['seq_raw', 'seq_strong', 'seq_all']:
    objs, QUAL, FR = load_objects(LV)
    M = collapse([o for o in objs.values() if sum(1 for f in o['faces'] if f['seq']) >= 2])
    P(f' -- {LV}: multi-sided objects {len(M)}; types {collections.Counter(o["type"] for o in M).most_common(6)}')
    for ES in ['FULL', 'MID', 'DES']:
        items = [(o['oid'], f['k'], f[ES], (o['site'], o['type'], min(len(f[ES]), 5))) for o in M for f in o['faces'] if f['complete'] and f['seq']]
        R[f'{LV}_{ES}_all'] = within(f'{LV} {ES} all multi-sided', items, NPERM)
        if ES == 'DES':
            for nm, sel in [('Harappa tablets', lambda o: o['site'] == 'Harappa' and o['ot'] == 'tablet'),
                            ('Mohenjo-daro objects', lambda o: o['site'] == 'Mohenjo-daro'),
                            ('seals + sealings', lambda o: o['ot'] in ('seal', 'sealing'))]:
                it = [x for x in items if sel(objs[x[0]])]
                R[f'{LV}_{ES}_{nm}'] = within(f'{LV} {ES} {nm}', it, NPERM)
    if LV == 'seq_raw':
        items = [(o['oid'], f['k'], f['DES'], (o['site'], o['type'], min(len(f['DES']), 5))) for o in M for f in o['faces'] if f['complete'] and f['seq']]
        P('   DES elements most shared across faces of one object:', shared_elements(items).most_common(12))
        items = [(o['oid'], f['k'], f['MID'], 0) for o in M for f in o['faces'] if f['complete'] and f['seq']]
        P('   MID elements most shared across faces of one object:', shared_elements(items).most_common(12))

P('\n##### (a) IM77 multi-sided texts (sides >= 1; bridged to W space; identical side-sets per site x type collapsed)')
T, pp = im77_sides()
P(f'  share of IM77 tokens resting on proposed (S-DARK-27) bridge entries: {pp:.3f}')
seen = set(); keep = []
for tn, L in T.items():
    L = [s for s in L if s['side'] >= 1 and s['seq']]
    if len(L) < 2: continue
    key = (L[0]['site'], L[0]['ot'], tuple(sorted(tuple(s['seq']) for s in L)))
    if key in seen: continue
    seen.add(key); keep.append((tn, L))
P(f'  IM77 multi-sided texts kept {len(keep)}; types {collections.Counter(L[0]["ot"] for _, L in keep).most_common(5)}')
for ES in ['FULL', 'MID', 'DES']:
    items = [(tn, s['side'], s[ES], (s['site'], s['ot'], min(len(s[ES]), 5))) for tn, L in keep for s in L]
    R[f'IM77_{ES}'] = within(f'IM77 {ES} multi-sided', items, NPERM)
items = [(tn, s['side'], s['DES'], (s['site'], s['ot'], min(len(s['DES']), 5))) for tn, L in keep for s in L if s['site'] == 'Harappa']
R['IM77_DES_Harappa'] = within('IM77 DES Harappa multi-sided', items, NPERM)
save('loop77_c1', R)
