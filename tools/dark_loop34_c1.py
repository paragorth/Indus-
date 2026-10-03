"""S-DARK-34 cycle 1: determinism table for SIGN targets (first, second, penultimate, last, closer) on seals.
Train Mohenjo-daro + Harappa seals; test (a) seals at the other sites, (b) the 324 IM77-only texts (seals + others).
Usage: python3 tools/dark_loop34_c1.py <seq_raw|seq_strong|seq_all>"""
import sys, json
sys.argv = [sys.argv[0], '1'] + sys.argv[1:]
from dark_loop34 import *

seals = [o for o in OBJ if o['ot'] == 'SEAL']
train = [o for o in seals if o['big']]
sites = [o for o in seals if not o['big']]
im77 = [o for o in NEW]
im77_seals = [o for o in NEW if o['ot'] == 'SEAL']
print(f'seals: train {len(train)}, held-out sites {len(sites)}; IM77-only all {len(im77)}, seals {len(im77_seals)}')
log = open(SP + f'loop34_c1_{LV}.txt', 'w')
def P(s): print(s); log.write(s + '\n'); log.flush()
P(f'# S-DARK-34 cycle 1, level {LV}: train seals {len(train)}, held-out site seals {len(sites)}, IM77-only {len(im77)} (seals {len(im77_seals)})')
rows = {}
for t in ('first_sign', 'second_sign', 'penult_sign', 'last_sign', 'closer'):
    P(f'--- target {t}')
    r_text, _ = evaluate(t, train, {'sites': sites, 'im77': im77}, use_facts=False); P('TEXT-ONLY  ' + fmt(r_text))
    r_full, _ = evaluate(t, train, {'sites': sites}, use_facts=True); P('TEXT+FACTS ' + fmt(r_full))
    r_ck, _ = evaluate(t, train, {'sites': sites, 'im77': im77}, use_facts=False, plant='checksum', report_feats=False); P('PLANT-CK   ' + fmt(r_ck))
    r_cs, _ = evaluate(t, train, {'sites': sites, 'im77': im77}, use_facts=False, plant='classsign', report_feats=False); P('PLANT-CS   ' + fmt(r_cs))
    r_sh, _ = evaluate(t, train, {'sites': sites, 'im77': im77}, use_facts=False, shuffle=True, report_feats=False); P('SHUFFLED   ' + fmt(r_sh))
    rows[t] = dict(text=r_text, full=r_full, checksum=r_ck, classsign=r_cs, shuffled=r_sh)
json.dump(rows, open(SP + f'loop34_c1_{LV}.json', 'w'), indent=1, default=str)
