"""pe37: bone-derived targets (Malyan Banesh) via herd-demography ABC.
Writes data/pe37_bone_targets.json (derived numbers only)."""
import time
from pe37_common import *

rows = malyan_rows()
out = {'source': 'Zeder, Tal-e Malyan Zooarchaeology, Open Context project d6b25ec9-2884-4e3c-00e8-0c5a6472fa63 '
                  '(export tables 394c525a-151a-7107-e310-55b48f304fd3, 6c6d9656-ad63-14c0-b24f-4153086c3620); '
                  'Banesh = records dated -3400/-2600; Kaftari = -2400/-1600 (comparison only)',
       'fusion_groups': {g: [e + ':' + p for e, p in v] for g, v in FUSION_GROUPS.items()},
       'fusion_age_years': FUSION_AGE}
for lab, ed, unit in (('banesh_all', '-3400.0', None), ('banesh_unit10', '-3400.0', 'Unit 10'),
                      ('banesh_unit20', '-3400.0', 'Unit 20'), ('kaftari_all', '-2400.0', None)):
    bc = bone_counts(rows, ed, unit)
    t0 = time.time()
    r = demography_abc(bc, n=int(sys.argv[1]) if len(sys.argv) > 1 else 60000, seed=7)
    my, sy = wstats(r['young'], r['w'])
    ma, sa = wstats(r['adultF'], r['w'])
    mb, sb = wstats(r['births'], r['w'])
    sheep = bc['ovis'] / (bc['ovis'] + bc['capra'])
    # NISP sheep share: binomial sd inflated x3 for identification bias
    ssd = 3 * math.sqrt(sheep * (1 - sheep) / (bc['ovis'] + bc['capra']))
    cat = bc['bos'] / (bc['bos'] + bc['caprine'])
    csd = 3 * math.sqrt(cat * (1 - cat) / (bc['bos'] + bc['caprine']))
    bc.update({'living_young_share': [my, sy], 'living_adultF_share': [ma, sa],
               'births_per_female_year': [mb, sb], 'abc_ess': r['ess'], 'abc_valid': r['n_valid'],
               'sheep_share_nisp': [sheep, ssd], 'cattle_share_nisp': [cat, csd]})
    out[lab] = bc
    print(lab, bc['fusion_caprine'], bc['pelvis_sex'], 'young %.3f+-%.3f adF %.3f+-%.3f births %.2f ess %.0f valid %d sheep %.3f+-%.3f cattle %.4f (%.0fs)'
          % (my, sy, ma, sa, mb, r['ess'], r['n_valid'], sheep, ssd, cat, time.time() - t0), flush=True)
dump(out, TARGETS)
