#!/usr/bin/env python3
"""la66: freeze the Linear A factor table (descriptors, candidates, identifiers) with a sha256 for testing on
new documents.  Test rule: for each listed feature, new entries carrying it are compared with their
same-document, same-commodity siblings; a feature is supported when >= 70 % of >= 10 new cases move in the
stated direction, killed when <= 50 %."""
import os, json, math, hashlib
from la66_lib import CK, DATA, la_rows, factor_boot

C2 = json.load(open(os.path.join(CK, 'c2_class.json')))
C3 = json.load(open(os.path.join(CK, 'c3.json')))
grade = {'W:KI': 'C', 'S:E': 'C-', 'W:*516': 'C-', 'W:A-DU': 'C-', 'W:TA': 'C-'}
tab = []
for f, v in C3['LA_factors'].items():
    tab.append(dict(feature=f, log_factor=round(v[0], 3), factor=round(math.exp(v[0]), 2),
                    ci95=[round(math.exp(v[1]), 2), round(math.exp(v[2]), 2)], n_docs=v[3],
                    direction='down' if v[0] < 0 else 'up', status='stable' if f in C2['descriptors'] else 'candidate',
                    grade=grade.get(f, 'C-')))
k = C3['LA_kuro']
frozen = dict(name='la66 Linear A word weights', date='2026-10-06',
              target='log integer quantity, within document x (commodity x fraction use)',
              test_rule=__doc__.split('Test rule: ')[1].replace('\n', ' '),
              factors=tab,
              internal_check=dict(feature='W:KU-RO', factor=round(math.exp(k[0]), 2), ci95=[round(math.exp(k[1]), 2), round(math.exp(k[2]), 2)], n_docs=k[3]),
              identifiers=C2['identifiers'])
body = json.dumps(frozen, sort_keys=True, ensure_ascii=False)
h = hashlib.sha256(body.encode()).hexdigest()
frozen['sha256_of_body_without_this_field'] = h
json.dump(frozen, open(os.path.join(DATA, 'la66_frozen_factors.json'), 'w'), indent=1, ensure_ascii=False)
print(h)
for t in tab:
    print(t)
