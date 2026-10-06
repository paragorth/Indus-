#!/usr/bin/env python3
"""pe63 cycle 1: random-template dossier search. usage: pe63_c1.py CORPUS MODE SEED N_TEMPLATES
CORPUS in PE / DR (Ur III Drehem) / UM (Ur III Umma); MODE real / null (line-shuffled corpus) / plant."""
import sys, os, json, random, time
import pe63_common as C

corp, mode, sd, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out = os.path.join(C.CK, 'c1_%s_%s_%d.json' % (corp, mode, sd))
if os.path.exists(out):
    sys.exit(0)
T = {'PE': C.pe_tabs, 'DR': lambda: C.ur3_tabs('Puzr', 1500, 'DR'),
     'UM': lambda: C.ur3_tabs('Umma', 1500, 'UM')}[corp]()
rng = random.Random(C.seed('pe63c1-%s-%s-%d' % (corp, mode, sd)))
truth = None
if mode == 'null':
    T = C.shuffle_lines(T, rng)
elif mode == 'plant':
    T, truth = C.plant_dossier(T, rng, k=6, tag='PL%d_' % sd)
t0 = time.time()
ix = C.Index(T)
best = C.search(ix, n, rng)
res = [{'G': [T[i]['id'] for i in G], 'h': h, 'tm': list(tm)} for G, (h, tm) in best.items()]
res.sort(key=lambda r: -r['h'])
json.dump({'corp': corp, 'mode': mode, 'seed': sd, 'n': n, 'groups': res[:20000], 'n_groups': len(res),
           'truth': truth, 'secs': time.time() - t0,
           'tabs': T if mode == 'plant' else None}, open(out, 'w'))
print(corp, mode, sd, len(res), round(time.time() - t0))
