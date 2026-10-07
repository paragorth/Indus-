"""v84 cycle 2 summary: by page mechanism, how many settings reach the Voynich PB band, the shape line, all bands."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v84_c2 as C
from collections import defaultdict

if __name__ == '__main__':
    C.init()
    pop = [r for r in C.load('c2_sweep.jsonl') + C.load('c2_climb.jsonl') if 'err' not in r]
    by = defaultdict(list)
    for r in pop: by[C.mechs(r['P'])].append(r)
    print('n', len(pop))
    for m, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        inb = [r for r in rs if C.PB_BAND[0] <= r['pb'] <= C.PB_BAND[1]]
        size = [r for r in rs if C.flags(r)[0]]; shape = [r for r in rs if C.flags(r)[1]]
        okr = [r for r in rs if r['ok'] and r['on_ok']]
        pbs = np.array([r['pb'] for r in rs]); fr = np.array([r['far'] / max(r['pb'], 1e-3) for r in rs])
        b = max(rs, key=C.fit)
        print('%-16s n %4d  pb med %.3f [%.3f-%.3f]  inPB %3d  bands+onset %3d  SIZE %3d  SHAPE %3d  far/pb med %.2f | best fit %.2f pb %.3f far %.2f bad %s on %s' % (
            m, len(rs), np.median(pbs), np.percentile(pbs, 10), np.percentile(pbs, 90), len(inb), len(okr), len(size), len(shape),
            np.median(fr), C.fit(b), b['pb'], b['far'] / max(b['pb'], 1e-3), b['bad'], b['on_ok']))
    if len(sys.argv) > 1:
        fin = [r for r in C.load('c2_final.jsonl') if 'err' not in r]
        byP = defaultdict(list)
        for r in fin: byP[json.dumps(r['P'], sort_keys=True)].append(r)
        for k, rs in byP.items():
            P = json.loads(k)
            print(C.mechs(P), 'pb', [round(r['pb'], 3) for r in rs], 'swap', [round(r.get('pb_swap', 0), 3) for r in rs],
                  'far/pb', [round(r['far'] / max(r['pb'], 1e-3), 2) for r in rs], 'bad', [r['bad'] for r in rs],
                  'on', [r['on_ok'] for r in rs], 'size', [C.flags(r)[0] for r in rs], 'shape', [C.flags(r)[1] for r in rs])
