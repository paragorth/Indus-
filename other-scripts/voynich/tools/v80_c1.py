"""v80 cycle 1: massive random table-hypothesis search on every corpus (and on interior-shuffled nulls).
usage: python3 v80_c1.py NHYP corpus1 corpus2 ...   (a name ending in ~sh = interior-shuffle null of that corpus)"""
import sys, json, os, time
import v80_lib as L

N = int(sys.argv[1]); names = sys.argv[2:]
C = L.corpora()
H = L.random_hyps(N, 80)
for nm in names:
    out = os.path.join(L.CK, sys.argv[0].endswith('c1.py') and os.environ.get('V80DIR', 'c1') or 'c1', nm.replace('~', '_').replace('@', '-') + '.json')
    if os.path.exists(out): continue
    base = nm.split('~')[0].split('@')[0]
    P = C[base][1]
    if '@' in nm:
        sec = nm.split('@')[1].split('~')[0]; P = [p for p in P if p['sec'] in sec]
    if '~sh' in nm: P = L.shuffle_interior(P, 8801 + int(nm.split('~sh')[1] or 0))
    T = L.flatten(P)
    t = time.time()
    R = L.run_hyps(T, H)
    json.dump(R, open(out, 'w'))
    print(nm, len(R), '%.0fs' % (time.time() - t), flush=True)
