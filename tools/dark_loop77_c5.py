"""Loop 77 cycle 5: repeat-controlled multi-sided test. S-DARK-37 showed that two faces of one object are often the same
text, a legend and its abbreviation, or a one-sign variant. Those pairs share elements because they are the same
designation, the analogue of PE's identical middles (counted apart in 73.2), not of different entries sharing an element.
Here a face pair is a REPEAT when one FULL text is a contiguous substring of the other, or the two differ by one sign
substitution (same length, Hamming 1); repeat faces are given the same designation so pair_stats counts them as
identical. Then the S-DARK-73.2 statistic is rerun (DES and MID), three merge levels + IM77.
Usage: python3 tools/dark_loop77_c5.py [nperm]
"""
import sys
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop77_common import *
R = {}
P(f'== Loop 77 cycle 5: repeat-controlled multi-sided sharing; nperm {NPERM}')
def sub(a, b): return len(a) <= len(b) and any(tuple(b[k:k + len(a)]) == tuple(a) for k in range(len(b) - len(a) + 1))
def repeat(a, b):
    if not a or not b: return False
    if sub(a, b) or sub(b, a): return True
    return len(a) == len(b) and sum(1 for x, y in zip(a, b) if x != y) == 1
def items_from(groups, ES):
    """groups: list of (gid, site, type, [(k, FULL, ES-set)]). Repeat faces are given one designation."""
    out = []; nrep = 0
    for gid, site, typ, fs in groups:
        des = [list(f) for f in fs]
        for i in range(len(fs)):
            for j in range(i):
                if fs[i][2] != fs[j][2] and repeat(fs[i][1], fs[j][1]):
                    des[i][2] = des[j][2]; nrep += 1; break
        for k, full, d in des:
            out.append((gid, k, d, (site, typ, min(len(d), 5))))
    return out, nrep
for LV in ['seq_raw', 'seq_strong', 'seq_all']:
    objs, QUAL, FR = load_objects(LV)
    M = collapse([o for o in objs.values() if sum(1 for f in o['faces'] if f['seq']) >= 2])
    for ES in ['MID', 'DES']:
        G = [(o['oid'], o['site'], o['type'], [(f['k'], tuple(f['seq']), f[ES]) for f in o['faces'] if f['complete'] and f['seq']]) for o in M]
        it, nrep = items_from(G, ES)
        P(f' -- {LV} {ES}: faces re-labelled as repeats {nrep}')
        R[f'{LV}_{ES}_all'] = within(f'{LV} {ES} multi-sided, repeats as identical', it, NPERM)
        for nm, sel in [('Mohenjo-daro', lambda o: o['site'] == 'Mohenjo-daro'), ('seals + sealings', lambda o: o['ot'] in ('seal', 'sealing'))]:
            ids = {o['oid'] for o in M if sel(o)}
            R[f'{LV}_{ES}_{nm}'] = within(f'{LV} {ES} {nm}, repeats as identical', [x for x in it if x[0] in ids], NPERM)
T, pp = im77_sides()
seen = set(); G = {'MID': [], 'DES': []}
for tn, L in T.items():
    L = [s for s in L if s['side'] >= 1 and s['seq']]
    if len(L) < 2: continue
    key = (L[0]['site'], L[0]['ot'], tuple(sorted(tuple(s['seq']) for s in L)))
    if key in seen: continue
    seen.add(key)
    for ES in G: G[ES].append((tn, L[0]['site'], L[0]['ot'], [(s['side'], tuple(s['seq']), s[ES]) for s in L]))
for ES in G:
    it, nrep = items_from(G[ES], ES)
    P(f' -- IM77 {ES}: repeats {nrep}')
    R[f'IM77_{ES}'] = within(f'IM77 {ES} multi-sided, repeats as identical', it, NPERM)
save('loop77_c5', R)
