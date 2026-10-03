"""Loop 54 cycle 2: held-out check on the 324 IM77-only texts (S-DARK-27.3 'certainly new', Mahadevan numbers).
Claims translated W -> M through the completed bridge (loop27_sets.json); chains fitted on these 324 texts alone, grouped by
object type (site x type is too sparse at n = 324). Also the per-claim token counts (power).
Usage: python3 tools/dark_loop54_c2.py [nnull]
Output: data/derived/dark/loop54_c2_im77.txt / .json"""
import sys, time, json, random, collections
from dark_loop54_common import *
from dark_loop54_claims import claims

NN = int(sys.argv[1]) if len(sys.argv) > 1 else 200
t0 = time.time()
T = im77_new_texts()
br = bridge_w2m()
# bridge additions that the alignment could not validate but S-DARK-27.3 lists as contextual: W34 = M95; W91 twins = M3? no: keep unbridged
S = m_sets(br); lib = mk(S)
missing = []
def tr(ws):
    m = to_m(ws, br)
    if m is None:
        missing.append(tuple(sorted(ws))); return set(w for w in ws if w in br) and to_m({w for w in ws if w in br}, br) or {-1}
    return m
CL_ = claims(lib, S, tr)
print('IM77-only texts', len(T), 'sites', collections.Counter(m['site'] for m, _ in T).most_common(), 'types', collections.Counter(m['ot'] for m, _ in T).most_common())
print('unbridged sign sets (partially translated):', sorted(set(missing)))
obs = [c['f'](T) for c in CL_]
rnd = random.Random(27)
nulls = {}
for name, order, end in (('M1', 1, False), ('M2', 2, False), ('M1E', 1, True)):
    ch = Chain(T, order=order, end=end, group=lambda m: m['ot'])
    vals = [[] for _ in CL_]
    for it in range(NN):
        G = ch.corpus(T, rnd)
        for i, c in enumerate(CL_): vals[i].append(c['f'](G))
    nulls[name] = vals
    print(name, 'done', round(time.time() - t0), flush=True)
rows = []
out = [f'# Loop 54 cycle 2: the {len(T)} IM77-only texts (M numbers; claims via the completed bridge); chains fitted on these texts per object type; {NN} corpora each. inside = within the 2.5-97.5% band; n = texts/tokens the claim rests on is given by the observed denominator where finite.',
       '| entry | claim | kind | observed | M1 med [band] | M1 | M2 | M1E | verdict |', '|---|---|---|---|---|---|---|---|---|']
for i, c in enumerate(CL_):
    r = {'entry': c['entry'], 'claim': c['claim'], 'kind': c['kind'], 'obs': obs[i]}
    for name in nulls: r[name] = summarize(obs[i], nulls[name][i])
    m1 = r['M1']['inside']; m2 = r['M2']['inside']
    if obs[i] != obs[i]: v = 'no tokens'
    elif c['kind'] == 'adjacency': v = 'adjacency fact' + ('' if m1 else ' (outside M1: ' + r['M1']['dir'] + ')')
    elif m1 is None: v = 'untestable'
    elif m1 and m2: v = 'chain-explained'
    elif m1: v = 'inside M1, outside M2'
    elif m2: v = 'beyond M1, within M2'
    else: v = 'beyond both chains (' + r['M1']['dir'] + ')'
    r['verdict'] = v; rows.append(r)
    def band(k): return f"{fmt(r[k]['med'])} [{fmt(r[k]['lo'])}-{fmt(r[k]['hi'])}]"
    def flag(k): return ('in' if r[k]['inside'] else r[k]['dir']) if r[k]['inside'] is not None else '-'
    out.append(f"| {c['entry']} | {c['claim']} | {c['kind']} | {fmt(obs[i])} | {band('M1')} | {flag('M1')} | {flag('M2')} | {flag('M1E')} | {v} |")
open(DARK + 'loop54_c2_im77.txt', 'w').write('\n'.join(out) + '\n')
json.dump(rows, open(DARK + 'loop54_c2_im77.json', 'w'), default=str)
print('written', round(time.time() - t0))
