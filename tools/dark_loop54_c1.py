"""Loop 54 cycle 1: every dictionary / grammar claim on Mohenjo-daro + Harappa (die regime) against
Markov-1 (site x type, lengths kept), Markov-2 (same), Markov-1 with END state (lengths free) and the S366 M* generator.
Usage: python3 tools/dark_loop54_c1.py <seq_raw|seq_strong|seq_all> [nnull] [ngen] [fitset: big|heldout|all]
Output: data/derived/dark/loop54_c1_<level>_<fitset>.txt and .json"""
import sys, time, json, random
from dark_loop54_common import *
from dark_loop54_claims import claims

level = sys.argv[1] if len(sys.argv) > 1 else 'seq_all'
NN = int(sys.argv[2]) if len(sys.argv) > 2 else 200
NG = int(sys.argv[3]) if len(sys.argv) > 3 else 100
fitset = sys.argv[4] if len(sys.argv) > 4 else 'big'
t0 = time.time()
Tall = load_corpus(level, 'die')
if fitset == 'big': T = [x for x in Tall if x[0]['site'] in BIG]
elif fitset == 'heldout': T = [x for x in Tall if x[0]['site'] not in BIG]
else: T = Tall
S = w_sets(); lib = mk(S); CL_ = claims(lib, S)
obs = [c['f'](T) for c in CL_]
rnd = random.Random(54)
nulls = {}
for name, order, end in (('M1', 1, False), ('M2', 2, False), ('M1E', 1, True)):
    ch = Chain(T, order=order, end=end)
    vals = [[] for _ in CL_]
    for it in range(NN):
        G = ch.corpus(T, rnd)
        for i, c in enumerate(CL_): vals[i].append(c['f'](G))
    nulls[name] = vals
    print(name, 'done', round(time.time() - t0), flush=True)
if fitset == 'big' and NG > 0:
    SA, model = s366_generator(level)
    vals = [[] for _ in CL_]
    for it in range(NG):
        G = s366_corpus(SA, model, T, rnd)
        for i, c in enumerate(CL_): vals[i].append(c['f'](G))
    nulls['G'] = vals
    print('G done', round(time.time() - t0), flush=True)

rows = []
out = [f'# Loop 54 cycle 1: claims on {fitset} ({len(T)} texts, die regime, {level}); nulls M1 (order-1, site x type, lengths kept), M2 (order-2), M1E (order-1 with END, lengths free), G (S366 M* generator fitted on MD+H); {NN} corpora per chain, {NG} for G. inside = observed within the 2.5-97.5% band.',
       '| entry | grade | claim | kind | observed | M1 med [band] | M1 | M2 med [band] | M2 | M1E | G med [band] | G | verdict | outside fact |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for i, c in enumerate(CL_):
    r = {'entry': c['entry'], 'grade': c['grade'], 'claim': c['claim'], 'kind': c['kind'], 'obs': obs[i], 'outside': c['outside'], 'srow': c['srow']}
    for name in nulls:
        r[name] = summarize(obs[i], nulls[name][i])
    m1 = r['M1']['inside']; m2 = r['M2']['inside']
    if c['kind'] == 'adjacency': v = 'adjacency fact (chain by construction)' if m1 else 'adjacency fact, yet outside M1 band (' + r['M1']['dir'] + ')'
    elif m1 is None: v = 'untestable'
    elif m1 and m2: v = 'chain-explained'
    elif m1 and not m2: v = 'inside M1, outside M2'
    elif (not m1) and m2: v = 'beyond M1, within M2'
    else: v = 'beyond both chains (' + r['M1']['dir'] + ')'
    if 'M1E' in r and r['M1E']['inside'] is not None and not m1 and r['M1E']['inside']: v += '; M1E reproduces (end-state fact)'
    if 'G' in r and r['G']['inside'] is not None and not r['G']['inside']: v += '; outside S366 generator too'
    r['verdict'] = v; rows.append(r)
    def band(k): return f"{fmt(r[k]['med'])} [{fmt(r[k]['lo'])}-{fmt(r[k]['hi'])}]" if k in r else '-'
    def flag(k): return ('in' if r[k]['inside'] else r[k]['dir']) if k in r and r[k]['inside'] is not None else '-'
    out.append(f"| {c['entry']} | {c['grade']} | {c['claim']} | {c['kind']} | {fmt(obs[i])} | {band('M1')} | {flag('M1')} | {band('M2')} | {flag('M2')} | {flag('M1E')} | {band('G')} | {flag('G')} | {v} | {c['outside']} |")
open(DARK + f'loop54_c1_{level}_{fitset}.txt', 'w').write('\n'.join(out) + '\n')
json.dump(rows, open(DARK + f'loop54_c1_{level}_{fitset}.json', 'w'), default=str)
print('written', round(time.time() - t0))
