"""Summarise pe39 runs: python3 pe39_sum.py TAG [min_n] [agree]"""
import sys, os, json, glob, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe39_common import CK

TAG = sys.argv[1]
MINN = int(sys.argv[2]) if len(sys.argv) > 2 else 15
AG = float(sys.argv[3]) if len(sys.argv) > 3 else 0.75
gold = json.load(open(os.path.join(CK, 'gold_ur3_pc.json')))


def correct(st, s, t):
    a = s.split('|', 1)[1]; b = t.split('|', 1)[1]
    if st.startswith('PLANT'):
        return a == b
    if st.startswith('UR3'):
        return (b in gold[a]) if a in gold else None
    return None


runs = collections.defaultdict(list)
for fn in sorted(glob.glob(os.path.join(CK, TAG + '_*.json'))):
    r = json.load(open(fn))
    runs[(r['set'], r['arch'])].append(r)
out = {}
for key, R in sorted(runs.items()):
    st = key[0]
    row = {'n_seeds': len(R), 'rt_ho_A': round(np.mean([r['rt_ho_A'] for r in R]), 3),
           'rt_ho_B': round(np.mean([r['rt_ho_B'] for r in R]), 3),
           'numcopy': round(np.mean([r['numcopy_AB'] for r in R]), 3)}
    # per-run precision on frequent sources
    pr = []
    for r in R:
        c = [correct(st, s, t[0]) for s, t in r['lexAB'].items() if t[4] >= MINN]
        c = [x for x in c if x is not None]
        if c:
            pr.append(np.mean(c))
    if pr:
        row['prec_freq'] = round(float(np.mean(pr)), 3)
    # stability across seeds
    tg = collections.defaultdict(list)
    ns = {}
    for r in R:
        for s, t in r['lexAB'].items():
            if t[4] >= MINN:
                tg[s].append(t[0]); ns[s] = t[4]
    stable = []
    for s, ts in tg.items():
        m, c = collections.Counter(ts).most_common(1)[0]
        agree = c / len(R)
        if agree >= AG:
            gains = [r['gainAB'][s][1] for r in R if s in r['gainAB'] and r['gainAB'][s][0] == m]
            stable.append((s, m, round(agree, 2), ns[s], round(float(np.mean(gains)), 3) if gains else None,
                           correct(st, s, m)))
    row['n_sources'] = len(tg)
    row['n_stable'] = len(stable)
    ks = [x[5] for x in stable if x[5] is not None]
    if ks:
        row['stable_prec'] = '%d/%d' % (sum(ks), len(ks))
    kg = [x[5] for x in stable if x[5] is not None and x[4] is not None and x[4] > 0.1]
    if kg:
        row['stable_gain_prec'] = '%d/%d' % (sum(kg), len(kg))
    pp = [np.mean([c for _, _, c, n in r['profile_known'] if n >= MINN]) for r in R if r.get('profile_known')]
    if pp:
        row['profile_prec'] = round(float(np.mean(pp)), 3)
    allg = [g[1] for r in R for g in r['gainAB'].values()]
    row['gain_mean_all'] = round(float(np.mean(allg)), 4) if allg else None
    row['stable'] = sorted(stable, key=lambda x: -x[3])
    out['%s/%s' % key] = row
    print('%-12s %s seeds %d rtA %.3f rtB %.3f numcopy %.2f prec_freq %s sources %d stable %d stable_prec %s gain_all %s' % (
        st, key[1], len(R), row['rt_ho_A'], row['rt_ho_B'], row['numcopy'], row.get('prec_freq'), len(tg), len(stable),
        row.get('stable_prec'), row['gain_mean_all']), 'stable&gain>0.1 prec', row.get('stable_gain_prec'), 'profile_prec', row.get('profile_prec'))
    for x in row['stable'][:25]:
        print('    ', x)
json.dump(out, open(os.path.join(CK, 'sum_%s.json' % TAG), 'w'), indent=0)
