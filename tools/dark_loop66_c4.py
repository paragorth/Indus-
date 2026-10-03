"""S-DARK-66 cycle 4: does the same-shape-family (Wells hundreds block) avoidance at distance >= 2 replicate by site?
Cycle 1 found same-WBLOCK middle pairs at distance >= 2 below both the site x type x length permutation null and a
Markov-2 middle chain, at all three merge levels.  Here the same statistic is computed separately on Mohenjo-daro,
Harappa and the held-out sites (everything else), each subset with its own permutation (within site x type x middle
length) and its own Markov-2 middle chain (fitted on that subset only, per site x type).  Also: the same statistic with
DESC classes (positive control that the per-site machinery can see a class effect) and a block-shuffled control
(Wells blocks relabelled at random among signs with the same block sizes; the avoidance should vanish).
Usage: python3 tools/dark_loop66_c4.py <seq_raw|seq_strong|seq_all> [nperm] [nmarkov]
"""
import sys, json, collections, random, time
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop66_common import *

LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
NM = int(sys.argv[3]) if len(sys.argv) > 3 else 300
rnd = random.Random(66004)
out = [f'# S-DARK-66 cycle 4 ({LV}, perm {NP}, markov {NM}) {time.strftime("%Y-%m-%dT%H:%M")}']
def P(s): print(s, flush=True); out.append(s)

T = parse_all(load_wells(LV))
midtok = collections.Counter(x for t in T for x in t['mid'])
ELEM = sorted(w for w, c in midtok.items() if c >= 5)
WB = wblock_classes(ELEM)
WB_NONUM = {w: c for w, c in WB.items() if c != 'num'}
DESC, _ = desc_classes(); DESC = {w: c for w, c in DESC.items() if w in midtok}
# block-shuffled control: same block sizes, labels permuted among non-numeral signs
labs = [WB_NONUM[w] for w in sorted(WB_NONUM)]; rs = random.Random(7); rs.shuffle(labs)
WB_SHUF = dict(zip(sorted(WB_NONUM), labs))
CLS = {'WBLOCK': WB, 'WBLOCK-no-num': WB_NONUM, 'WBLOCK-shuffled': WB_SHUF, 'DESC': DESC}

def same_share(pairs_per_text, cls):
    s = n = 0
    for ps in pairs_per_text:
        for a, b in ps:
            ca, cb = cls.get(a), cls.get(b)
            if ca is None or cb is None: continue
            n += 1; s += (ca == cb)
    return s, n

def markov_mid(TT, rnd, order=2):
    groups = collections.defaultdict(list)
    for t in TT: groups[(t['site'], t['ot'])].append(t['mid'])
    models = {g: ({o: markov_fit(seqs, o) for o in range(1, order + 1)}, collections.Counter(x for s in seqs for x in s))
              for g, seqs in groups.items()}
    new = []
    for t in TT:
        backoff, uni = models[(t['site'], t['ot'])]
        s = list(t['seq'])
        if t['mid']:
            for pos, x in zip(t['midpos'], markov_gen(None, len(t['mid']), rnd, order, uni, backoff)): s[pos] = x
        new.append(s)
    return new

SUBSETS = {'Mohenjo-daro': [t for t in T if t['site'] == 'Mohenjo-daro'],
           'Harappa': [t for t in T if t['site'] == 'Harappa'],
           'held-out sites': [t for t in T if t['site'] not in BIG],
           'all': T}
RES = {}
for sname, TT in SUBSETS.items():
    pf = lambda seqs: [nonadj_pairs(dict(seq=s, midpos=t['midpos'])) for s, t in zip(seqs, TT)]
    obs = {c: same_share([nonadj_pairs(t) for t in TT], cls) for c, cls in CLS.items()}
    NA = {c: [] for c in CLS}; NB = {c: [] for c in CLS}
    for _ in range(NP):
        ps = pf(permute_middles(TT, rnd))
        for c, cls in CLS.items(): NA[c].append(same_share(ps, cls)[0])
    for _ in range(NM):
        ps = pf(markov_mid(TT, rnd, 2))
        for c, cls in CLS.items(): NB[c].append(same_share(ps, cls)[0])
    P(f'\n## {sname}: {len(TT)} texts')
    RES[sname] = {}
    for c in CLS:
        s, n = obs[c]
        if n == 0: P(f'   {c}: no classed pairs'); continue
        ea, eb = np.mean(NA[c]), np.mean(NB[c])
        pa, pb = pval(s, NA[c], 'lo'), pval(s, NB[c], 'lo')
        P(f'   {c:16s} same-class {s} / {n} pairs ({s / n:.3f}); permutation {ea:.1f} (O/E {s / max(ea, 1e-9):.2f}) P(lo) = {pa:.3f}; '
          f'Markov-2 {eb:.1f} (O/E {s / max(eb, 1e-9):.2f}) P(lo) = {pb:.3f}')
        RES[sname][c] = dict(obs=s, n=n, E_perm=ea, P_perm=pa, E_mk2=eb, P_mk2=pb)

open(DARK + f'loop66_c4_{LV}.txt', 'w').write('\n'.join(out) + '\n')
json.dump(dict(level=LV, nperm=NP, nmarkov=NM, res=RES), open(DARK + f'loop66_c4_{LV}.json', 'w'))
