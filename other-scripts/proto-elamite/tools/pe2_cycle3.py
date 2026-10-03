"""PE2 cycle 3: per-sign nulls, known-function partners, sensitivity.

 (1) Per-sign null for the cycle-2 leaders: held-out rank of the real partner
     (pairing on half A without system features) vs the partner chosen after
     PE-A numeral groups are shuffled within tablet (20 shuffles x 10 splits).
 (2) Known-function partners: for every PE sign whose no-system partner (mode
     over 10 splits, >= 5/10) is a PC sign with a conventional commodity name,
     compare held-out system-profile similarity to that partner against all other
     named commodity signs (rank among them; random = 0.5).
     Direction test from cycle 2 redone with a draw-from-same-pool null.
 (3) Sensitivity: PC restricted to Uruk III (closer in date); and K_PC = 100.
"""
import json, random
import numpy as np
from collections import defaultdict, Counter
from pe2_common import *
from pe2_cycle2 import pair, NOSYS, SYSF, KPE, KPC

LEAD = ['M036', 'M002', 'M243', 'M010', 'M297', 'M376', 'M388', 'M354', 'M347', 'M124']


def shuffled(A, seed):
    rr = random.Random(seed); ov = {}
    for ti, t in enumerate(A):
        ents = [li for li, l in enumerate(t['lines']) if l['signs'] and l['nums']]
        sy = [system_of(t['lines'][li]['nums']) for li in ents]; rr.shuffle(sy)
        ov.update({(ti, li): x for li, x in zip(ents, sy)})
    return ov


def run(PE, Ppc, npc, kpc=KPC, nshuf=20, label=''):
    Pall, nall = profiles(PE)
    core = top_signs(Pall, nall, KPE)
    pcs = top_signs(Ppc, npc, kpc)
    Z = matrix(Ppc, pcs, SYSF); idx = {s: i for i, s in enumerate(pcs)}
    func = [s for s in pcs if s in PC_FUNC]
    cap_pc = np.mean([Ppc[s]['sys_C'] for s in pcs])
    real = defaultdict(list); null = defaultdict(list); partners = defaultdict(list)
    dir_hits = []; dir_null = []
    fr = defaultdict(list)
    rng = np.random.default_rng(7)
    for seed in range(10):
        A, B = split_tablets(PE, seed)
        Pa, na = profiles(A, min_n=5); Pb, nb = profiles(B, min_n=5)
        use = [s for s in core if s in Pa and s in Pb]
        m, _ = pair(Pa, na, Ppc, npc, NOSYS, kb=kpc, restrict_a=use)
        S = cos_sim(matrix(Pb, use, SYSF), Z)
        cap_pe = np.mean([Pb[s]['sys_C'] for s in use])
        for i, s in enumerate(use):
            partners[s].append(m[s])
            real[s].append(float((S[i] > S[i, idx[m[s]]]).mean()))
            if m[s] in PC_FUNC:
                hit = (Pb[s]['sys_C'] > cap_pe) == (Ppc[m[s]]['sys_C'] > cap_pc)
                dir_hits.append(hit)
                draws = rng.choice(func, 200)
                dir_null.append(np.mean([(Pb[s]['sys_C'] > cap_pe) == (Ppc[d]['sys_C'] > cap_pc) for d in draws]))
                fi = [idx[f] for f in func if f != m[s]]
                fr[s].append(float((S[i, fi] > S[i, idx[m[s]]]).mean()))
        for k in range(nshuf):
            P0, n0 = profiles(A, min_n=5, sys_override=shuffled(A, 1000 * seed + k))
            u0 = [s for s in use if s in P0]
            m0, _ = pair(P0, n0, Ppc, npc, NOSYS, kb=kpc, restrict_a=u0)
            for i, s in enumerate(use):
                if s in m0:
                    null[s].append(float((S[i] > S[i, idx[m0[s]]]).mean()))
    per = {}
    for s in core:
        if s not in real: continue
        mode = Counter(partners[s]).most_common(1)[0]
        r = np.mean(real[s]); nl = np.array(null[s])
        # per-sign p: share of null shuffles' 10-split means <= real mean (bootstrap of means)
        nm = np.array([np.mean(rng.choice(nl, len(real[s]))) for _ in range(2000)])
        per[s] = {'n': int(nall[s]), 'partner': mode[0], 'mode': mode[1], 'rank_real': round(float(r), 3),
                  'rank_null': round(float(nl.mean()), 3), 'p': float((nm <= r).mean()),
                  'top10': int(sum(x < 0.1 for x in real[s])), 'func': PC_FUNC.get(mode[0], ''),
                  'func_rank_among_named': round(float(np.mean(fr[s])), 3) if fr[s] else None}
    out = {'label': label, 'dir_hits': [int(sum(dir_hits)), len(dir_hits)],
           'dir_null_expected': round(float(np.sum(dir_null)), 1), 'per_sign': per,
           'mean_rank_real': round(float(np.mean([v for x in real.values() for v in x])), 3),
           'mean_rank_null': round(float(np.mean([v for x in null.values() for v in x])), 3),
           'n_signs_p01': sum(1 for v in per.values() if v['p'] < 0.01),
           'n_signs_p05': sum(1 for v in per.values() if v['p'] < 0.05)}
    return out


def main():
    PE = load_pe(); PC = load_pc()
    Ppc, npc = profiles(PC)
    res = {'main': run(PE, Ppc, npc, label='all PC, K=200')}
    PC3 = [t for t in PC if t['period'].startswith('Uruk III')]
    P3, n3 = profiles(PC3)
    res['uruk3'] = run(PE, P3, n3, nshuf=10, label='Uruk III only, K=200')
    res['k100'] = run(PE, Ppc, npc, kpc=100, nshuf=10, label='all PC, K=100')
    json.dump(res, open(os.path.join(DATA, 'pe2_cycle3.json'), 'w'), indent=1)
    for k, v in res.items():
        print(k, {a: b for a, b in v.items() if a != 'per_sign'})
        good = sorted(v['per_sign'].items(), key=lambda x: x[1]['p'])
        for s, d in good[:12]:
            print('  ', s, d)
        for s in LEAD:
            if s in v['per_sign'] and s not in dict(good[:12]):
                print('  *', s, v['per_sign'][s])
        fs = [(s, d) for s, d in v['per_sign'].items() if d['func'] and d['mode'] >= 5]
        print('  known-function partners (mode>=5):', fs)


if __name__ == '__main__':
    main()
