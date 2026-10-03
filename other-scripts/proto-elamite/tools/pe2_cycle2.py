"""PE2 cycle 2: held-out commodity prediction and pair stability.

For 10 random tablet splits of PE (A = training half, B = held-out half):
 (i)  stability: pair top-60 PE signs into top-200 PC signs using A's profiles
      and, separately, B's profiles (all features). Share of PE signs with the
      same PC partner in both halves. Reference: same statistic for two halves
      of PC itself (planted-style).
 (ii) held-out commodity test: pair on A WITHOUT own-system features
      (position, tablet context, header, reverse, magnitude), then ask whether
      the partner's numeral-system profile (PC) predicts the PE sign's system
      profile on held-out B: rank of the partner among all 200 PC signs by
      cosine(sys_PE_B, sys_PC). Null: rank of a random PC sign (uniform, mean 0.5).
      A second null re-runs the pairing with PE-A systems shuffled within tablet.
 (iii) known-function partners: for partners listed in PC_FUNC, direction test:
      is the PE sign's held-out capacity share above the PE baseline exactly when
      the partner's capacity share is above the PC baseline?
"""
import json, random
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from pe2_common import *

KPE, KPC = 60, 200
NOSYS = [f for f in FEATS if not f.startswith('sys_')]
SYSF = ['sys_' + s for s in SYS]


def pair(Pa, na, Pb, nb, feats, ka=KPE, kb=KPC, restrict_a=None):
    sa = restrict_a if restrict_a else top_signs(Pa, na, ka)
    sb = top_signs(Pb, nb, kb)
    S = cos_sim(matrix(Pa, sa, feats), matrix(Pb, sb, feats))
    r, c = linear_sum_assignment(-S)
    return {sa[i]: sb[j] for i, j in zip(r, c)}, sb


def main():
    PE = load_pe(); PC = load_pc()
    Ppc, npc = profiles(PC)
    Pall, nall = profiles(PE)
    core = top_signs(Pall, nall, KPE)
    pcs = top_signs(Ppc, npc, KPC)
    Zpc_sys = matrix(Ppc, pcs, SYSF)
    pc_idx = {s: i for i, s in enumerate(pcs)}
    cap_pc_base = np.mean([Ppc[s]['sys_C'] for s in pcs])
    stab = defaultdict(int); ranks = defaultdict(list); ranks_null = []; dirs = defaultdict(list)
    partners = defaultdict(list); stab_share = []; pcstab = []
    for seed in range(10):
        A, B = split_tablets(PE, seed)
        Pa, na = profiles(A, min_n=5); Pb, nb = profiles(B, min_n=5)
        use = [s for s in core if s in Pa and s in Pb]
        # (i) stability, full features
        mA, _ = pair(Pa, na, Ppc, npc, FEATS, restrict_a=use)
        mB, _ = pair(Pb, nb, Ppc, npc, FEATS, restrict_a=use)
        same = [s for s in use if mA[s] == mB[s]]
        stab_share.append(len(same) / len(use))
        for s in same: stab[s] += 1
        # PC self-stability reference
        X, Y = split_tablets(PC, seed)
        Px, nx = profiles(X); Py, ny = profiles(Y)
        ux = [s for s in top_signs(Ppc, npc, KPE) if s in Px and s in Py]
        m1, _ = pair(Px, nx, Ppc, npc, FEATS, restrict_a=ux)
        m2, _ = pair(Py, ny, Ppc, npc, FEATS, restrict_a=ux)
        pcstab.append(sum(m1[s] == m2[s] for s in ux) / len(ux))
        # (ii) held-out: pair on A without sys, predict B's sys
        mA2, _ = pair(Pa, na, Ppc, npc, NOSYS, restrict_a=use)
        Zb = matrix(Pb, use, SYSF)
        sims = cos_sim(Zb, Zpc_sys)
        cap_pe_base = np.mean([Pb[s]['sys_C'] for s in use])
        for i, s in enumerate(use):
            p = mA2[s]; partners[s].append(p)
            j = pc_idx[p]
            rk = float((sims[i] > sims[i, j]).mean())   # 0 = best
            ranks[s].append(rk)
            if p in PC_FUNC:
                dirs[s].append(((Pb[s]['sys_C'] > cap_pe_base) == (Ppc[p]['sys_C'] > cap_pc_base), p))
        # null: PE-A systems shuffled within tablet before pairing (ctx then also shuffled)
        rr = random.Random(500 + seed); ov = {}
        for ti, t in enumerate(A):
            ents = [li for li, l in enumerate(t['lines']) if l['signs'] and l['nums']]
            sy = [system_of(t['lines'][li]['nums']) for li in ents]; rr.shuffle(sy)
            ov.update({(ti, li): x for li, x in zip(ents, sy)})
        P0, n0 = profiles(A, min_n=5, sys_override=ov)
        u0 = [s for s in use if s in P0]
        m0, _ = pair(P0, n0, Ppc, npc, NOSYS, restrict_a=u0)
        for i, s in enumerate(use):
            if s in m0:
                ranks_null.append(float((sims[i] > sims[i, pc_idx[m0[s]]]).mean()))
    allr = [r for v in ranks.values() for r in v]
    res = {'stability_PE_mean': float(np.mean(stab_share)), 'stability_PC_selfhalves_mean': float(np.mean(pcstab)),
           'heldout_rank_mean': float(np.mean(allr)), 'heldout_rank_null_withintablet': float(np.mean(ranks_null)),
           'heldout_rank_random': 0.5, 'heldout_share_top10pct': float(np.mean([r < 0.1 for r in allr])),
           'heldout_share_top10pct_null': float(np.mean([r < 0.1 for r in ranks_null]))}
    persign = {}
    for s in core:
        if s not in ranks: continue
        pc_mode = max(set(partners[s]), key=partners[s].count)
        d = dirs.get(s, [])
        persign[s] = {'n': nall[s], 'stable_full_of10': stab[s], 'nosys_partner_mode': pc_mode,
                      'mode_count': partners[s].count(pc_mode), 'heldout_rank_mean': round(float(np.mean(ranks[s])), 3),
                      'top10_of10': int(sum(r < 0.1 for r in ranks[s])),
                      'func_partner_dir_hits': [int(sum(x for x, _ in d)), len(d)],
                      'func_partners': sorted(set(p for _, p in d))}
    res['per_sign'] = persign
    nd = [x for v in dirs.values() for x, _ in v]
    res['func_direction_hits'] = [int(sum(nd)), len(nd)]
    json.dump(res, open(os.path.join(DATA, 'pe2_cycle2.json'), 'w'), indent=1)
    print({k: v for k, v in res.items() if k != 'per_sign'})
    for s, v in sorted(persign.items(), key=lambda x: x[1]['heldout_rank_mean'])[:60]:
        print(s, v)


if __name__ == '__main__':
    main()
