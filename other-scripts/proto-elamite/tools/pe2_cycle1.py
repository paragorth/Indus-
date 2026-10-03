"""PE2 cycle 1: role profiles + assignment PE <-> proto-cuneiform, with controls.

Pairing: z-scored role profiles (position, numeral system, tablet context, header,
reverse, magnitude), cosine similarity, Hungarian assignment of the top-K PE signs
into the top-K' PC signs.
Controls:
 C1 'shuffle profiles within each system': every feature column permuted
    independently across signs inside each system (marginals kept, joint role
    structure destroyed). Real assignment score vs 200 such runs.
 C2 entry-level null: PE numeral groups shuffled between entries of the same
    tablet, profiles rebuilt.
 C3 planted: PC tablets split in two; one half relabelled and used as the
    'unknown' script (sub-sampled to PE size). Recovery = share of its top-60
    signs paired to their own true identity. Also a cross-site version
    (Uruk vs the other sites).
"""
import json, random, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from pe2_common import *

KPE, KPC = 60, 200


def assign(Pa, na, Pb, nb, ka, kb, feats):
    sa = top_signs(Pa, na, ka); sb = top_signs(Pb, nb, kb)
    A = matrix(Pa, sa, feats); B = matrix(Pb, sb, feats)
    S = cos_sim(A, B)
    r, c = linear_sum_assignment(-S)
    return sa, sb, S, r, c


def colshuffle(X, rng):
    Y = X.copy()
    for j in range(Y.shape[1]):
        rng.shuffle(Y[:, j])
    return Y


def planted(PC, seed, feats, sub=1585, cross=False):
    rng = random.Random(seed)
    if cross:
        A = [t for t in PC if t['site'].startswith('Uruk')]
        B = [t for t in PC if not t['site'].startswith('Uruk')]
    else:
        A, B = split_tablets(PC, seed)
    B = rng.sample(B, min(sub, len(B)))
    Pa, na = profiles(A); Pb, nb = profiles(B)
    sa, sb, S, r, c = assign(Pb, nb, Pa, na, KPE, KPC, feats)   # B (relabelled) is the 'unknown'
    hits = sum(1 for i, j in zip(r, c) if sa[i] == sb[j])
    inA = sum(1 for s in sa if s in sb)
    # nearest-neighbour recovery too (no one-to-one constraint)
    nn = sum(1 for i in range(len(sa)) if sb[int(np.argmax(S[i]))] == sa[i])
    return hits, nn, inA, len(sa)


def main():
    out = {}
    PE = load_pe(); PC = load_pc()
    Ppe, npe = profiles(PE); Ppc, npc = profiles(PC)
    rows = []
    for gname, feats in (('all', FEATS), ('no_pos', [f for f in FEATS if not f.startswith('pos')])):
        sa, sb, S, r, c = assign(Ppe, npe, Ppc, npc, KPE, KPC, feats)
        real = S[r, c].mean()
        rng = np.random.default_rng(1)
        A = matrix(Ppe, sa, feats); B = matrix(Ppc, sb, feats)
        null = []
        for _ in range(200):
            S0 = cos_sim(colshuffle(A, rng), colshuffle(B, rng))
            r0, c0 = linear_sum_assignment(-S0)
            null.append(S0[r0, c0].mean())
        null = np.array(null)
        # C2 entry-level null on PE
        null2 = []
        for k in range(30):
            rr = random.Random(100 + k); ov = {}
            for ti, t in enumerate(PE):
                ents = [li for li, l in enumerate(t['lines']) if l['signs'] and l['nums']]
                sy = [system_of(t['lines'][li]['nums']) for li in ents]
                rr.shuffle(sy)
                ov.update({(ti, li): s for li, s in zip(ents, sy)})
            P2, n2 = profiles(PE, sys_override=ov)
            # ctx/mag/pos unchanged except sys; rebuild and assign
            s2a, s2b, S2, r2, c2 = assign(P2, n2, Ppc, npc, KPE, KPC, feats)
            null2.append(S2[r2, c2].mean())
        null2 = np.array(null2)
        res = {'real': round(float(real), 4), 'C1_mean': round(float(null.mean()), 4), 'C1_max': round(float(null.max()), 4),
               'C1_p': float((null >= real).mean()), 'C2_mean': round(float(null2.mean()), 4),
               'C2_max': round(float(null2.max()), 4), 'C2_p': float((null2 >= real).mean())}
        # per-pair p: PE sign i's assigned sim vs its best sim under PC column shuffle
        perp = {}
        bests = np.zeros((200, len(sa)))
        for k in range(200):
            bests[k] = cos_sim(A, colshuffle(B, rng)).max(1)
        for i, j in zip(r, c):
            p = float((bests[:, i] >= S[i, j]).mean())
            perp[sa[i]] = {'pc': sb[j], 'sim': round(float(S[i, j]), 3), 'p': p, 'n_pe': npe[sa[i]], 'n_pc': npc[sb[j]],
                           'func': PC_FUNC.get(sb[j], '')}
        res['pairs'] = perp
        res['n_pairs_p01'] = sum(1 for v in perp.values() if v['p'] < 0.01)
        out[gname] = res
        print(gname, {k: v for k, v in res.items() if k != 'pairs'})
    # C3 planted
    pl = []
    for seed in range(5):
        pl.append(planted(PC, seed, FEATS))
    plx = planted(PC, 0, FEATS, cross=True)
    out['planted'] = {'random_half': pl, 'cross_site': plx,
                      'note': '(hungarian hits, nearest-neighbour hits, relabelled signs present in reference top-200, n)'}
    print('planted', pl, 'cross-site', plx)
    json.dump(out, open(os.path.join(DATA, 'pe2_cycle1.json'), 'w'), indent=1)
    good = sorted(out['all']['pairs'].items(), key=lambda x: x[1]['p'])
    for s, v in good[:25]:
        print(s, v)


if __name__ == '__main__':
    main()
