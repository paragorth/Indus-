"""pe20 cycle 3: make the flock predict numbers it has not seen.

  W  : wether-aware model (adds W, non-breeding adults, mean omega*F, omega 0.05-3).
       Does it repair the Ur III control (u8 = F, udu-nita2 = M/W)? Does it change PE?
  LOO: leave-one-record-out imputation. For each herd record, the assignment is
       re-fitted on the other records (annealed Gibbs), rate posteriors come from the
       other records, and each observed cell of the left-out record is predicted from
       the record's other cells: p(v | rest) ~ g(v) * exp(LLR(record with v)).
       Gain = mean log2 p_bio - log2 g (bits per cell). Nulls: within-sign shuffles;
       controls: planted herds, Ur III Girsu/Umma subsamples of 10 records.
  PRED: predictions for the broken cells of MDP 17,096+325+380 (checkable by collation).
Usage: python3 pe20_cycle3.py W | LOO | PRED
"""
import sys
import pe20_common as P
from pe20_common import (np, json, os, DATA, CK, PE_SIGNS, UR_SIGNS, MAIN, Scorer, pe_records,
                         to_matrix, ur_herd_records, maximise, shuffle_within_sign, planted, dump,
                         lnb, EPS, K_GRID, F_GRID, R_GRID, logsumexp)

VMAX = 300


def cond_pred(V, r, j, a, lg):
    """log p(V[r,j] | rest of record r, other records) under assignment a, and log g."""
    v0 = V[r, j]
    others = np.delete(V, r, axis=0)
    base = Scorer(others, lg=lg)
    vals = np.arange(VMAX + 1, dtype=float)
    # LLR of the left-out record for each candidate value, with parameter posteriors
    # from the other records: log sum_theta w(theta) p(rec | theta) / g
    rows = np.repeat(V[r:r + 1], len(vals), 0)
    rows[:, j] = vals
    tot = np.zeros(len(vals))
    for s in (0, 1):
        slot = {}
        for jj, st in enumerate(a):
            if st and (st - 1) // P.NC == s:
                slot[P.CATS[(st - 1) % P.NC]] = jj
        if 'F' not in slot:
            continue
        i = slot['F']
        Fv = rows[:, i][:, None]

        def add(child, prior, grid, kgrid, half=False):
            nonlocal tot
            x = rows[:, child][:, None]
            ok = ~np.isnan(rows[:, child]) & ~np.isnan(rows[:, i])
            m = grid[None, :] * np.nan_to_num(Fv) / (2 if half else 1) + EPS
            ll = lnb(np.nan_to_num(x), m, kgrid[None, :]) - lg(np.nan_to_num(x))[:, None] * 0
            post = prior - logsumexp(prior)
            contrib = logsumexp(ll + post[None, :], axis=1)
            tot = tot + np.where(ok, contrib - lg(np.nan_to_num(rows[:, child])), 0.0)
        if 'M' in slot:
            add(slot['M'], base.Mp[i, slot['M']], base.rg, base.rk)
        if 'W' in slot:
            wg = np.repeat(np.geomspace(0.05, 3.0, 10), len(K_GRID))
            add(slot['W'], base.Wp[i, slot['W']], wg, np.tile(K_GRID, 10))
        if 'Y' in slot:
            add(slot['Y'], base.Yp[i, slot['Y']], base.fg, base.fk)
        for c in ('YF', 'YM'):
            if c in slot:
                # sex-specific young treated as half the young (pair term approximated)
                add(slot[c], base.Yh[i, slot[c]], base.fg, base.fk, half=True)
    logp = lg(vals) + tot
    logp -= logsumexp(logp)
    return float(logp[int(v0)]), float(lg(np.array([v0]))[0]), logp


def loo(V, rng, lg=None):
    lg = lg or Scorer(V).lg
    gains = []
    for r in range(V.shape[0]):
        rest = np.delete(V, r, axis=0)
        v, a = maximise(Scorer(rest, lg=lg), rng, restarts=4)
        for j in range(V.shape[1]):
            if np.isnan(V[r, j]) or V[r, j] > VMAX or a[j] == 0:
                continue
            lp, lgv, _ = cond_pred(V, r, j, a, lg)
            gains.append((lp - lgv) / np.log(2))
    return float(np.mean(gains)) if gains else 0.0, len(gains)


def part_W():
    rng = np.random.default_rng(31)
    P.set_cats(['F', 'M', 'W', 'YF', 'YM', 'Y', 'T'])
    out = {}
    U = ur_herd_records()
    Vu = to_matrix(U, UR_SIGNS)
    su = Scorer(Vu)
    v, a = maximise(su, rng, restarts=4)
    inv = {P.state_name(k): k for k in range(P.NSTATE)}
    tr = lambda names: su.score([inv[n] for n in names])
    out['ur3_full'] = {'max': v, 'assign': dict(zip(UR_SIGNS, [P.state_name(x) for x in a])),
                       'truth_M': tr(['F0', 'M0', 'YM0', 'YF0', 'F1', 'M1', 'YM1', 'X']),
                       'truth_W': tr(['F0', 'W0', 'YM0', 'YF0', 'F1', 'W1', 'YM1', 'X']),
                       'truth_sheep_only_W': tr(['F0', 'W0', 'YM0', 'YF0', 'X', 'X', 'X', 'X']),
                       'udu_as_F_sheep_only': tr(['W0', 'F0', 'YM0', 'YF0', 'X', 'X', 'X', 'X'])}
    print('ur3', out['ur3_full'], flush=True)
    rich = [i for i in range(len(U)) if np.sum(~np.isnan(Vu[i])) >= 4]
    subs = []
    for rep in range(20):
        idx = rng.choice(rich, 10, replace=False)
        v, a = maximise(Scorer(Vu[idx]), rng)
        names = [P.state_name(x) for x in a]
        subs.append(names)
    out['ur3_sub_u8_is_F'] = float(np.mean([n[0].startswith('F') for n in subs]))
    out['ur3_sub_udu_is_F'] = float(np.mean([n[1].startswith('F') for n in subs]))
    out['ur3_sub_young_ok'] = float(np.mean([n[2].startswith('Y') + n[3].startswith('Y') for n in subs]) / 2)
    out['ur3_sub_ud5_is_F'] = float(np.mean([n[4].startswith('F') for n in subs]))
    main = [r for r in pe_records() if r[0] == MAIN]
    Vm = to_matrix(main, PE_SIGNS)
    sm = Scorer(Vm)
    best = (-1e9, None)
    marg = np.zeros((8, P.NSTATE))
    for ch in range(4):
        v, a = maximise(sm, rng)
        m, ab, vb = P.gibbs(sm, 2000, rng, init=a)
        marg += m / 4
        if vb > best[0]:
            best = (vb, ab)
    out['pe_max'] = best[0]
    out['pe_assign'] = dict(zip(PE_SIGNS, [P.state_name(x) for x in best[1]]))
    out['pe_marg'] = {s: dict(zip(['X'] + P.CATS, np.round(m, 3).tolist()))
                      for s, m in zip(PE_SIGNS, P.collapse_marg(marg))}
    nul = [maximise(Scorer(shuffle_within_sign(Vm, rng), lg=sm.lg), rng, restarts=3)[0] for _ in range(100)]
    out['pe_null_within_max'] = float(np.max(nul))
    out['pe_p_within'] = float((1 + np.sum(np.array(nul) >= best[0])) / 101)
    print(json.dumps(out, indent=0, default=float), flush=True)
    dump(out, os.path.join(DATA, 'pe20_cycle3_W.json'))


def part_LOO():
    rng = np.random.default_rng(32)
    main = [r for r in pe_records() if r[0] == MAIN]
    Vm = to_matrix(main, PE_SIGNS)
    lg = Scorer(Vm).lg
    out = {}
    out['pe'] = loo(Vm, rng, lg)
    print('pe', out['pe'], flush=True)
    out['pe_null_within'] = [loo(shuffle_within_sign(Vm, rng), rng, lg)[0] for _ in range(20)]
    print('null', out['pe_null_within'], flush=True)
    mask = ~np.isnan(Vm)
    out['planted'] = [loo(planted(mask, rng), rng)[0] for _ in range(8)]
    print('planted', out['planted'], flush=True)
    U = ur_herd_records()
    Vu = to_matrix(U, UR_SIGNS)
    rich = [i for i in range(len(U)) if np.sum(~np.isnan(Vu[i])) >= 4]
    ur, urn = [], []
    for rep in range(8):
        idx = rng.choice(rich, 10, replace=False)
        ur.append(loo(Vu[idx], rng)[0])
        urn.append(loo(shuffle_within_sign(Vu[idx], rng), rng)[0])
    out['ur3'] = ur
    out['ur3_null_within'] = urn
    out['p_pe_vs_within'] = float((1 + sum(x >= out['pe'][0] for x in out['pe_null_within'])) / 21)
    print(json.dumps(out, default=float), flush=True)
    dump(out, os.path.join(DATA, 'pe20_cycle3_LOO.json'))


def part_PRED():
    """Predict the broken cells of 17,096+ under the cycle-1 MAP."""
    c1 = json.load(open(os.path.join(DATA, 'pe20_cycle1_pe.json')))
    inv = {P.state_name(k): k for k in range(P.NSTATE)}
    a = [inv[c1['max_assign'][s]] for s in PE_SIGNS]
    main = [r for r in pe_records() if r[0] == MAIN]
    Vm = to_matrix(main, PE_SIGNS)
    lg = Scorer(Vm).lg
    preds = []
    for r, rec in enumerate(main):
        for j, s in enumerate(PE_SIGNS):
            if not np.isnan(Vm[r, j]) or a[j] == 0:
                continue
            W = Vm.copy()
            W[r, j] = 0.0
            _, _, logp = cond_pred(W, r, j, a, lg)
            p = np.exp(logp)
            cdf = np.cumsum(p)
            gq = np.exp(lg(np.arange(VMAX + 1.0)))
            gq /= gq.sum()
            gc = np.cumsum(gq)
            preds.append({'block': rec[1], 'sign': s, 'role': c1['max_assign'][s],
                          'median': int(np.searchsorted(cdf, 0.5)),
                          'p10_p90': [int(np.searchsorted(cdf, 0.1)), int(np.searchsorted(cdf, 0.9))],
                          'generic_p10_p90': [int(np.searchsorted(gc, 0.1)), int(np.searchsorted(gc, 0.9))],
                          'other_cells': {PE_SIGNS[k]: (None if np.isnan(Vm[r, k]) else int(Vm[r, k]))
                                          for k in range(8)}})
            print(preds[-1], flush=True)
    dump({'map': c1['max_assign'], 'predictions': preds}, os.path.join(DATA, 'pe20_cycle3_PRED.json'))


def split_prob(V, rng, lg, sweeps=800):
    sc = Scorer(V, lg=lg)
    v, a = maximise(sc, rng, restarts=3)
    hits, n = 0, 0
    for sw in range(sweeps):
        _, a, _ = P.gibbs(sc, 1, rng, init=a, burn=0)
        if sw >= sweeps // 4 and sw % 4 == 0:
            sp = [(-1 if x == 0 else (x - 1) // P.NC) for x in a]
            bs = {sp[i] for i in range(4) if sp[i] >= 0}
            ts = {sp[i] for i in range(4, 8) if sp[i] >= 0}
            hits += (len(bs) == 1 and len(ts) == 1 and bs != ts)
            n += 1
    return hits / n


def part_SPLIT():
    """Is the blind base / ~a grouping (P 0.92 in cycle 2) a property of the coupling?"""
    rng = np.random.default_rng(33)
    main = [r for r in pe_records() if r[0] == MAIN]
    Vm = to_matrix(main, PE_SIGNS)
    lg = Scorer(Vm).lg
    out = {'real': split_prob(Vm, rng, lg)}
    out['within'] = [split_prob(shuffle_within_sign(Vm, rng), rng, lg) for _ in range(20)]
    out['all'] = [split_prob(P.shuffle_all(Vm, rng), rng, lg) for _ in range(20)]
    # column-permuted control: relabel which signs carry ~a (random 4/4 split of the 8 columns)
    perm = []
    for _ in range(20):
        pm = rng.permutation(8)
        perm.append(split_prob(Vm[:, pm], rng, lg))
    out['random_split_of_columns'] = perm
    out['p_within'] = float((1 + sum(x >= out['real'] for x in out['within'])) / 21)
    out['p_all'] = float((1 + sum(x >= out['real'] for x in out['all'])) / 21)
    out['p_colperm'] = float((1 + sum(x >= out['real'] for x in perm)) / 21)
    print(json.dumps(out, default=float), flush=True)
    dump(out, os.path.join(DATA, 'pe20_cycle3_SPLIT.json'))


if __name__ == '__main__':
    {'W': part_W, 'LOO': part_LOO, 'PRED': part_PRED, 'SPLIT': part_SPLIT}[sys.argv[1]]()
