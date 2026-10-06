"""pe69 cycle 2: use what survives of the rule as a decoder.

2a decoded table: every M288 slot on a non-joined tablet with its run, implied per-head rations (M288 / team sum,
   M288 / last line) and the rule that explains it (TEAM60 / LAST60 / LAST120 / CONST (same value as other slots of
   the tablet) / none). -> data/pe69_decoded_slots.tsv
2b does the M288 value scale with the team or with the last line?  log M288 ~ b_sum log(sum) + b_last log(last) on
   runs with >= 2 count lines and sum != last; tablet bootstrap. Controls: the same corpus with M288 replaced by
   60 x sum (planted team) or 60 x last (planted per-line) on the same slots, plus noise slots.
   Within tablets (>= 3 clean slots): rank correlation of M288 with sum and with last vs within-tablet permutation.
2c what distinguishes slots the per-line / team rules fail on: excess hits per feature level vs the analytic
   re-deal expectation (a feature that only makes chance hits likelier shows no excess).
2d other multipliers among failing slots: ratios M288 / last recurring on >= 3 tablets vs the re-deal null.
2e pe62 check: M056-final count lines directly followed by an M288 line (pe62 kill line: 'an M056 line plus a
   separate M288 allotment for the same unit').
"""
import os, sys, json, math, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe69_lib import *  # noqa

NREP = int(os.environ.get('NREP', 1000))


def classify(m, s, last, const):
    if not np.isfinite(m):
        return 'BROKEN_M288'
    lab = []
    if np.isfinite(last) and abs(m - 60 * last) < 1e-9:
        lab.append('LAST60')
    if np.isfinite(last) and abs(m - 120 * last) < 1e-9:
        lab.append('LAST120')
    if np.isfinite(s) and abs(m - 60 * s) < 1e-9:
        lab.append('TEAM60')
    if not lab and const:
        lab.append('CONST')
    return '+'.join(lab) or 'NONE'


def main():
    T = tablets()
    S = slots(T)
    m, A = feats(S)
    nc = np.array([nclean(s) for s in S])
    tab = np.array([s['id'] for s in S])
    last, ssum = A['LAST'], A['SUM_BEFORE']
    res = {}
    # ---------- 2a
    vals_by_tab = defaultdict(list)
    for i, s in enumerate(S):
        if np.isfinite(m[i]):
            vals_by_tab[s['id']].append(m[i])
    rows, lab = [], []
    for i, s in enumerate(S):
        const = np.isfinite(m[i]) and sum(1 for v in vals_by_tab[s['id']] if v == m[i]) >= 2
        c = classify(m[i], ssum[i], last[i], const)
        lab.append(c)
        lf = s['before'][-1]['fin'] if s['before'] else None
        rows.append([s['id'], s['line'], int(nc[i]), '' if not np.isfinite(ssum[i]) else int(ssum[i]),
                     '' if not np.isfinite(last[i]) else int(last[i]), lf or '-', ' '.join(s['m_signs']),
                     '' if not np.isfinite(m[i]) else int(m[i]),
                     '' if not (np.isfinite(m[i]) and np.isfinite(ssum[i]) and ssum[i]) else round(m[i] / ssum[i], 2),
                     '' if not (np.isfinite(m[i]) and np.isfinite(last[i]) and last[i]) else round(m[i] / last[i], 2), c])
    with open(os.path.join(DATA, 'pe69_decoded_slots.tsv'), 'w') as f:
        f.write('# pe69 decoded M288 slots (non-joined tablets). values: count lines in the pe59 sex2 count map; M288 in '
                'N39C (pe59 capacity ladder). class: which rule reproduces the M288 value\n')
        f.write('\t'.join(['tablet', 'line', 'n_count_lines', 'team_sum', 'last_count', 'last_final_sign', 'm288_line_signs',
                           'm288_N39C', 'per_head_team', 'per_head_last', 'class']) + '\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
    lab = np.array(lab)
    clean = np.isfinite(m)
    cnt = Counter(lab[clean].tolist())
    res['2a'] = {'slots': len(S), 'clean': int(clean.sum()), 'tablets_with_clean_m288': len(vals_by_tab),
                 'classes': dict(cnt.most_common())}
    # per-head ration distribution where the run is a single count line
    one = clean & (nc == 1) & np.isfinite(last) & (last > 0)
    ph = Counter(np.round(m[one] / last[one], 2).tolist())
    res['2a']['per_head_single_line_top'] = ph.most_common(12)
    print('2a', json.dumps(res['2a']), flush=True)

    # ---------- 2b scaling
    D = clean & (nc >= 2) & np.isfinite(ssum) & np.isfinite(last) & (ssum != last) & (last > 0) & (m > 0)

    def fit(mv, idx):
        X = np.column_stack([np.ones(len(idx)), np.log(ssum[idx]), np.log(last[idx])])
        y = np.log(mv[idx])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return b[1], b[2]

    idx = np.where(D)[0]
    rng = np.random.default_rng(seed('pe69-c2'))
    tabs = sorted(set(tab[idx].tolist()))

    def boot(mv):
        bs = []
        for _ in range(NREP):
            pick = rng.choice(tabs, len(tabs), replace=True)
            ii = np.concatenate([idx[tab[idx] == t] for t in pick])
            try:
                bs.append(fit(mv, ii))
            except Exception:
                pass
        bs = np.array(bs)
        return {'b_sum': round(float(fit(mv, idx)[0]), 3), 'b_last': round(float(fit(mv, idx)[1]), 3),
                'b_sum_ci': [round(float(x), 3) for x in np.percentile(bs[:, 0], [2.5, 97.5])],
                'b_last_ci': [round(float(x), 3) for x in np.percentile(bs[:, 1], [2.5, 97.5])]}

    out = {'n_runs': int(D.sum()), 'n_tablets': len(tabs), 'real': boot(m)}
    # controls: planted team / per-line on a random 30% of the slots, rest kept as is
    for nm, mk in (('planted_team_30pct', lambda: 60 * ssum), ('planted_last_30pct', lambda: 60 * last)):
        mv = m.copy()
        sel = idx[rng.random(len(idx)) < 0.3]
        mv[sel] = mk()[sel]
        out[nm] = boot(mv)
    res['2b_regression'] = out
    print('2b', json.dumps(out), flush=True)
    # within tablet rank correlation
    from scipy.stats import spearmanr
    W = defaultdict(list)
    for i in np.where(clean & np.isfinite(ssum) & np.isfinite(last))[0]:
        W[tab[i]].append(i)
    W = {k: v for k, v in W.items() if len(v) >= 3}

    def wcorr(mv, x):
        rs = []
        for k, v in W.items():
            a, b = mv[v], x[v]
            if len(set(a.tolist())) > 1 and len(set(b.tolist())) > 1:
                rs.append(spearmanr(a, b)[0])
        return float(np.mean(rs)) if rs else float('nan'), len(rs)

    wc = {}
    for nm, x in (('sum', ssum), ('last', last)):
        r0, n0 = wcorr(m, x)
        nul = []
        for _ in range(NREP):
            mv = m.copy()
            for k, v in W.items():
                mv[v] = mv[rng.permutation(v)]
            nul.append(wcorr(mv, x)[0])
        nul = np.array(nul)
        wc[nm] = {'mean_rho': round(r0, 3), 'tablets': n0, 'null_mean': round(float(np.nanmean(nul)), 3),
                  'p': float(((nul >= r0).sum() + 1) / (NREP + 1))}
    const_tabs = sum(1 for k, v in W.items() if len(set(m[v].tolist())) == 1)
    wc['tablets_ge3_slots'] = len(W)
    wc['tablets_all_m288_equal'] = const_tabs
    # null for constancy: values re-dealt across all slots
    pool = m[clean]
    cn = []
    for _ in range(NREP):
        cn.append(sum(1 for k, v in W.items() if len(set(rng.choice(pool, len(v)).tolist())) == 1))
    wc['const_null_mean'] = float(np.mean(cn))
    res['2b_within'] = wc
    print('2b within', json.dumps(wc), flush=True)

    # ---------- 2c features of failing slots
    hit = np.array(['LAST60' in c or 'LAST120' in c or 'TEAM60' in c for c in lab])
    ok = clean & np.isfinite(last)
    vc = Counter(np.round(pool, 6).tolist())
    pe = np.array([(vc.get(round(60 * l, 6), 0) + vc.get(round(120 * l, 6), 0) +
                    (vc.get(round(60 * s, 6), 0) if np.isfinite(s) and s not in (l, 2 * l) else 0)) / len(pool)
                   if np.isfinite(l) else np.nan for l, s in zip(last, ssum)])
    broken_tab = {t['id'] for t in T if any(e['k'].startswith('BROKEN') for e in t['E'])}

    def lastcls(s):
        f = s['before'][-1]['fin'] if s['before'] else None
        if f is None:
            return 'bare number'
        if f == 'M056':
            return 'M056'
        if f in PERSON:
            return 'PERSON sign'
        return 'other sign'

    F = {'n_count_lines': [('1' if n == 1 else '2' if n == 2 else '3+' if n >= 3 else '0/mixed') for n in nc],
         'last_final': [lastcls(s) for s in S],
         'last_value': [('1' if l == 1 else '2-9' if l < 10 else '10+') if np.isfinite(l) else '-' for l in last],
         'M288_line': ['M288 alone' if s['m_signs'] == ['M288'] else 'signs + M288' for s in S],
         'M288_shape': ['has N39B/N24' if s['m_shape'] and set(s['m_shape']) & {'N39B', 'N24'} else 'N01/N14 only' for s in S],
         'tablet_damaged': ['damaged' if s['id'] in broken_tab else 'intact' for s in S],
         'first_slot': ['first' if s['first_on_tablet'] else 'later' for s in S],
         'surface': [s['before'][-1]['surf'] if s['before'] else '-' for s in S]}
    fc = {}
    for fn, vals in F.items():
        vals = np.array(vals)
        rr = {}
        for v in sorted(set(vals[ok].tolist())):
            q = ok & (vals == v)
            h = int(hit[q].sum())
            e = float(np.nansum(pe[q]))
            rr[v] = {'n': int(q.sum()), 'hits': h, 'expected': round(e, 2), 'hit_rate': round(h / q.sum(), 3),
                     'excess_rate': round((h - e) / q.sum(), 3)}
        fc[fn] = rr
    res['2c'] = fc
    for fn, rr in fc.items():
        print('2c', fn, json.dumps(rr), flush=True)

    # ---------- 2d other multipliers on failing slots
    fail = ok & ~hit & (last > 0)
    rt = defaultdict(set)
    for i in np.where(fail)[0]:
        rt[round(m[i] / last[i], 3)].add(tab[i])
    obs = {r: len(v) for r, v in rt.items() if len(v) >= 3}
    nulmax = []
    for _ in range(NREP):
        mv = redeal_m(m, rng)
        rr = defaultdict(set)
        for i in np.where(fail)[0]:
            q = round(mv[i] / last[i], 3)
            if q not in (60.0, 120.0) and abs(mv[i] - 60 * ssum[i]) > 1e-9:
                rr[q].add(tab[i])
        nulmax.append(max((len(v) for v in rr.values()), default=0))
    nulmax = np.array(nulmax)
    res['2d'] = {'n_failing': int(fail.sum()), 'ratios_on_ge3_tablets': sorted(obs.items(), key=lambda kv: -kv[1]),
                 'null_max_tablets_median': float(np.median(nulmax)), 'null_max_95': float(np.percentile(nulmax, 95)),
                 'p_by_ratio': {str(r): float(((nulmax >= n).sum() + 1) / (NREP + 1)) for r, n in obs.items()}}
    print('2d', json.dumps(res['2d']), flush=True)

    # ---------- 2e M056-final lines directly before an M288 line
    pairs = []
    for t in T:
        E = t['E']
        for j in range(len(E) - 1):
            if E[j]['fin'] == 'M056' and E[j]['k'] == 'CNT' and E[j + 1]['k'] == 'M288':
                pairs.append({'id': t['id'], 'm056_count': float(E[j]['v']), 'm288': float(E[j + 1]['v']),
                              'ratio': round(float(E[j + 1]['v'] / E[j]['v']), 3)})
    m056_all = sum(1 for t in T for e in t['E'] if e['fin'] == 'M056')
    rc = Counter(p['ratio'] for p in pairs)
    # null: same M056 counts, M288 values re-dealt from all clean M288 lines; statistic = most common ratio's tablets
    nm = []
    for _ in range(NREP):
        rr = Counter(round(float(rng.choice(pool)) / p['m056_count'], 3) for p in pairs)
        nm.append(max(rr.values()) if rr else 0)
    nm = np.array(nm)
    top = rc.most_common(1)[0] if rc else (None, 0)
    res['2e'] = {'m056_final_lines': m056_all, 'm056_count_then_m288': len(pairs), 'ratios': rc.most_common(),
                 'top_ratio_p': float(((nm >= top[1]).sum() + 1) / (NREP + 1)), 'pairs': pairs}
    print('2e', json.dumps({k: v for k, v in res['2e'].items() if k != 'pairs'}), flush=True)
    json.dump(res, open(os.path.join(CK, 'c2.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
