"""pe68 cycle 3.
3b  Can a fragment's predicted loss FIND its join partner?  For each of the 29 real fragments, every other
    PE tablet (1,571 non-joined + the true partner = the other fragments of the same tablet) is scored by
    the bits its lines cost under the fragment's frozen completion model (per scored item, size-neutral).
    Rank of the true partner: READ vs CACHE vs CORPUS (the corpus null ranks by typicality only) vs
    READ with shuffled roles.  Control: the same on geometry-matched pseudo-fragments of held-out PE,
    proto-cuneiform and Ur III tablets (partner = the hidden rest of the same tablet).
3c  Frozen predictions for broken, unjoined tablets: (i) total gaps (a written total minus the clean
    entries = the value of the lost entries), (ii) team-sum M288 (cycle 3a), (iii) top join candidates
    among the existing fragments (mutual top-20 pairs of fragmentary Susa tablets).
"""
import os, sys, json, math, random, re, time
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe68_lib import *  # noqa
from pe68_c2 import setup, real_cases, signflip, NSHUF
import pe59_lib as P

F = json.load(open(os.path.join(DATA, 'pe68_frozen_join_predictions.json')))
mR, mC = F['models']['READ'], F['models']['CACHE']


def per_item_bits(V, H, st, rates, m, rs_list, ridx=0):
    p = precompute({'V': V, 'H': H}, st, rs_list if ridx else rs_list[:1])
    s = score_model(p, m, st, rates, ridx)
    n = (p['h_cap'] + p['h_cnt']) + len(p['u']) + len(p['hx']) + len(p['hh'])
    return (s['total'] / n) if n else None


def rank_test(cases, pool, st, rates, rs_list, rng, max_pool=600, nshuf=5):
    out = defaultdict(list)
    for c in cases:
        cand = [t['lines'] for t in rng.sample(pool, min(max_pool, len(pool))) if t['id'] != c['id']]
        allc = [c['H']] + cand
        for nm, m, ridx in [('READ', mR, 0), ('CACHE', mC, 0), ('CORPUS', CORPUS_NULL, 0)] + \
                [('SHUF', mR, k) for k in range(1, nshuf + 1)]:
            sc = [per_item_bits(c['V'], H, st, rates, m, rs_list, ridx) for H in allc]
            if sc[0] is None:
                continue
            v = np.array([x if x is not None else 1e9 for x in sc])
            pct = float((v[1:] < v[0]).mean())          # share of decoys that fit better (0 = partner ranked first)
            out[nm].append(pct)
    return {k: np.array(v) for k, v in out.items()}


def main_rank():
    res = {}
    rng = np.random.default_rng(seed('pe68-c3'))
    for name in ('PE', 'PC', 'U3'):
        pyr = random.Random(seed('pe68-setup') if name == 'PE' else seed('pe68-setup-' + name))
        T, tr, ho, st, rs_list, ctr, cte = setup(name, pyr)
        rates = sys_rates([precompute(c, st, rs_list[:1]) for c in ctr])
        pool = T
        sets = {'pseudo': [c for c in cte if len(c['H']) >= 2][:150]}
        if name == 'PE':
            sets['real'] = [c for c in real_cases() if c['H']]
            pool = corpus('PE')
            pool = [t for t in pool if t['id'] not in JOINED_IDS]
        for sname, cases in sets.items():
            r = rank_test(cases, pool, st, rates, rs_list, random.Random(seed('rk' + name + sname)))
            R = {k: {'mean_pct': float(v.mean()), 'top1pct': float((v <= 0.01).mean()), 'n': len(v)}
                 for k, v in r.items()}
            for k in ('CACHE', 'CORPUS'):
                n = min(len(r['READ']), len(r[k]))
                d, p = signflip(r[k][:n] - r['READ'][:n], rng)
                R['READ_vs_' + k] = {'d': d, 'p': p}
            res['%s_%s' % (name, sname)] = R
            print(name, sname, json.dumps(R), flush=True)
    json.dump(res, open(os.path.join(CK, 'c3_rank.json'), 'w'), indent=1)


# ------------------------------------------------------------------ 3c frozen predictions for broken tablets
def atf_breaks():
    """P-number -> set of break markers from the raw ATF."""
    br = defaultdict(set)
    cur = None
    for line in open(os.path.join(DATA, 'pe_raw.atf'), errors='replace'):
        if line.startswith('&P'):
            cur = line[1:8]
        elif cur and line.startswith('$'):
            s = line.lower()
            for k in ('beginning broken', 'rest broken', 'lines broken', 'broken'):
                if k in s:
                    br[cur].add(k)
                    break
        elif cur and '[...]' in line:
            br[cur].add('[...]')
    return br


def main_freeze():
    cap, cnt = P.pe_maps()
    cm = cnt['sex2']
    raw = {t['id']: t for t in P.build_pe()}
    U = {t['id']: t for t in corpus('PE')}
    br = atf_breaks()
    totals, teams = [], []
    for pid, t in raw.items():
        if pid in JOINED_IDS or not br.get(pid):
            continue
        L = t['lines']
        # (i) total gap
        T_ = [l for l in L if l['role'] == 'T']
        E = [l for l in L if l['role'] == 'E']
        if T_ and T_[0]['numclean'] and E:
            m = cap if P.tablet_type(t) == 'CAPT' else cm
            tv = P.value(T_[0]['nums'], m)
            vis = [P.value(l['nums'], m) for l in E if l['numclean']]
            nb = sum(1 for l in E if not l['numclean'])
            lost_lines = bool(br[pid] & {'beginning broken', 'rest broken', 'lines broken'})
            if tv is not None and all(v is not None for v in vis) and (nb or lost_lines):
                gap = tv - sum(vis)
                totals.append({'tablet': pid, 'system': 'CAP (N39C units)' if m is cap else 'count (sex2)',
                               'total': str(tv), 'visible_sum': str(sum(vis)), 'predicted_lost_sum': str(gap),
                               'n_broken_entries': nb, 'lines_lost': lost_lines,
                               'status': 'consistent' if gap >= 0 else 'IMPOSSIBLE (visible entries exceed total)'})
        # (ii) team-sum M288: broken M288 value predicted from a fully visible run, or a broken count from a visible M288
        acc, okc, run = 0, True, []
        for i, l in enumerate(L):
            if l['role'] != 'E':
                continue
            if l['signs'] and l['signs'][-1] == 'M288':
                if run:
                    nbk = [j for j in run if not L[j]['numclean']]
                    if not l['numclean'] and not nbk and acc:
                        teams.append({'tablet': pid, 'line': i, 'kind': 'lost M288 value',
                                      'predicted_N39C': str(60 * acc), 'run_sum': str(acc)})
                    elif l['numclean'] and len(nbk) == 1:
                        mv = P.value(l['nums'], cap)
                        if mv is not None:
                            need = Fr(mv) / 60 - acc
                            teams.append({'tablet': pid, 'line': nbk[0], 'kind': 'lost count in team',
                                          'predicted_count': str(need), 'M288_line': i, 'M288_N39C': str(mv), 'visible_run_sum': str(acc),
                                          'status': 'consistent' if (need > 0 and need.denominator == 1 and need <= 30) else 'rule fails here (not a small whole count)'})
                acc, run = Fr(0), []
                continue
            run.append(i)
            if l['numclean'] and P.ncls(l['nums']) == 'AMB':
                acc += P.value(l['nums'], cm)
            elif l['numclean']:
                run = []
                acc = Fr(0)
    # (iii) join candidates among fragmentary Susa tablets (READ model, symmetric per-item bits, mutual top-20)
    pyr = random.Random(seed('pe68-setup'))
    T, tr, ho, st, rs_list, ctr, cte = setup('PE', pyr)
    rates = sys_rates([precompute(c, st, rs_list[:1]) for c in ctr])
    frag = [t for t in U.values() if t['id'] not in JOINED_IDS and t['site'].startswith('Susa')
            and br.get(t['id'], set()) & {'beginning broken', 'rest broken', 'lines broken'}
            and sum(1 for l in t['lines'] if l['cls']) >= 2]
    n = len(frag)
    S = np.full((n, n), np.inf)
    t0 = time.time()
    for a in range(n):
        for b in range(n):
            if a != b:
                x = per_item_bits(frag[a]['lines'], frag[b]['lines'], st, rates, mR, rs_list)
                if x is not None:
                    S[a, b] = x
    Z = (S - np.nanmean(np.where(np.isinf(S), np.nan, S), 0)) / (np.nanstd(np.where(np.isinf(S), np.nan, S), 0) + 1e-9)
    sym = Z + Z.T
    rk = np.argsort(np.argsort(sym, 1), 1)
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            if rk[a, b] < 20 and rk[b, a] < 20 and np.isfinite(sym[a, b]):
                pairs.append((float(sym[a, b]), frag[a]['id'], frag[b]['id']))
    pairs.sort()
    print('fragments', n, 'mutual pairs', len(pairs), '%.0fs' % (time.time() - t0))
    out = {'made': time.strftime('%Y-%m-%d %H:%M'), 'model': 'pe68 READ (frozen in pe68_frozen_join_predictions.json, '
           'sha %s)' % F['sha256'][:16],
           'total_gaps': totals, 'team_sum_M288': teams,
           'join_candidates_top40': [{'z': round(z, 3), 'a': a, 'b': b} for z, a, b in pairs[:40]],
           'kill_lines': {
               'total_gaps': 'a new join or collation that restores the lost entries of a tablet listed as consistent, '
                             'with a sum different from predicted_lost_sum, kills the total rule for that tablet; '
                             'if >= 3 of the first 5 restored tablets miss, the rule (pe59 A: a single reverse numeric '
                             'line is the sum of the entries) is wrong as stated',
               'team_sum': 'a restored M288 value or team count that differs from the prediction in >= 3 of the first 5 '
                           'restorations kills the team-sum form (C+ from cycle 3a)',
               'join_candidates': 'if physical checking of the top 40 pairs finds no join, the method has no join-finding '
                                  'power (expected: chance level, see c3 rank test)'}}
    h = sha(out)
    out['sha256'] = h
    json.dump(out, open(os.path.join(DATA, 'pe68_frozen_broken_predictions.json'), 'w'), indent=1)
    print('totals', len(totals), Counter(x['status'] for x in totals), 'teams', len(teams),
          Counter(x['kind'] for x in teams), 'sha', h[:16])


from fractions import Fraction as Fr  # noqa: E402

if __name__ == '__main__':
    {'rank': main_rank, 'freeze': main_freeze}[sys.argv[1]]()
