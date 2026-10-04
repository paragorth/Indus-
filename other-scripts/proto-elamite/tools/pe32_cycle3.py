"""pe32 cycle 3: DO ALLOTMENT SIZES SORT THE CLASS INTO SUB-CLASSES?  Events (counted sign, count x, amount y);
per-unit amount r = y / x.  Blind method, identical for Ur III and PE:
  D  = number of events whose r equals the leave-one-out modal r of their counted sign (signs with >= 2 events);
  null: r re-dealt among events (global) and within tablets; sub-classes = signs grouped by modal r.
Ur III control (CDLI '-ta' ration / wage lines; truth used only for scoring: gurusz, geme2, dumu must come out
as separate groups ordered gurusz > geme2 > dumu, at full size and at PE size (subsamples with as many events
and signs as the PE data).  Planted PE control: C14 split into two planted sub-classes at 120 and 30 N39C per
unit on 70% of their events.  Real PE: M288 entries after count lines (pe27/pe28 pairs + held-out), the C14
class alone and all counted signs; plus the pe16 companion pattern (a smaller capacity line right after a
standard M288 line) by counted sign."""
import os, sys, json, time
import numpy as np
from fractions import Fraction as Fr
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe32_common import *  # noqa
import pe32_ur3

NR = int(os.environ.get('NR', 1000))
t0 = time.time()


def loo_modes(ev):
    by = defaultdict(list)
    for i, e in enumerate(ev):
        by[e['s']].append(i)
    return by


def D_stat(ev, rs, by):
    d = 0
    for s, idx in by.items():
        if len(idx) < 2:
            continue
        c = Counter(rs[i] for i in idx)
        for i in idx:
            c[rs[i]] -= 1
            top = max(c.values())
            m = min(r for r, v in c.items() if v == top) if top > 0 else None
            if m is not None and rs[i] == m:
                d += 1
            c[rs[i]] += 1
    return d


def test(ev, rng, nr=NR, label=''):
    by = loo_modes(ev)
    rs = [e['r'] for e in ev]
    d = D_stat(ev, rs, by)
    nulls = {}
    tb = defaultdict(list)
    for i, e in enumerate(ev):
        tb[e['t']].append(i)
    for mode in ('global', 'tablet'):
        N = np.zeros(nr)
        for k in range(nr):
            rr = list(rs)
            if mode == 'global':
                p = rng.permutation(len(rs)); rr = [rs[j] for j in p]
            else:
                for idx in tb.values():
                    p = rng.permutation(len(idx))
                    for a, b in zip(idx, p):
                        rr[a] = rs[idx[b]]
            N[k] = D_stat(ev, rr, by)
        nulls[mode] = {'mean': round(float(N.mean()), 1), 'p': float((1 + (N >= d).sum()) / (1 + nr))}
    groups = defaultdict(list)
    for s, idx in by.items():
        if len(idx) >= 2:
            c = Counter(rs[i] for i in idx)
            top = max(c.values())
            m = min(r for r, v in c.items() if v == top)
            groups[str(m)].append((s, len(idx), round(top / len(idx), 2)))
    out = {'label': label, 'events': len(ev), 'signs2': sum(1 for v in by.values() if len(v) >= 2), 'D': d,
           'null': nulls, 'groups': {k: sorted(v, key=lambda a: -a[1]) for k, v in groups.items()}}
    return out


def mode_of(ev, s):
    c = Counter(e['r'] for e in ev if e['s'] == s)
    if not c:
        return None
    top = max(c.values())
    return min(r for r, v in c.items() if v == top)


def ur_truth_ok(ev):
    g, w, d = mode_of(ev, 'gurusz'), mode_of(ev, 'geme2'), mode_of(ev, 'dumu')
    if None in (g, w, d):
        return None
    return g > w > d


if __name__ == '__main__':
    rng = np.random.default_rng(33)
    res = {}
    # ---------------- Ur III
    U = [{'s': e['pfin'], 'r': Fr(e['r']), 't': e['tid']} for e in pe32_ur3.build()]
    res['ur3_full'] = test(U, rng, nr=200, label='UR3 full')
    res['ur3_full']['modes'] = {s: str(mode_of(U, s)) for s in ('gurusz', 'geme2', 'dumu')}
    res['ur3_full']['truth_order'] = ur_truth_ok(U)
    print('UR3 full', res['ur3_full']['D'], res['ur3_full']['null'], res['ur3_full']['modes'],
          res['ur3_full']['truth_order'], flush=True)
    # ---------------- PE events
    T = table()
    E = m288_events(T)
    P = [{'s': e['pfin'], 'r': Fr(e['y']) / e['x'], 't': e['tid'], 'x': e['x'], 'y': e['y'], 'seen': e['seen']}
         for e in E if e['x']]
    PC = [e for e in P if e['s'] in C14]
    n_pe, s_pe = len(PC), len({e['s'] for e in PC})
    # Ur III at PE size: sample tablets until the event count matches the class data, recipient words kept
    tabs = defaultdict(list)
    for e in U:
        tabs[e['t']].append(e)
    tl = sorted(tabs)
    ok, pv, sizes = [], [], []
    for k in range(int(os.environ.get('NSUB', 100))):
        rng.shuffle(tl)
        sub = []
        for t in tl:
            sub += tabs[t]
            if len(sub) >= n_pe:
                break
        r = test(sub, rng, nr=100)
        pv.append(r['null']['tablet']['p'])
        ok.append(ur_truth_ok(sub))
        sizes.append(len(sub))
    okv = [o for o in ok if o is not None]
    res['ur3_pesize'] = {'n_events': n_pe, 'frac_p05_tablet': float(np.mean(np.array(pv) <= 0.05)),
                         'truth_order_frac': float(np.mean(okv)) if okv else None, 'n_with_all3': len(okv)}
    print('UR3 at PE size', res['ur3_pesize'], flush=True)
    # ---------------- planted PE sub-classes
    A, B = set(C14[0::2]), set(C14[1::2])
    PP = []
    for e in PC:
        e2 = dict(e)
        if rng.random() < 0.7:
            e2['r'] = Fr(120) if e['s'] in A else Fr(30)
        PP.append(e2)
    r = test(PP, rng, label='PLANT')
    corr = sum(1 for k, v in r['groups'].items() for s, _, _ in v if (k == '120' and s in A) or (k == '30' and s in B))
    r['correct_signs'] = corr
    res['plant'] = r
    print('PLANT', r['D'], r['null'], 'correctly grouped signs', corr, 'of', r['signs2'], flush=True)
    # ---------------- real PE
    res['pe_C14'] = test(PC, rng, label='PE C14')
    res['pe_all'] = test(P, rng, label='PE all')
    for k in ('pe_C14', 'pe_all'):
        print(k, res[k]['events'], res[k]['signs2'], res[k]['D'], res[k]['null'], flush=True)
        for g, v in sorted(res[k]['groups'].items(), key=lambda a: -len(a[1]))[:8]:
            print('   ', g, v[:10])
    # per-unit amount profile of the class (x >= 2 lines are the only ones that show scaling)
    prof = defaultdict(Counter)
    for e in PC:
        prof[e['s']][('x1' if e['x'] == 1 else 'x2+') + ':' + str(e['r'])] += 1
    res['pe_C14_profile'] = {s: dict(c.most_common(6)) for s, c in prof.items()}
    # ---------------- pe16 companion: smaller capacity line right after a standard M288 line
    comp = Counter(); tot = Counter()
    for t in T:
        L = t['L']
        for k in range(1, len(L) - 1):
            l = L[k]
            if l['fin'] != 'M288' or l['cap'] is None:
                continue
            p = L[k - 1]
            if p['sys'] != 'CNT' or not p['cnt'] or l['cap'] != STD * p['cnt']:
                continue
            nx = L[k + 1]
            tot[p['fin']] += 1
            if nx['cap'] is not None and nx['sys'] == 'CAP' and nx['cap'] < l['cap']:
                comp[(p['fin'], nx['fin'], nx['cap'])] += 1
    res['companion'] = {'std_lines_by_sign': dict(tot), 'smaller_next': [(list(k), v) for k, v in comp.most_common()]}
    print('companion', res['companion'], flush=True)
    json.dump(res, open(os.path.join(CK, 'cycle3.json'), 'w'), indent=1, default=str)
    print('done', round(time.time() - t0))
