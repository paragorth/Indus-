"""S-DARK-9: TIME and DEPTH arrows. Any reading feature of a text vs depth (ft below datum; Spearman / Kruskal) and vs
fine period (ordinal), within Mohenjo-daro and Harappa separately, with permutation stratified by excavation area
(depth at Mohenjo-daro is mostly an area effect: DKG(S) mean 12.9 ft vs ~4.5 ft elsewhere). FDR (BH) over arrows per
city x outcome x level. Replication = q<0.1 in both cities with the same sign; 'local' = both q<0.1, opposite sign.
Usage: python3 tools/strat_dark_time.py CYCLE [nrandom] [control] [typefilter]
  CYCLE 1 named arrows; 2 random arrows (strat_dark reading kinds); 3 named arrows restricted to a type (SEAL / TAB);
  CYCLE 4 survivors in detail (per-period means, merge dependence)."""
import json, csv, re, sys, random, collections, math
import numpy as np
CYCLE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NRAND = int(sys.argv[2]) if len(sys.argv) > 2 else 300
CONTROL = 'control' in sys.argv
TYPEF = next((a[5:] for a in sys.argv if a.startswith('type=')), None)
NPERM = 2000
rng = random.Random(9); nrng = np.random.default_rng(9)

C = json.load(open('data/derived/merged-corpus-canonical.json'))
raw = {r['cisi']: r for r in csv.DictReader(open('data/raw/inscriptions.csv')) if r['cisi']}
bridge = json.load(open('data/derived/bridge_extended.json'))
M2W = collections.defaultdict(set)
for w, ms in bridge.items():
    for m in ms: M2W[m].add(int(w))
NUM = {1: 1, 3: 3, 4: 4, 5: 5, 16: 6, 17: 7, 18: 8, 31: 1, 32: 2, 33: 3, 34: 4}
def fam(s): return s // 100 if s >= 100 else 0
OPEN = {817, 861, 820}
CLOSER_M = {342, 162, 169, 15, 254, 12, 211}
CLOSERS = set().union(*(M2W[m] for m in CLOSER_M)) | {740, 390, 405, 520}
SUFFIX = set(M2W[176]) | set(M2W[1])
FINE = {'Period 2': 0, 'Period 3B-1': 1, 'Period 3B-2': 2, 'Period 3C-1': 4, 'Period 3C-2': 5, 'Period 3C-3': 6,
        'Period 3C-4': 7, 'Period 4': 8, 'Period 5A': 9}
def depth_ft(s):
    m = re.match(r'^-([\d.]+)\s*ft', s, re.I)
    if not m: return None
    try: return float(m.group(1).replace('..', '.'))
    except ValueError: return None
def fnum(x):
    try: v = float(x); return v if v > 0 else None
    except (TypeError, ValueError): return None

# middles over the whole corpus (all sites) per level, for uniqueness
def middle(s):
    s = list(s)
    if s and s[0] in OPEN: s = s[1:]
    while s and (s[-1] in SUFFIX or s[-1] in CLOSERS or s[-1] in NUM): s = s[:-1]
    return tuple(s)
MIDCOUNT = {lv: collections.Counter(middle(r[lv]) for r in C if r.get(lv)) for lv in ('seq_raw', 'seq_strong', 'seq_all')}

OBJ = []
for r in C:
    if not r.get('seq_raw') or len(r['seq_raw']) < 1: continue
    if r['site'] not in ('Mohenjo-daro', 'Harappa'): continue
    x = raw.get(r['cisi'], {})
    t = r['type'].split(':')[0]
    if TYPEF and t != TYPEF: continue
    OBJ.append(dict(site=r['site'], area=r.get('area-section') or '--', type=r['type'], t0=t,
                    emblem=(r.get('symbol') or '').split(':')[0] or None, depth=depth_ft(x.get('depth', '')),
                    fine=FINE.get(x.get('time', '')), h=fnum(x.get('horizontal(mm)')), v=fnum(x.get('vertical(mm)')),
                    th=fnum(x.get('thickness(mm)')), seq_raw=r['seq_raw'], seq_strong=r['seq_strong'], seq_all=r['seq_all']))
if CONTROL:  # whole-machine control: shuffle depth and fine period within site x area
    for key in ('depth', 'fine'):
        g = collections.defaultdict(list)
        for o in OBJ: g[(o['site'], o['area'])].append(o)
        for os_ in g.values():
            vals = [o[key] for o in os_]; rng.shuffle(vals)
            for o, v in zip(os_, vals): o[key] = v

# ---------------- arrows ----------------
def named_arrows(lv):
    A = {}
    A['len'] = ('num', lambda o: len(o[lv]))
    A['numeral_sum'] = ('num', lambda o: sum(NUM.get(x, 0) for x in o[lv]))
    A['numeral_max'] = ('num', lambda o: max([NUM.get(x, 0) for x in o[lv]]))
    A['has_numeral'] = ('num', lambda o: int(any(x in NUM for x in o[lv])))
    A['w390_count'] = ('num', lambda o: sum(x in (390, 405) for x in o[lv]))
    A['w390_present'] = ('num', lambda o: int(any(x in (390, 405) for x in o[lv])))
    def num_before_390(o):
        s = o[lv]
        for i in range(1, len(s)):
            if s[i] in (390, 405) and s[i - 1] in NUM: return NUM[s[i - 1]]
        return None
    A['numeral_before_390'] = ('num', num_before_390)
    A['jar_final'] = ('num', lambda o: int(o[lv][-1] == 740))
    A['arrow_final'] = ('num', lambda o: int(o[lv][-1] == 520))
    A['closer_last_sign'] = ('cat', lambda o: o[lv][-1] if o[lv][-1] in CLOSERS else 'other')
    A['last_sign'] = ('cat', lambda o: o[lv][-1])
    A['first_sign'] = ('cat', lambda o: o[lv][0])
    A['opener_present'] = ('num', lambda o: int(o[lv][0] in OPEN))
    A['jar_count'] = ('num', lambda o: sum(x == 740 for x in o[lv]))
    A['middle_unique'] = ('num', lambda o: int(MIDCOUNT[lv][middle(o[lv])] == 1) if middle(o[lv]) else None)
    A['middle_reuse_log'] = ('num', lambda o: math.log(MIDCOUNT[lv][middle(o[lv])]) if middle(o[lv]) else None)
    A['middle_len'] = ('num', lambda o: len(middle(o[lv])))
    A['n_families'] = ('num', lambda o: len({fam(x) for x in o[lv]}))
    A['repeats'] = ('num', lambda o: len(o[lv]) - len(set(o[lv])))
    A['fish_present'] = ('num', lambda o: int(any(x in (220, 231, 232, 233, 235, 240, 226) for x in o[lv])))
    A['size_h'] = ('num', lambda o: o['h']); A['size_v'] = ('num', lambda o: o['v']); A['size_th'] = ('num', lambda o: o['th'])
    A['size_area'] = ('num', lambda o: o['h'] * o['v'] if o['h'] and o['v'] else None)
    A['type'] = ('cat', lambda o: o['t0']); A['type2'] = ('cat', lambda o: o['type'])
    A['is_seal'] = ('num', lambda o: int(o['t0'] == 'SEAL')); A['is_tablet'] = ('num', lambda o: int(o['t0'] == 'TAB'))
    A['emblem'] = ('cat', lambda o: o['emblem'])
    A['unicorn'] = ('num', lambda o: int(o['emblem'] == 'Bull1') if o['emblem'] not in (None, 'None', '-') else None)
    return A

# random arrows: reuse strat_dark reading kinds
src = open('tools/strat_dark.py').read()
exec(src[src.index('def make_reading'):src.index('# ---------- MI')])
def random_arrows(lv, n):
    A = {}
    r = random.Random(1000 + n)
    while len(A) < n:
        kind, p = make_reading(r)
        name = f'{kind}{json.dumps(p, sort_keys=True)}'
        if name in A: continue
        numeric = kind in ('mod', 'nfam', 'rep', 'nnum', 'sumnum', 'maxw', 'lenpar', 'run', 'contains')
        A[name] = ('num' if numeric else 'cat', (lambda k, q: lambda o: read(k, q, o[lv]))(kind, p))
    return A

# ---------------- statistics ----------------
def rank(a): return np.argsort(np.argsort(a, kind='stable')) + 1.0
def spearman(x, y):
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0: return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])
def kruskal_H(groups_idx, y):
    ry = rank(y); n = len(y); gm = ry.mean()
    return float(sum(len(ix) * (ry[ix].mean() - gm) ** 2 for ix in groups_idx))
def strat_perm(y, strata, nperm):
    """(nperm, n) matrix of y permuted within strata (vectorised)"""
    idx = collections.defaultdict(list)
    for i, s in enumerate(strata): idx[s].append(i)
    out = np.tile(y, (nperm, 1))
    for v in idx.values():
        ix = np.array(v)
        if len(ix) < 2: continue
        order = np.argsort(nrng.random((nperm, len(ix))), axis=1)
        out[:, ix] = y[ix][order]
    return out
def within_area_rho(x, y, strata):
    xs = np.array(x, float); ys = np.array(y, float); st = np.array(strata)
    xr, yr = rank(xs), rank(ys)
    for s in set(strata):
        m = st == s; xr[m] -= xr[m].mean(); yr[m] -= yr[m].mean()
    if xr.std() == 0 or yr.std() == 0: return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])
def test(kind, xs, ys, strata, nperm):
    ys = np.array(ys, float)
    if kind == 'num':
        xs = np.array(xs, float)
        if len(set(xs)) < 2: return None
        obs = spearman(xs, ys)
        rx = rank(xs); rx = (rx - rx.mean()) / (rx.std() + 1e-12)
        RY = strat_perm(rank(ys), strata, nperm)
        RY = (RY - RY.mean(1, keepdims=True)) / (RY.std(1, keepdims=True) + 1e-12)
        rhos = RY @ rx / len(xs)
        ge = int((np.abs(rhos) >= abs(obs) - 1e-12).sum())
        return dict(stat=obs, within=within_area_rho(xs, ys, strata), p=(ge + 1) / (nperm + 1), n=len(xs))
    cnt = collections.Counter(xs); xs = [x if cnt[x] >= 5 else 'other' for x in xs]
    groups = collections.defaultdict(list)
    for i, x in enumerate(xs): groups[x].append(i)
    if len(groups) < 2: return None
    gi = [np.array(v) for v in groups.values()]
    obs = kruskal_H(gi, ys)
    RY = strat_perm(rank(ys), strata, nperm)  # ranks of permuted y = permuted ranks
    G = np.zeros((len(gi), len(xs)))
    for j, ix in enumerate(gi): G[j, ix] = 1.0 / len(ix)
    means = RY @ G.T                               # (nperm, k)
    ng = np.array([len(ix) for ix in gi])
    H = ((means - RY.mean(1, keepdims=True)) ** 2 * ng).sum(1)
    ge = int((H >= obs - 1e-9).sum())
    # direction summary: group with highest / lowest mean rank
    ry = rank(ys); means = {k: ry[np.array(v)].mean() for k, v in groups.items()}
    hi = max(means, key=means.get); lo = min(means, key=means.get)
    return dict(stat=obs, within=None, p=(ge + 1) / (nperm + 1), n=len(xs), hi=str(hi), lo=str(lo), k=len(groups))
def bh(ps):
    m = len(ps); order = np.argsort(ps); q = np.empty(m); prev = 1.0
    for rank_, i in enumerate(reversed(order)):
        prev = min(prev, ps[i] * m / (m - rank_)); q[i] = prev
    return q

def run(arrows, outcome, lv, nperm, minn=40):
    res = {}
    for site in ('Mohenjo-daro', 'Harappa'):
        objs = [o for o in OBJ if o['site'] == site and o[outcome] is not None]
        rows = {}
        for name, (kind, f) in arrows.items():
            xs, ys, st = [], [], []
            for o in objs:
                x = f(o)
                if x is None: continue
                xs.append(x if kind == 'num' else str(x)); ys.append(o[outcome]); st.append(o['area'])
            if len(xs) < minn: continue
            r = test(kind, xs, ys, st, nperm)
            if r: rows[name] = r
        if rows:
            q = bh(np.array([r['p'] for r in rows.values()]))
            for (name, r), qq in zip(rows.items(), q): r['q'] = float(qq)
        res[site] = rows
    return res

def report(res, lv, outcome, out):
    md, hp = res['Mohenjo-daro'], res['Harappa']
    both = sorted(set(md) & set(hp))
    out.append(f'\n### {outcome} | {lv} | arrows MD {len(md)} HP {len(hp)}')
    out.append(f'{"arrow":42s} {"MD rho/H":>9s} {"MDwithin":>8s} {"MD p":>7s} {"MD q":>6s} | {"HP rho/H":>9s} {"HPwithin":>8s} {"HP p":>7s} {"HP q":>6s}  verdict')
    both_sig = []; opp = []
    for a in both:
        m, h = md[a], hp[a]
        v = ''
        if m['q'] < 0.1 and h['q'] < 0.1:
            if m['within'] is not None:
                same = np.sign(m['within']) == np.sign(h['within']) and m['within'] != 0
                v = 'BOTH-SAME' if same else 'BOTH-OPPOSITE'
                (both_sig if same else opp).append(a)
            else:
                v = f'BOTH-CAT (MD hi={m["hi"]} lo={m["lo"]}; HP hi={h["hi"]} lo={h["lo"]})'; both_sig.append(a)
        elif m['q'] < 0.1 or h['q'] < 0.1: v = 'one city'
        wm = f'{m["within"]:+.3f}' if m['within'] is not None else f'k={m["k"]}'
        wh = f'{h["within"]:+.3f}' if h['within'] is not None else f'k={h["k"]}'
        if v or min(m['p'], h['p']) < 0.05:
            out.append(f'{a[:42]:42s} {m["stat"]:+9.3f} {wm:>8s} {m["p"]:7.4f} {m["q"]:6.3f} | {h["stat"]:+9.3f} {wh:>8s} {h["p"]:7.4f} {h["q"]:6.3f}  {v}')
    out.append(f'single-city q<0.1: MD {sum(r["q"]<0.1 for r in md.values())}, HP {sum(r["q"]<0.1 for r in hp.values())}; both-same {both_sig}; both-opposite {opp}')
    return both_sig, opp

if __name__ == '__main__':
    out = []
    tag = f'cycle{CYCLE}' + ('_control' if CONTROL else '') + (f'_type{TYPEF}' if TYPEF else '')
    out.append(f'# S-DARK-9 {tag}: objects MD {sum(o["site"]=="Mohenjo-daro" for o in OBJ)} HP {sum(o["site"]=="Harappa" for o in OBJ)}; '
               f'with depth MD {sum(o["site"]=="Mohenjo-daro" and o["depth"] is not None for o in OBJ)} HP {sum(o["site"]=="Harappa" and o["depth"] is not None for o in OBJ)}; '
               f'with fine period MD {sum(o["site"]=="Mohenjo-daro" and o["fine"] is not None for o in OBJ)} HP {sum(o["site"]=="Harappa" and o["fine"] is not None for o in OBJ)}')
    out.append(f'closer set W {sorted(CLOSERS)}; suffix W {sorted(SUFFIX)}; depth = ft below datum (larger = deeper = older); fine period 0=P2 .. 8=P4 (larger = later). '
               f'Permutation: {NPERM}x, depth/period shuffled within site x area-section. Sign of rho: + means feature rises with depth (older) / with period (later).')
    summary = collections.defaultdict(list)
    for lv in ('seq_raw', 'seq_strong', 'seq_all'):
        arrows = named_arrows(lv) if CYCLE in (1, 3, 4) else random_arrows(lv, NRAND)
        for outcome in ('depth', 'fine'):
            res = run(arrows, outcome, lv, NPERM if CYCLE != 2 else 1000)
            bs, op = report(res, lv, outcome, out)
            summary[outcome].append((lv, bs, op))
            if CYCLE == 4:
                for a in set(bs) | set(op):
                    for site in ('Mohenjo-daro', 'Harappa'):
                        objs = [o for o in OBJ if o['site'] == site and o[outcome] is not None]
                        kind, f = arrows[a]
                        g = collections.defaultdict(list)
                        for o in objs:
                            x = f(o)
                            if x is None: continue
                            b = (int(o['depth'] // 4) * 4 if outcome == 'depth' else o['fine'])
                            g[b].append(x)
                        out.append(f'  {a} @ {site}: ' + '; '.join(f'{b}:{np.mean(v):.2f}(n{len(v)})' for b, v in sorted(g.items()) if len(v) >= 8))
    out.append('\n## cross-level summary')
    for outcome, L in summary.items():
        for lv, bs, op in L: out.append(f'{outcome} {lv}: both-same {bs}; both-opposite {op}')
    import os; os.makedirs('data/derived/dark', exist_ok=True)
    path = f'data/derived/dark/loop9_{tag}.txt'
    open(path, 'w').write('\n'.join(out) + '\n'); print('\n'.join(out)); print('->', path)
