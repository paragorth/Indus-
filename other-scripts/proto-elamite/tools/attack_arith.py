#!/usr/bin/env python3
"""Attack 2: arithmetic as a decoder (joint exhaustive fit of numeral values).

Case building (fixed before any fitting):
  * Edge marks: lines on the top/left edge that carry no signs (mostly '1(N34)'
    and '2(N01)') are NOT totals and are dropped. Earlier test c treated some of
    them as totals.
  * A tablet is usable only if no line has a lacuna and the ATF has no
    '$ ... broken / missing' state line, and all numerals are readable (no 'n').
    strict = also no '#' or '?' on any numeral.
  * Off-obverse numeric lines R1..Rk give the total hypotheses:
      k = 1  'single'     R1 = sum of the obverse entries of the same system.
      k >= 2 'persys'     every R line = sum of the obverse entries of its own
                          system (only if the R lines have different systems).
      k >= 2 'last'/'first' (secondary set) last R = obverse + earlier R lines;
                          or first R = obverse entries.
  * System of a numeral group: 'cap' (capacity codes), 'cap@' (hatched),
    'amb' (only N01/N14/N45/N34/N48), 'cnt' (has N51/N08/N23/...).
    Count total: members = cnt + amb lines.
    Capacity total: 3 member variants are allowed (any may hold):
      v1 lines with capacity codes only; v2 + amb lines whose final sign is not
      a 'counted' class sign (test b); v3 + all amb lines.
  * Conversion case: capacity total, entries all count-only (kept apart).

Fit: exhaustive grids. A tablet adds up if, for some hypothesis, every pair
(sum, total) agrees exactly. 'one error' = at most one pair is off, by exactly
one unit of one code present on that pair.
Out of sample: 50 random halves of tablets. Values fitted on one half (all tied
best value sets kept, accuracy averaged over them), scored on the other half.
Control: same fitted values, totals swapped among test tablets of the same class.
Fit-to-noise: best in-sample score after shuffling totals (whole grid freedom).
"""
import collections, itertools, json, os, random, re
import numpy as np
from fractions import Fraction as F
from common import load, base, is_sign, DATA

rng = random.Random(11)
T = load()

BROKEN = set()
_cur = None
for _l in open(os.path.join(DATA, 'pe_raw.atf'), encoding='utf-8'):
    if _l.startswith('&'):
        _cur = _l[1:].split()[0]
    elif _l.startswith('$') and re.search(r'broken|missing|not given', _l):
        BROKEN.add(_cur)

CNT_ORDER = ['N01', 'N14', 'N45', 'N34', 'N48', 'N51', 'N08', 'N08A', 'N8B', 'N02']
CAP_ORDER = ['N39C', 'N30D', 'N30C', 'N24', 'N39B', 'N01', 'N14', 'N45', 'N34', 'N48']
AMBIG = {'N01', 'N14', 'N45', 'N34', 'N48'}
CAPSET = {'N39B', 'N30C', 'N24', 'N30D', 'N39C', 'N39A', 'N28', 'N29B', 'N39N'}
COUNTED = {'M263', 'M346', 'M264', 'M003', 'M376', 'M032', 'M373', 'M102', 'M362', 'M317', 'M149', 'M046'}
CAP_TIED = {'M297', 'M002', 'M036', 'M243'}
R2_12 = list(range(2, 13))


def final_sign(l):
    s = [base(x) for x in l['signs'] if is_sign(x)]
    return s[-1] if s else None


def system(l):
    codes = {c for _, c in l['numerals']}
    if any('@' in c for c in codes):
        return 'cap@'
    if codes & CAPSET:
        return 'cap'
    if codes <= AMBIG:
        return 'amb'
    return 'cnt'


def vec(lines):
    v = collections.Counter()
    for l in lines:
        for n, c in l['numerals']:
            v[c.split('@')[0]] += n
    return dict(v)


def line_clean(l, strict):
    tail = l['raw'].split(',')[-1]
    pat = r'[\[#?]' if strict else r'[\[?]'
    return (not l['lacuna'] and '...' not in l['raw'] and not re.search(pat, tail)
            and all(n is not None and c != 'n' for n, c in l['numerals']))


def tot_class(tl, ent):
    s = system(tl)
    if s != 'amb':
        return 'cnt' if s == 'cnt' else ('cap@' if s == 'cap@' else 'cap')
    caps = sum(system(l) in ('cap', 'cap@') for l in ent)
    return 'cap' if caps > len(ent) / 2 else 'cnt'


def member_variants(ent, cls):
    if cls == 'cnt':
        return [[l for l in ent if system(l) in ('cnt', 'amb')]]
    if cls == 'cap@':
        return [[l for l in ent if system(l) == 'cap@']]
    v1 = [l for l in ent if system(l) == 'cap']
    v2 = v1 + [l for l in ent if system(l) == 'amb' and final_sign(l) not in COUNTED]
    v3 = v1 + [l for l in ent if system(l) == 'amb']
    out = []
    for v in (v1, v2, v3):
        if v and all(set(map(id, v)) != set(map(id, w)) for w in out):
            out.append(v)
    return out


def has_gap(t):
    prev = {}
    for l in t['lines']:
        m = re.match(r'(\d+)', str(l.get('label', '')))
        if not m:
            continue
        key = (l['surface'], l['column']); n = int(m.group(1))
        if key in prev and n > prev[key] + 1:
            return True
        prev[key] = n
    return False


def annotation(l):
    # numeral written before the signs with no comma: not an entry (e.g. '2(N14) x')
    return any(is_sign(s) or s == 'x' for s in l['signs']) and not l.get('has_comma', True)


n_gap = n_annot = 0
tablets, conversions, conversions_relaxed = [], [], []
for t in T:
    if has_gap(t):
        n_gap += 1
        continue
    lines = [l for l in t['lines'] if l['numerals'] and not (
        l['surface'] in ('top', 'left', 'seal') and not any(is_sign(s) for s in l['signs']))]
    n_annot += sum(annotation(l) for l in lines)
    lines = [l for l in lines if not annotation(l)]
    O = [l for l in lines if l['surface'] == 'obverse']
    R = [l for l in lines if l['surface'] != 'obverse']
    if not R or not O:
        continue
    # relaxed conversion set: involved lines clean, other damage on the tablet allowed
    if len(R) >= 1 and tot_class(R[-1], O) == 'cap' and all(system(l) in ('cnt', 'amb') for l in O) \
            and all(line_clean(l, False) for l in O + [R[-1]]):
        conversions_relaxed.append({'id': t['id'], 'heads': vec(O), 'tot': vec([R[-1]]),
                                    'tot_signs': [base(s) for s in R[-1]['signs'] if is_sign(s)],
                                    'ent_final': [final_sign(l) for l in O], 'broken_elsewhere': t['id'] in BROKEN})
    if t['id'] in BROKEN or any(l['lacuna'] or '...' in l['raw'] for l in t['lines']):
        continue
    if not all(line_clean(l, False) for l in lines):
        continue
    strict = all(line_clean(l, True) for l in lines)
    struct_pairs = []
    if len(R) == 1:
        struct_pairs.append(('single', [(R[0], O)]))
    else:
        cl = [tot_class(r, O) for r in R]
        if len(set(cl)) == len(cl):
            struct_pairs.append(('persys', [(r, O) for r in R]))
        struct_pairs.append(('last', [(R[-1], O + R[:-1])]))
        struct_pairs.append(('first', [(R[0], O)]))
    for sname, pairs in struct_pairs:
        # conversion: single capacity total with count-only entries
        if sname in ('single', 'last') and len(pairs) == 1:
            tl, ent = pairs[0]
            if tot_class(tl, ent) == 'cap' and all(system(l) in ('cnt', 'amb') for l in ent):
                conversions.append({'id': t['id'], 'struct': sname, 'heads': vec(ent),
                                    'tot': vec([tl]), 'tot_signs': [base(s) for s in tl['signs'] if is_sign(s)],
                                    'ent_final': [final_sign(l) for l in ent], 'strict': strict})
                continue
        # hypotheses = product of member variants over pairs
        per_pair = []
        for tl, ent in pairs:
            cls = tot_class(tl, ent)
            per_pair.append([(cls, m, tl) for m in member_variants(ent, cls)])
        hyps = []
        for combo in itertools.product(*per_pair):
            if any(len(m) < 2 for _, m, _ in combo):
                continue
            hyps.append([{'cls': cls, 'sum': vec(m), 'tot': vec([tl]),
                          'tot_signs': [base(s) for s in tl['signs'] if is_sign(s)],
                          'ent_final': [final_sign(l) for l in m], 'n_members': len(m)}
                         for cls, m, tl in combo])
        if not hyps:
            continue
        clsset = sorted(set(p['cls'] for p in hyps[0]))
        tablets.append({'id': t['id'], 'struct': sname, 'strict': strict, 'n_R': len(R),
                        'classes': clsset, 'hyps': hyps})

main_structs = ('single', 'persys')
print('tablets', collections.Counter((t['struct'], tuple(t['classes'])) for t in tablets))
print('conversions', len(conversions), 'relaxed', len(conversions_relaxed), 'gap tablets skipped', n_gap, 'annotation lines', n_annot)


# ---------------- grids ----------------
def count_grid():
    rows, labels = [], []
    for n14 in R2_12:
        for n45 in sorted(set([n14 * k for k in R2_12] + [60, 100, 120, 600, 1000])):
            if n45 <= n14:
                continue
            for n34 in sorted(set([n45 * k for k in R2_12] + [60, 100, 120, 600, 1000, 3600])):
                if n34 <= n45:
                    continue
                for n48 in sorted(set([n34 * 2, n34 * 3, n34 * 6, n34 * 10, 3600, 10000])):
                    for n51 in (120, 2 * n34):
                        for fr in (1 / 2, 1 / 3, 1 / 5, 1 / 6, 1 / 10):
                            rows.append([1, n14, n45, n34, n48, n51, fr, fr, fr, fr])
                            labels.append((n14, n45, n34, n48, n51, round(fr, 4)))
    return np.array(rows, dtype=float), labels


def cap_values(r):
    """r = (N30D/N39C, N30C/N30D, N24/N30C, N39B/N24, N01/N39B, N14/N01, N45/N14, N34/N45, N48/N34)"""
    v = [1.0]
    for x in r:
        v.append(v[-1] * x)
    return v


def cap_grid_lower(upper=(3, 10, 6)):
    rows, labels = [], []
    for lo in itertools.product(R2_12, R2_12, R2_12, R2_12, R2_12, R2_12):
        rows.append(cap_values(lo + upper)); labels.append(lo + upper)
    return np.array(rows), labels


def cap_grid_upper(lo6):
    rows, labels = [], []
    for up in itertools.product(R2_12, R2_12, (2, 3, 6, 10)):
        rows.append(cap_values(tuple(lo6) + up)); labels.append(tuple(lo6) + up)
    return np.array(rows), labels


# ---------------- evaluation ----------------
def flatten(tabs, codes, pairs_override=None):
    """Return pair matrix D, presence P, and per-tablet hypothesis structure."""
    ix = {c: i for i, c in enumerate(codes)}
    Drows, Prows, struct, keep = [], [], [], []
    for k, t in enumerate(tabs):
        hs = []
        ok_t = True
        for h in t['hyps']:
            idxs = []
            for p in h:
                d = np.zeros(len(codes)); pr = np.zeros(len(codes), dtype=bool)
                for code, n in p['sum'].items():
                    if code not in ix:
                        ok_t = False; break
                    d[ix[code]] += n; pr[ix[code]] = True
                for code, n in p['tot'].items():
                    if code not in ix:
                        ok_t = False; break
                    d[ix[code]] -= n; pr[ix[code]] = True
                idxs.append(len(Drows)); Drows.append(d); Prows.append(pr)
            hs.append(idxs)
        if ok_t:
            struct.append(hs); keep.append(k)
    return np.array(Drows), np.array(Prows), struct, keep


def evaluate(D, P, struct, V):
    """V (m x codes). Returns tablet-level exact and <=1-error matrices (m x ntab)."""
    res = V @ D.T
    ex = np.isclose(res, 0, atol=1e-7)
    ar = np.abs(res)
    er = np.zeros_like(ex)
    for j in range(V.shape[1]):
        er |= P[:, j][None, :] & np.isclose(ar, V[:, j][:, None], atol=1e-7)
    TE = np.zeros((V.shape[0], len(struct)), dtype=bool)
    TR = np.zeros_like(TE)
    for k, hs in enumerate(struct):
        for idxs in hs:
            e = ex[:, idxs]
            TE[:, k] |= e.all(1)
            TR[:, k] |= ((e | er[:, idxs]).all(1) & ((~e).sum(1) <= 1))
    return TE, TR


def chunked(D, P, struct, V, chunk=4000):
    A, B = [], []
    for i in range(0, len(V), chunk):
        a, b = evaluate(D, P, struct, V[i:i + chunk]); A.append(a); B.append(b)
    return np.vstack(A), np.vstack(B)


def tied_best(E, cols, R=None):
    s = E[:, cols].sum(1)
    rows = np.where(s == s.max())[0]
    if R is not None:
        s2 = R[rows][:, cols].sum(1)
        rows = rows[s2 == s2.max()]
    return rows, int(s.max())


def shuffled_tabs(tabs, ids):
    """Swap totals among the given tablets of the same class (pair-level)."""
    pool = collections.defaultdict(list)
    for k in ids:
        for h in tabs[k]['hyps']:
            for p in h:
                pool[p['cls']].append(p['tot'])
    out = []
    for k in ids:
        t = tabs[k]
        nh = []
        draw = {}
        for h in t['hyps']:
            np_ = []
            for j, p in enumerate(h):
                if j not in draw:
                    cand = [x for x in pool[p['cls']] if x != p['tot']] or pool[p['cls']]
                    draw[j] = rng.choice(cand)
                np_.append(dict(p, tot=draw[j]))
            nh.append(np_)
        out.append(dict(t, hyps=nh))
    return out


def run_system(name, tabs, codes, V, L, reps=50):
    D, P, struct, keep = flatten(tabs, codes)
    tabs = [tabs[k] for k in keep]
    E, R = chunked(D, P, struct, V)
    allc = np.arange(len(tabs))
    rows, top = tied_best(E, allc, R)
    res = {'n_tablets': len(tabs), 'n_strict': int(sum(t['strict'] for t in tabs)),
           'full_best_exact': top, 'full_best_one_error': int(R[rows[0]].sum()),
           'n_tied_best': int(len(rows)), 'tied_best_labels_sample': [L[r] for r in rows[:20]]}
    # out-of-sample
    acc, acc1, acc_s, ctrl, ctrl1 = [], [], [], [], []
    for rep in range(reps):
        perm = list(allc); rng.shuffle(perm)
        tr = np.array(sorted(perm[:len(perm) // 2])); te = np.array(sorted(perm[len(perm) // 2:]))
        brow, _ = tied_best(E, tr, R)
        acc.append(E[np.ix_(brow, te)].mean()); acc1.append(R[np.ix_(brow, te)].mean())
        tes = [k for k in te if tabs[k]['strict']]
        if tes:
            acc_s.append(E[np.ix_(brow, tes)].mean())
        sh = shuffled_tabs(tabs, list(te))
        Ds, Ps, ss, kk = flatten(sh, codes)
        e, r = evaluate(Ds, Ps, ss, V[brow[:300]])
        ctrl.append(e.mean()); ctrl1.append(r.mean())
    res['oos'] = {'test_exact': float(np.mean(acc)), 'test_exact_sd': float(np.std(acc)),
                  'test_exact_strict': float(np.mean(acc_s)) if acc_s else None,
                  'test_one_error': float(np.mean(acc1)),
                  'control_exact': float(np.mean(ctrl)), 'control_one_error': float(np.mean(ctrl1)),
                  'share_splits_over_70pct': float(np.mean(np.array(acc) > 0.7))}
    # fit to noise
    fn = []
    for _ in range(5):
        sh = shuffled_tabs(tabs, list(allc))
        Ds, Ps, ss, kk = flatten(sh, codes)
        e, r = chunked(Ds, Ps, ss, V)
        fn.append(int(e.sum(1).max()))
    res['fit_to_noise_best_exact'] = fn
    print(name, json.dumps({k: v for k, v in res.items() if k != 'tied_best_labels_sample'}))
    return res, tabs, E, R, rows


out = {'n_conversion': len(conversions)}

# ===== COUNT systems =====
Vn, Ln = count_grid()
cnt_main = [t for t in tablets if t['struct'] in main_structs and t['classes'] == ['cnt']]
cnt_sec = [t for t in tablets if t['struct'] in ('last', 'first') and t['classes'] == ['cnt']]
print('count grid', len(Vn), 'main', len(cnt_main), 'secondary', len(cnt_sec))
out['count_main'], cnt_tabs, En, Rn, rows_n = run_system('COUNT main', cnt_main, CNT_ORDER, Vn, Ln)


def marg(E, L, k, cols=None):
    s = E.sum(1) if cols is None else E[:, cols].sum(1)
    m = collections.defaultdict(int)
    for row, lab in zip(s, L):
        m[lab[k]] = max(m[lab[k]], int(row))
    return m


mk = {}
for k, nm in enumerate(['N14', 'N45', 'N34', 'N48', 'N51', 'frac']):
    m = marg(En, Ln, k)
    top = max(m.values())
    mk[nm] = {'best_score': top, 'values_reaching_best': sorted(a for a, b in m.items() if b == top)[:40],
              'n_values_reaching_best': sum(b == top for b in m.values()), 'n_values': len(m),
              'score_for': {str(a): m[a] for a in (2, 3, 5, 6, 10, 12, 20, 30, 60, 100, 120, 200, 300, 360, 600, 1000, 3600, 0.5, 0.2, 0.1) if a in m}}
out['count_marginals'] = mk
print('count marginals', json.dumps(mk)[:1500])
# high-unit tablets only
hi = [k for k, t in enumerate(cnt_tabs) if any(set(p['sum']) | set(p['tot']) & {'N45', 'N34', 'N48'} for h in t['hyps'] for p in h)]
hi = [k for k, t in enumerate(cnt_tabs) if any(({'N45', 'N34', 'N48'} & (set(p['sum']) | set(p['tot']))) for h in t['hyps'] for p in h)]
out['count_high_unit_tablets'] = {'n': len(hi), 'best_exact_on_them': int(En[:, hi].sum(1).max()),
                                  'ids': [cnt_tabs[k]['id'] for k in hi]}
for nm, k in (('N45', 1), ('N34', 2)):
    m = marg(En, Ln, k, hi)
    out['count_high_unit_tablets']['marg_' + nm] = {str(a): b for a, b in sorted(m.items()) if b >= max(m.values()) - 1}
print('high-unit tablets', out['count_high_unit_tablets'])

# named systems
NAMED = {'S (10,600?,60)': [1, 10, 600, 60, 3600, 120, .5, .5, .5, .5],
         'S-strict N45>N34 ignored: (10,-,60) N45=100': [1, 10, 100, 60, 3600, 120, .5, .5, .5, .5],
         'D (10,100,1000)': [1, 10, 100, 1000, 10000, 120, .5, .5, .5, .5],
         'D3 (10,100,300)': [1, 10, 100, 300, 1800, 120, .5, .5, .5, .5]}
Dn, Pn, sn, _ = flatten(cnt_tabs, CNT_ORDER)
named = {}
for nm, v in NAMED.items():
    e, r = evaluate(Dn, Pn, sn, np.array([v], dtype=float))
    named[nm] = {'exact': int(e.sum()), 'one_error': int(r.sum()), 'n': len(cnt_tabs),
                 'high_unit_exact': int(e[0, hi].sum())}
out['count_named_systems'] = named
print('named', named)

# secondary multi-line structures (are they totals at all?)
if cnt_sec:
    Ds, Ps, ss, kk = flatten(cnt_sec, CNT_ORDER)
    bestv = Vn[rows_n[:1]]
    e, r = evaluate(Ds, Ps, ss, bestv)
    sh = shuffled_tabs([cnt_sec[k] for k in kk], list(range(len(kk))))
    Ds2, Ps2, ss2, _ = flatten(sh, CNT_ORDER)
    e2, _ = evaluate(Ds2, Ps2, ss2, bestv)
    out['count_multiline_last_first'] = {'n': len(kk), 'exact': int(e.sum()), 'one_error': int(r.sum()),
                                         'shuffled_control_exact': int(e2.sum())}
    print('secondary', out['count_multiline_last_first'])

# ===== CAPACITY =====
cap_main = [t for t in tablets if t['struct'] in main_structs and t['classes'] != ['cnt']
            and all(c in ('cap', 'cap@', 'cnt') for c in t['classes'])]
Vc, Lc = cap_grid_lower()
print('cap grid', len(Vc), 'tablets', len(cap_main))
# capacity tablets may also carry count pairs (persys); give them a combined vector space:
CODES_ALL = CAP_ORDER + ['N51', 'N08', 'N08A', 'N8B', 'N02', 'cntN01', 'cntN14', 'cntN45', 'cntN34']


def recode(tabs):
    """Count-class pairs get their own code names so capacity and count units stay separate."""
    out_ = []
    for t in tabs:
        nh = []
        for h in t['hyps']:
            nh.append([p if p['cls'] != 'cnt' else dict(
                p, sum={('cnt' + c if c in AMBIG else c): n for c, n in p['sum'].items()},
                tot={('cnt' + c if c in AMBIG else c): n for c, n in p['tot'].items()}) for p in h])
        out_.append(dict(t, hyps=nh))
    return out_


def with_cnt(V):
    extra = np.tile(np.array([120, .5, .5, .5, .5, 1, 10, 100, 300], dtype=float), (len(V), 1))
    return np.hstack([V, extra])


cap_main = recode(cap_main)
Vc_all = with_cnt(Vc)
out['cap_stage1'], cap_tabs, Ec, Rc, rows_c = run_system('CAP stage1 (lower 6 ratios, upper 3/10/6)', cap_main, CODES_ALL, Vc_all, Lc, reps=50)
cm = {}
for k, nm in enumerate(['N30D/N39C', 'N30C/N30D', 'N24/N30C', 'N39B/N24', 'N01/N39B', 'N14/N01']):
    m = marg(Ec, Lc, k)
    cm[nm] = {str(a): b for a, b in sorted(m.items())}
out['cap_marginals'] = cm
print('cap marginals', cm)
lo6 = Lc[rows_c[0]][:6]
Vu, Lu = cap_grid_upper(lo6)
Du, Pu, su, ku = flatten(cap_tabs, CODES_ALL)
eu, ru = evaluate(Du, Pu, su, with_cnt(Vu))
su_ = eu.sum(1)
out['cap_stage2_upper'] = {'lower_fixed': lo6, 'best_exact': int(su_.max()),
                           'n_upper_sets_tied': int((su_ == su_.max()).sum()),
                           'example': Lu[int(np.argmax(su_))]}
print('cap upper', out['cap_stage2_upper'])

# ===== a-priori value sets from notation bundling (max repetitions + 1), not fitted on totals =====
bund = collections.defaultdict(collections.Counter)
for t in T:
    for l in t['lines']:
        if not l['numerals'] or any(n is None for n, _ in l['numerals']):
            continue
        if l['surface'] in ('top', 'left') and not any(is_sign(s) for s in l['signs']):
            continue
        sk = 'cap' if system(l) in ('cap', 'cap@') else ('cnt' if system(l) in ('cnt', 'amb') else None)
        for n, c in l['numerals']:
            bund[(sk, c.split('@')[0])][n] += 1


def max_rep(sk, c, min_share=0.01):
    d = bund[(sk, c)]; tot = sum(d.values())
    return max([k for k, v in d.items() if v / tot >= min_share] or [0])


out['bundling_max_repetition'] = {'%s %s' % k: dict(sorted(v.items())) for k, v in bund.items() if sum(v.values()) >= 10}
CAP_NAMED = {
    'bundling A (N30C=2 N30D)': (2, 2, 3, 2, 5, 6, 10, 3, 10),
    'bundling B (N30C=3 N30D)': (2, 3, 3, 2, 5, 6, 10, 3, 10),
    'best-fit example': tuple(Lc[rows_c[0]]),
}
Dc_, Pc_, sc_, _ = flatten(cap_tabs, CODES_ALL)
capn = {}
for nm, r in CAP_NAMED.items():
    e, rr = evaluate(Dc_, Pc_, sc_, with_cnt(np.array([cap_values(r)])))
    capn[nm] = {'ratios(N30D/N39C,N30C/N30D,N24/N30C,N39B/N24,N01/N39B,N14/N01,N45/N14,N34/N45,N48/N34)': r,
                'exact': int(e.sum()), 'one_error': int(rr.sum()), 'n': len(cap_tabs),
                'exact_ids': [cap_tabs[k]['id'] for k in np.where(e[0])[0]]}
    # shuffled control for an a-priori set
    cc = []
    for _ in range(200):
        sh = shuffled_tabs(cap_tabs, list(range(len(cap_tabs))))
        D2, P2, s2, _ = flatten(sh, CODES_ALL)
        e2, _ = evaluate(D2, P2, s2, with_cnt(np.array([cap_values(r)])))
        cc.append(int(e2.sum()))
    capn[nm]['shuffled_control_mean'] = float(np.mean(cc))
out['cap_named_sets'] = capn
print('cap named', capn)
cntn_ctrl = {}
for nm, v in NAMED.items():
    cc = []
    for _ in range(200):
        sh = shuffled_tabs(cnt_tabs, list(range(len(cnt_tabs))))
        D2, P2, s2, _ = flatten(sh, CNT_ORDER)
        e2, _ = evaluate(D2, P2, s2, np.array([v], dtype=float))
        cc.append(int(e2.sum()))
    out['count_named_systems'][nm]['shuffled_control_mean'] = float(np.mean(cc))
print('count named with control', out['count_named_systems'])

# ===== diagnosis per tablet =====
def pair_val(p, Vd):
    return sum(n * Vd[c] for c, n in p['sum'].items()), sum(n * Vd[c] for c, n in p['tot'].items())


VN = dict(zip(CNT_ORDER, Vn[rows_n[0]]))
VC = dict(zip(CODES_ALL, with_cnt(np.array([cap_values(CAP_NAMED['bundling A (N30C=2 N30D)'])]))[0]))
out['best_count_values'] = {k: float(v) for k, v in VN.items()}
out['cap_values_used_for_diagnosis(bundling A)'] = {k: float(v) for k, v in VC.items() if not k.startswith('cnt')}
out['best_cap_label_example'] = Lc[rows_c[0]]


def fix_suggestion(p, Vd):
    s, tt = pair_val(p, Vd)
    diff = s - tt
    codes = sorted(set(p['sum']) | set(p['tot']), key=lambda c: -Vd[c])
    sug = []
    for c in codes:
        if abs(abs(diff) - Vd[c]) < 1e-7:
            if diff > 0:
                sug.append('total should have one more %s (or one entry one %s fewer)' % (c, c))
            else:
                sug.append('total should have one %s fewer (or one entry one %s more)' % (c, c))
    return s, tt, sug


diag = []
for tabs, E, R, rows, Vd in ((cnt_tabs, En, Rn, rows_n, VN), (cap_tabs, Ec, Rc, rows_c, VC)):
    b = rows[0]
    for k, t in enumerate(tabs):
        h = t['hyps'][0]
        best_h = h
        for hh in t['hyps']:
            if all(abs(pair_val(p, Vd)[0] - pair_val(p, Vd)[1]) < 1e-7 for p in hh):
                best_h = hh
        ex_ = any(all(abs(pair_val(p, Vd)[0] - pair_val(p, Vd)[1]) < 1e-7 for p in hh) for hh in t['hyps'])
        rec = {'id': t['id'], 'struct': t['struct'], 'classes': t['classes'], 'strict': t['strict'],
               'exact': ex_, 'pairs': []}
        for p in best_h:
            s, tt, sug = fix_suggestion(p, Vd)
            rec['pairs'].append({'cls': p['cls'], 'sum': p['sum'], 'tot': p['tot'], 'sum_val': s, 'tot_val': tt,
                                 'tot_signs': p['tot_signs'], 'suggest': sug})
        rec['one_error'] = (not ex_) and len(rec['pairs']) >= 1 and all(
            p['suggest'] or abs(p['sum_val'] - p['tot_val']) < 1e-7 for p in rec['pairs']) and sum(
            abs(p['sum_val'] - p['tot_val']) > 1e-7 for p in rec['pairs']) == 1
        diag.append(rec)
out['tablets'] = diag
n_need_err = sum(d['one_error'] for d in diag)
out['n_tablets_needing_one_error'] = n_need_err

# ===== 3a. decimal vs sexagesimal notation by class sign =====
cls_groups = {'capacity-tied': CAP_TIED, 'counted': {'M263', 'M346', 'M264', 'M003'}, 'fractional': {'M376'}}
lines_all = []
for t in T:
    for l in t['lines']:
        if not l['numerals'] or system(l) not in ('amb', 'cnt') or final_sign(l) is None:
            continue
        d = {c: n for n, c in l['numerals'] if n}
        if any(n is None for n, _ in l['numerals']):
            continue
        # decimal-marked: 6-9 N14 or an N45 (=100); sexagesimal-marked: N34 with N14<=5 and no N45
        dec = d.get('N14', 0) >= 6 or 'N45' in d
        sex = 'N34' in d and 'N45' not in d and d.get('N14', 0) <= 5
        if not (dec or sex):
            continue
        lines_all.append((t['id'], final_sign(l), dec))
ds = {}
for g, S in cls_groups.items():
    xs = [x for x in lines_all if x[1] in S]
    ds[g] = {'n_lines_value_ge_60': len(xs), 'decimal_marked': sum(x[2] for x in xs)}
xs = [x for x in lines_all if x[1] not in set().union(*cls_groups.values())]
ds['other final signs'] = {'n_lines_value_ge_60': len(xs), 'decimal_marked': sum(x[2] for x in xs)}
per_sign = collections.defaultdict(lambda: [0, 0])
for _, fs, dec in lines_all:
    per_sign[fs][0] += 1; per_sign[fs][1] += dec
ds['per_sign_n>=8'] = {s: {'n': a, 'decimal': b} for s, (a, b) in sorted(per_sign.items(), key=lambda x: -x[1][0]) if a >= 8}
# permutation test: does decimal share differ between sign groups beyond within-tablet shuffling?
def group_stat(rows_):
    by = collections.defaultdict(list)
    for _, fs, dec in rows_:
        by[fs].append(dec)
    # chi-square-like: sum over signs with n>=5 of n*(p - pbar)^2
    allp = np.mean([r[2] for r in rows_])
    return sum(len(v) * (np.mean(v) - allp) ** 2 for v in by.values() if len(v) >= 5)
obs = group_stat(lines_all)
bytab = collections.defaultdict(list)
for i, r in enumerate(lines_all):
    bytab[r[0]].append(i)
null = []
for _ in range(1000):
    dec = [r[2] for r in lines_all]
    for idxs in bytab.values():
        vals = [dec[i] for i in idxs]; rng.shuffle(vals)
        for i, v in zip(idxs, vals):
            dec[i] = v
    null.append(group_stat([(a, b, d) for (a, b, _), d in zip(lines_all, dec)]))
ds['sign_effect_stat'] = float(obs)
ds['within_tablet_null_mean'] = float(np.mean(null))
ds['p_within_tablet'] = float((np.sum(np.array(null) >= obs) + 1) / 1001)
tabdec = collections.Counter()
for t_id, idxs in bytab.items():
    v = [lines_all[i][2] for i in idxs]
    tabdec['all decimal' if all(v) else ('all sexagesimal-marked' if not any(v) else 'mixed')] += 1
ds['tablet_level'] = dict(tabdec)
# targeted: M376 (fractional class) vs all other signs, within-tablet permutation
def m376_gap(dec_list):
    a = [d for (_, fs, _), d in zip(lines_all, dec_list) if fs == 'M376']
    b = [d for (_, fs, _), d in zip(lines_all, dec_list) if fs != 'M376']
    return (np.mean(b) - np.mean(a)) if a else 0
obs376 = m376_gap([r[2] for r in lines_all])
null376 = []
for _ in range(2000):
    dec = [r[2] for r in lines_all]
    for idxs in bytab.values():
        vals = [dec[i] for i in idxs]; rng.shuffle(vals)
        for i, v in zip(idxs, vals):
            dec[i] = v
    null376.append(m376_gap(dec))
ds['M376_vs_rest_gap'] = float(obs376)
ds['M376_p_within_tablet'] = float((np.sum(np.array(null376) >= obs376) + 1) / 2001)
ds['M376_tablets'] = len(set(r[0] for r in lines_all if r[1] == 'M376'))
# tablet homogeneity: mixed tablets vs expectation when marks are shuffled across tablets
multi = [idxs for idxs in bytab.values() if len(idxs) >= 2]
def n_mixed(dec):
    return sum(0 < sum(dec[i] for i in idxs) < len(idxs) for idxs in multi)
obsm = n_mixed([r[2] for r in lines_all])
nm = []
for _ in range(2000):
    dec = [r[2] for r in lines_all]; rng.shuffle(dec); nm.append(n_mixed(dec))
ds['tablets_with_2plus_marked_lines'] = len(multi)
ds['mixed_observed'] = obsm
ds['mixed_expected_if_random'] = float(np.mean(nm))
ds['p_fewer_mixed'] = float((np.sum(np.array(nm) <= obsm) + 1) / 2001)
out['decimal_vs_sexagesimal'] = ds
print('dec/sex', json.dumps(ds)[:1200])

# ===== 3b. conversion tablets: constant ration? =====
def heads_val(v):
    return sum(n * VN.get(c, np.nan) for c, n in v.items())


cap_rows_try = rows_c[:min(len(rows_c), 3000)]


def rates_for(convs, tots, Vd):
    rr = []
    for c, tt in zip(convs, tots):
        h = heads_val(c['heads'])
        try:
            tv = sum(n * Vd[k] for k, n in tt.items())
        except KeyError:
            rr.append(None); continue
        rr.append(F(int(round(tv)), int(round(h))) if (h == h and h > 0) else None)
    return rr


def conv_test(convs, label):
    best = (0, None, 0, None)
    for r_ in cap_rows_try:
        Vd = dict(zip(CODES_ALL, Vc_all[r_]))
        rr = [x for x in rates_for(convs, [c['tot'] for c in convs], Vd) if x is not None]
        if rr:
            m, k = collections.Counter(rr).most_common(1)[0]
            if k > best[0]:
                best = (k, m, len(rr), r_)
    ctrl = []
    for _ in range(100):
        tots = [c['tot'] for c in convs]; rng.shuffle(tots)
        b = 0
        for r_ in cap_rows_try[::max(1, len(cap_rows_try) // 300)]:
            Vd = dict(zip(CODES_ALL, Vc_all[r_]))
            rr = [x for x in rates_for(convs, tots, Vd) if x is not None]
            if rr:
                b = max(b, collections.Counter(rr).most_common(1)[0][1])
        ctrl.append(b)
    Vd = dict(zip(CODES_ALL, Vc_all[best[3] if best[3] is not None else rows_c[0]]))
    rr = rates_for(convs, [c['tot'] for c in convs], Vd)
    res = {'n_tablets': len(convs), 'best_modal_rate_count': best[0], 'n_with_rate': best[2],
           'modal_rate_in_smallest_units': str(best[1]),
           'control_mean_best_modal': float(np.mean(ctrl)),
           'control_p': float((np.sum(np.array(ctrl) >= best[0]) + 1) / 101),
           'tablets': [{'id': c['id'], 'heads': c['heads'], 'heads_val': heads_val(c['heads']), 'tot': c['tot'],
                        'tot_signs': c['tot_signs'], 'rate': str(x)} for c, x in zip(convs, rr)]}
    print(label, {k: v for k, v in res.items() if k != 'tablets'})
    return res


out['conversion_clean'] = conv_test(conversions, 'conversion clean')
out['conversion_relaxed'] = conv_test(conversions_relaxed, 'conversion relaxed')
out['conversion_relaxed']['total_head_signs'] = dict(collections.Counter(
    (c['tot_signs'][0] if c['tot_signs'] else None) for c in conversions_relaxed))

json.dump(out, open(os.path.join(DATA, 'attack_arith.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('done')
