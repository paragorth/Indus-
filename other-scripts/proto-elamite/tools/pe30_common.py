"""pe30 shared code: WHAT THE TOTALS LEAVE OUT.

Hypothesis: when a written total misses about one entry (pe24), a whole category
of entries is left out BY RULE (the official's own share, the first or last
entry, entries with a given sign, entries in another unit, duplicates ...), or a
fixed carried-over amount is added.

Cases. Tier 1 = the 49 clean tablets of pe24 (same cleaning). Tier 2 adds
damaged tablets whose every numeral is still readable (sign damage, '#', '?',
editor-restored numerals in '[...]' allowed; a line whose numeral is lost, a
'$ broken/missing' state line or a gap in line numbers excludes the tablet: a
lost entry would absorb any difference). Tier 3 (cycle 3 only) = tablets with
exactly one lost entry numeral, scored as inequalities.

Rule = which member entries are left out of the sum (coefficient 0), plus an
optional constant added to the sum. A tablet CLOSES under a rule if for some
member hypothesis and some value set (pe24) every (total, members) pair has
total == sum(kept members) + c exactly.

Rule families (all global, the same for every tablet):
  EX(f)      leave out the entries that carry feature f
  EX(f|g)    leave out entries with f or with g
  KEEP(f)    sum only the entries that carry f (Ur III 'one category' totals)
  C(c)       all entries plus a constant c (count pairs only; in N01 units)
  EX(f)+C(c)
Features of an entry (computed per member list): position (P_first, P_last,
P_second, P_penult, P_col1, P_collast), sign tokens (S_), base signs (B_), first
and last base sign (F_, L_), variant markers (K_), sign count (N0 bare, N1,
N3p), numeral units (U_), system (Y_), size (Z_max, Z_min, Z_one, Z_big = more
than half the sum), duplicates (D_sig same signs as another entry, D_val same
value), damage (X_dmg, X_rest), relation to the total line (T_share shares a
sign with the total, T_last same final sign, T_none shares none).
"""
import collections, itertools, json, os, re, sys
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe30_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa: E402
from pe24_common import (system, final_sign, tot_class, member_variants, has_gap,  # noqa: E402
                         annotation, value, CNT_SETS, CAPV, MARK)

CONSTS = (1, 2, 3, 5, 10, -1, -2, -3, -5, -10)


# ------------------------------------------------------------------ cases
def line_ok(l, tier):
    if any(n is None or c == 'n' for n, c in l['numerals']):
        return False
    tail = l['raw'].split(',')[-1]
    if tier == 1:
        return (not l['lacuna'] and '...' not in l['raw'] and not re.search(r'[\[?]', tail))
    return '...' not in tail and 'x(' not in tail


def lost_numeral(l):
    """A line whose numeral part is lost (signs maybe present)."""
    tail = l['raw'].split(',')[-1] if ',' in l['raw'] else ''
    return ',' in l['raw'] and '...' in tail and not l['numerals']


def member_feats(m, ent_all, tl):
    """m: list of lines (members, in tablet order); returns list of feature lists."""
    n = len(m)
    cols = [l['column'] for l in m]
    c1, cl = min(cols), max(cols)
    vals = [value(l, CNT_SETS[0]) if system(l) in ('cnt', 'amb') else value(l, CAPV) for l in m]
    tot = sum(v for v in vals if v is not None)
    sigs = [tuple(x for x in l['signs'] if is_sign(x)) for l in m]
    sc = collections.Counter(sigs); vc = collections.Counter(vals)
    tsg = {base(x) for x in tl['signs'] if is_sign(x)}
    tlast = final_sign(tl)
    vmax = max(v for v in vals if v is not None); vmin = min(v for v in vals if v is not None)
    out = []
    for i, l in enumerate(m):
        f = []
        if i == 0: f.append('P_first')
        if i == n - 1: f.append('P_last')
        if i == 1: f.append('P_second')
        if i == n - 2: f.append('P_penult')
        if c1 != cl:
            f.append('P_col1' if l['column'] == c1 else ('P_collast' if l['column'] == cl else 'P_colmid'))
        toks = list(sigs[i])
        bs = [base(x) for x in toks]
        f += ['S_' + x for x in sorted(set(toks))]
        f += ['B_' + x for x in sorted(set(bs))]
        if bs:
            f += ['F_' + bs[0], 'L_' + bs[-1]]
        f += ['K_' + k for k in sorted({k for x in toks for k in MARK.findall(x)})]
        f.append('N0' if not toks else ('N1' if len(toks) == 1 else ('N2' if len(toks) == 2 else 'N3p')))
        f += ['U_' + c.split('@')[0] for c in sorted({c for _, c in l['numerals']})]
        f.append('Y_' + system(l))
        v = vals[i]
        if v is not None:
            if v == vmax: f.append('Z_max')
            if v == vmin: f.append('Z_min')
            if v == 1: f.append('Z_one')
            if 2 * v > tot: f.append('Z_big')
            if vc[v] > 1: f.append('D_val')
        if sc[sigs[i]] > 1 and toks: f.append('D_sig')
        if l.get('damaged') or l['lacuna']: f.append('X_dmg')
        if '[' in l['raw'].split(',')[-1]: f.append('X_rest')
        if tsg & set(bs): f.append('T_share')
        else: f.append('T_none')
        if bs and tlast and bs[-1] == tlast: f.append('T_last')
        out.append(sorted(set(f)))
    return out


def build(tier=2):
    """Cases {'id','tier','hyps':[[pair]]}; pair {'cls','T':[nv],'E':[[nv]...],'F':[[feat]...]}"""
    T = load()
    broken = set(); cur = None
    for raw in open(os.path.join(DATA, 'pe_raw.atf'), encoding='utf-8'):
        if raw.startswith('&'):
            cur = raw[1:].split()[0]
        elif raw.startswith('$') and re.search(r'broken|missing|not given|traces', raw):
            broken.add(cur)
    out = []
    for t in T:
        if has_gap(t) or t['id'] in broken:
            continue
        lines = [l for l in t['lines'] if l['numerals'] and not (
            l['surface'] in ('top', 'left', 'seal') and not any(is_sign(s) for s in l['signs']))]
        lines = [l for l in lines if not annotation(l)]
        O = [l for l in lines if l['surface'] == 'obverse']
        R = [l for l in lines if l['surface'] != 'obverse']
        if not R or not O:
            continue
        t1 = (not any(l['lacuna'] or '...' in l['raw'] for l in t['lines'])
              and all(line_ok(l, 1) for l in lines))
        if tier == 1 and not t1:
            continue
        if not t1:
            if any(lost_numeral(l) for l in t['lines']):
                continue
            if any(l['lacuna'] and not l['numerals'] and l['surface'] in ('obverse', 'reverse')
                   and re.fullmatch(r'[\s\[\].]*', l['raw'].replace('...', '')) for l in t['lines']):
                continue
            if not all(line_ok(l, 2) for l in lines):
                continue
        if len(R) == 1:
            pairs = [(R[0], O)]
        else:
            cl = [tot_class(r, O) for r in R]
            if len(set(cl)) != len(cl):
                continue
            pairs = [(r, O) for r in R]
        per_pair = [[(tot_class(tl, ent), m, tl) for m in member_variants(ent, tot_class(tl, ent))]
                    for tl, ent in pairs]
        hyps = []
        for combo in itertools.product(*per_pair):
            if any(len(m) < 2 for _, m, _ in combo):
                continue
            if any(cls == 'cap' and all(system(l) in ('cnt', 'amb') for l in m) for cls, m, _ in combo):
                continue
            h = []; ok = True
            for cls, m, tl in combo:
                Tv, Ev = [], []
                for vs in CNT_SETS:
                    vv = vs if cls == 'cnt' else CAPV
                    Tv.append(value(tl, vv)); Ev.append([value(l, vv) for l in m])
                if any(a is None for a in Tv) or any(x is None for b in Ev for x in b):
                    ok = False; break
                h.append({'cls': cls, 'T': [str(x) for x in Tv], 'E': [[str(x) for x in e] for e in zip(*Ev)],
                          'F': member_feats(m, O, tl)})
            if ok:
                hyps.append(h)
        if hyps:
            out.append({'id': t['id'], 'tier': 1 if t1 else 2, 'hyps': hyps})
    return out


def load_cases(tier=2):
    p = os.path.join(CK, 'cases_t%d.json' % tier)
    if not os.path.exists(p):
        json.dump(build(tier), open(p, 'w'))
    return json.load(open(p))


# ------------------------------------------------------------------ compiled scoring
def compile_cases(C):
    """Float arrays x6 (fractions 1/2, 1/6 become integers)."""
    out = []
    for c in C:
        H = []
        for h in c['hyps']:
            P = []
            for p in h:
                Tm = np.array([float(Fr(x)) * 6 for x in p['T']])
                Em = np.array([[float(Fr(x)) * 6 for x in e] for e in p['E']])
                P.append((Tm, Em, p['F'], p['cls']))
            H.append(P)
        out.append(H)
    return out


def feature_list(C, min_tab=2, prefixes=None):
    cnt = collections.Counter()
    for c in C:
        fs = set()
        for h in c['hyps']:
            for p in h:
                for fl in p['F']:
                    fs.update(fl)
        cnt.update(fs)
    fl = sorted(f for f, n in cnt.items() if n >= min_tab and (prefixes is None or f[:2] in prefixes))
    return fl, cnt


def tablet_tensor(H, fix, F, consts=CONSTS):
    """For one tablet: closure under
       base (scalar), EX1 (F,), EX2 (F,F) union, KEEP1 (F,), C (nc,), EX1+C (F,nc).
    Features not on the tablet behave as no-op (EX) / empty sum (KEEP)."""
    nc = len(consts)
    base = False; ex1 = np.zeros(F, bool); keep = np.zeros(F, bool)
    ex2 = np.zeros((F, F), bool); cc = np.zeros(nc, bool); exc = np.zeros((F, nc), bool)
    loc = sorted({fix[f] for P in H for (_, _, Fl, _) in P for fl in Fl for f in fl if f in fix})
    li = {g: i for i, g in enumerate(loc)}
    L = len(loc)
    cvec = np.array(consts, float) * 6
    # local closure arrays, any over hyps
    lb = False; le1 = np.zeros(L, bool); le2 = np.zeros((L, L), bool); lk = np.zeros(L, bool)
    lc = np.zeros(nc, bool); lec = np.zeros((L, nc), bool)
    keep_nonloc = False
    for P in H:
        ok_b = None; ok_e1 = None; ok_e2 = None; ok_k = None; ok_c = None; ok_ec = None; ok_kn = None
        for (Tm, Em, Fl, cls) in P:
            ne = len(Fl)
            M = np.zeros((ne, L), bool)
            for e, fl in enumerate(Fl):
                for f in fl:
                    if f in li:
                        M[e, li[f]] = True
            S = Em.sum(0)                                  # (nv,)
            ex = M.T.astype(float) @ Em                    # (L, nv) sum of entries with f
            both = np.einsum('ea,eb,ev->abv', M, M, Em) if L else np.zeros((0, 0, len(Tm)))
            g = lambda x: np.isclose(x, Tm, atol=1e-6)
            b = g(S)                                       # (nv,)
            e1 = np.isclose(S[None] - ex, Tm[None], atol=1e-6)            # (L, nv)
            e2 = np.isclose(S[None, None] - ex[:, None] - ex[None, :] + both, Tm[None, None], atol=1e-6)
            k = np.isclose(ex, Tm[None], atol=1e-6)
            kn = np.isclose(0 * S, Tm, atol=1e-6)
            if cls == 'cnt':
                cv = cvec
            else:
                cv = np.zeros(nc)                          # constants only for count pairs
            c_ = np.isclose(S[None] + cv[:, None], Tm[None], atol=1e-6)       # (nc, nv)
            ec = np.isclose(S[None, None] - ex[:, None] + cv[None, :, None], Tm[None, None], atol=1e-6)
            and_ = lambda a, x: x if a is None else (a & x)
            ok_b = and_(ok_b, b); ok_e1 = and_(ok_e1, e1); ok_e2 = and_(ok_e2, e2)
            ok_k = and_(ok_k, k); ok_c = and_(ok_c, c_); ok_ec = and_(ok_ec, ec); ok_kn = and_(ok_kn, kn)
        lb |= bool(ok_b.any()); le1 |= ok_e1.any(-1); le2 |= ok_e2.any(-1); lk |= ok_k.any(-1)
        lc |= ok_c.any(-1); lec |= ok_ec.any(-1); keep_nonloc |= bool(ok_kn.any())
    base = lb
    ex1[:] = lb; keep[:] = keep_nonloc; cc[:] = lc
    exc[:] = lc[None]
    ex2[:] = lb
    loc = np.array(loc, int)
    if L:
        ex1[loc] = le1; keep[loc] = lk; exc[loc] = lec
        # pairs: one local -> single; both local -> pair
        ex2[loc, :] = le1[:, None]
        ex2[:, loc] = le1[None, :]
        ex2[np.ix_(loc, loc)] = le2
    return base, ex1, ex2, keep, cc, exc


class Scorer:
    """Rule x tablet closure matrix (bool). Rules indexed in one flat list."""

    def __init__(self, C, feats, consts=CONSTS, pairs=True):
        self.feats = feats; self.consts = consts
        fix = {f: i for i, f in enumerate(feats)}
        F = len(feats); nc = len(consts)
        iu = np.triu_indices(F, 1) if pairs else (np.array([], int), np.array([], int))
        names = ['BASE'] + ['EX(%s)' % f for f in feats] + ['KEEP(%s)' % f for f in feats] + \
                ['C(%+d)' % c for c in consts] + \
                ['EX(%s)+C(%+d)' % (f, c) for f in feats for c in consts] + \
                ['EX(%s|%s)' % (feats[a], feats[b]) for a, b in zip(*iu)]
        fam = ['BASE'] + ['EX1'] * F + ['KEEP'] * F + ['C'] * nc + ['EXC'] * (F * nc) + ['EX2'] * len(iu[0])
        self.names = names; self.fam = np.array(fam)
        comp = compile_cases(C)
        M = np.zeros((len(names), len(C)), bool)
        for k, H in enumerate(comp):
            b, e1, e2, kp, cc, ec = tablet_tensor(H, fix, F, consts)
            M[:, k] = np.concatenate([[b], e1, kp, cc, ec.ravel(), e2[iu]])
        self.M = M
        cx = {'BASE': 0, 'EX1': 1, 'KEEP': 1, 'C': 1, 'EXC': 2, 'EX2': 2}
        self.cx = np.array([cx[f] for f in fam])

    def fit(self, idx, rng=None, maxtie=200):
        S = self.M[:, idx].sum(1)
        best = S.max()
        cand = np.where(S == best)[0]
        cand = cand[self.cx[cand] == self.cx[cand].min()]
        return cand, int(best - S[0])


def split_eval(sc, ntab, rng, nsplit=100, mask=None):
    """Fit on a random half (max closures, then simplest family; ties averaged),
    held-out gain over BASE on the other half."""
    Mm = sc.M if mask is None else sc.M[mask]
    rows = np.arange(len(sc.names)) if mask is None else np.where(mask)[0]
    cxm = sc.cx[rows]
    gains, ins, picks = [], [], collections.Counter()
    for s in range(nsplit):
        perm = rng.permutation(ntab)
        a, b = perm[:ntab // 2], perm[ntab // 2:]
        Sa = Mm[:, a].sum(1); Sb = Mm[:, b].sum(1)
        best = Sa.max()
        cand = np.where(Sa == best)[0]
        cand = cand[cxm[cand] == cxm[cand].min()]
        b0 = Sb[0]
        ins.append(best - Sa[0])
        gains.append(float(Sb[cand].mean() - b0))
        for c in cand:
            picks[sc.names[rows[c]]] += 1.0 / len(cand)
    return np.array(ins), np.array(gains), picks


def summary(sc, C, rng, nsplit=100, mask=None):
    S = sc.M.sum(1)
    Mm = S if mask is None else np.where(mask, S, -1)
    best = Mm.max()
    cand = np.where(Mm == best)[0]
    cand = cand[sc.cx[cand] == sc.cx[cand].min()]
    ins, gain, picks = split_eval(sc, len(C), rng, nsplit, mask)
    top = np.argsort(-Mm, kind='stable')[:12]
    return {'n': len(C), 'base': int(S[0]), 'best': int(best), 'gain_full': int(best - S[0]),
            'n_best_rules': int(len(cand)), 'best_rules': [sc.names[i] for i in cand[:15]],
            'top': [(sc.names[i], int(S[i])) for i in top],
            'heldout_gain': float(gain.mean()), 'heldout_sd': float(gain.std()),
            'heldout_pos_share': float((gain > 0).mean()),
            'insample_half_gain': float(ins.mean()),
            'top_split_picks': [(k, round(v, 1)) for k, v in picks.most_common(8)],
            'n_rules': len(sc.names)}


# ------------------------------------------------------------------ nulls
def rand_totals(C, rng):
    out = []
    for c in C:
        hs = []
        for h in c['hyps']:
            nh = []
            for p in h:
                r = rng.uniform(0.5, 1.5)
                T = []
                for x in p['T']:
                    t = Fr(x)
                    v = Fr(round(float(t) * r)) if t >= 1 else t * Fr(round(r * 4), 4)
                    T.append(str(max(v, Fr(1))))
                nh.append(dict(p, T=T))
            hs.append(nh)
        out.append(dict(c, hyps=hs))
    return out


SIGNPFX = ('S_', 'B_', 'F_', 'L_', 'K_', 'T_')


def shuffle_labels(C, rng):
    """Random sign labels: the sign-derived features (S_ B_ F_ L_ K_ T_) of every
    member are re-dealt among all members of all tablets (values, positions,
    units, sizes kept). Members are keyed per tablet by their feature tuple so a
    line keeps one label set across hypotheses."""
    keys = []
    for k, c in enumerate(C):
        seen = set()
        for h in c['hyps']:
            for p in h:
                for fl in p['F']:
                    key = (k, tuple(fl))
                    if key not in seen:
                        seen.add(key); keys.append(key)
    labs = [tuple(f for f in key[1] if f[:2] in SIGNPFX) for key in keys]
    perm = [labs[i] for i in rng.permutation(len(labs))]
    mp = {key: perm[i] for i, key in enumerate(keys)}
    out = []
    for k, c in enumerate(C):
        hs = []
        for h in c['hyps']:
            nh = []
            for p in h:
                F = [sorted([f for f in fl if f[:2] not in SIGNPFX] + list(mp[(k, tuple(fl))])) for fl in p['F']]
                nh.append(dict(p, F=F))
            hs.append(nh)
        out.append(dict(c, hyps=hs))
    return out


def shuffle_entries(C, rng):
    """Entries (value + all features) re-dealt among tablets of the same pair class,
    first hypothesis only, member counts kept; position features recomputed."""
    pool = collections.defaultdict(list)
    for c in C:
        for p in c['hyps'][0]:
            for e, fl in zip(p['E'], p['F']):
                pool[p['cls']].append((e, [f for f in fl if not f.startswith('P_')]))
    for v in pool.values():
        rng.shuffle(v)
    it = {k: iter(v) for k, v in pool.items()}
    out = []
    for c in C:
        nh = []
        for p in c['hyps'][0]:
            got = [next(it[p['cls']]) for _ in p['E']]
            n = len(got)
            F = []
            for i, (_, fl) in enumerate(got):
                pf = (['P_first'] if i == 0 else []) + (['P_last'] if i == n - 1 else []) + \
                     (['P_second'] if i == 1 else []) + (['P_penult'] if i == n - 2 else [])
                F.append(sorted(fl + pf))
            nh.append(dict(p, E=[g[0] for g in got], F=F))
        out.append(dict(c, hyps=[nh]))
    return out


def shuffle_order(C, rng):
    """Entry order permuted within each tablet (one permutation per tablet,
    applied to every hypothesis by line identity); position features recomputed."""
    out = []
    for c in C:
        hs = []
        for h in c['hyps']:
            nh = []
            for p in h:
                n = len(p['E']); perm = rng.permutation(n)
                E = [p['E'][i] for i in perm]
                F = []
                for j, i in enumerate(perm):
                    fl = [f for f in p['F'][i] if f not in ('P_first', 'P_last', 'P_second', 'P_penult')]
                    fl += (['P_first'] if j == 0 else []) + (['P_last'] if j == n - 1 else []) + \
                          (['P_second'] if j == 1 else []) + (['P_penult'] if j == n - 2 else [])
                    F.append(sorted(fl))
                nh.append(dict(p, E=E, F=F))
            hs.append(nh)
        out.append(dict(c, hyps=hs))
    return out


def plant(C, pred, const=0):
    """On tablets that close at baseline, totals recomputed leaving out the entries
    with pred(feature list) (and adding const N01 on count pairs)."""
    comp = compile_cases(C)
    out = []; n = 0
    for c, H in zip(C, comp):
        closes = False
        for P in H:
            ok = None
            for (Tm, Em, Fl, cls) in P:
                g = np.isclose(Em.sum(0), Tm)
                ok = g if ok is None else ok & g
            closes |= bool(ok.any())
        hit = any(pred(fl) for h in c['hyps'] for p in h for fl in p['F'])
        if not closes or not (hit or const):
            out.append(c); continue
        n += 1
        hs = []
        for h in c['hyps']:
            nh = []
            for p in h:
                keep = [not pred(fl) for fl in p['F']]
                nv = len(p['T'])
                T = []
                for v in range(nv):
                    s = sum(Fr(e[v]) for e, k in zip(p['E'], keep) if k)
                    if p['cls'] == 'cnt':
                        s += const
                    T.append(str(s))
                nh.append(dict(p, T=T))
            hs.append(nh)
        out.append(dict(c, hyps=hs))
    return out, n
