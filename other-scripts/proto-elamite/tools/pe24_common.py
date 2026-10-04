"""pe24 shared code: THE ~a SIGNS ARE RED INK.

Hypothesis: sign variants (~a, ~b, @g ...) or particular signs flag a different
accounting state (owed vs present, dead vs alive), so their entries count AGAINST
the written total (-1), or not at all (0), instead of +1.

Cases (fixed before any fitting, cleaning copied from attack_arith.py):
  tablets with obverse entries and reverse total line(s), no lacuna, no broken
  state line, no gap in line numbers, every numeral readable; edge marks and
  numeral-before-sign annotations dropped. Structures 'single' (1 reverse line =
  total of the obverse) and 'persys' (several reverse lines of different systems,
  each the total of its own system). Capacity totals allow the 3 member variants
  of attack_arith. Values: count system decimal (N14 10, N45 100, N34 300, N48
  3000) or sexagesimal (N14 10, N34 60, N45 600, N48 3600), chosen per tablet,
  fractions 1/2 or 1/6 of N01 (4 value sets per count tablet); capacity a-priori
  bundling chain (one set). A tablet CLOSES if for some hypothesis and some value
  set every (total, members) pair satisfies  total == sum_e c_e * v_e  exactly.

Entry coefficient: c_e = product over the entry's features f of c_f (default +1);
c_f in {+1, -1, 0} (or a ratio set in cycle 2). Features:
  'mk'  variant markers on the entry's signs ('~a', '~b', '@g', ...)
  'sg'  sign tokens themselves (plain and variant, e.g. 'M362', 'M362~a')
"""
import collections, itertools, json, os, re, sys
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe24_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa: E402

CAPSET = {'N39B', 'N30C', 'N24', 'N30D', 'N39C', 'N39A', 'N28', 'N29B', 'N39N'}
AMBIG = {'N01', 'N14', 'N45', 'N34', 'N48'}
COUNTED = {'M263', 'M346', 'M264', 'M003', 'M376', 'M032', 'M373', 'M102', 'M362', 'M317', 'M149', 'M046'}
CAPV = {"N39C": 1, "N30D": 2, "N30C": 4, "N24": 12, "N39B": 24, "N01": 120, "N14": 720,
        "N45": 7200, "N34": 21600, "N48": 216000, "N51": 120, "N08": Fr(1, 2), "N08A": Fr(1, 2),
        "N8B": Fr(1, 2), "N02": Fr(1, 2)}
CNT_SETS = []
for big in ({'N14': 10, 'N45': 100, 'N34': 300, 'N48': 3000},
            {'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600}):
    for fr in (Fr(1, 2), Fr(1, 6)):
        d = {'N01': 1, 'N51': 120, 'N08': fr, 'N08A': fr, 'N8B': fr, 'N02': fr}
        d.update(big)
        CNT_SETS.append(d)
MARK = re.compile(r'(~[A-Za-z0-9]+|@[a-z]+)')


def system(l):
    codes = {c for _, c in l['numerals']}
    if any('@' in c for c in codes):
        return 'cap@'
    if codes & CAPSET:
        return 'cap'
    if codes <= AMBIG:
        return 'amb'
    return 'cnt'


def final_sign(l):
    s = [base(x) for x in l['signs'] if is_sign(x)]
    return s[-1] if s else None


def line_clean(l):
    tail = l['raw'].split(',')[-1]
    return (not l['lacuna'] and '...' not in l['raw'] and not re.search(r'[\[?]', tail)
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
    return any(is_sign(s) or s == 'x' for s in l['signs']) and not l.get('has_comma', True)


def value(l, vs):
    tot = Fr(0)
    for n, c in l['numerals']:
        c = c.split('@')[0]
        if c not in vs:
            return None
        tot += n * Fr(vs[c])
    return tot


def features(l):
    toks = [s for s in l['signs'] if is_sign(s)]
    mk = sorted({m for s in toks for m in MARK.findall(s)})
    sg = sorted(set(toks))
    return mk, sg


def build_pe():
    """List of tablets: {'id', 'cls', 'hyps': [[pair,...],...]} with pair =
    {'T': [value per value set], 'E': [[value per value set] per member],
     'mk': [[markers] per member], 'sg': [[signs] per member]}. Value sets: the
    count sets for count pairs, the capacity set for capacity pairs; a tablet with
    pairs of both kinds uses the product (cnt set index only matters for cnt)."""
    T = load()
    broken = set(); cur = None
    for raw in open(os.path.join(DATA, 'pe_raw.atf'), encoding='utf-8'):
        if raw.startswith('&'):
            cur = raw[1:].split()[0]
        elif raw.startswith('$') and re.search(r'broken|missing|not given', raw):
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
        if any(l['lacuna'] or '...' in l['raw'] for l in t['lines']):
            continue
        if not all(line_clean(l) for l in lines):
            continue
        if len(R) == 1:
            pairs = [(R[0], O)]
        else:
            cl = [tot_class(r, O) for r in R]
            if len(set(cl)) != len(cl):
                continue
            pairs = [(r, O) for r in R]
        per_pair = []
        for tl, ent in pairs:
            cls = tot_class(tl, ent)
            if cls == 'cnt' and all(system(l) in ('cnt', 'amb') for l in ent) and False:
                pass
            per_pair.append([(cls, m, tl) for m in member_variants(ent, cls)])
        hyps = []
        for combo in itertools.product(*per_pair):
            if any(len(m) < 2 for _, m, _ in combo):
                continue
            # conversion cases (capacity total, count-only entries) are not totals
            if any(cls == 'cap' and all(system(l) in ('cnt', 'amb') for l in m) for cls, m, _ in combo):
                continue
            h = []
            ok = True
            for cls, m, tl in combo:
                sets = CNT_SETS if cls == 'cnt' else [CAPV]
                Tv, Ev = [], []
                for vs in (CNT_SETS if True else sets):
                    vv = vs if cls == 'cnt' else CAPV
                    a = value(tl, vv); b = [value(l, vv) for l in m]
                    Tv.append(a); Ev.append(b)
                if any(a is None for a in Tv) or any(x is None for b in Ev for x in b):
                    ok = False; break
                f = [features(l) for l in m]
                h.append({'cls': cls, 'T': Tv, 'E': [list(x) for x in zip(*Ev)],
                          'mk': [x[0] for x in f], 'sg': [x[1] for x in f]})
            if ok:
                hyps.append(h)
        if hyps:
            out.append({'id': t['id'], 'hyps': hyps,
                        'cls': '+'.join(sorted({p['cls'] for p in hyps[0]}))})
    return out


# ------------------------------------------------------------- compiled evaluator
class Problem:
    """Compiled cases for fast scoring. Each case = list of hypotheses, each a
    list of pairs; a pair has target T[v] and member values E[e][v] (as floats
    scaled to integers by a common denominator) and the feature ids of each member."""

    def __init__(self, cases, kind='mk', min_tab=1):
        self.ids = [c['id'] for c in cases]
        self.kind = kind
        cnt = collections.Counter()
        for c in cases:
            fs = set()
            for h in c['hyps']:
                for p in h:
                    for f in p[kind]:
                        fs.update(f)
            cnt.update(fs)
        self.feats = sorted(f for f, n in cnt.items() if n >= min_tab)
        self.fix = {f: i for i, f in enumerate(self.feats)}
        self.ntab_feat = np.array([cnt[f] for f in self.feats])
        self.cases = []
        self.tabs_of_feat = collections.defaultdict(set)
        for k, c in enumerate(cases):
            H = []
            for h in c['hyps']:
                P = []
                for p in h:
                    nv = len(p['T'])
                    Tm = np.array([float(x) * 6 for x in p['T']])        # x6: fractions 1/2, 1/6
                    Em = np.array([[float(x) * 6 for x in e] for e in p['E']])  # (n_e, nv)
                    F = [[self.fix[f] for f in fl if f in self.fix] for fl in p[kind]]
                    for fl in F:
                        for f in fl:
                            self.tabs_of_feat[f].add(k)
                    P.append((Tm, Em, F))
                H.append(P)
            self.cases.append(H)

    def coefs(self, P, c):
        return np.array([np.prod([c[f] for f in fl]) if fl else 1.0 for fl in P[2]])

    def closes(self, k, c):
        for H in self.cases[k]:
            ok = None
            for P in H:
                ce = self.coefs(P, c)
                s = ce @ P[1]
                good = np.abs(s - P[0]) < 1e-6
                ok = good if ok is None else (ok & good)
            if ok is not None and ok.any():
                return True
        return False

    def score(self, c, tabs=None):
        tabs = range(len(self.cases)) if tabs is None else tabs
        return sum(self.closes(k, c) for k in tabs)

    def baseline(self):
        return np.ones(len(self.feats))


def with_totals(cases, newT):
    """Copy of cases with pair totals replaced: newT[k] = function(pair) -> list of T."""
    out = []
    for k, c in enumerate(cases):
        hs = []
        for h in c['hyps']:
            hs.append([dict(p, T=newT(k, p)) for p in h])
        out.append(dict(c, hyps=hs))
    return out


def shuffle_markers(cases, rng):
    """Variant markers re-dealt among all sign tokens of all members (counts kept).
    Each member's marker set = the markers of its re-dealt tokens; signs 'sg' get
    the re-dealt marker appended to their base."""
    toks = []
    for c in cases:
        for h in c['hyps'][:1]:
            for p in h:
                for s in p['sg']:
                    for x in s:
                        toks.append(x)
    marks = [''.join(MARK.findall(x)) for x in toks]
    perm = list(marks); rng.shuffle(perm)
    # deterministic map per (tablet, member position, token index) via a global counter
    it = iter(perm)
    out = []
    for c in cases:
        cache = {}
        hs = []
        for hi, h in enumerate(c['hyps']):
            nh = []
            for p in h:
                mk, sg = [], []
                for s in p['sg']:
                    key = tuple(s)
                    if key not in cache:
                        news = []
                        for x in s:
                            b = MARK.sub('', x) if not x.startswith('|') else x
                            try:
                                m = next(it)
                            except StopIteration:
                                m = ''
                            news.append(b + m if not x.startswith('|') else x)
                        cache[key] = news
                    ns = cache[key]
                    sg.append(sorted(set(ns)))
                    mk.append(sorted({m for x in ns for m in MARK.findall(x)}))
                nh.append(dict(p, mk=mk, sg=sg))
            hs.append(nh)
        out.append(dict(c, hyps=hs))
    return out


# ------------------------------------------------------------- lookup-table search
VALS3 = (1.0, -1.0, 0.0)


def tables(P, feats, vals=VALS3):
    """For each tablet: list of relevant feature positions (indices into feats)
    and closure table over all len(vals)^r configurations (other features +1).
    Vectorised: member coefficients for all configurations at once."""
    pos = {f: j for j, f in enumerate(feats)}
    vals = np.array(vals, float)
    nv = len(vals)
    out = []
    for k in range(len(P.cases)):
        rel = sorted({pos[P.feats[f]] for H in P.cases[k] for Pp in H for fl in Pp[2] for f in fl
                      if P.feats[f] in pos})
        rix = {P.fix[feats[j]]: i for i, j in enumerate(rel)}
        r = len(rel)
        cfg = np.array(list(itertools.product(range(nv), repeat=r)), int)[:, ::-1] if r else np.zeros((1, 0), int)
        V = vals[cfg]
        tab = np.zeros(len(cfg), bool)
        for H in P.cases[k]:
            ok = None
            for (Tm, Em, F) in H:
                co = np.ones((len(cfg), len(F)))
                for e, fl in enumerate(F):
                    for f in fl:
                        if f in rix:
                            co[:, e] *= V[:, rix[f]]
                good = np.abs(co @ Em - Tm[None, :]) < 1e-6        # (ncfg, nvalset)
                ok = good if ok is None else (ok & good)
            tab |= ok.any(1)
        out.append((np.array(rel, int), tab))
    return out


def score_all(A, tabs, idx, nv=3):
    """A: (n_assign, n_feat) ints in 0..nv-1; returns (n_assign, len(idx)) closure."""
    out = np.zeros((A.shape[0], len(idx)), bool)
    for j, k in enumerate(idx):
        rel, tab = tabs[k]
        if len(rel) == 0:
            out[:, j] = tab[0]
            continue
        code = np.zeros(A.shape[0], np.int64)
        for i, r in enumerate(rel):
            code += A[:, r].astype(np.int64) * nv ** i
        out[:, j] = tab[code]
    return out


def all_assign(n, nv=3):
    return np.array(list(itertools.product(range(nv), repeat=n)), np.int8)


def split_eval(tabs, A, ntab, rng, nsplit=50, maxtie=300):
    """Fit on random half (max closures, then fewest non-+1), held-out gain over
    the all-+1 baseline (A row 0 must be all-+1), averaged over tied fits."""
    nonp = (A != 0).sum(1)
    ins, gain, picks = [], [], collections.Counter()
    for s in range(nsplit):
        perm = rng.permutation(ntab)
        a, b = perm[:ntab // 2], perm[ntab // 2:]
        Sa = score_all(A, tabs, a).sum(1)
        Sb = score_all(A, tabs, b).sum(1)
        best = Sa.max()
        cand = np.where(Sa == best)[0]
        cand = cand[nonp[cand] == nonp[cand].min()]
        if len(cand) > maxtie:
            cand = rng.choice(cand, maxtie, replace=False)
        ins.append(best - Sa[0])
        gain.append(float(Sb[cand].mean() - Sb[0]))
        for c in cand:
            picks[tuple(A[c])] += 1.0 / len(cand)
    return np.array(ins), np.array(gain), picks
