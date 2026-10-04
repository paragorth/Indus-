"""pe12: THE ENTRIES CARRY CHECK DIGITS.  Shared code.

Hypothesis: some sign slot in each entry is not content but a control mark computed
from the rest of the entry (a checksum, a parity / residue mark, a size-class mark),
added so an auditor could catch copying errors.

Engine.  An entry = (tablet, sign string, numeral group, system, value, line index).
A SLOT picks one sign of the entry (first, last = before-number, first of multi-sign,
last of multi-sign, first sign of the line after the header).  A HYPOTHESIS is a
function x = f(rest of entry) with a small integer range; its rule is the lookup table
P(slot sign | x, system), fitted on training tablets.  Families:
  ARITH  : value mod k, digit counts mod k, digit sums mod k, running (cumulative)
           tablet sum mod k, and tens of thousands of RANDOM sparse modular hashes
           (sum a_i * var_i) mod k                         -> check digits
  SIZE   : magnitude bands, thresholds, top numeral code, fraction flag  -> size class
  CTX    : sign count, other-slot sign, header sign, line index, list length
           (agreement with the rest of the entry, list position)
  SEQ    : previous / next entry's sign in the same slot (runs; reported apart)
  pairs  : random products of two single features (cross-family rules)
Score = held-out gain in bits per entry over the baseline P(sign | tablet type, system).
Tablets are split A (fit) / B (select top hypotheses) / C (test).  Family statistic =
C-gain of the B-best hypothesis (and mean of the B-top-10).  NULL: the whole search is
re-run with the slot sign shuffled among entries of the same (tablet, system) (strict)
or the same (tablet type, system) (loose); this corrects for search size.
"""
import json, os, sys, math
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa: E402
from pe5_common import pe_system, pe_value, VSETS, CSETS, FRACV  # noqa: E402

PEDATA = os.path.join(HERE, '..', 'data')
CKPT = os.path.join(PEDATA, 'pe12_ckpt')
os.makedirs(CKPT, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'

CNT_CODES = ['N01', 'N14', 'N45', 'N34', 'N48']
CAP_CODES = ['N39C', 'N30D', 'N30C', 'N24', 'N39B']
SLOTS = ['FIRST', 'LAST', 'FIRSTm', 'LASTm', 'AFTERHDR']


# ------------------------------------------------------------------ corpus -> entries
def pe_entries(T=None, corpus='PE'):
    """List of entry dicts.  Works for PE and for proto-cuneiform (same json shape)."""
    if T is None:
        T = load()
    out = []
    vs_main, vs_alt = VSETS['D3'], VSETS['SEX']
    cs_main, cs_alt = CSETS['NOT'], CSETS['PCS']
    if corpus == 'PC':
        vs_main, vs_alt = VSETS['SEX'], VSETS['D3']
        cs_main, cs_alt = CSETS['PCS'], CSETS['NOT']
    for t in T:
        lines = t['lines']
        num = [(i, l) for i, l in enumerate(lines) if l['numerals']]
        obv = [x for x in num if x[1]['surface'] == 'obverse']
        off = [x for x in num if x[1]['surface'] != 'obverse']
        tot_idx = off[0][0] if len(off) == 1 and len(obv) >= 2 else None
        hdr = 'none'
        if lines and not lines[0]['numerals']:
            s = [base(x) for x in lines[0]['signs'] if is_sign(x)]
            hdr = s[0] if s else 'none'
        ents = []
        for i, l in num:
            if l['surface'] not in ('obverse', 'reverse') or i == tot_idx:
                continue
            raw = l['signs']
            if corpus == 'PE':
                if any(not is_sign(x) for x in raw):
                    continue
                sg = [base(x) for x in raw]
            else:
                if any(x.lower() == 'x' or x == '...' for x in raw):
                    continue
                sg = [x.split('~')[0] for x in raw]
            if not sg:
                continue
            sysn = pe_system(l['numerals'])
            if sysn is None:
                continue
            v1 = pe_value(l['numerals'], sysn, vs_main, cs_main)
            v2 = pe_value(l['numerals'], sysn, vs_alt, cs_alt)
            if v1 is None or v1 <= 0 or v2 is None:
                continue
            dig = Counter()
            for n, c in l['numerals']:
                dig[c] += n
            ents.append(dict(tab=t['id'], signs=sg, sys=sysn, v=v1, v2=v2, dig=dig,
                             frac=int(any(c in FRACV for c in dig)), line=i, hdr=hdr,
                             first_after_hdr=(hdr != 'none' and not ents)))
        n = len(ents)
        for j, e in enumerate(ents):
            e['idx'] = j
            e['n'] = n
        out.extend(ents)
    return out


def tablet_types(E, nhdr=12):
    hc = Counter()
    seen = set()
    for e in E:
        if e['tab'] not in seen:
            seen.add(e['tab'])
            hc[e['hdr']] += 1
    top = {h for h, _ in hc.most_common(nhdr + 1) if h != 'none'}
    maj = defaultdict(Counter)
    for e in E:
        maj[e['tab']][e['sys']] += 1
    ty = {}
    for e in E:
        h = e['hdr'] if e['hdr'] in top else ('none' if e['hdr'] == 'none' else 'oth')
        ty[e['tab']] = h + '/' + maj[e['tab']].most_common(1)[0][0]
    return ty


def slot_entries(E, slot):
    """(entries, slot sign per entry, other-sign per entry)."""
    keep, y, oth = [], [], []
    for e in E:
        s = e['signs']
        if slot == 'FIRST':
            k, o = s[0], (s[-1] if len(s) > 1 else 'NONE')
        elif slot == 'LAST':
            k, o = s[-1], (s[0] if len(s) > 1 else 'NONE')
        elif slot == 'FIRSTm':
            if len(s) < 2:
                continue
            k, o = s[0], s[-1]
        elif slot == 'LASTm':
            if len(s) < 2:
                continue
            k, o = s[-1], s[0]
        elif slot == 'AFTERHDR':
            if not e['first_after_hdr']:
                continue
            k, o = s[0], (s[-1] if len(s) > 1 else 'NONE')
        keep.append(e); y.append(k); oth.append(o)
    return keep, y, oth


# ------------------------------------------------------------------ features
def _code(vals, cap=40):
    """Map arbitrary hashable values to ints, rare values pooled into one code."""
    if isinstance(vals, np.ndarray) and vals.dtype.kind in 'iub':
        u, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
        rank = np.empty(len(u), dtype=np.int64)
        o = np.argsort(-cnt, kind='stable')
        rank[o] = np.arange(len(u))
        return np.minimum(rank[inv], cap - 1).astype(np.int64), min(cap, len(u))
    c = Counter(vals)
    top = [v for v, _ in c.most_common(cap - 1)]
    m = {v: i for i, v in enumerate(top)}
    return np.array([m.get(v, cap - 1) for v in vals], dtype=np.int64), min(cap, len(top) + 1)


def base_vars(ents):
    """Integer variables for random modular hashes (name -> int array)."""
    V = {}
    for c in CNT_CODES + CAP_CODES:
        V['c_' + c] = np.array([e['dig'].get(c, 0) for e in ents])
    V['c_frac'] = np.array([sum(e['dig'].get(c, 0) for c in FRACV) for e in ents])
    V['ntok'] = np.array([sum(e['dig'].values()) for e in ents])
    V['ncode'] = np.array([len(e['dig']) for e in ents])
    V['vint'] = np.array([int(e['v']) for e in ents])
    V['v2int'] = np.array([int(e['v2']) for e in ents])
    V['nsg'] = np.array([len(e['signs']) for e in ents])
    V['idx'] = np.array([e['idx'] for e in ents])
    V['ridx'] = np.array([e['n'] - 1 - e['idx'] for e in ents])
    V['nent'] = np.array([e['n'] for e in ents])
    # running checksums on the tablet (needs all entries of the tablet, slot-independent)
    return V


def add_running(V, ents, allE):
    """Cumulative tablet sums up to (excl./incl.) the entry, and tablet sum, from ALL entries."""
    bytab = defaultdict(list)
    for e in allE:
        bytab[e['tab']].append(e)
    cum_ex, tot = {}, {}
    for t, L in bytab.items():
        s = 0
        for e in sorted(L, key=lambda z: z['idx']):
            cum_ex[(t, e['idx'])] = s
            s += int(e['v'])
        tot[t] = s
    V['cum_ex'] = np.array([cum_ex[(e['tab'], e['idx'])] for e in ents])
    V['cum_in'] = V['cum_ex'] + V['vint']
    V['tabsum'] = np.array([tot[e['tab']] for e in ents])
    V['prev_v'] = np.array([cum_ex[(e['tab'], e['idx'])] - cum_ex.get((e['tab'], e['idx'] - 1), 0)
                            if e['idx'] > 0 else 0 for e in ents])
    return V


def single_features(ents, oth, V, allE, slot_of_entry):
    """name -> (codes, card, family)."""
    F = {}
    def put(name, vals, fam, cap=40):
        c, k = _code(vals if isinstance(vals, np.ndarray) else list(vals), cap)
        if k >= 2:
            F[name] = (c, k, fam)
    # ARITH
    for var in ['vint', 'v2int', 'ntok', 'c_N01', 'c_N14', 'c_N45', 'c_N34', 'c_N39B', 'c_N24',
                'c_N30C', 'c_N30D', 'cum_ex', 'cum_in', 'tabsum', 'prev_v']:
        for k in range(2, 14):
            put('%s%%%d' % (var, k), V[var] % k, 'ARITH')
    dsum = sum(V['c_' + c] for c in CNT_CODES + CAP_CODES)
    for k in range(2, 14):
        put('digsum%%%d' % k, dsum % k, 'ARITH')
        put('lastdig%%%d' % k, (V['vint'] % 10) % k, 'ARITH')
    # SIZE
    lv = np.log2(np.maximum(np.array([e['v'] for e in ents]), 0.01))
    put('log2band', np.floor(lv).astype(int), 'SIZE')
    put('log10band', np.floor(lv / math.log2(10)).astype(int), 'SIZE')
    put('v_is1', (V['vint'] == 1) & (V['c_frac'] == 0), 'SIZE')
    put('frac', V['c_frac'] > 0, 'SIZE')
    put('ncode', V['ncode'], 'SIZE')
    put('ntokband', np.minimum(V['ntok'], 12), 'SIZE')
    top = []
    order = ['N48', 'N34', 'N45', 'N14', 'N01', 'N39B', 'N24', 'N30C', 'N30D', 'N39C']
    for e in ents:
        top.append(next((c for c in order if e['dig'].get(c)), 'frac'))
    put('topcode', top, 'SIZE')
    for q in [1, 2, 3, 5, 10, 20, 30, 50, 100, 300]:
        put('v>%d' % q, np.array([e['v'] for e in ents]) > q, 'SIZE')
    # CTX
    put('nsg', np.minimum(V['nsg'], 6), 'CTX')
    put('othsign', oth, 'CTX', cap=30)
    put('hdr', [e['hdr'] for e in ents], 'CTX', cap=20)
    put('idx', np.minimum(V['idx'], 15), 'CTX')
    put('ridx', np.minimum(V['ridx'], 15), 'CTX')
    put('isfirst', V['idx'] == 0, 'CTX')
    put('islast', V['ridx'] == 0, 'CTX')
    put('nentband', np.minimum(V['nent'], 20), 'CTX')
    put('idx%2', V['idx'] % 2, 'CTX')
    put('idx%3', V['idx'] % 3, 'CTX')
    # SEQ: neighbour entries' sign in the same slot
    pos = {(e['tab'], e['idx']): s for e, s in zip(allE, slot_of_entry)}
    put('prevsign', [pos.get((e['tab'], e['idx'] - 1), 'NONE') for e in ents], 'SEQ', cap=30)
    put('nextsign', [pos.get((e['tab'], e['idx'] + 1), 'NONE') for e in ents], 'SEQ', cap=30)
    return F


def random_hashes(V, n, rng, vars_=None):
    """n random sparse modular hashes: (sum a_i var_i + b) mod k."""
    # numeral-only variables (positional / sign-count variables belong to CTX, not ARITH)
    vars_ = vars_ or ['c_N01', 'c_N14', 'c_N45', 'c_N34', 'c_N39B', 'c_N24', 'c_N30C', 'c_N30D',
                      'c_frac', 'ntok', 'vint', 'v2int', 'cum_ex', 'tabsum', 'prev_v']
    M = np.stack([V[v] for v in vars_]).astype(np.int64)
    H, desc = [], []
    for _ in range(n):
        k = int(rng.integers(2, 14))
        m = int(rng.integers(1, 4))
        idx = rng.choice(len(vars_), m, replace=False)
        a = rng.integers(1, k, size=m) if k > 2 else np.ones(m, dtype=int)
        h = (a @ M[idx]) % k
        H.append(h)
        desc.append('(' + '+'.join('%d*%s' % (ai, vars_[i]) for ai, i in zip(a, idx)) + ')%%%d' % k)
    return np.stack(H), desc


# ------------------------------------------------------------------ scorer
def baseline_probs(y, Ky, strat, train, alpha=2.0):
    """P(y | stratum) fitted on train, smoothed to global; returns (N x Ky) ... as row probs."""
    g = np.bincount(y[train], minlength=Ky) + 0.5
    g = g / g.sum()
    S = strat.max() + 1
    cnt = np.zeros((S, Ky))
    np.add.at(cnt, (strat[train], y[train]), 1)
    P = (cnt + alpha * g) / (cnt.sum(1, keepdims=True) + alpha)
    return P  # S x Ky


def score_batch(X, Kx, y, Ky, sysv, P0, strat, fit, ev, alpha=4.0):
    """X: H x N codes (< Kx).  Lookup P(y | x, sys) fitted on `fit`, mixed into baseline
    P0[strat]; returns mean held-out gain (bits/entry) on `ev` for each of H hypotheses."""
    H = X.shape[0]
    Kc = Kx * 2
    xc = X * 2 + sysv[None, :]                       # condition on system as well
    f = np.asarray(fit); v = np.asarray(ev)
    off = (np.arange(H)[:, None] * Kc)
    nxy = np.bincount(((off + xc[:, f]) * Ky + y[f][None, :]).ravel(),
                      minlength=H * Kc * Ky).reshape(H, Kc, Ky)
    nx = nxy.sum(2)
    xv = xc[:, v]
    yv = y[v]
    pb = P0[strat[v], yv]                            # baseline prob of the true sign
    num = nxy[np.arange(H)[:, None], xv, yv[None, :]] + alpha * pb[None, :]
    den = nx[np.arange(H)[:, None], xv] + alpha
    return (np.log2(num / den) - np.log2(pb)[None, :]).mean(1)


def split_tabs(tabs, rng):
    t = np.array(sorted(tabs))
    rng.shuffle(t)
    n = len(t)
    return set(t[:n // 2]), set(t[n // 2:3 * n // 4]), set(t[3 * n // 4:])


def shuffle_within(y, groups, rng):
    y2 = y.copy()
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    for g, L in idx.items():
        L = np.array(L)
        y2[L] = y[rng.permutation(L)]
    return y2
