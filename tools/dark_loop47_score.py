"""Loop 47: re-score the loop 36 museum seals after re-reading (loop47_newfinds_v2.csv) with the loop 42 rules, and test the
'copy of a published seal' hypothesis (nearest-neighbour edit distance to the whole corpus vs the leave-one-out distribution
for excavated seals of the same length).

Usage: python3 tools/dark_loop47_score.py [seq_raw|seq_strong|seq_all]
Imports tools/dark_loop42.py (Rules, FIT, HELD, MM) at the chosen level. Sources: data/raw/inscriptions.csv (S-DARK-23 caution),
data/derived/dark/loop47_newfinds_v2.csv (column signs_W, '?' = unread, confidence_per_sign A/B/C).
"""
import sys, csv, collections, random, importlib, math
LEVEL = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
sys.argv = ['dark_loop42.py', '0', LEVEL]          # cycle '0' = no cycle work at import
sys.path.insert(0, '/home/user/Indus-/tools')
L42 = importlib.import_module('dark_loop42')
ROOT = '/home/user/Indus-/'
R = L42.Rules(L42.FIT); MM = L42.MM; T = L42.T; FIT = L42.FIT; HELD = L42.HELD
rng = random.Random(47)

def parse(row, strict):
    signs = row['signs_W'].split('-'); conf = (row['confidence_per_sign'] or '').split('-'); o = []
    for i, s in enumerate(signs):
        c = conf[i] if i < len(conf) else '?'
        if s in ('?', '') or '/' in s or not s.strip().isdigit(): o.append(None); continue
        if strict and c not in ('A', 'B'): o.append(None); continue
        o.append(MM(int(s)))
    return o

def viol_gapped(seq_with_none):
    conc = [x for x in seq_with_none if x is not None]
    if len(conc) < 2: return {}
    V = R.violations(conc, complete=False)
    real_adj = set((a, b) for a, b in zip(seq_with_none, seq_with_none[1:]) if a is not None and b is not None)
    if 'R4_BIGRAM' in V:
        V['R4_BIGRAM'] = [p for p in V['R4_BIGRAM'] if p in real_adj]
        if not V['R4_BIGRAM']: del V['R4_BIGRAM']
    if 'R7_FROZEN' in V and None in seq_with_none: del V['R7_FROZEN']
    return V

def lev(a, b):
    """sign-level edit distance; None in a is a wildcard (matches any sign at cost 0)"""
    n, m = len(a), len(b); D = list(range(m + 1))
    for i in range(1, n + 1):
        prev = D[:]; D[0] = i
        for j in range(1, m + 1):
            sub = 0 if (a[i-1] is None or a[i-1] == b[j-1]) else 1
            D[j] = min(prev[j] + 1, D[j-1] + 1, prev[j-1] + sub)
    return D[m]

def nn(seq, pool, skip=None):
    best = (99, None)
    for t in pool:
        if t is skip: continue
        d = min(lev(seq, t['seq']), lev(seq, t['seq'][::-1]))
        if d < best[0]: best = (d, t)
    return best

rows = list(csv.DictReader(open(ROOT + 'data/derived/dark/loop47_newfinds_v2.csv')))
rows = [r for r in rows if r['signs_W'] and 'not transcribed' not in r['signs_W']]
out = [f'# loop 47 re-score  level={LEVEL}  (rules fitted on FIT = MD+H register texts, n={len(FIT)}; T={len(T)} corpus texts)']
out.append(f'R1 repeatable (exempt) signs at this level: {sorted(R.repeatable)}')

# in-sample per-rule rates for reference (texts >= 3 signs)
def rates(ts):
    s3 = [t for t in ts if len(t['seq']) >= 3]; c = collections.Counter()
    for t in s3:
        for k in R.violations(t['seq'], t['complete']): c[k] += 1
    return {k: c[k] / len(s3) for k in L42.RULES}, len(s3)
rf, nf = rates(FIT); rh, nh = rates(HELD)
out.append('reference per-rule rates (>=3 signs): FIT in-sample n=%d ' % nf + ' '.join(f'{k}={rf[k]:.3f}' for k in L42.RULES))
out.append('                                      HELD-out   n=%d ' % nh + ' '.join(f'{k}={rh[k]:.3f}' for k in L42.RULES))

out.append('')
out.append('object            mode    reading (merged)                n  score  rules')
flag = collections.Counter()
for r in rows:
    for mode in ('strict', 'lenient'):
        s = parse(r, mode == 'strict'); conc = [x for x in s if x is not None]
        if len(conc) < 2: continue
        V = viol_gapped(s)
        if len(conc) >= 3 and r['tier'].startswith('B'): flag[(mode, len(V) >= 1)] += 1; flag[(mode, 'R1', 'R1_REPEAT' in V)] += 1
        out.append(f"  {r['id']:16s} {mode:7s} {'-'.join('?' if x is None else str(x) for x in s):30s} {len(conc)}  {len(V)}      " + '; '.join(f'{k}:{v}' for k, v in V.items()))
for mode in ('strict', 'lenient'):
    n = flag[(mode, True)] + flag[(mode, False)]
    out.append(f'tier-B museum pieces >=3 signs ({mode}): any rule {flag[(mode, True)]}/{n}; R1 repeat {flag[(mode, "R1", True)]}/{n}')

# ---- copy-of-a-published-seal test: NN edit distance vs leave-one-out distribution for excavated seals
out.append('')
out.append('nearest corpus text (sign-level edit distance, either orientation; ? = wildcard) and the leave-one-out reference for excavated')
out.append('FIT seals of the same length (fraction of excavated seals whose nearest other corpus text is at least as close):')
seals_fit = [t for t in FIT if t['tc'] == 'SEAL' and len(t['seq']) >= 3]
ref = collections.defaultdict(list)
sample = rng.sample(seals_fit, min(300, len(seals_fit)))
for t in sample:
    d, _ = nn(t['seq'], T, skip=t)
    ref[min(len(t['seq']), 6)].append(d)
for r in rows:
    s = parse(r, False); conc = [x for x in s if x is not None]
    if len(conc) < 3: continue
    d, t = nn(s, T)
    L = min(len(s), 6); dist = ref[L]
    frac = sum(x <= d for x in dist) / len(dist) if dist else float('nan')
    out.append(f"  {r['id']:16s} {'-'.join('?' if x is None else str(x) for x in s):28s} NN d={d} -> {t['cisi'] or t['id']} {t['site']} {t['type']} {'-'.join(map(str, t['seq']))}   | excavated seals of length {L}: median NN d={sorted(dist)[len(dist)//2]}, P(d_exc <= {d}) = {frac:.2f}")
out.append('  (a modern copy of a published seal would sit at d = 0-1; genuine excavated seals of 4-6 signs sit at d = 1-3)')
path = ROOT + f'data/derived/dark/loop47_score_{LEVEL}.txt'
open(path, 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
