"""S-DARK-24 cycle 2: (1) per-sign reliability (agreement rate between Wells and Mahadevan per sign, binomial
vs the corpus rate); (2) are the confused pairs and Mahadevan's lumps grammatically interchangeable? Context
sharing = mean cosine of left/right neighbour profiles over the whole Wells corpus (seq_raw) vs 300 frequency-
matched partners, calibrated on the S268 strong merges; site and medium split of the two forms."""
import json, collections, random, math
import numpy as np
R = '/home/user/Indus-/'; OUT = R + 'data/derived/dark/'
rng = random.Random(24)
corpus = json.load(open(R + 'data/derived/merged-corpus-canonical.json'))
corr = {int(k): set(v) for k, v in json.load(open(OUT + 'loop24_corr.json')).items()}
levels = json.load(open(R + 'data/derived/sign_allographs_levels.json'))
P = [r for r in json.load(open(OUT + 'loop24_pairs.json')) if r['accepted']]
clean = [r for r in P if r['cost'] <= 1.0]
L = []
def say(s): print(s); L.append(s)

# ---------------- (1) per-sign reliability ----------------
def outcomes(pairs):
    W = collections.defaultdict(collections.Counter); M = collections.defaultdict(collections.Counter)
    partners = collections.defaultdict(collections.Counter)
    for r in pairs:
        ops = [o for o in r['ops'] if o[0] != 'L']
        for o in ops:
            k, w, m = o[0], o[1], o[2]
            if k == 'S':
                if w and m:
                    if w not in corr: W[w]['unbr'] += 1
                    elif m in corr[w]: W[w]['agree'] += 1; M[m]['agree'] += 1
                    else: W[w]['sub'] += 1; M[m]['sub'] += 1; partners[w][m] += 1
                elif w and not m: W[w]['Mlost'] += 1
                elif m and not w: M[m]['Wlost'] += 1
            elif k == 'W' and w: W[w]['Wonly'] += 1
            elif k == 'M' and m: M[m]['Monly'] += 1
    return W, M, partners
W, M, partners = outcomes(clean)
tot = collections.Counter()
for c in W.values(): tot.update(c)
base = tot['agree'] / (tot['agree'] + tot['sub'] + tot['Wonly'])
say(f'(1) PER-SIGN RELIABILITY on {len(clean)} clean pairs (cost <= 1.0). Wells-side outcome per token: agree / substituted / Wells-only. Corpus rate agree = {base:.4f}')
def binom_low(k, n, p):
    # P(X <= k)
    return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k + 1))
rows = []
for w, c in W.items():
    n = c['agree'] + c['sub'] + c['Wonly']
    if n < 15: continue
    pv = binom_low(c['agree'], n, base)
    rows.append((pv, w, n, c['agree'] / n, c['sub'], c['Wonly'], c['Mlost'], dict(partners[w].most_common(3))))
rows.sort()
say(f'  signs with >= 15 aligned tokens: {len(rows)}; Bonferroni threshold {0.05/len(rows):.5f}')
say('  least reliable Wells signs (binomial P of agreement count <= observed, given corpus rate):')
for pv, w, n, rate, sub, wo, ml, pt in rows[:25]:
    say(f'    W{w}: agree {rate:.3f} of {n} (sub {sub}, Wells-only {wo}, Mahadevan-lost {ml}) P={pv:.4f} partners {pt}')
sig = [r for r in rows if r[0] < 0.05 / len(rows)]
say(f'  significantly unreliable after Bonferroni: ' + ', '.join(f'W{r[1]} ({r[3]:.2f}, n={r[2]})' for r in sig))
say('  most reliable high-frequency signs (n >= 100): ' + ', '.join(f'W{w} {rate:.3f}/{n}' for pv, w, n, rate, *_ in sorted(rows, key=lambda z: -z[2]) if n >= 100))
# frequency dependence
xs = [math.log(r[2]) for r in rows]; ys = [r[3] for r in rows]
from scipy.stats import spearmanr
rho, pr = spearmanr(xs, ys)
say(f'  agreement rate vs log token count over {len(rows)} signs: Spearman rho {rho:+.3f} (P={pr:.3f})')
# M side
totm = collections.Counter()
for c in M.values(): totm.update(c)
basem = totm['agree'] / (totm['agree'] + totm['sub'] + totm['Monly'])
rowsm = []
for m, c in M.items():
    n = c['agree'] + c['sub'] + c['Monly']
    if n >= 15: rowsm.append((binom_low(c['agree'], n, basem), m, n, c['agree'] / n, c['sub'], c['Monly']))
rowsm.sort()
say(f'  Mahadevan-side (M sign tokens, agree rate {basem:.4f}); least reliable: ' + '; '.join(f'M{m} {rate:.2f}/{n} (sub {s}, M-only {mo}) P={pv:.3f}' for pv, m, n, rate, s, mo in rowsm[:12]))
json.dump([dict(w=w, n=n, agree=rate, sub=sub, wonly=wo, mlost=ml, P=pv) for pv, w, n, rate, sub, wo, ml, pt in rows],
          open(OUT + 'loop24_sign_reliability.json', 'w'))

# ---------------- (2) interchangeability ----------------
seqs = [tuple(x['seq_raw']) for x in corpus]
sc = collections.Counter(s for q in seqs for s in q)
left = collections.defaultdict(collections.Counter); right = collections.defaultdict(collections.Counter)
site_of = collections.defaultdict(collections.Counter); med_of = collections.defaultdict(collections.Counter)
for x in corpus:
    q = x['seq_raw']
    for i, s in enumerate(q):
        if i > 0: left[s][q[i-1]] += 1
        if i < len(q) - 1: right[s][q[i+1]] += 1
        site_of[s][x['site']] += 1; med_of[s][x['type'].split(':')[0]] += 1
def cos(a, b):
    ks = set(a) | set(b)
    if not ks: return float('nan')
    na = math.sqrt(sum(v*v for v in a.values())); nb = math.sqrt(sum(v*v for v in b.values()))
    if not na or not nb: return float('nan')
    return sum(a[k] * b[k] for k in ks) / (na * nb)
def ctx(a, b):
    v = [cos(left[a], left[b]), cos(right[a], right[b])]
    v = [x for x in v if not math.isnan(x)]
    return sum(v) / len(v) if v else float('nan')
def split(a, b):
    return 1 - cos(site_of[a], site_of[b]), 1 - cos(med_of[a], med_of[b])
signs_by_freq = sorted(sc, key=lambda s: sc[s])
def freq_matched(b, exclude):
    f = sc[b]
    cands = [s for s in sc if s not in exclude and 0.5 * f <= sc[s] <= 2 * f] or [s for s in sc if s not in exclude and abs(math.log(sc[s] + 1) - math.log(f + 1)) < 1.0]
    return rng.choice(cands)
def test(a, b, n=300):
    o = ctx(a, b)
    nl = [ctx(a, freq_matched(b, {a, b})) for _ in range(n)]
    nl = [x for x in nl if not math.isnan(x)]
    p = (1 + sum(1 for x in nl if x >= o)) / (len(nl) + 1)
    ss, ms = split(a, b)
    return o, (sorted(nl)[len(nl)//2] if nl else float('nan')), p, ss, ms
say('')
say('(2) INTERCHANGEABILITY: ctx = mean cosine of left/right neighbour profiles (whole Wells corpus, seq_raw); null = 300 partners frequency-matched to b; split = 1 - cosine of site / medium profiles (0 = same distribution)')
strong = [(m['form'], m['into']) for m in levels['merges'] if m['level'] == 'strong']
cal = [(a, b, *test(a, b)) for a, b in strong]
npass = sum(1 for c in cal if c[4] <= 0.05)
say(f'  calibration, S268 strong merges: {npass} of {len(cal)} pass at P <= 0.05; ' + '; '.join(f'{a}/{b} ctx {o:.2f} (null {nm:.2f}, P={p:.3f}) split site {ss:.2f} med {ms:.2f}' for a, b, o, nm, p, ss, ms in cal))
rand = []
for _ in range(100):
    a = rng.choice([s for s in sc if sc[s] >= 10]); b = freq_matched(a, {a}); rand.append(ctx(a, b))
rand = [x for x in rand if not math.isnan(x)]
say(f'  reference: 100 random frequency-matched pairs ctx mean {np.mean(rand):.2f}, 95th pct {np.percentile(rand, 95):.2f}')
groups = {
 'residual confusions (sporadic substitutions, cycle 1)': [(740, 741), (220, 233), (220, 231), (220, 235), (233, 235), (231, 233), (803, 806), (550, 798), (900, 920), (31, 32), (368, 520), (740, 742), (741, 742), (226, 236), (840, 845), (400, 413), (90, 100), (90, 93), (690, 692), (700, 702), (1, 2), (1, 31), (33, 55), (900, 904)],
 'Mahadevan LUMPS (one M number, two Wells signs)': [(154, 156), (156, 158), (154, 158), (615, 617), (510, 511), (904, 927), (405, 407), (406, 407), (405, 406), (176, 388), (336, 337), (812, 814), (526, 527), (27, 28), (13, 3), (790, 625), (632, 630), (632, 636), (772, 773), (371, 370), (717, 711), (317, 315), (297, 298)],
 'Mahadevan SPLITS (project merges / candidates)': [(384, 388), (803, 806), (390, 405), (390, 407), (151, 156), (550, 798)],
 'S-DARK-11.3 merge candidates': [(384, 388), (27, 28), (390, 407), (525, 526), (525, 527)],
}
res = {}
for g, prs in groups.items():
    say(f'  -- {g}')
    for a, b in prs:
        if a not in sc or b not in sc: say(f'    {a}/{b}: sign absent from seq_raw'); continue
        o, nm, p, ss, ms = test(a, b); res[(a, b)] = (o, nm, p, ss, ms)
        verdict = 'interchangeable' if p <= 0.05 else ('weak' if p <= 0.15 else 'different slot/words')
        say(f'    W{a}/W{b} (n {sc[a]}/{sc[b]}): ctx {o:.2f} vs null {nm:.2f} P={p:.3f}; split site {ss:.2f} medium {ms:.2f} -> {verdict}')
# bridge-error check: W798 vs W550 (bridge equated them) and W806 vs W803
say('  -- bridge-error check: context of the two signs the bridge equated')
for a, b in ((798, 550), (806, 803), (798, 415), (550, 415)):
    o, nm, p, ss, ms = test(a, b)
    say(f'    W{a}/W{b}: ctx {o:.2f} vs null {nm:.2f} P={p:.3f}; right partners W{a}: {dict(right[a].most_common(4))}; W{b}: {dict(right[b].most_common(4))}')
open(OUT + 'loop24_cycle2_log.txt', 'w').write('\n'.join(L) + '\n')
