"""S-DARK-68: HOW UNIFORM ARE REAL NAME POOLS ACROSS CITIES 600 KM APART?
Same statistic as S-DARK-48 (tools/dark_loop48.py): between-group divergence as a multiple of the divergence between two random
halves of ONE group, equal-n draws (n texts per side), reported per component with the excess in bits, z, P and a bootstrap CI
of the multiple. Applied to real personal-name populations with geography, at three distance bands.

Units (so the ladder can be read against the Indus sign level):
  letters   a name = sequence of letters (inventory 26)
  bigrams   a name = sequence of adjacent letter pairs (inventory ~400 used types, ~5-6 tokens per name: the closest match to a
            seal text of ~5 signs from a ~400-sign inventory)
  whole     a name = one token (the name itself); unigram JSD = name-pool divergence; reuse = share of distinct names attested in the other group
  elements  (multi-element names: Latin, medieval) a person = sequence of name elements (praenomen, nomen, cognomen ...)
Components (as in S-DARK-48): uni, bi, rho (1 - Spearman on units with >= 5 pooled tokens), init, fin, len, reuse (within / between,
> 1 = local), and 'noise-matched' multiple: the multiple at the n where the within-group unigram JSD equals the Indus within value
(0.142 bits for MD vs H seals at n = 200 seq_raw), so corpora with different inventory sizes are compared at equal noise floor.

Usage: python3 tools/dark_loop68.py <cycle> [nsub] [nperm]
  cycle 1: SSA given names by US state (public domain), cohorts 1950s / 2010s / 1910s, states >= 150k births, all pairs, bands by centroid distance
  cycle 2: EDH Latin name elements by province (CC BY-SA 4.0) + designed-code populations (aircraft registrations by owner state)
  cycle 3: Indus multiples with bootstrap CIs at seq_raw / seq_strong / seq_all / IM77 through the identical engine; Ur III and Linear B names re-run as name pools
"""
import sys, json, gzip, collections, math, os, re, csv
import numpy as np
from scipy.stats import spearmanr

ROOT = '/home/user/Indus-/'
DARK = ROOT + 'data/derived/dark/'
C68 = DARK + 'loop68_corpora/'
CY = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 60
NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 300
rng = np.random.default_rng(680 + CY)
INDUS_WITHIN_UNI = 0.142   # S-DARK-48.1 seals MD vs H, n = 200, seq_raw: within-city unigram JSD (bits)
INDUS_WITHIN_NAME = 0.2556 # same, NAME-slot tokens
KEYS = ['uni', 'bi', 'rho', 'init', 'fin', 'len', 'reuse']

# ------------------------------------------------------------------ divergence engine
def jsd(p, q):
    p = p / p.sum(); q = q / q.sum(); m = (p + q) / 2
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def jsd_counter(c1, c2):
    keys = list(set(c1) | set(c2))
    if not keys: return 0.0
    p = np.array([c1.get(k, 0) for k in keys], float); q = np.array([c2.get(k, 0) for k in keys], float)
    if p.sum() == 0 or q.sum() == 0: return 0.0
    return jsd(p, q)

class Pop:
    """a population of texts (tuples of unit tokens) with multiplicities; texts drawn with probability proportional to count"""
    def __init__(self, counter, vocab):
        self.texts = list(counter.keys()); self.w = np.array([counter[t] for t in self.texts], float); self.w /= self.w.sum()
        self.N = int(sum(counter.values()))
        self.enc = [np.array([vocab.setdefault(u, len(vocab)) for u in t], dtype=np.int64) for t in self.texts]
        self.vocab = vocab
    def draw(self, n, replace=None):
        # with replacement over distinct texts weighted by count (population >> n): equivalent to sampling n individuals
        return rng.choice(len(self.texts), n, replace=True, p=self.w)
    def draw2(self, n):
        """two disjoint samples of n individuals (within-null): sample 2n individuals, split"""
        idx = rng.choice(len(self.texts), 2 * n, replace=True, p=self.w)
        return idx[:n], idx[n:]

def counts(arrs, V):
    if not arrs: return np.zeros(V)
    return np.concatenate(arrs) if False else np.bincount(np.concatenate(arrs), minlength=V).astype(float)

def bigram_counts(arrs, V):
    bs = [a[:-1] * V + a[1:] for a in arrs if len(a) > 1]
    if not bs: return collections.Counter()
    return collections.Counter(np.concatenate(bs).tolist())

def components(PA, ia, PB, ib, V, whole=False):
    A = [PA.enc[i] for i in ia]; B = [PB.enc[i] for i in ib]
    ua = counts(A, V); ub = counts(B, V)
    out = {'uni': jsd(ua, ub)}
    if not whole:
        out['bi'] = jsd_counter(bigram_counts(A, V), bigram_counts(B, V))
        pooled = ua + ub; m = pooled >= 5
        if m.sum() >= 5:
            r = spearmanr(ua[m], ub[m]).correlation
            out['rho'] = 1 - (0.0 if np.isnan(r) else r)
        else: out['rho'] = float('nan')
        out['init'] = jsd(np.bincount([a[0] for a in A], minlength=V).astype(float), np.bincount([b[0] for b in B], minlength=V).astype(float))
        out['fin'] = jsd(np.bincount([a[-1] for a in A], minlength=V).astype(float), np.bincount([b[-1] for b in B], minlength=V).astype(float))
        la = np.bincount(np.minimum([len(a) for a in A], 12), minlength=13).astype(float)
        lb = np.bincount(np.minimum([len(b) for b in B], 12), minlength=13).astype(float)
        out['len'] = jsd(la, lb)
    sa = set(tuple(PA.texts[i]) for i in ia); sb = set(tuple(PB.texts[i]) for i in ib)
    out['reuse'] = 0.5 * (len(sa & sb) / len(sa) + len(sa & sb) / len(sb))
    return out

def pair_test(PA, PB, n, nsub=NSUB, nperm=NPERM, whole=False, boot=200):
    V = len(PA.vocab)
    between = collections.defaultdict(list); within = collections.defaultdict(list)
    for _ in range(nsub):
        for k, v in components(PA, PA.draw(n), PB, PB.draw(n), V, whole).items(): between[k].append(v)
    for P in (PA, PB):
        for _ in range(nperm):
            a, b = P.draw2(n)
            for k, v in components(P, a, P, b, V, whole).items(): within[k].append(v)
    res = {'n': n, 'comp': {}}
    for k in between:
        bt = np.array(between[k], float); wt = np.array(within[k], float)
        bm = float(np.nanmean(bt)); wm = float(np.nanmean(wt)); wsd = float(np.nanstd(wt))
        if k == 'reuse':
            eps = 0.5 / n
            ratio = (wm + eps) / (bm + eps); excess = wm - bm
            P = float((np.sum(wt <= np.median(bt)) + 1) / (len(wt) + 1))
            # bootstrap CI of the ratio over the draw sets
            bs = []
            for _ in range(boot):
                b1 = rng.choice(bt, len(bt)); w1 = rng.choice(wt, len(wt))
                bs.append((np.nanmean(w1) + eps) / (np.nanmean(b1) + eps))
        else:
            ratio = bm / wm if wm > 0 else float('nan'); excess = bm - wm
            P = float((np.sum(wt >= np.nanmedian(bt)) + 1) / (len(wt) + 1))
            bs = []
            for _ in range(boot):
                b1 = rng.choice(bt, len(bt)); w1 = rng.choice(wt, len(wt))
                bs.append(np.nanmean(b1) / np.nanmean(w1))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        res['comp'][k] = dict(between=bm, within=wm, within_sd=wsd, excess=excess, ratio=ratio, lo=float(lo), hi=float(hi),
                              z=((bm - wm) / wsd if wsd > 0 else float('nan')), P=P)
    return res

def noise_matched_n(P, target, whole=False, nperm=60):
    """n (individuals) at which the within-population unigram JSD equals target bits (log-interpolated over a grid)"""
    grid = [3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500, 800, 1200, 2000]
    V = len(P.vocab); vals = []
    for n in grid:
        v = []
        for _ in range(nperm):
            a, b = P.draw2(n); v.append(components(P, a, P, b, V, True)['uni'])
        vals.append(np.mean(v))
        if vals[-1] < target * 0.6: break
    vals = np.array(vals); g = np.array(grid[:len(vals)], float)
    if (vals > target).all(): return int(g[-1])
    if (vals < target).all(): return int(g[0])
    i = np.where(vals < target)[0][0]
    # interpolate in log n between grid[i-1] and grid[i]
    f = (vals[i - 1] - target) / (vals[i - 1] - vals[i])
    return int(round(math.exp(math.log(g[i - 1]) + f * (math.log(g[i]) - math.log(g[i - 1])))))

def fmt(res):
    lines = []
    for k, c in res['comp'].items():
        lines.append(f"    {k:6s} between {c['between']:.4f} within {c['within']:.4f} (sd {c['within_sd']:.4f}) excess {c['excess']:+.4f} x{c['ratio']:.2f} [{c['lo']:.2f}-{c['hi']:.2f}] z {c['z']:+.1f} P {c['P']:.3f}")
    return '\n'.join(lines)

# ------------------------------------------------------------------ tokenisers
def tok_letters(name): return tuple(name.lower())
def tok_bigrams(name):
    s = name.lower()
    return tuple(s[i:i + 2] for i in range(len(s) - 1)) if len(s) > 1 else (s,)
def tok_whole(name): return (name.lower(),)
TOK = {'letters': tok_letters, 'bigrams': tok_bigrams, 'whole': tok_whole}

def make_pops(groups, tok, vocab=None):
    """groups: dict label -> Counter(name -> count). returns dict label -> Pop over a shared vocab"""
    vocab = {} if vocab is None else vocab
    return {g: Pop(collections.Counter({tok(nm): c for nm, c in cnt.items()}), vocab) for g, cnt in groups.items()}

def haversine(a, b):
    la1, lo1 = map(math.radians, a); la2, lo2 = map(math.radians, b)
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(d))

# approximate population centroids / largest-city coordinates of US states (lat, lon)
STATE_XY = {'AL': (33.5, -86.8), 'AK': (61.2, -149.9), 'AZ': (33.4, -112.1), 'AR': (34.7, -92.3), 'CA': (36.0, -119.5), 'CO': (39.7, -105.0),
            'CT': (41.6, -72.7), 'DE': (39.7, -75.6), 'DC': (38.9, -77.0), 'FL': (28.3, -81.6), 'GA': (33.7, -84.4), 'HI': (21.3, -157.9),
            'ID': (43.6, -116.2), 'IL': (41.6, -88.0), 'IN': (39.8, -86.2), 'IA': (41.9, -93.1), 'KS': (38.5, -97.0), 'KY': (38.0, -85.0),
            'LA': (30.5, -91.2), 'ME': (44.3, -69.8), 'MD': (39.2, -76.7), 'MA': (42.3, -71.5), 'MI': (42.7, -83.9), 'MN': (45.0, -93.3),
            'MS': (32.7, -89.7), 'MO': (38.6, -91.5), 'MT': (46.6, -111.0), 'NE': (41.2, -96.6), 'NV': (36.5, -116.0), 'NH': (43.1, -71.5),
            'NJ': (40.3, -74.5), 'NM': (34.8, -106.3), 'NY': (41.2, -74.5), 'NC': (35.6, -79.6), 'ND': (47.2, -99.0), 'OH': (40.3, -82.6),
            'OK': (35.6, -97.0), 'OR': (45.2, -122.8), 'PA': (40.5, -77.0), 'RI': (41.8, -71.4), 'SC': (34.0, -81.0), 'SD': (43.9, -99.0),
            'TN': (35.8, -86.4), 'TX': (30.9, -97.5), 'UT': (40.7, -111.9), 'VT': (44.3, -72.7), 'VA': (37.7, -77.8), 'WA': (47.4, -121.5),
            'WV': (38.5, -81.0), 'WI': (43.4, -88.7), 'WY': (42.5, -107.0)}
BANDS = [('<350 km', 0, 350), ('350-900 km (~600)', 350, 900), ('900-1500 km', 900, 1500), ('>1500 km', 1500, 1e9)]
def band(d):
    for nm, lo, hi in BANDS:
        if lo <= d < hi: return nm

def band_summary(R, dist, out, label, keys=KEYS):
    out.append(f'\n  BAND SUMMARY {label}: mean multiple (sd) over pairs; reuse = within/between')
    out.append('  ' + f"{'band':20s}{'pairs':>6s}" + ''.join(f'{k:>14s}' for k in keys))
    S = {}
    for nm, lo, hi in BANDS:
        prs = [p for p in R if lo <= dist[p] < hi]
        if not prs: continue
        row = []
        for k in keys:
            v = [R[p]['comp'][k]['ratio'] for p in prs if k in R[p]['comp'] and np.isfinite(R[p]['comp'][k]['ratio'])]
            row.append(f'{np.mean(v):7.2f} ({np.std(v):.2f})' if v else f"{'-':>14s}")
            S.setdefault(nm, {})[k] = (float(np.mean(v)), float(np.std(v)), len(v)) if v else None
        out.append('  ' + f'{nm:20s}{len(prs):6d}' + ''.join(row))
    out.append('  excess (bits; reuse = within - between share)')
    for nm, lo, hi in BANDS:
        prs = [p for p in R if lo <= dist[p] < hi]
        if not prs: continue
        row = []
        for k in keys:
            v = [R[p]['comp'][k]['excess'] for p in prs if k in R[p]['comp'] and np.isfinite(R[p]['comp'][k]['excess'])]
            row.append(f'{np.mean(v):14.4f}' if v else f"{'-':>14s}")
        out.append('  ' + f'{nm:20s}{len(prs):6d}' + ''.join(row))
    return S

# ------------------------------------------------------------------ cycle 1: SSA given names by state
def load_ssa(cohort, min_births=150000, max_states=24, sexes=('M', 'F')):
    D = json.load(gzip.open(C68 + 'ssa_state_cohorts.json.gz', 'rt'))[cohort]
    G = {}
    for st, cnt in D.items():
        c = collections.Counter()
        for k, v in cnt.items():
            sx, nm = k.split(':', 1)
            if sx in sexes: c[nm] += v
        if sum(c.values()) >= min_births and st in STATE_XY: G[st] = c
    keep = sorted(G, key=lambda s: -sum(G[s].values()))[:max_states]
    return {s: G[s] for s in keep}

def true_jsd(ca, cb):
    keys = list(set(ca) | set(cb))
    return jsd(np.array([ca.get(k, 0) for k in keys], float), np.array([cb.get(k, 0) for k in keys], float))

def cycle1():
    out = [f'# S-DARK-68 cycle 1 (nsub {NSUB}, nperm {NPERM}): US given names by state (SSA, public domain; names with >= 5 births per state-year)']
    R = {}
    for cohort, min_b, n_list in (('1950s', 150000, (200, 30)), ('2010s', 150000, (200,)), ('1910s', 40000, (200,))):
        G = load_ssa(cohort, min_b)
        states = sorted(G, key=lambda s: -sum(G[s].values()))
        out.append(f'\n== cohort {cohort}: {len(states)} states with >= {min_b} recorded births: ' + ', '.join(f'{s} {sum(G[s].values())//1000}k' for s in states))
        pairs = [(a, b) for i, a in enumerate(states) for b in states[i + 1:]]
        dist = {f'{a} vs {b}': haversine(STATE_XY[a], STATE_XY[b]) for a, b in pairs}
        for unit in (('bigrams', 'letters', 'whole') if cohort == '1950s' else ('bigrams', 'whole')):
            pops = make_pops(G, TOK[unit])
            whole = unit == 'whole'
            for n in n_list:
                Rn = {}
                for a, b in pairs:
                    Rn[f'{a} vs {b}'] = pair_test(pops[a], pops[b], n, whole=whole)
                keys = ['uni', 'reuse'] if whole else KEYS
                S = band_summary(Rn, dist, out, f'{cohort} {unit} n={n}', keys)
                R[f'{cohort}|{unit}|n{n}'] = {'bands': S, 'pairs': {p: {'dist': dist[p], 'comp': r['comp']} for p, r in Rn.items()}}
                # the pairs nearest the Indus geometry: 500-700 km, both large
                near = sorted([p for p in Rn if 500 <= dist[p] <= 700], key=lambda p: dist[p])
                out.append(f'  pairs at 500-700 km ({unit}, n={n}): ' + '; '.join(
                    f"{p} {dist[p]:.0f}km uni x{Rn[p]['comp']['uni']['ratio']:.2f}" + (f" reuse x{Rn[p]['comp']['reuse']['ratio']:.2f}" if 'reuse' in Rn[p]['comp'] else '') for p in near[:14]))
            # population-level ('true') JSD per band, from the full counts, and the noise-matched multiple at the Indus within value
            if unit in ('bigrams', 'whole'):
                tj = {}
                for a, b in pairs:
                    ca = collections.Counter(); cb = collections.Counter()
                    for nm, c in G[a].items():
                        for u in TOK[unit](nm): ca[u] += c
                    for nm, c in G[b].items():
                        for u in TOK[unit](nm): cb[u] += c
                    tj[f'{a} vs {b}'] = true_jsd(ca, cb)
                out.append(f'  population-level unigram JSD ({unit}; full state counts, millions of births, bias ~ 0) by band: ' + '; '.join(
                    f"{nm} {np.mean([tj[p] for p in tj if lo <= dist[p] < hi]):.4f} bits (n pairs {sum(lo <= dist[p] < hi for p in tj)})" for nm, lo, hi in BANDS if any(lo <= dist[p] < hi for p in tj)))
                R[f'{cohort}|{unit}|trueJSD'] = tj
            if unit == 'bigrams' and cohort == '1950s':
                target = INDUS_WITHIN_UNI
                nm_n = noise_matched_n(pops[states[0]], target)
                out.append(f'\n  NOISE-MATCHED design ({unit}): within-state unigram JSD = {target} bits (Indus MD-H seals within, n=200) at n = {nm_n} names')
                Rn = {}
                for a, b in pairs:
                    Rn[f'{a} vs {b}'] = pair_test(pops[a], pops[b], nm_n)
                S = band_summary(Rn, dist, out, f'{cohort} {unit} noise-matched n={nm_n}')
                R[f'{cohort}|{unit}|noisematched_n{nm_n}'] = {'bands': S, 'pairs': {p: {'dist': dist[p], 'comp': r['comp']} for p, r in Rn.items()}}
            if unit == 'whole' and cohort == '1950s':
                target = INDUS_WITHIN_NAME
                nm_n = noise_matched_n(pops[states[0]], target, whole=True)
                out.append(f'\n  NOISE-MATCHED design ({unit}): within-state name-pool JSD = {target} bits (Indus MD-H NAME-slot within, n=200) at n = {nm_n} names')
                Rn = {}
                for a, b in pairs:
                    Rn[f'{a} vs {b}'] = pair_test(pops[a], pops[b], nm_n, whole=True)
                S = band_summary(Rn, dist, out, f'{cohort} {unit} noise-matched n={nm_n}', ['uni', 'reuse'])
                R[f'{cohort}|{unit}|noisematched_n{nm_n}'] = {'bands': S, 'pairs': {p: {'dist': dist[p], 'comp': r['comp']} for p, r in Rn.items()}}
    json.dump(R, open(DARK + 'loop68_c1.json', 'w'), default=float)
    open(DARK + 'loop68_c1_log.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))

if __name__ == '__main__':
    {1: cycle1}[CY]()
