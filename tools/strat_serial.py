"""Serial-number hypothesis test: are Indus seal texts issue numbers of licences (mixed-radix serials:
closer/opener = high-order digits, middle = low-order digits) rather than names or words?
Predictions and tests (each with a control):
 a. Same-closer seal pairs at one site: smaller stratigraphic distance when middles share a prefix (shuffle time 2,000x)
 b. Digit uniformity: per-slot sign distribution vs uniform and vs Ur III name-syllable Zipf shape
 c. Growth: rarefaction of distinct middles vs seals, against Ur III names and planted mixed-radix serials
 d. Issue order: within a closer, middle 'value' (signs as digits in frequency order) vs stratigraphic rank (Spearman, permutation)
Runs on seq_raw / seq_strong / seq_all. Output: data/derived/strat_serial.txt
"""
import json, collections, random, math, re, statistics as st, sys, os
random.seed(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)

corpus = json.load(open(f'{ROOT}/data/derived/merged-corpus-canonical.json'))
parsed = {r['cisi']: r for r in json.load(open(f'{ROOT}/data/derived/parsed_texts.json'))}
# stratigraphic ordinals: HARP-style 'time' field (both cities) and Vats strata at Harappa (VII deepest = earliest)
TIME = {'Period 3A': 0, 'Period 3B': 1, 'Period 3B-1': 1, 'Period 3B-2': 1.5, 'Period 3B/C': 2, 'Period 3C': 3.5,
        'Period 3C-1': 3, 'Period 3C-2': 4, 'Period 3C-3': 5, 'Period 3C-4': 6, 'Period 4': 7, 'Period 5A': 8, 'Period 5B': 8}
STRAT = {f'Stratum {r}': i for i, r in enumerate(['VII', 'VI', 'V', 'IV', 'III', 'II', 'I'])}
MDPER = {'Early': 0, 'Intermediate': 1, 'Late': 2}

def strat_rank(r, scheme):
    if scheme == 'time': return TIME.get(r['time'])
    if scheme == 'strata': return STRAT.get(r['phase'].strip())
    if scheme == 'mdper': return MDPER.get(r['period'].strip())

def parse(seq, slots):
    """Return opener-part, middle (NAME-slot run), closer-part using parsed slot labels aligned to seq (same length)."""
    if len(seq) != len(slots): return None
    mid = [s for s, l in zip(seq, slots) if l == 'NAME']
    closer = tuple(s for s, l in zip(seq, slots) if l in ('TITLE', 'CLOSER', 'SUFFIX'))
    opener = tuple(s for s, l in zip(seq, slots) if l in ('OPENER', 'MARKER'))
    return opener, tuple(mid), closer

seals = [r for r in corpus if str(r['type']).startswith('SEAL') and r['cisi'] in parsed]
P(f'Seals with parsed slots: {len(seals)} (corpus seals {sum(1 for r in corpus if str(r["type"]).startswith("SEAL"))})')
for site in ('Mohenjo-daro', 'Harappa'):
    s = [r for r in seals if r['site'] == site]
    P(f'  {site}: {len(s)} seals; time-field dated {sum(strat_rank(r,"time") is not None for r in s)}; '
      f'Vats strata {sum(strat_rank(r,"strata") is not None for r in s)}; MD period {sum(strat_rank(r,"mdper") is not None for r in s)}')

# Ur III / OB seal-legend owner names (CDLI line 1), syllables split on '-'
NAMEF = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/seal_line1.txt'
names = []
if os.path.exists(NAMEF):
    for l in open(NAMEF, errors='ignore'):
        l = re.sub(r'[#\[\]!?<>]', '', l.strip().lower())
        if not l or ' ' in l or 'x' in l.split('-'): continue
        t = tuple(x for x in l.split('-') if x)
        if 2 <= len(t) <= 6: names.append(t)
P(f'Ur III/OB seal-owner names loaded: {len(names)}')

def spearman(x, y):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v); i = 0
        while i < len(o):
            j = i
            while j+1 < len(o) and v[o[j+1]] == v[o[i]]: j += 1
            for k in range(i, j+1): r[o[k]] = (i+j)/2
            i = j+1
        return r
    rx, ry = rk(x), rk(y); n = len(x); mx, my = sum(rx)/n, sum(ry)/n
    sx = math.sqrt(sum((a-mx)**2 for a in rx)); sy = math.sqrt(sum((b-my)**2 for b in ry))
    return sum((a-mx)*(b-my) for a, b in zip(rx, ry))/(sx*sy) if sx and sy else 0.0

def norm_entropy(counter):
    n = sum(counter.values()); k = len(counter)
    H = -sum(c/n*math.log2(c/n) for c in counter.values())
    return H, math.log2(k) if k > 1 else 1, n, k

def uniform_null_entropy(n, k, reps=200):
    vals = []
    for _ in range(reps):
        c = collections.Counter(random.randrange(k) for _ in range(n)); vals.append(norm_entropy(c)[0]/math.log2(k))
    return st.mean(vals)

def rarefaction(items, points, reps=30):
    out = {}
    for p in points:
        if p > len(items): break
        out[p] = st.mean(len(set(random.sample(items, p))) for _ in range(reps))
    return out

for VAR in ('seq_raw', 'seq_strong', 'seq_all'):
    P(f'\n================ variant {VAR} ================')
    recs = []
    for r in seals:
        seq = r[VAR]; slots = parsed[r['cisi']]['slots']
        # slot labels were parsed on 'seq'; map positions when lengths agree (merges do not change length)
        pr = parse(seq, slots)
        if pr is None or not pr[1]: continue
        recs.append(dict(site=r['site'], seq=tuple(seq), opener=pr[0], mid=pr[1], closer=pr[2], r=r))
    P(f'seals with a non-empty middle: {len(recs)}; mean middle length {st.mean(len(x["mid"]) for x in recs):.2f}')

    # ---------- a. prefix-sharing pairs vs stratigraphic distance ----------
    P('\n-- a. same-closer pairs: |delta strat| when middles share first sign (or first two) vs not --')
    for site, schemes in (('Mohenjo-daro', ('time', 'mdper')), ('Harappa', ('time', 'strata'))):
        for scheme in schemes:
            sub = [x for x in recs if x['site'] == site and x['closer'] and strat_rank(x['r'], scheme) is not None]
            groups = collections.defaultdict(list)
            for x in sub: groups[x['closer']].append(x)
            groups = {k: v for k, v in groups.items() if len(v) >= 2}
            def stat(assign, depth):
                sh, nsh = [], []
                for k, v in groups.items():
                    for i in range(len(v)):
                        for j in range(i+1, len(v)):
                            d = abs(assign[id(v[i])] - assign[id(v[j])])
                            (sh if v[i]['mid'][:depth] == v[j]['mid'][:depth] and len(v[i]['mid']) >= depth and len(v[j]['mid']) >= depth else nsh).append(d)
                return (st.mean(sh) if sh else float('nan')), (st.mean(nsh) if nsh else float('nan')), len(sh), len(nsh)
            base = {id(x): strat_rank(x['r'], scheme) for x in sub}
            for depth in (1, 2):
                o_sh, o_nsh, n_sh, n_nsh = stat(base, depth)
                if n_sh < 5: P(f'  {site} [{scheme}] prefix{depth}: only {n_sh} sharing pairs, skipped'); continue
                obs = o_nsh - o_sh  # positive = sharing pairs are closer in time (serial prediction)
                ge = 0; N = 2000
                for _ in range(N):
                    vals = [base[id(x)] for x in sub]; random.shuffle(vals)
                    a = {id(x): v for x, v in zip(sub, vals)}
                    s_sh, s_nsh, _, _ = stat(a, depth)
                    ge += (s_nsh - s_sh) >= obs
                P(f'  {site} [{scheme}] n={len(sub)} seals, {len(groups)} closer groups; prefix{depth}: sharing pairs {n_sh} mean|dt|={o_sh:.2f} '
                  f'vs non-sharing {n_nsh} mean|dt|={o_nsh:.2f}; diff={obs:+.2f}; P(sharing closer in time | shuffle)={ge/N:.3f}')

    # ---------- b. digit uniformity per slot ----------
    P('\n-- b. per-slot sign distributions: entropy / max entropy, vs uniform null and Ur III syllables --')
    slotc = collections.defaultdict(collections.Counter)
    for x in recs:
        sl = parsed[x['r']['cisi']]['slots']
        for s, l in zip(x['seq'], sl): slotc[l][s] += 1
    # middle digit positions separately: 1st, 2nd, last middle sign
    for x in recs:
        slotc['MID-pos1'][x['mid'][0]] += 1
        if len(x['mid']) > 1: slotc['MID-pos2'][x['mid'][1]] += 1; slotc['MID-last'][x['mid'][-1]] += 1
    def shape(c):
        H, Hmax, n, k = norm_entropy(c); top = sorted(c.values(), reverse=True)
        return H/Hmax, n, k, top[0]/n, sum(top[:5])/n, sum(1 for v in c.values() if v == 1)/k
    rows = []
    for l in ('OPENER', 'MARKER', 'NAME', 'MID-pos1', 'MID-pos2', 'MID-last', 'TITLE', 'CLOSER', 'SUFFIX'):
        if l not in slotc: continue
        hn, n, k, t1, t5, hap = shape(slotc[l]); un = uniform_null_entropy(n, k)
        rows.append((l, hn, un)); P(f'  {l:9s} n={n:5d} k={k:4d} H/Hmax={hn:.3f} (uniform digits with same n,k: {un:.3f}) top1={t1:.2f} top5={t5:.2f} singleton signs={hap:.2f}')
    if names:
        syl = collections.Counter(s for nm in names for s in nm); p1 = collections.Counter(nm[0] for nm in names)
        for lab, c in (('UrIII syllables all', syl), ('UrIII 1st syllable', p1)):
            hn, n, k, t1, t5, hap = shape(c); un = uniform_null_entropy(n, k)
            P(f'  {lab:19s} n={n:5d} k={k:4d} H/Hmax={hn:.3f} (uniform: {un:.3f}) top1={t1:.2f} top5={t5:.2f} singleton={hap:.2f}')
    # Zipf slope (log rank vs log freq, top 50) for NAME slot vs Ur III syllables vs uniform digits
    def zipf_slope(c, top=50):
        f = sorted(c.values(), reverse=True)[:top]
        xs = [math.log(i+1) for i in range(len(f))]; ys = [math.log(v) for v in f]
        mx, my = st.mean(xs), st.mean(ys)
        return sum((a-mx)*(b-my) for a, b in zip(xs, ys))/sum((a-mx)**2 for a in xs)
    n_name, k_name = sum(slotc['NAME'].values()), len(slotc['NAME'])
    unif = collections.Counter(random.randrange(k_name) for _ in range(n_name))
    P(f'  Zipf slope (top 50): Indus NAME slot {zipf_slope(slotc["NAME"]):.2f}; Ur III syllables {zipf_slope(syl) if names else float("nan"):.2f}; '
      f'uniform digits (same n,k) {zipf_slope(unif):.2f}  [near 0 = flat digits; about -1 = Zipfian words]')

    # ---------- c. growth of distinct middles ----------
    P('\n-- c. rarefaction: distinct middles vs seals sampled (Indus, Ur III names, planted serials) --')
    mids = [x['mid'] for x in recs]
    pts = [100, 250, 500, 1000, len(mids)]
    ind = rarefaction(mids, pts)
    nm_s = random.sample(names, min(len(names), len(mids))) if names else []
    ur = rarefaction(nm_s, pts) if nm_s else {}
    # planted serials: issue numbers 0..N-1 in mixed radix over the middle-sign inventory, length from Indus distribution
    inv = sorted(slotc['NAME'], key=lambda s: -slotc['NAME'][s]); K = len(inv)
    lens = [len(m) for m in mids]
    def serial(i, L):
        d = []
        for _ in range(L): d.append(inv[i % K]); i //= K
        return tuple(reversed(d))
    ser = [serial(i, L) for i, L in enumerate(lens)]
    # realistic serials: issue numbers from many issuing offices (= closers), each counting from 0, so low digits are shared
    off = collections.Counter(); ser2 = []
    for x in recs:
        c = off[x['closer']]; off[x['closer']] += 1; ser2.append((x['closer'], serial(c, len(x['mid']))))
    ser2m = [s for _, s in ser2]
    sr = rarefaction(ser, pts); sr2 = rarefaction(ser2m, pts)
    P('  n seals   Indus middles   UrIII names   serials(single counter)   serials(per-closer counters)')
    for p in pts:
        if p in ind: P(f'  {p:6d}   {ind[p]:9.1f}      {ur.get(p, float("nan")):9.1f}      {sr[p]:9.1f}                 {sr2[p]:9.1f}')
    def uniqfrac(ms): c = collections.Counter(ms); return sum(1 for m in ms if c[m] == 1)/len(ms)
    def slope(d):
        ks = sorted(d); return (d[ks[-1]]-d[ks[-2]])/(ks[-1]-ks[-2])
    P(f'  unique fraction: Indus middles {uniqfrac(mids):.3f}; Ur III names {uniqfrac(nm_s) if nm_s else float("nan"):.3f}; serials {uniqfrac(ser):.3f}; per-closer serials {uniqfrac(ser2m):.3f}')
    P(f'  final-segment slope (new distinct per added seal): Indus {slope(ind):.3f}; Ur III {slope(ur) if ur else float("nan"):.3f}; serials {slope(sr):.3f}; per-closer serials {slope(sr2):.3f}')
    # whole-text reuse: serials must not repeat a full text (closer+middle) on distinct seals
    full = collections.Counter(x['seq'] for x in recs)
    rep = sum(v for v in full.values() if v > 1)
    P(f'  full seal texts repeated on 2+ distinct seals: {sum(1 for v in full.values() if v>1)} texts, {rep} seals ({rep/len(recs):.1%}); '
      f'middles reused under a DIFFERENT closer: {sum(1 for m,c in collections.Counter(x["mid"] for x in recs).items() if c>1 and len(set(x["closer"] for x in recs if x["mid"]==m))>1)}')
    # within-middle digit repetition: a serial has no reason to avoid or favour repeated digits; language/names avoid repeats
    rep_in = sum(1 for m in mids if len(m) >= 2 and len(set(m)) < len(m))
    # compare observed repeated-digit rate with a sign-shuffled control (same lengths, same sign pool)
    pool = [s for m in mids for s in m]; random.shuffle(pool); i = 0; shuf = []
    for m in mids: shuf.append(tuple(pool[i:i+len(m)])); i += len(m)
    rep_sh = sum(1 for m in shuf if len(m) >= 2 and len(set(m)) < len(m))
    P(f'  middles with a repeated sign: observed {rep_in} vs sign-shuffled control {rep_sh} (ratio {rep_in/max(rep_sh,1):.2f}); serials give ratio about 1')

    # ---------- d. issue order: middle value vs stratigraphic rank within closer ----------
    P('\n-- d. within-closer Spearman(middle value as mixed-radix number, stratigraphic rank), permutation within closer --')
    rank_of = {s: i for i, s in enumerate(inv)}  # digit value in frequency order (commonest = 0)
    def value(m):
        v = 0
        for s in m: v = v*K + rank_of[s]
        return v
    for site, schemes in (('Mohenjo-daro', ('time', 'mdper')), ('Harappa', ('time', 'strata'))):
        for scheme in schemes:
            sub = [x for x in recs if x['site'] == site and x['closer'] and strat_rank(x['r'], scheme) is not None]
            groups = collections.defaultdict(list)
            for x in sub: groups[x['closer']].append(x)
            groups = {k: v for k, v in groups.items() if len(v) >= 4}
            if not groups: P(f'  {site} [{scheme}]: no closer group with 4+ dated seals'); continue
            def pooled(tvals):
                xs, ys = [], []
                for k, v in groups.items():
                    vv = [value(x['mid']) for x in v]; tt = [tvals[id(x)] for x in v]
                    # within-group centring by rank to pool
                    def rk(a):
                        o = sorted(a); return [(o.index(z) + len(o) - 1 - o[::-1].index(z))/2/(len(a)-1) for z in a]
                    xs += rk(vv); ys += rk(tt)
                return spearman(xs, ys), len(xs)
            base = {id(x): strat_rank(x['r'], scheme) for x in sub}
            obs, npool = pooled(base)
            # also value by length-normalised position (first digit only) and by plain middle length
            ge = 0; N = 2000
            for _ in range(N):
                a = {}
                for k, v in groups.items():
                    t = [base[id(x)] for x in v]; random.shuffle(t)
                    for x, tv in zip(v, t): a[id(x)] = tv
                ge += abs(pooled(a)[0]) >= abs(obs)
            per = [(k, len(v), spearman([value(x['mid']) for x in v], [base[id(x)] for x in v])) for k, v in groups.items()]
            pos = sum(1 for _, _, r in per if r > 0); neg = sum(1 for _, _, r in per if r < 0)
            P(f'  {site} [{scheme}]: {len(groups)} closer groups, {npool} seals; pooled Spearman rho={obs:+.3f}, two-sided P={ge/N:.3f}; '
              f'groups with rho>0: {pos}, rho<0: {neg}')
            # also: first-digit value alone, and middle length (serial counters fill short numbers first)
            def pooled_f(fn):
                xs, ys = [], []
                for k, v in groups.items():
                    for x in v: xs.append(fn(x)); ys.append(base[id(x)])
                return spearman(xs, ys)
            P(f'      first-middle-sign value rho={pooled_f(lambda x: rank_of[x["mid"][0]]):+.3f}; middle length rho={pooled_f(lambda x: len(x["mid"])):+.3f} (serials: longer numbers later)')

open(f'{ROOT}/data/derived/strat_serial.txt', 'w').write('\n'.join(OUT) + '\n')
