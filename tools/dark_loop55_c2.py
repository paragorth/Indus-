"""Loop 55 cycle 2: POWER CURVES. For sample sizes n in SIZES (Indus-shaped lengths), how often does each statistic
(a) beat the Markov-2 / Markov-2-END chains (|z| >= 3, outside range) in known language, in designed codes, in Indus; and
(b) separate language samples from designed-code samples on its raw value (AUC, and power at a false-alarm rate of 5%
set on the designed codes at the same n)? Reports the n needed for 80% power per statistic and whether 3,000 is enough.
Output: data/derived/dark/loop55_c2/<corpus>_n<n>_r<k>.json; summary -> loop55_c2.txt / .json.
Usage: python3 tools/dark_loop55_c2.py run [nres] [nnull]   |   python3 tools/dark_loop55_c2.py summary"""
import os, sys, json, random, math, collections, statistics as st, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark_loop55_common as C
from multiprocessing import Pool

OUTD = os.path.join(C.DARK, 'loop55_c2')
SIZES = [250, 500, 1000, 2000, 3000]
CORPORA = ['ur3_words', 'ur3_syll', 'ur3_names_syll', 'linb_syll', 'linb_words', 'latin_edh',
           'icd10', 'hts', 'aircraft_reg', 'unicode_names', 'heraldry',
           'proto_cuneiform', 'proto_elamite', 'khipu',
           'indus_seq_raw', 'indus_seq_strong', 'indus_seq_all']
NULLS = ('M2', 'M2E')


def job(args):
    name, n, r, nnull = args
    tag = f'{name}_n{n}_r{r}'
    path = os.path.join(OUTD, tag + '.json')
    if os.path.exists(path): return tag, 'cached'
    rnd = random.Random(7919 * r + 13 * n + 55)
    hist, copies, _ = C.indus_shape('seq_all')
    if name.startswith('indus_'):
        src = [t for t in C.load_indus(name[6:]) if 2 <= len(t[2]) <= 12]
        data = rnd.sample(src, min(n, len(src))); by_type = True
        meta = {'n': len(data), 'distinct_share': len(set(s for _, _, s in data)) / len(data), 'median_len': st.median([len(s) for _, _, s in data])}
    else:
        src = C.load_ref(name)
        data, meta = C.sample_indus_shaped(src, rnd, hist, copies, n=n, dup='natural')
        by_type = False
    t0 = time.time()
    obs, res = C.run_one(data, rnd, nnull=nnull, nulls=NULLS, by_type=by_type)
    json.dump({'corpus': name, 'class': C.TYPE[name], 'n': n, 'resample': r, 'meta': meta, 'obs': obs, 'nulls': res, 'secs': time.time() - t0}, open(path, 'w'))
    return tag, f'{time.time() - t0:.0f}s'


def run(nres=6, nnull=10):
    os.makedirs(OUTD, exist_ok=True)
    jobs = [(name, n, r, nnull) for r in range(nres) for n in SIZES for name in CORPORA]
    with Pool(8) as P:
        for tag, msg in P.imap_unordered(job, jobs):
            print(tag, msg, flush=True)


def auc(pos, neg):
    """P(pos > neg) with ties 0.5; nan if either empty."""
    if not pos or not neg: return float('nan')
    s = 0.0
    for p in pos:
        for q in neg: s += 1.0 if p > q else 0.5 if p == q else 0.0
    return s / (len(pos) * len(neg))


def summary(out=None):
    R = collections.defaultdict(list)   # (corpus, n) -> dicts
    for f in sorted(os.listdir(OUTD)):
        if f.endswith('.json'):
            d = json.load(open(os.path.join(OUTD, f))); R[(d['corpus'], d['n'])].append(d)
    L = [c for c in CORPORA if C.TYPE[c] == 'L']; D = [c for c in CORPORA if C.TYPE[c] == 'D']; A = [c for c in CORPORA if C.TYPE[c] == 'A']
    I = [c for c in CORPORA if C.TYPE[c] == 'I']
    lines = []; P = lines.append
    P('# Loop 55 cycle 2: power curves (Indus-shaped samples; M2 and M2E chains fitted on each sample, lengths kept / END-state).')
    # (a) beyond-chain detection rate by class and n
    P('')
    P('## (a) Mean number of the 55 statistics beyond BOTH chains (M2 and M2E), by class and sample size [min-max over corpora x resamples]')
    P('| n | L language | D designed code | A accounting | Indus (raw/strong/all) |')
    P('|---|---|---|---|---|')
    def beyond_both(d): return sum(1 for k in C.STATS if all(d['nulls'][nm][k]['beyond'] for nm in NULLS))
    for n in SIZES:
        def cell(cs):
            v = [beyond_both(d) for c in cs for d in R.get((c, n), [])]
            return f'{st.mean(v):.1f} [{min(v)}-{max(v)}]' if v else 'nan'
        ind = ' / '.join(cell([c]).split(' ')[0] for c in I)
        P(f'| {n} | {cell(L)} | {cell(D)} | {cell(A)} | {ind} |')
    # (b) per statistic: AUC L vs D on raw value at each n; power at FAR 5%; n80
    P('')
    P('## (b) Per statistic, raw value: AUC(language vs designed code) by n, power at 5% false-alarm (threshold = D-class 5th/95th pct at the same n), n for 80% power, and where Indus falls at n=3000')
    P('| statistic | family | AUC n=250 | 500 | 1000 | 2000 | 3000 | power@3000 | n80 | direction | Indus(all) share on the language side @3000 |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    fam = {k: f for f, ks in C.FAMILY.items() for k in ks}
    res = {}
    for k in C.STATS:
        aucs = {}; powers = {}; direc = None; ind_side = float('nan')
        for n in SIZES:
            lv = [d['obs'][k] for c in L for d in R.get((c, n), []) if not (isinstance(d['obs'][k], float) and math.isnan(d['obs'][k]))]
            dv = [d['obs'][k] for c in D for d in R.get((c, n), []) if not (isinstance(d['obs'][k], float) and math.isnan(d['obs'][k]))]
            a = auc(lv, dv); aucs[n] = a
            if not lv or not dv or math.isnan(a): powers[n] = float('nan'); continue
            hi = a >= 0.5   # language higher than codes?
            ds = sorted(dv); q = ds[int(0.95 * (len(ds) - 1))] if hi else ds[int(0.05 * (len(ds) - 1))]
            powers[n] = (sum(1 for x in lv if x > q) if hi else sum(1 for x in lv if x < q)) / len(lv)
            if n == 3000:
                direc = 'L>D' if hi else 'L<D'
                iv = [d['obs'][k] for d in R.get(('indus_seq_all', 3000), []) if not (isinstance(d['obs'][k], float) and math.isnan(d['obs'][k]))]
                ind_side = (sum(1 for x in iv if x > q) if hi else sum(1 for x in iv if x < q)) / len(iv) if iv else float('nan')
        n80 = next((n for n in SIZES if not math.isnan(powers.get(n, float('nan'))) and powers[n] >= 0.8), None)
        res[k] = {'auc': aucs, 'power': powers, 'n80': n80, 'direction': direc, 'indus_all_language_side': ind_side}
        P(f"| {k} | {fam.get(k, '?')} | " + ' | '.join(C.fmt(aucs.get(n, float('nan'))) for n in SIZES) +
          f" | {C.fmt(powers.get(3000, float('nan')))} | {n80 if n80 else '>3000'} | {direc} | {C.fmt(ind_side)} |")
    # (c) per statistic: beyond-chain (M2 & M2E) detection rate for L, D, I by n
    P('')
    P('## (c) Per statistic: share of samples beyond both chains, L / D / Indus(all), at n = 500, 1000, 3000')
    P('| statistic | L@500 | D@500 | I@500 | L@1000 | D@1000 | I@1000 | L@3000 | D@3000 | I@3000 |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    chain = {}
    for k in C.STATS:
        row = []; chain[k] = {}
        for n in (500, 1000, 3000):
            for cs, lab in ((L, 'L'), (D, 'D'), (['indus_seq_all'], 'I')):
                v = [all(d['nulls'][nm][k]['beyond'] for nm in NULLS) for c in cs for d in R.get((c, n), [])]
                x = sum(v) / len(v) if v else float('nan'); row.append(x); chain[k][f'{lab}@{n}'] = x
        P(f'| {k} | ' + ' | '.join(C.fmt(x) for x in row) + ' |')
    txt = '\n'.join(lines)
    out = out or os.path.join(C.DARK, 'loop55_c2.txt')
    open(out, 'w').write(txt + '\n')
    json.dump({'raw_separation': res, 'chain_detection': chain}, open(out.replace('.txt', '.json'), 'w'))
    print(txt)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if cmd == 'run': run(int(sys.argv[2]) if len(sys.argv) > 2 else 6, int(sys.argv[3]) if len(sys.argv) > 3 else 10)
    else: summary()
