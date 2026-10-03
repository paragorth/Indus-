"""Loop 55 cycle 1: the red-team battery on Indus-shaped samples of known writing, designed codes, accounting and Indus.

For every corpus and resample seed: subsample ~3,000 texts to the Indus die-regime length histogram (2-12 signs),
run the 55-statistic battery, and compare against SH (within-text shuffle), M1 and M2 (Markov chains fitted on the
sample itself, lengths kept, per type stratum). Count statistics 'beyond chain' (|z| >= 3 and outside the null range).
Output: data/derived/dark/loop55_c1/<corpus>_r<k>.json; summary by tools/dark_loop55_c1.py summary.
Usage: python3 tools/dark_loop55_c1.py run [nres] [nnull] [dup]   |   python3 tools/dark_loop55_c1.py summary [dup]"""
import os, sys, json, random, math, collections, statistics as st, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark_loop55_common as C
from multiprocessing import Pool

OUTD = os.path.join(C.DARK, 'loop55_c1')
CORPORA = ['ur3_words', 'ur3_syll', 'ur3_names_syll', 'linb_syll', 'linb_words', 'latin_edh',
           'icd10', 'hts', 'aircraft_reg', 'unicode_names', 'heraldry', 'chess_eco', 'chords',
           'proto_cuneiform', 'proto_elamite', 'khipu',
           'indus_seq_raw', 'indus_seq_strong', 'indus_seq_all']
NULLS = ('SH', 'M1', 'M2', 'M2E')


def job(args):
    name, r, nnull, dup = args
    tag = f'{name}_{dup}_r{r}'
    path = os.path.join(OUTD, tag + '.json')
    if os.path.exists(path): return tag, 'cached'
    rnd = random.Random(1000 * r + 55)
    hist, copies, _ = C.indus_shape('seq_all')
    if name.startswith('indus_'):
        src = [t for t in C.load_indus(name[6:]) if 2 <= len(t[2]) <= 12]
        data = rnd.sample(src, 3000); meta = {'n': 3000, 'shortfall0': 0.0, 'len_tvd': 0.0,
                                               'distinct_share': len(set(s for _, _, s in data)) / 3000,
                                               'median_len': st.median([len(s) for _, _, s in data])}
        by_type = True
    else:
        src = C.load_ref(name)
        data, meta = C.sample_indus_shaped(src, rnd, hist, copies, dup=dup)
        by_type = False
    t0 = time.time()
    obs, res = C.run_one(data, rnd, nnull=nnull, nulls=NULLS, by_type=by_type)
    out = {'corpus': name, 'class': C.TYPE[name], 'resample': r, 'dup': dup, 'meta': meta, 'obs': obs, 'nulls': res, 'secs': time.time() - t0}
    json.dump(out, open(path, 'w'))
    return tag, f'{time.time() - t0:.0f}s'


def run(nres=6, nnull=20, dup='natural'):
    os.makedirs(OUTD, exist_ok=True)
    jobs = [(name, r, nnull, dup) for r in range(nres) for name in CORPORA]
    with Pool(4) as P:
        for tag, msg in P.imap_unordered(job, jobs):
            print(tag, msg, flush=True)


def summary(dup='natural', out=None):
    rows = collections.defaultdict(list)
    for f in sorted(os.listdir(OUTD)):
        if not f.endswith('.json') or f'_{dup}_r' not in f: continue
        d = json.load(open(os.path.join(OUTD, f)))
        rows[d['corpus']].append(d)
    lines = []
    P = lines.append
    P(f'# Loop 55 cycle 1 (dup={dup}): statistics beyond chain per corpus. beyond = |z| >= 3 and outside the null range; mean over resamples [min-max].')
    P('| corpus | class | n | med len | distinct | len TVD | beyond SH | beyond M1 | beyond M2 | beyond M2E | beyond all chains | sum|z| M2 (cap 20) |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|')
    per_stat = collections.defaultdict(lambda: collections.defaultdict(list))   # stat -> corpus -> [beyond M2 flags]
    zvals = collections.defaultdict(lambda: collections.defaultdict(list))
    agg = {}
    for name in C.TYPE:
        if name not in rows: continue
        R = rows[name]
        def cnt(d, nm): return sum(1 for k in C.STATS if d['nulls'][nm][k]['beyond'])
        def both(d): return sum(1 for k in C.STATS if all(d['nulls'][nm][k]['beyond'] for nm in NULLS if nm != 'SH'))
        def sz(d, nm): return sum(min(abs(d['nulls'][nm][k]['z']), 20) for k in C.STATS if not math.isnan(d['nulls'][nm][k]['z']))
        def rng(v): return f'{st.mean(v):.1f} [{min(v)}-{max(v)}]'
        cs = {nm: [cnt(d, nm) for d in R] for nm in NULLS}; b = [both(d) for d in R]; s2 = [sz(d, 'M2') for d in R]
        m = R[0]['meta']
        P(f"| {name} | {C.TYPE[name]} | {int(st.mean(d['meta']['n'] for d in R))} | {st.mean(d['meta']['median_len'] for d in R):.1f} | {st.mean(d['meta']['distinct_share'] for d in R):.2f} | {m['len_tvd']:.2f} | "
          f"{rng(cs['SH'])} | {rng(cs['M1'])} | {rng(cs['M2'])} | {rng(cs['M2E'])} | {rng(b)} | {st.mean(s2):.0f} |")
        agg[name] = {'class': C.TYPE[name], 'nres': len(R), 'beyond': {nm: cs[nm] for nm in NULLS}, 'both': b}
        for d in R:
            for k in C.STATS:
                per_stat[k][name].append(all(d['nulls'][nm][k]['beyond'] for nm in NULLS if nm != 'SH'))
                zvals[k][name].append(d['nulls']['M2'][k]['z'])
    # detection rate by class
    P('')
    P('## Detection rate of beyond-chain structure by class (share of statistics beyond M2, pooled over corpora and resamples)')
    for cls in 'LDGAI':
        vals = [x for n, a in agg.items() if a['class'] == cls for x in a['beyond']['M2']]
        v1 = [x for n, a in agg.items() if a['class'] == cls for x in a['beyond']['M1']]
        ve = [x for n, a in agg.items() if a['class'] == cls for x in a['beyond']['M2E']]
        vb = [x for n, a in agg.items() if a['class'] == cls for x in a['both']]
        if vals: P(f'- {cls}: beyond M1 {st.mean(v1):.1f}/55, beyond M2 {st.mean(vals):.1f}/55 (range {min(vals)}-{max(vals)}), beyond M2E {st.mean(ve):.1f}/55, beyond all three chains {st.mean(vb):.1f}/55 ({len(vals)} samples)')
    # per statistic: which beat M2 in language but not in codes
    P('')
    P('## Per statistic: share of samples beyond ALL chains (M1, M2, M2E), by class (L language, D code, A accounting, I Indus); z of Indus seq_all (mean over resamples)')
    P('| statistic | family | L | D | G | A | Indus raw/strong/all | mean z Indus(all) vs M2 |')
    P('|---|---|---|---|---|---|---|---|')
    fam = {k: f for f, ks in C.FAMILY.items() for k in ks}
    table = []
    for k in C.STATS:
        def share(cls):
            v = [x for n, L in per_stat[k].items() if C.TYPE[n] == cls for x in L]
            return (sum(v) / len(v)) if v else float('nan')
        ind = '/'.join(f"{sum(per_stat[k][n]) / max(1, len(per_stat[k][n])):.1f}" for n in ('indus_seq_raw', 'indus_seq_strong', 'indus_seq_all') if n in per_stat[k])
        za = [z for z in zvals[k].get('indus_seq_all', []) if not math.isnan(z)]
        table.append((k, fam.get(k, '?'), share('L'), share('D'), share('G'), share('A'), ind, st.mean(za) if za else float('nan')))
    for k, f, l, d, g, a, ind, z in table:
        P(f'| {k} | {f} | {C.fmt(l)} | {C.fmt(d)} | {C.fmt(g)} | {C.fmt(a)} | {ind} | {C.fmt(z)} |')
    txt = '\n'.join(lines)
    out = out or os.path.join(C.DARK, f'loop55_c1_{dup}.txt')
    open(out, 'w').write(txt + '\n')
    json.dump({'agg': agg, 'per_stat': {k: {n: v for n, v in d.items()} for k, d in per_stat.items()},
               'z_M2': {k: {n: v for n, v in d.items()} for k, d in zvals.items()}}, open(out.replace('.txt', '.json'), 'w'))
    print(txt)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if cmd == 'run':
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 6, int(sys.argv[3]) if len(sys.argv) > 3 else 20, sys.argv[4] if len(sys.argv) > 4 else 'natural')
    else:
        summary(sys.argv[2] if len(sys.argv) > 2 else 'natural')
