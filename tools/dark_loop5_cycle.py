"""Arrow-in-the-dark loop 5, cycle driver. usage: python3 tools/dark_loop5_cycle.py CYCLE
Cycle 1: seal size (log h*v, length-residualised) ; 2: voucher count on the other face (S93) ;
3: counted numeral before W390/W405 ; 4: seal thickness + Indus weight-series hit rate.
Each arrow = (rule, seq level). Controls: permuted target (3x), sign shuffle (global, lengths kept),
planted synthetic target (recoverability). Fixed hypothesis: stroke count = value (S204 extension)."""
import sys, json, math, random, collections, time
sys.path.insert(0, 'tools')
from multiprocessing import Pool
import numpy as np
from scipy.stats import spearmanr
from dark_loop5_engine import *

CYC = int(sys.argv[1]); RESTARTS = 30; STEPS = 1500; VMAX = 12
OUT = f'data/derived/dark/loop5_cycle{CYC}.txt'
WEIGHTS = [1, 2, 4, 8, 16, 32, 64, 160, 200, 320, 640]
STROKES = {int(k): v for k, v in json.load(open('data/derived/dark/loop5_strokes.json')).items()}

objs = load_rows()
SIGNS = top_signs([o['seq_raw'] for o in objs])

def build(cyc):
    """Return list of (text, target) and a split function."""
    if cyc == 1:
        S = [o for o in objs if o['type'].startswith('SEAL') and o['h'] and o['v']]
        y = [math.log(o['h'] * o['v']) for o in S]
        return S, y, (lambda o: o['site'] in TRAIN_SITES), 'log(h*v) of seals, length-residualised'
    if cyc == 2:
        by = collections.defaultdict(list)
        for o in objs: by[o['obj']].append(o)
        S, y = [], []
        for fs in by.values():
            if len(fs) != 2: continue
            c = [count_face(f['seq_raw']) for f in fs]
            if (c[0] is None) != (c[1] is None):
                t = fs[1] if c[0] is not None else fs[0]; cnt = c[0] if c[0] is not None else c[1]
                if len(t['seq_raw']) >= 2: S.append(t); y.append(cnt)
        # held-out: Harappa Mound E (+ non-Harappa) vs the rest (Mound F, D, AB, unlabelled)
        return S, y, (lambda o: not (o['area'].startswith('E') or o['site'] != 'Harappa')), 'voucher count (tall N [+W700]) on other face; train Harappa non-E areas, test Mound E + other sites'
    if cyc == 3:
        S, y = [], []
        for o in objs:
            s = o['seq_raw']
            for i, a in enumerate(s):
                if a in (390, 405) and i > 0 and s[i - 1] in SHORT:
                    t = dict(o);
                    for k in ('seq_raw', 'seq_strong', 'seq_all'):
                        t[k] = [b for j, b in enumerate(o[k]) if j != i - 1]   # drop the numeral itself
                    S.append(t); y.append(VAL[s[i - 1]]); break
        return S, y, (lambda o: o['site'] == 'Mohenjo-daro'), 'count before W390/W405 (3-8) predicted by the rest of the text; train Mohenjo-daro, test Harappa + others'
    if cyc == 4:
        S = [o for o in objs if o['type'].startswith('SEAL') and o['th']]
        y = [math.log(o['th']) for o in S]
        return S, y, (lambda o: o['site'] in TRAIN_SITES), 'log thickness of seals, length-residualised'

def arrow(args):
    rule, key, kind, seed, S_tr, y_tr, S_te, y_te = args
    rng = np.random.default_rng(seed)
    seqs_tr = [o[key] for o in S_tr]; seqs_te = [o[key] for o in S_te]
    y_tr = np.array(y_tr, float); y_te = np.array(y_te, float)
    if kind == 'perm':
        y_tr = rng.permutation(y_tr); y_te = rng.permutation(y_te)
    elif kind == 'shuffle':
        pool = [a for s in seqs_tr + seqs_te for a in s if a not in NUMERALS and a not in EXCL]
        rng.shuffle(pool); it = iter(pool)
        def resh(s): return [a if (a in NUMERALS or a in EXCL) else next(it) for a in s]
        seqs_tr = [resh(s) for s in seqs_tr]; seqs_te = [resh(s) for s in seqs_te]
    elif kind == 'planted':
        truth = {a: int(rng.integers(1, VMAX + 1)) for a in SIGNS}
        def val(s):
            p = Problem([s], SIGNS, rule, 10); return p.values([truth[a] for a in SIGNS] + [1])[0]
        v_tr = np.array([val(s) for s in seqs_tr]); v_te = np.array([val(s) for s in seqs_te])
        sd = v_tr.std() if v_tr.std() > 0 else 1.0
        y_tr = v_tr + rng.normal(0, 1.5 * sd, len(v_tr)); y_te = v_te + rng.normal(0, 1.5 * sd, len(v_te))
        y_tr = length_residual(seqs_tr, y_tr); y_te = length_residual(seqs_te, y_te)
    res = run_arrow(seqs_tr, y_tr, seqs_te, y_te, SIGNS, rule, RESTARTS, VMAX, STEPS, seed0=seed * 100)
    if kind == 'planted':
        a = res['best_assign']; tv = [truth[s] for s in SIGNS]
        res['planted_value_rho'] = spearmanr(a[:len(SIGNS)], tv).correlation
        res['planted_true_test'] = spearmanr(v_te, y_te).correlation
    res.pop('best_assign', None) if kind != 'real' else None
    return (rule, key, kind, seed, res)

if __name__ == '__main__':
    S, y, is_train, desc = build(CYC)
    y = np.array(y, float)
    if CYC in (1, 4):
        y_res = length_residual([o['seq_raw'] for o in S], y)
    else:
        y_res = y
    tr = [i for i, o in enumerate(S) if is_train(o)]; te = [i for i, o in enumerate(S) if not is_train(o)]
    S_tr = [S[i] for i in tr]; S_te = [S[i] for i in te]
    lines = [f'# loop5 cycle {CYC}: {desc}', f'n_train={len(tr)} n_test={len(te)}; train sites={collections.Counter(S[i]["site"] for i in tr).most_common(3)}; test={collections.Counter(S[i]["site"] for i in te).most_common(6)}',
             f'target distribution: {collections.Counter(np.round(y, 1)).most_common(8) if CYC in (2, 3) else ("mean %.2f sd %.2f" % (y.mean(), y.std()))}']
    L_tr = [len(o['seq_raw']) for o in S_tr]; L_te = [len(o['seq_raw']) for o in S_te]
    lines.append(f'length baseline: rho(len, raw target) train={spearmanr(L_tr, y[tr]).correlation:.3f} test={spearmanr(L_te, y[te]).correlation:.3f}; after residualising test={spearmanr(L_te, y_res[te]).correlation:.3f}')
    # fixed hypothesis: stroke count = value
    for rule in ('sum', 'prod', 'pos'):
        r_tr = fixed_score([o['seq_raw'] for o in S_tr], y_res[tr], SIGNS, STROKES, rule)
        r_te = fixed_score([o['seq_raw'] for o in S_te], y_res[te], SIGNS, STROKES, rule)
        r_raw = fixed_score([o['seq_raw'] for o in S], y, SIGNS, STROKES, rule)
        lines.append(f'FIXED stroke-count=value [{rule}]: train rho={r_tr[0]:.3f} (p={r_tr[1]:.2g}); test rho={r_te[0]:.3f} (p={r_te[1]:.2g}); all texts vs raw target rho={r_raw[0]:.3f} (p={r_raw[1]:.2g})')
    jobs = []
    for rule in ('sum', 'prod', 'pos'):
        for key in (('seq_raw', 'seq_strong', 'seq_all') if rule == 'sum' else ('seq_raw',)):
            jobs.append((rule, key, 'real', 1, S_tr, list(y_res[tr]), S_te, list(y_res[te])))
        jobs.append((rule, 'seq_raw', 'perm', 10, S_tr, list(y_res[tr]), S_te, list(y_res[te])))
        jobs.append((rule, 'seq_raw', 'shuffle', 20, S_tr, list(y_res[tr]), S_te, list(y_res[te])))
    jobs.append(('sum', 'seq_raw', 'planted', 30, S_tr, list(y_res[tr]), S_te, list(y_res[te])))
    t0 = time.time()
    with Pool(4) as P:
        results = P.map(arrow, jobs, chunksize=1)
    lines.append(f'arrows: {sum(1 for r in results if r[2]=="real")} real (sum x 3 seq levels + prod + pos), restarts={RESTARTS}, steps={STEPS}, values 1..{VMAX} on {len(SIGNS)} signs + OTHER; wall {time.time()-t0:.0f}s')
    best_real = {}; null_max = collections.defaultdict(list)
    for rule, key, kind, seed, r in results:
        tag = f'{rule:4s} {key:10s} {kind:8s}'
        extra = ''
        if kind == 'planted':
            extra = f' | truth test rho={r["planted_true_test"]:.3f}, recovered-values rho vs truth={r["planted_value_rho"]:.3f}'
        lines.append(f'{tag}: train={r["train"]:.3f} test(best-train)={r["test"]:.3f} test_max={r["test_max"]:.3f} test_mean={r["test_mean"]:.3f} stable>80%={r["stable_frac"]:.2f} ({len(r["stable_signs"])} signs){extra}')
        if kind == 'real': best_real[(rule, key)] = r
        if kind in ('perm', 'shuffle'): null_max[rule].append(r['test_max'])
    lines.append('## verdict per rule (real test_max over 30 restarts vs controls\' test_max; 1 perm + 1 shuffle = 2 null arrows per rule, each 30 restarts)')
    for rule in ('sum', 'prod', 'pos'):
        reals = [(k, best_real[(rule, k)]) for k in ('seq_raw', 'seq_strong', 'seq_all') if (rule, k) in best_real]
        lines.append(f'{rule}: real test(best-train) ' + ', '.join(f'{k}={r["test"]:.3f}' for k, r in reals) +
                     f'; real test_max raw={best_real[(rule,"seq_raw")]["test_max"]:.3f}; null test_max values={[round(v,3) for v in null_max[rule]]} (max {max(null_max[rule]):.3f})')
    r = best_real[('sum', 'seq_raw')]
    lines.append(f'best sum assignment (seq_raw, best-train restart): ' + ' '.join(f'W{s}={v}' for s, v in zip(SIGNS, r['best_assign'])))
    lines.append(f'stable signs (sum, seq_raw): {r["stable_signs"]}')
    open(OUT, 'w').write('\n'.join(lines) + '\n')
    json.dump({f'{a}|{b}|{c}|{d}': r for a, b, c, d, r in results}, open(OUT.replace('.txt', '.json'), 'w'), default=float)
    print('\n'.join(lines))
