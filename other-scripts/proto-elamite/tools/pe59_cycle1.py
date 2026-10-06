"""pe59 cycle 1: freeze the reading, fit the generative grammar on the training half, score held-out tablets
against simpler models, ablations, the same grammar with sign roles shuffled, and random metrologies.

usage: python3 pe59_cycle1.py [PE|PC] [n_shuffles]
"""
import sys, os, json, math, random, time
from collections import Counter
from multiprocessing import Pool
import numpy as np
from fractions import Fraction as Fr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe59_lib import (READING, build_pe, build_pc, split, pe_roles, pc_roles, pe_maps, pc_maps, shuffle_roles, sha,
                      seed, tablet_type, ncls, value, CK, DATA, LOOPS, PC_KNOWN)
from pe59_models import Grammar

CORPUS = sys.argv[1] if len(sys.argv) > 1 else 'PE'
NSH = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def corpus():
    if CORPUS == 'PE':
        T = build_pe(); return T, pe_roles(), pe_maps()
    T = build_pc(); return T, pc_roles(), pc_maps()


def fit(tr, roles, maps):
    g = Grammar(tr, roles, maps, corpus=CORPUS)
    g.fit_signs()
    g.num_bits_train = g.fit_nums()
    g.par_ent = g.num_par_G
    g.par_ent_N2 = g.num_par_N2
    g.tot_bits_train = g.fit_nums(total=True)
    g.par_tot = g.num_par_G
    g.par_tot_N2 = g.num_par_N2
    g.num_par_G, g.num_par_N2 = g.par_ent, g.par_ent_N2
    return g


def nb_allsigns(tr):
    """Data-only tablet-type classifier: naive Bayes over the set of signs on the tablet (temperature fitted)."""
    cnt = {'CAPT': Counter(), 'CNTT': Counter()}; n = Counter()
    for t in tr:
        tau = tablet_type(t)
        if tau:
            n[tau] += 1
            cnt[tau].update(set(s for l in t['lines'] for s in l['signs']))

    def post(t, th, own=False):
        ss = set(s for l in t['lines'] for s in l['signs'])
        tau0 = tablet_type(t) if own else None
        nC = n['CAPT'] - (tau0 == 'CAPT'); nN = n['CNTT'] - (tau0 == 'CNTT')
        lo = math.log((nC + 1) / (nN + 1))
        for s in ss:
            cC = cnt['CAPT'][s] - (tau0 == 'CAPT'); cN = cnt['CNTT'][s] - (tau0 == 'CNTT')
            lo += th * (math.log((cC + 0.5) / (nC + 1)) - math.log((cN + 0.5) / (nN + 1)))
        return 1 / (1 + math.exp(-max(-30, min(30, lo))))
    best = None
    for th in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        v = 0
        for t in tr:
            tau = tablet_type(t)
            if tau:
                p = post(t, th, own=True)
                v -= math.log(max(1e-12, p if tau == 'CAPT' else 1 - p))
        if best is None or v < best[0]:
            best = (v, th)
    th = best[1]
    return lambda t: post(t, th)


def tau_metrics(probs, tabs):
    ll, acc, n = 0, 0, 0
    for t, p in zip(tabs, probs):
        tau = tablet_type(t)
        if tau is None:
            continue
        n += 1
        q = p if tau == 'CAPT' else 1 - p
        ll -= math.log2(max(1e-12, q))
        acc += q > 0.5
    return {'n': n, 'bits': ll / max(1, n), 'acc': acc / max(1, n)}


def naive_closes(t, g):
    """Pre-reading arithmetic: plain numerals always counts (sexagesimal or decimal), capacity codes on their own chain."""
    capsmall = {c: v / 120 for c, v in g.capmap.items() if c not in ('N01', 'N14', 'N45', 'N34', 'N48')}
    ents = [l for l in t['lines'] if l['role'] == 'E' and l['numclean']]
    tot = [l for l in t['lines'] if l['role'] == 'T' and l['numclean']]
    if len(ents) < 2 or not tot or not all(l['numclean'] for l in t['lines'] if l['role'] == 'E'):
        return None
    for m in g.cntmaps.values():
        mm = dict(m); mm.update(capsmall)
        vals = [value(l['nums'], mm) for l in ents]
        T = value(tot[0]['nums'], mm)
        if T is not None and all(v is not None for v in vals) and sum(vals) == T:
            return True
    return False


def closure(g, tabs, taus, rng, ndraw=200):
    """Closure rate under the reading's tau (MAP), naive arithmetic, either tau (flexible), and random-total nulls."""
    res = {'read': [], 'naive': [], 'any': [], 'cap': [], 'cnt': []}
    elig = []
    for t, pt in zip(tabs, taus):
        c = g.closes(t, 'CAPT' if pt > 0.5 else 'CNTT')
        if c is None:
            continue
        elig.append((t, pt))
        res['read'].append(c)
        res['naive'].append(bool(naive_closes(t, g)))
        a, b = g.closes(t, 'CAPT'), g.closes(t, 'CNTT')
        res['any'].append(bool(a or b)); res['cap'].append(bool(a)); res['cnt'].append(bool(b))
    out = {k: (sum(v), len(v)) for k, v in res.items()}
    # null: each tablet's total replaced by the total of another eligible tablet
    totals = [[l for l in t['lines'] if l['role'] == 'T'][0] for t, _ in elig]
    null = {'read': [], 'any': [], 'naive': []}
    for d in range(ndraw):
        perm = list(range(len(elig))); rng.shuffle(perm)
        cr = ca = cn = 0
        for (t, pt), j in zip(elig, perm):
            t2 = dict(t); t2['lines'] = [l if l['role'] != 'T' else totals[j] for l in t['lines']]
            cr += bool(g.closes(t2, 'CAPT' if pt > 0.5 else 'CNTT'))
            ca += bool(g.closes(t2, 'CAPT') or g.closes(t2, 'CNTT'))
            cn += bool(naive_closes(t2, g))
        null['read'].append(cr); null['any'].append(ca); null['naive'].append(cn)
    out['null'] = {k: (float(np.mean(v)), float(np.percentile(v, 95))) for k, v in null.items()}
    return out


def heldout(g, ho):
    sb, per = g.score_signs(ho)
    nb, nrows = g.score_nums(ho, parG=g.par_ent)
    g.num_par_N2 = g.par_tot_N2
    tb, trows = g.score_nums(ho, total=True, parG=g.par_tot)
    g.num_par_N2 = g.par_ent_N2
    taus_signs = [g.tau_post(t, with_nums=False) for t in ho]
    taus_full = [g.tau_post(t, with_nums=True) for t in ho]
    return {'sign': {k: float(v.mean()) for k, v in sb.items()}, 'n_sign': int(len(sb['G'])),
            'num': {k: [float(v[:, 0].mean()), float(v[:, 1].mean())] for k, v in nb.items()}, 'n_num': len(nrows),
            'tot': {k: [float(v[:, 0].mean()), float(v[:, 1].mean())] for k, v in tb.items()} if len(trows) else {},
            'n_tot': len(trows), 'tau_signs': tau_metrics(taus_signs, ho)}, (sb, per, nb, nrows, tb, trows,
                                                                                 taus_signs, taus_full)


def boot(per_tab_a, per_tab_b, rng, B=2000):
    ids = list(per_tab_a)
    d = np.array([per_tab_a[i] - per_tab_b[i] for i in ids])
    n = np.array([1] * len(ids))
    bs = []
    for _ in range(B):
        ix = rng.integers(0, len(ids), len(ids))
        bs.append(d[ix].sum())
    return float(d.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def shuffled_job(args):
    k, = args
    T, roles, maps = corpus()
    tr, ho = split(T)
    rng = random.Random(seed('pe59shuf%s%d' % (CORPUS, k)))
    R = shuffle_roles(roles, T, rng)
    g = fit(tr, R, maps)
    s, raw = heldout(g, ho)
    rr = random.Random(seed('pe59clos%d' % k))
    s['closure'] = closure(g, ho, raw[7], rr, ndraw=20)
    return s


def metro_job(args):
    k, = args
    T, roles, maps = corpus()
    tr, ho = split(T)
    rng = random.Random(seed('pe59metro%s%d' % (CORPUS, k)))
    cap, cnt = maps
    vals = sorted(cap.values())
    # random ladder: random integer ratios between consecutive codes, same code order
    codes = sorted(cap, key=lambda c: cap[c])
    v = Fr(1); new = {}
    for c in codes:
        new[c] = v; v *= rng.choice([2, 3, 4, 5, 6, 8, 10, 12])
    cnt2 = {}
    for nm, m in cnt.items():
        cs = sorted(m, key=lambda c: m[c]); vv = Fr(1); mm = {}
        for c in cs:
            mm[c] = vv if m[c] >= 1 else m[c]
            if m[c] >= 1:
                vv *= rng.choice([2, 3, 5, 6, 10, 12])
        cnt2[nm] = mm
    g = fit(tr, roles, (new, cnt2))
    s, raw = heldout(g, ho)
    rr = random.Random(seed('pe59mclos%d' % k))
    s['closure'] = closure(g, ho, raw[7], rr, ndraw=20)
    return s


def main():
    t0 = time.time()
    T, roles, maps = corpus()
    tr, ho = split(T)
    out = {'corpus': CORPUS, 'n_train': len(tr), 'n_heldout': len(ho)}
    if CORPUS == 'PE':
        fz = {'reading': READING, 'weights_file_sha': sha(json.load(open(os.path.join(DATA, 'pe52_frozen_weights.json')))),
              'train_ids_sha': sha(sorted(t['id'] for t in tr)), 'heldout_ids_sha': sha(sorted(t['id'] for t in ho))}
        fz['sha256'] = sha(fz)
        fn = os.path.join(DATA, 'pe59_reading_frozen.json')
        if os.path.exists(fn):
            old = json.load(open(fn))
            assert old['sha256'] == fz['sha256'], 'reading changed after freezing'
        else:
            json.dump(fz, open(fn, 'w'), indent=1, sort_keys=True)
        out['reading_sha'] = fz['sha256']
    else:
        out['reading_sha'] = sha({'PC_KNOWN': {k: sorted(v) for k, v in PC_KNOWN.items()}})
    g = fit(tr, roles, maps)
    out['train'] = {'sign': {k: float(v) for k, v in g.sign_train_bits.items()}, 'num': g.num_bits_train,
                    'tot': g.tot_bits_train, 'theta': g.theta}
    params = {'sign_S3': [float(x) for x in g.sign_par_S3], 'sign_G': [float(x) for x in g.sign_par_G],
              'num_G': g.par_ent, 'num_N2': g.par_ent_N2, 'tot_G': g.par_tot, 'tot_N2': g.par_tot_N2, 'theta': g.theta}
    out['params'] = params
    out['params_sha'] = sha(params)
    print('fitted', CORPUS, out['params_sha'][:16], time.time() - t0, flush=True)
    json.dump(out, open(os.path.join(CK, 'c1_%s_frozen_params.json' % CORPUS), 'w'), indent=1, default=str)
    # ---------------- held-out (only now)
    s, raw = heldout(g, ho)
    out['heldout'] = s
    sb, per, nb, nrows, tb, trows, taus_signs, taus_full = raw
    rng = np.random.default_rng(seed('pe59boot'))
    # per-tablet sums for bootstrap
    pt = {}
    i = 0
    for tid, n in per:
        pt[tid] = {lv: float(sb[lv][i:i + n].sum()) for lv in sb}
        i += n
    out['boot_sign'] = {lv: boot({k: v['G'] for k, v in pt.items()}, {k: v[lv] for k, v in pt.items()}, rng)
                        for lv in ('S2', 'S3')}
    pn = {}
    for j, r in enumerate(nrows):
        d = pn.setdefault(r['id'], {lv: 0.0 for lv in nb})
        for lv in nb:
            d[lv] += float(nb[lv][j].sum())
    out['boot_num'] = {lv: boot({k: v['G'] for k, v in pn.items()}, {k: v[lv] for k, v in pn.items()}, rng)
                       for lv in ('N1', 'N2')}
    # tau classifiers from signs only
    nb_post = nb_allsigns(tr)
    out['tau_nb_allsigns'] = tau_metrics([nb_post(t) for t in ho], ho)
    maj = sum(1 for t in tr if tablet_type(t) == 'CAPT') / max(1, sum(1 for t in tr if tablet_type(t)))
    out['tau_majority'] = tau_metrics([maj] * len(ho), ho)
    # closure
    out['closure'] = closure(g, ho, taus_full, random.Random(seed('pe59clos')))
    # ablations (no refit): drop each numeral component on held-out
    abl = {}
    for comp in ('pi_ln', 'pi_rate', 'pi_copy'):
        P = dict(g.par_ent); P[comp] = 0.0
        bb, _ = g.score_nums(ho, parG=P)
        abl['no_' + comp] = [float(bb['G'][:, 0].mean()), float(bb['G'][:, 1].mean())]
    P = dict(g.par_ent); P['lam'] = 1.0
    bb, _ = g.score_nums(ho, parG=P)
    abl['no_tau_roles'] = [float(bb['G'][:, 0].mean()), float(bb['G'][:, 1].mean())]
    out['ablation_num'] = abl
    # rate rule hits on held-out
    hits = [r for r in nrows if r['rate'] is not None]
    out['rate_rule'] = {'applicable': len(hits), 'exact_hits': sum(1 for r in hits if r['rate'] > 0)}
    copies = [r for r in nrows if r['copy'] is not None]
    out['copy_rule'] = {'applicable': len(copies), 'hits': sum(1 for r in copies if r['copy'] > 0)}
    print('real done', time.time() - t0, flush=True)
    json.dump(out, open(os.path.join(CK, 'c1_%s.json' % CORPUS), 'w'), indent=1, default=str)
    # refit ablations: no weights, no AMB rule (plain numerals always counts)
    R0 = dict(roles); R0['weights'] = {}
    g0 = fit(tr, R0, maps)
    s0, _ = heldout(g0, ho)
    out['refit_no_weights'] = s0
    # controls
    with Pool(2) as pool:
        sh = pool.map(shuffled_job, [(k,) for k in range(NSH)])
        me = pool.map(metro_job, [(k,) for k in range(max(4, NSH // 2))])
    out['shuffled'] = sh
    out['metro'] = me
    out['time'] = time.time() - t0
    json.dump(out, open(os.path.join(CK, 'c1_%s.json' % CORPUS), 'w'), indent=1, default=str)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
