"""v30 cycle 2: (a) planted rewrite on Voynich A (positive control on the Voynich alphabet);
(b) transfer: rules learned on one section/genre, applied to unseen sections, vs random rule sets;
(c) massive random guessing: 2,000 random 6-rule sets per pair, fit gain vs held gain.
Usage: python3 v30_c2.py a|b|c
"""
import sys, os, json, random, time, collections
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v30_lib import Search, run_search, split_pages, distance, eval_rules, apply_word, rules_by_first
from v30_ladder import make_split, corpora, CK

PLANT = [('o', 'a', '#', None), ('dy', 'dey', None, '#'), ('k', 't', None, None), ('aiin', 'ain', None, None)]


def plant(pages):
    rb = rules_by_first(PLANT)
    return [[apply_word(w, rb) for w in p] for p in pages]


def part_a(seed, guided=False):
    C = corpora()
    rng = random.Random(500 + seed)
    idx = list(range(len(C['V_A']))); rng.shuffle(idx); h = len(idx) // 2
    X = [C['V_A'][i] for i in idx[:h]]; Y = plant([C['V_A'][i] for i in idx[h:]])
    xf, xh, _ = split_pages(X, 3000, 1500, 2 * seed + 1); yf, yh, _ = split_pages(Y, 3000, 1500, 2 * seed + 2)
    floor = distance(yf[:1500], yh)
    path = run_search(xf, yf, xh, yh, kmax=15, n_cand=200, seed=seed, guided=guided)
    found = path[-1]['rules']
    rec = [p for p in PLANT if tuple(p) in {tuple(r) for r in found}]
    return dict(seed=seed, floor=floor, path=path, recovered=rec)


def fit_all(X, Y, kmax=20, seed=0):
    S = Search(X, Y, seed=seed)
    fails = 0
    while len(S.rules) < kmax and fails < 2:
        if S.step(n_cand=200):
            fails = 0
        else:
            fails += 1
    return S


def part_b(job):
    """job = (name, X_fit corpus, Y_fit corpus, [(X_test, Y_test), ...])"""
    name, xs, ys, tests, seed = job
    C = corpora()
    words = lambda keys: [w for k in keys for p in C[k] for w in p]
    pages = lambda keys: [p for k in keys for p in C[k]]
    X = words(xs); Y = words(ys)
    S = fit_all(X, Y, kmax=20, seed=seed)
    rng = random.Random(seed)
    res = dict(name=name, rules=S.rules, tests=[])
    for xt, yt in tests:
        Xt = pages([xt]); Yt = pages([yt])
        xw = [w for p in Xt for w in p][:2500]; yw = [w for p in Yt for w in p][:2500]
        n = min(len(xw), len(yw)); xw, yw = xw[:n], yw[:n]
        d0 = distance(xw, yw)[0]; d1 = eval_rules(S.rules, xw, yw)[0]
        # random rule sets of the same size from the same proposal pool
        rnd = []
        for _ in range(100):
            S.rules_backup = S.rules; S.rules = []
            rr = [S.propose() for _ in range(len(S.rules_backup))]
            S.rules = S.rules_backup
            rnd.append(eval_rules(rr, xw, yw)[0])
        # shuffled-rule control: same lhs set, rhs permuted among rules
        rhs = [r[1] for r in S.rules]; shf = []
        for _ in range(50):
            rng.shuffle(rhs)
            shf.append(eval_rules([(r[0], h, r[2], r[3]) for r, h in zip(S.rules, rhs)], xw, yw)[0])
        res['tests'].append(dict(x=xt, y=yt, n=n, d0=d0, d1=d1, rnd=rnd, shf=shf))
        print(f'{name} {xt}->{yt} n {n} d0 {d0:.3f} learned {d1:.3f} random med {np.median(rnd):.3f} best {min(rnd):.3f} shuffled-rhs med {np.median(shf):.3f}', flush=True)
    return res


def part_c(job):
    name, seed, nsets, size = job
    xf, xh, yf, yh = make_split(name, seed)
    S = Search(xf, yf, seed=seed)
    base_fit = S.score; base_held = distance(xh, yh)[0]
    rng = random.Random(seed)
    out = []
    for i in range(nsets):
        rr = []
        while len(rr) < size:
            r = S.propose()
            if r not in rr:
                rr.append(r)
        fit, _ = S.try_rules(rr)
        out.append((base_fit - fit, rr))
    out.sort(key=lambda t: -t[0])
    top = out[:max(20, nsets // 100)]
    held_top = [base_held - eval_rules(rr, xh, yh)[0] for _, rr in top]
    rand20 = rng.sample(out, 40)
    held_rand = [base_held - eval_rules(rr, xh, yh)[0] for _, rr in rand20]
    fits = np.array([o[0] for o in out])
    return dict(name=name, seed=seed, frac_fit_pos=float((fits > 0).mean()), fit_q=[float(np.quantile(fits, q)) for q in (0.5, 0.9, 0.99)],
                held_top=held_top, held_rand=held_rand, fit_top=[o[0] for o in top], fit_rand=[o[0] for o in rand20],
                top_rules=[o[1] for o in top[:5]])


if __name__ == '__main__':
    part = sys.argv[1]
    if part == 'a':
        with Pool(2) as p:
            R = p.map(part_a, [0, 1])
        json.dump(R, open(os.path.join(CK, 'c2a.json'), 'w'), default=float)
        for r in R:
            p = r['path']
            print('seed', r['seed'], 'F', round(r['floor'][0], 3), 'D0', round(p[0]['held'], 3), 'Dk', round(p[-1]['held'], 3), 'k', p[-1]['k'], 'recovered', r['recovered'])
            print('   rules', p[-1]['rules'])
    elif part == 'ag':
        with Pool(2) as p:
            R = p.starmap(part_a, [(0, True), (1, True)])
        json.dump(R, open(os.path.join(CK, 'c2ag.json'), 'w'), default=float)
        for r in R:
            p = r['path']
            print('GUIDED seed', r['seed'], 'F', round(r['floor'][0], 3), 'D0', round(p[0]['held'], 3), 'Dk', round(p[-1]['held'], 3), 'k', p[-1]['k'], 'recovered', r['recovered'])
            print('   rules', p[-1]['rules'])
    elif part == 'b':
        jobs = [('VOY_herb', ['V_Aherb'], ['V_Bherb'], [('V_Apharm', 'V_Bbio'), ('V_Apharm', 'V_Bstar'), ('V_Apharm', 'V_B3')], 0),
                ('GER_bav1', ['G_Bav1'], ['G_Alem'], [('G_Bav2', 'G_Alem'), ('G_Bav2', 'G_Rip')], 0),
                ('LAT_ita', ['I_Lat'], ['I_Ita'], [('I_Lat', 'I_Com2')], 0),
                ('CZ_reform', ['C_Mod'], ['C_Old'], [('C_Mod2', 'C_Old')], 0)]
        with Pool(2) as p:
            R = p.map(part_b, jobs)
        json.dump(R, open(os.path.join(CK, 'c2b.json'), 'w'), default=float, ensure_ascii=False)
    elif part == 'c':
        names = ['V_AB', 'N_AA', 'N_BB', 'V_B2B3', 'L_ComCom', 'L_CzReform', 'L_BavBav', 'L_BavAlem', 'L_LatIta', 'U_GerIta', 'K_Swap3']
        jobs = [(n, s, 2000, 6) for s in (0,) for n in names]
        with Pool(2) as p:
            R = p.map(part_c, jobs)
        json.dump(R, open(os.path.join(CK, 'c2c.json'), 'w'), default=float, ensure_ascii=False)
        for r in R:
            print(f"{r['name']:10s} frac fit>0 {r['frac_fit_pos']:.2f} fit q50/90/99 {r['fit_q'][0]:+.4f} {r['fit_q'][1]:+.4f} {r['fit_q'][2]:+.4f} | top1% held gain {np.mean(r['held_top']):+.4f} (pos {np.mean(np.array(r['held_top'])>0):.2f}) | random held {np.mean(r['held_rand']):+.4f}")
