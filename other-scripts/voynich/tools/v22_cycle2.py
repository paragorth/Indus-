"""v22 cycle 2: (a) IT2a replication; (b) whole pages as units (paragraph order shuffled within page as null);
(c) does Currier A run the same program as Currier B? Transfer of stage tilts (log P(f|stage,lp) - log P(f|lp))
learned on one language to held-out pages of the other, against the same transfer onto line-shuffled targets;
(d) the program itself: full-data fit with many restarts, restart stability, stage profiles (q-/a-/l- drift)."""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import v22_lib as L

SL = [2, 3, 4, 6, 9]
R = 8


def shuffle_paras(C, rng):
    out = []
    for p in C:
        q = dict(p); ps = [list(map(list, pa)) for pa in p['paras']]; rng.shuffle(ps); q['paras'] = ps; out.append(q)
    return out


def t_eval(spec):
    name, kind, level, mode, seed = spec
    f = f'c2_{name}.json'
    got = L.load(f)
    if got: return got
    t0 = time.time()
    C = L.voynich_pages(kind) if kind in ('ZL3b', 'IT2a') else L.herbal_pages(kind)
    rng = random.Random(seed)
    coder = L.Coder(L.units(C, level))
    if mode == 'real': U = L.units(C, level)
    elif mode == 'lineshuf': U = L.shuffle_lines(L.units(C, level), rng)
    elif mode == 'wordshuf': U = L.shuffle_words_lp(L.units(C, level), rng)
    elif mode == 'parashuf': U = L.units(shuffle_paras(C, rng), level)
    elif mode == 'inl': U = L.shuffle_inline(L.units(C, level), rng)
    elif mode == 'inl_lineshuf': U = L.shuffle_lines(L.shuffle_inline(L.units(C, level), rng), rng)
    elif mode.startswith('plant') or mode.startswith('sham'):
        # plantE<rho>: very elastic stage durations (Dirichlet 0.5) and strong tilt 2.0
        el = 'E' in mode
        rho = float(mode.replace('plant', '').replace('sham', '').replace('E', ''))
        U, truth = L.plant_program(L.units(C, level), np.random.default_rng(seed), S=4, rho=rho,
                                   tilt=(2.0 if el else 1.0) if mode.startswith('plant') else 0.0,
                                   conc=0.5 if el else 2.0)
    codes = L.code_all(U, coder)
    res = L.evaluate(codes, coder.sizes, L.folds_by_unit(U, 5, 1000 + seed), SL, R, seed)
    if mode.startswith('plant'):
        seq = L.Seq(codes)
        m = L.fit_hmm(seq, coder.sizes, 4, 3 * R, np.random.default_rng(seed))
        res['ari4'] = L.ari(np.concatenate(truth), np.concatenate(L.viterbi_paths(m, seq, 4)))
    res.update(name=name, mode=mode, secs=time.time() - t0)
    L.save(f, res)
    print(name, f"{res['delta']:+.1f}", res['best_hmm_k'], f"{res['best_hmm']:.1f}", f"{res['secs']:.0f}s", flush=True)
    return res


def tilt_model(model, E0, S):
    """stage tilt T_k[s,lp,v] = log P(v|s,lp) - log P(v|lp)"""
    E, lA, _ = model
    return [E[k] - E0[k][None] for k in range(len(E))], lA


def apply_tilt(T, lA, E0t):
    E = []
    for k in range(len(E0t)):
        z = E0t[k][None] + T[k]
        E.append(z - np.log(np.exp(z).sum(-1, keepdims=True)))
    return (E, lA, 0.0)


def t_transfer(spec):
    name, S, seed, nshuf = spec
    f = f'c2_{name}.json'
    got = L.load(f)
    if got: return got
    t0 = time.time()
    C = L.voynich_pages('ZL3b')
    U = L.units(C, 'para'); coder = L.Coder(U)
    langs = {'A': [u for u in U if u['lang'] == 'A'], 'B': [u for u in U if u['lang'] == 'B']}
    nrng = np.random.default_rng(seed)
    out = {}
    folds = {g: L.folds_by_unit(langs[g], 5, 77) for g in 'AB'}
    acc = {}
    for fo in range(5):
        tr = {g: [c for c, h in zip(L.code_all(langs[g], coder), folds[g]) if h != fo] for g in 'AB'}
        teU = {g: [u for u, h in zip(langs[g], folds[g]) if h == fo] for g in 'AB'}
        E0 = {g: L.fit_m0(tr[g], coder.sizes) for g in 'AB'}
        M = {g: L.fit_hmm(L.Seq(tr[g]), coder.sizes, S, R, nrng) for g in 'AB'}
        TT = {g: tilt_model(M[g], E0[g], S) for g in 'AB'}
        for tgt in 'AB':
            variants = [('real', teU[tgt])] + [(f'shuf{i}', L.shuffle_lines(teU[tgt], random.Random(seed * 100 + fo * 10 + i)))
                                              for i in range(nshuf)]
            for vn, uu in variants:
                cc = L.code_all(uu, coder); sq = L.Seq(cc)
                l0, n = L.ll_m0(E0[tgt], cc)
                for src in 'AB':
                    l, _ = L.ll_hmm(apply_tilt(*TT[src], E0[tgt]), sq, S)
                    key = f'{src}->{tgt}:{vn}'
                    a = acc.setdefault(key, [0.0, 0]); a[0] += l - l0; a[1] += n
    for k, (g, n) in acc.items(): out[k] = g / n / L.LN2 * 1000
    out.update(name=name, S=S, secs=time.time() - t0)
    L.save(f, out)
    print(name, json.dumps({k: round(v, 1) for k, v in out.items() if ':' in k}), flush=True)
    return out


def t_profile(spec):
    name, kind, level, S, restarts, seed = spec
    f = f'c2_{name}.json'
    got = L.load(f)
    if got: return got
    t0 = time.time()
    C = L.voynich_pages(kind)
    U = L.units(C, level); coder = L.Coder(U); codes = L.code_all(U, coder); seq = L.Seq(codes)
    nrng = np.random.default_rng(seed)
    # many restarts, keep the top 5 polished solutions to measure stability
    sols = []
    for r in range(restarts):
        E, lA = L._init_random(seq, coder.sizes, S, nrng)
        for it in range(15):
            ll, g, x = L._fb(L._logB(seq, E), seq.offs, lA, S, True); E, lA = L._m_step(seq, g, x, coder.sizes, S)
        sols.append((ll, E, lA))
    sols.sort(key=lambda z: -z[0])
    top = []
    for ll, E, lA in sols[:5]:
        for it in range(40):
            ll, g, x = L._fb(L._logB(seq, E), seq.offs, lA, S, True); E, lA = L._m_step(seq, g, x, coder.sizes, S)
        top.append((E, lA, ll))
    top.sort(key=lambda z: -z[2])
    paths = [np.concatenate(L.viterbi_paths(m, seq, S)) for m in top]
    aris = [L.ari(paths[0], p) for p in paths[1:]]
    best = top[0]; path = paths[0]
    words = [w for u in U for l in u['lines'] for w in l]
    rel = np.concatenate([c['r'] for c in codes]); li = np.concatenate([c['li'] for c in codes])
    lp = seq.lp
    lang = np.concatenate([[u['lang']] * len(c['lp']) for u, c in zip(U, codes)])
    st = []
    # paragraph-level: how many stages does a paragraph visit, at which relative position do stages start
    for s in range(S):
        m = path == s
        ws = [w for w, k in zip(words, m) if k]
        n = len(ws)
        fst = np.array([w[0] for w in ws]) if n else np.array([])
        st.append({'stage': s, 'n': int(n), 'share': n / len(words), 'rel_mean': float(rel[m].mean()) if n else None,
                   'line_mean': float(li[m].mean()) if n else None,
                   'q': float((fst == 'q').mean()) if n else None, 'a_l': float(np.isin(fst, ['a', 'l']).mean()) if n else None,
                   'o': float((fst == 'o').mean()) if n else None, 'ch': float((fst == 'C').mean()) if n else None,
                   'sh': float((fst == 'S').mean()) if n else None, 'd': float((fst == 'd').mean()) if n else None,
                   'gallows_any': float(np.mean([any(g in w for g in 'tkpfTKPF') for w in ws])) if n else None,
                   'wlen': float(np.mean([len(w) for w in ws])) if n else None,
                   'lineinit': float((lp[m] == 0).mean()) if n else None,
                   'langA': float((lang[m] == 'A').mean()) if n else None,
                   'stay': float(np.exp(best[1][s, 0]))})
    base = {'q': float(np.mean([w[0] == 'q' for w in words])), 'a_l': float(np.mean([w[0] in 'al' for w in words])),
            'wlen': float(np.mean([len(w) for w in words]))}
    visits = []
    for q in range(len(seq.offs) - 1):
        p = path[seq.offs[q]:seq.offs[q + 1]]; visits.append(len(set(p.tolist())))
    out = {'name': name, 'S': S, 'restarts': restarts, 'top_ll': [t[2] for t in top], 'ari_top': aris, 'stages': st,
           'base': base, 'visits_mean': float(np.mean(visits)), 'visits_hist': np.bincount(visits).tolist(),
           'secs': time.time() - t0}
    L.save(f, out)
    print(name, 'ARI', [round(a, 2) for a in aris], flush=True)
    return out


def run(spec):
    k = spec[0]
    if k == 'eval': return t_eval(spec[1:])
    if k == 'transfer': return t_transfer(spec[1:])
    return t_profile(spec[1:])


if __name__ == '__main__':
    S_best = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    jobs = [('eval', 'ZL_inl', 'ZL3b', 'para', 'inl', 20),
            ('eval', 'ZL_inl_lineshuf_0', 'ZL3b', 'para', 'inl_lineshuf', 21),
            ('eval', 'LA_inl', 'LA', 'para', 'inl', 20),
            ('eval', 'LA_inl_lineshuf_0', 'LA', 'para', 'inl_lineshuf', 21),
            ('eval', 'IT_inl', 'IT', 'para', 'inl', 20),
            ('eval', 'IT_inl_lineshuf_0', 'IT', 'para', 'inl_lineshuf', 21),
            ('eval', 'ZL_inl_lineshuf_1', 'ZL3b', 'para', 'inl_lineshuf', 22),
            ('eval', 'ZL_plantE0.5_0', 'ZL3b', 'para', 'plantE0.5', 80),
            ('eval', 'ZL_shamE0.5_0', 'ZL3b', 'para', 'shamE0.5', 80),
            ('eval', 'ZL_plantE0.25_0', 'ZL3b', 'para', 'plantE0.25', 81),
            ('eval', 'ZL_shamE0.25_0', 'ZL3b', 'para', 'shamE0.25', 81),
            ('profile', 'ZL_profile', 'ZL3b', 'para', S_best, 300, 7),
            ('transfer', 'ZL_AB_transfer', S_best, 3, 2),
            ('eval', 'IT2_real', 'IT2a', 'para', 'real', 0),
            ('eval', 'IT2_lineshuf_0', 'IT2a', 'para', 'lineshuf', 100),
            ('eval', 'IT2_wordshuf_0', 'IT2a', 'para', 'wordshuf', 200),
            ('eval', 'ZLpage_real', 'ZL3b', 'page', 'real', 0),
            ('eval', 'ZLpage_parashuf_0', 'ZL3b', 'page', 'parashuf', 900),
            ('eval', 'ZLpage_wordshuf_0', 'ZL3b', 'page', 'wordshuf', 910),
            ]
    with Pool(2) as P:
        res = P.map(run, jobs, chunksize=1)
    L.save('c2_all.json', res)
