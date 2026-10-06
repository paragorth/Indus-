#!/usr/bin/env python3
"""v83 cycle 2: every B-grade Voynich result re-run with the loop's own code on several versions of the text
(legacy, all, clean, agree, and random-thinning controls thin-clean-1..3, thin-agree-1..3; see v83_harness).
Usage: python3 v83_c2.py [test,...]   (jobs = test x version x transcription; 2 workers; resumable)
Out: data/v83_ckpt/c2/<test>__<version>__<name>.json"""
import os, sys, json, time, random, math, collections, traceback
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(_v, '1')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import v83_harness as H
import v83_parse as P

OUT = os.path.join(P.CK, 'c2'); os.makedirs(OUT, exist_ok=True)
NAMES = ['ZL3b', 'IT2a']


# ------------------------------------------------------------------ tests (each returns a flat dict of numbers)
def t_v78(name):
    """v75 doubling + v78 no neighbour avoidance: E1c tokens, nested shuffles (v78_c1.summarize)."""
    import v72_lib as V, v78_c1 as C
    pages = V.voynich(name)
    r, ci = C.summarize(pages, B=100)
    out = {k: float(v) for k, v in r.items()}
    out.update({k + '_lo': float(ci[k][0]) for k in ci}); out.update({k + '_hi': float(ci[k][1]) for k in ci})
    out['ntok'] = sum(len(l['w']) for p in pages for l in p['lines'])
    return out


def t_v61(name):
    """v61: MI(last glyph of line-final word; first glyph of next line) vs line starts permuted in page."""
    import v61_lib as L, v61_xline as X
    r = X.run(L.load_vms(name))
    ex = r['cross_mi'] - r['cross_null']
    return dict(cross_excess=ex, cross_z=ex / r['cross_null_sd'], within_excess=r['within_mi_same_n'] - r['within_null_same_n'],
                n_cross=r['n_cross'])


def t_v72(name):
    """v72: frozen E1c extraction, ladder trained on discovery leaves, scored on holdout leaves."""
    import v72_lib as L
    S = L.voynich(name)
    X = L.extract(S, L.RULES['E1c_keepd'])
    half = [L.leaf_half(p['id']) for p in S]
    tr = [p for p, h in zip(X, half) if h == 0]; te = [p for p, h in zip(X, half) if h == 1]
    lad = L.ladder(tr, te, W=20, seed=0); lad.pop('per_tok')
    return dict(excess=lad['gain_msg'] - lad['gain_msgsh'], gain_page=lad['gain_page'], gain_big=lad['gain_big'],
                gain_msg=lad['gain_msg'], ntest=sum(len(l['w']) for p in te for l in p['lines']))


def t_v82(name):
    """v81 frozen onset test (3 frozen definitions) + v82 B measurement: diagonal share, line-break crossing."""
    import v72_lib as V, v82_lib as K, v81_lib as L81
    sys.argv = sys.argv[:1]
    import v82_c2 as C2
    pages = V.voynich(name); c = K.Mini(pages); rng = np.random.default_rng(822)
    out = {}
    for di, (g, z) in enumerate(K.frozen(c, nrep=20)):
        out['gain_d%d' % di] = g; out['z_d%d' % di] = z
    for di, d in enumerate(K.DEFS[:1]):
        o = L81.onset(c, d)
        D, Nm, noise = C2.ext_table(c, o, o, rng)
        pos = D.clip(min=0)
        out['diag'] = float(np.trace(pos) / (pos.sum() + 1e-9))
        out['asym'] = C2.asym(D); out['asym_noise'] = float(np.mean([C2.asym(x) for x in noise]))
        cr = C2.cross(c, o, rng)
        out['cross_gain'], out['cross_z'] = cr['cross'][0], cr['cross'][1]
        out['inline_gain'], out['inline_z'] = cr['inline'][0], cr['inline'][1]
    return out


def t_v54(name):
    """v54: near-repeated 3-word passages (<= 3 edits) real vs nulls; witness-word test (variants per exact excess)."""
    import v54_lib as V, v54_c1 as C1, v54_c5 as C5
    pages = V.voynich(name)
    out = {}
    n_real = len(V.families(pages)[1])
    nulls = {'lineshuf': V.null_lineshuf(pages, 1), 'posshuf': C1.null_posshuf(pages, 1), 'pageshuf': C1.null_pageshuf(pages, 1),
             'markov': V.null_markov(pages, 1)}
    best = 0
    for k, q in nulls.items():
        n = len(V.families(q)[1]); out['pairs_' + k] = n; best = max(best, n)
    out['pairs_real'] = n_real; out['ratio_best_null'] = n_real / max(1, best)
    r = C5.run(pages, nperm=60)
    out.update(var_per_exact=r['var_per_exact_excess'] or float('nan'), var_se=r['var_per_exact_excess_se'] or float('nan'),
               exact_z=r['exact_z'], variant_z=r['variant_z'])
    return out


def t_v56(name):
    """v56: page-to-page topical tie (table test z_hit, all tokens, skeleton and word keys, 3 page splits)."""
    import v56_lib as L, v56_c3 as C3
    C = L.voynich(name)
    E = L.build_R(C); TT = L.token_table(C, E)
    out = {}
    for split in range(3):
        trp = L.split_pages(E, seed=split); tok_tr = trp[E['tok_page']]
        m = np.ones(len(TT['words']), bool)
        for key in ('skel', 'word'):
            r = C3.table_test(E, TT, C, m, tok_tr, key=key)
            out['z_hit_%s_%d' % (key, split)] = r['z_hit']; out['z_spec_%s_%d' % (key, split)] = r['z_spec']
    out['z_hit_skel'] = float(np.mean([out['z_hit_skel_%d' % s] for s in range(3)]))
    out['z_hit_word'] = float(np.mean([out['z_hit_word_%d' % s] for s in range(3)]))
    return out


def _v59_voynich(name, lang):
    import re
    pages = collections.OrderedDict()
    for l in H.derived_lines(name):
        if l['ltype'] != 'P' or l['lang'] != lang: continue
        ws = [w for w in l['words'] if w and '?' not in w and re.fullmatch(r'[a-z]+', w)]
        if not ws: continue
        p = pages.setdefault(l['folio'], {'id': l['folio'], 'sec': str(l['illus']), 'hand': str(l['hand']),
                                          'quire': l['quire'], 'lines': []})
        p['lines'].append(ws)
    return [p for p in pages.values() if sum(map(len, p['lines'])) >= 10]


def t_v59(name):
    """v59: B herbal pages lean to A-herbal (vs A-pharma) vocabulary: AUC, label-permutation z; first/last 3 glyphs."""
    sys.argv = sys.argv[:1]
    import v59_c3b as C
    C.NPERM = 1000
    rng = np.random.default_rng(77)
    A = _v59_voynich(name, 'A'); B = _v59_voynich(name, 'B')
    out = {}
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        r = C.run('id', A, 'H', 'P', B, {'H'}, {'B', 'S', 'C'}, lambda w: w, rng)
        out['auc'], out['z'] = r['auc'], r['z']
        sk = lambda w: w[:3]
        A3 = [dict(p, lines=[[sk(w) for w in l] for l in p['lines']]) for p in A]
        r = C.run('f3', A3, 'H', 'P', B, {'H'}, {'B', 'S', 'C'}, sk, rng); out['auc_first3'], out['z_first3'] = r['auc'], r['z']
        sk2 = lambda w: w[-3:]
        A4 = [dict(p, lines=[[sk2(w) for w in l] for l in p['lines']]) for p in A]
        r = C.run('l3', A4, 'H', 'P', B, {'H'}, {'B', 'S', 'C'}, sk2, rng); out['auc_last3'], out['z_last3'] = r['auc'], r['z']
    out['n_pos'], out['n_neg'] = r['n_pos'], r['n_neg']
    return out


def t_v79(name):
    """v79: line-start chain within paragraphs: repeat of the mark above (avoidance) and first-order table MI,
    each against 50 within-paragraph shuffles of the body-line marks."""
    import v72_lib as V, v79_lib as L
    pages = V.voynich(name)
    T, al = L.chain_table(pages)
    rng = np.random.default_rng(79)

    def stats(T):
        g, pa, ps = T['g'], T['para'], T['ps']
        ok = (pa[1:] == pa[:-1]) & (ps[1:] == 0) & (ps[:-1] == 0)
        a, b = g[:-1][ok], g[1:][ok]
        rep = float((a == b).mean())
        from collections import Counter
        j = Counter(zip(a.tolist(), b.tolist())); n = len(a)
        ca = Counter(a.tolist()); cb = Counter(b.tolist())
        mi = sum(v / n * math.log2(v * n / (ca[x] * cb[y])) for (x, y), v in j.items())
        return rep, mi, n
    rep, mi, n = stats(T)
    nul = [stats(L.shuffle_within_para(T, rng)) for _ in range(50)]
    nr = np.array([x[0] for x in nul]); nm = np.array([x[1] for x in nul])
    return dict(rep=rep, rep_null=float(nr.mean()), rep_ratio=rep / nr.mean(), rep_z=float((rep - nr.mean()) / (nr.std() + 1e-9)),
                mi=mi, mi_null=float(nm.mean()), mi_z=float((mi - nm.mean()) / (nm.std() + 1e-9)), n=n)


def t_v68(name):
    """v68: line-initial pool = line-first glyph MI (v53 panel, 2,000-token chunks, mean), target 0.114, every coded
    plaintext <= 0.044. Plus neighbour coupling by Currier group (proxy for v68 c3: adjacent-word edit similarity minus
    within-line shuffle, per 2,000-token chunk)."""
    import vlib, v53_lib as L53
    recs = vlib.load_voynich(name, ltypes=('P',))
    lines = []
    for r in recs:
        ws = [L53.vglyphs(w) for w, u in zip(r['words'], r['uncertain']) if not u and '?' not in w]
        ws = [w for w in ws if w]
        if ws: lines.append((r['illus'], r['lang'], ws))
    ch = L53.chunks([l[2] for l in lines], 2000)
    mis = [L53.panel(c, 2000)['mi_lf'] for c in ch if L53.panel(c, 2000)]
    out = dict(mi_lf=float(np.mean(mis)), mi_lf_min=float(np.min(mis)), mi_lf_max=float(np.max(mis)), nchunks=len(mis))
    # neighbour coupling by Currier group: junction MI (last glyph -> next first glyph, inside lines) minus the same
    # after shuffling words inside each line; 2,000-token samples of whole lines, 10 draws (v68 c3 'step' proxy)
    rng = random.Random(68)
    import v61_lib as L61
    grp = collections.defaultdict(list)
    for il, lg, ws in lines:
        g = {('B', 'B'): 'Bbio', ('S', 'B'): 'Bstars', ('H', 'B'): 'Bherbal', ('H', 'A'): 'Aherbal', ('P', 'A'): 'Apharma'}.get((il, lg))
        if g: grp[g].append(ws)
    for g, ls in grp.items():
        vals = []
        for _ in range(10):
            idx = list(range(len(ls))); rng.shuffle(idx); sel = []; n = 0
            for i in idx:
                sel.append(ls[i]); n += len(ls[i])
                if n >= 2000: break
            obs = L61.coupling_mi([{'words': l} for l in sel])
            sh = [rng.sample(l, len(l)) for l in sel]
            vals.append(obs - L61.coupling_mi([{'words': l} for l in sh]))
        out['nb_' + g] = float(np.mean(vals))
    a = np.mean([out['nb_Aherbal'], out['nb_Apharma']]); b = np.mean([out['nb_Bbio'], out['nb_Bstars'], out['nb_Bherbal']])
    out['nb_ratio_B_over_A'] = float(b / a) if a > 0 else float('inf')
    return out


def _v71_sections(name):
    import v71_lib as L
    d = H.derived_lines(name)
    out = {}
    for sec, (il, lg) in L.SECTIONS.items():
        pages, cur_fol = [], None
        for r in d:
            if r['ltype'] != 'P' or r['illus'] != il or r['lang'] != lg: continue
            ws = [L.vglyphs(w) for w, u in zip(r['words'], r['uncertain']) if not u and '?' not in w]
            if not ws: continue
            if r['folio'] != cur_fol:
                pages.append([]); cur_fol = r['folio']
            if r['para_start'] or not pages[-1]:
                pages[-1].append([])
            pages[-1][-1].append(ws)
        out[sec] = pages
    return out


def t_v71(name):
    """v71: page-order trace S4 (held-out damage when pages are shuffled inside a 2-3k-word chunk), 120 model types;
    reference: the self-citation generator (window 60) on the same chunks (computed on every version)."""
    import v71_lib as L
    M = L.model_types(120)
    V = _v71_sections(name)
    s4, g4 = [], []
    for sec, pages in V.items():
        for i, ch in enumerate(L.split_chunks(pages)):
            fp = L.summarise(L.fingerprint(ch, M), M); s4.append(fp['S4|ALL'])
            if i == 0:
                gp = L.gen_selfcit(ch, 11); fg = L.summarise(L.fingerprint(gp, M), M); g4.append(fg['S4|ALL'])
    return dict(s4_mean=float(np.mean(s4)), s4_min=float(np.min(s4)), s4_max=float(np.max(s4)), nchunks=len(s4),
                selfcit_s4_mean=float(np.mean(g4)))


TESTS = dict(v78=t_v78, v61=t_v61, v72=t_v72, v82=t_v82, v54=t_v54, v56=t_v56, v59=t_v59, v79=t_v79, v68=t_v68,
             v71=t_v71)


def job(a):
    test, ver, name = a
    fn = os.path.join(OUT, '%s__%s__%s.json' % (test, ver, name))
    if os.path.exists(fn): return a, 'skip'
    H.set_version(ver)
    t0 = time.time()
    try:
        r = TESTS[test](name)
    except Exception as e:
        return a, 'ERR ' + traceback.format_exc()[-800:]
    r = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in r.items()}
    r['_sec'] = time.time() - t0
    json.dump(r, open(fn, 'w'))
    return a, '%.0fs %s' % (time.time() - t0, json.dumps({k: round(v, 4) for k, v in r.items() if isinstance(v, float)})[:400])


if __name__ == '__main__':
    tests = sys.argv[1].split(',') if len(sys.argv) > 1 else list(TESTS)
    vers = sys.argv[2].split(',') if len(sys.argv) > 2 else H.VERSIONS_MAIN + H.THIN
    names = sys.argv[3].split(',') if len(sys.argv) > 3 else NAMES
    jobs = [(t, v, n) for t in tests for v in vers for n in names]
    if os.environ.get('V83_SERIAL'):
        for j in jobs: print(*job(j), flush=True)
    else:
        from multiprocessing import Pool
        with Pool(2, maxtasksperchild=1) as Pp:
            for a, s in Pp.imap_unordered(job, jobs):
                print(a, s, flush=True)
