"""v43 cycle 1: sliding-window island scan (reading order) and group scan (section, quire, hand, Currier language,
bifolio), with controls: size-matched windows of 10 real languages (must all be language-like), planted islands of
Latin / German transliterated into Voynich units (must be found at true place and size), uniform generator texts
(trigram resynthesis, global junction forger; must give no island).
Usage: python3 v43_cycle1.py [nworkers]"""
import sys, time, random, json
import numpy as np
from collections import Counter, defaultdict
import v43_lib as L

NW = int(sys.argv[1]) if len(sys.argv) > 1 else 2
OUT = 'v43_cycle1.txt'
t0 = time.time()


def log(*a):
    print(round(time.time() - t0), *a, flush=True)


def feats_cached(name, chs):
    F = L.load(f'F_{name}.json')
    if F is None or len(F) != len(chs):
        F = L.feats_pool(chs, NW); L.save(f'F_{name}.json', F)
    return F


# ---------------- Voynich
P = L.voynich('ZL3b')
S = L.stream_of_pages(P)
chV = L.chunks(S)
FV = feats_cached('ZL', chV)
prV = L.probs(FV); csV = L.lscore(prV)
log('voynich chunks', len(chV), 'mean s', csV.mean())
chunk_page = [S[i * L.N + L.N // 2][1] for i in range(len(chV))]

# ---------------- languages
lang_ws = {}
lang_cs = {}
for name in L.LANGS:
    lines = L.corpus_lines(name, maxtok=14400)
    SL = L.stream_of_lines(lines)
    ch = L.chunks(SL)
    F = feats_cached(name, ch)
    cs = L.lscore(L.probs(F, exclude=(name,)))
    lang_cs[name] = cs
    lang_ws[name] = {W: L.window_scores(cs, W)[::W] for W in (6, 12, 24)}
    log(name, len(ch), 'mean s', round(cs.mean(), 3), 'min win12', round(lang_ws[name][12].min(), 3))
res = {'lang': {n: {W: v.tolist() for W, v in d.items()} for n, d in lang_ws.items()}}
TAU = {}
for W in (6, 12, 24):
    allw = np.concatenate([lang_ws[n][W] for n in L.LANGS])
    TAU[W] = dict(p5=float(np.percentile(allw, 5)), p1=float(np.percentile(allw, 1)), min=float(allw.min()),
                  frac_half=float((allw >= 0.5).mean()), n=len(allw))
log('TAU', TAU)
res['tau'] = TAU
tau = {W: min(0.5, TAU[W]['p5']) for W in TAU}   # language band edge: 0.5 or the languages' 5th pct, whichever lower

# ---------------- Voynich scan
scan = {}
for W in (6, 12, 24):
    ws = L.window_scores(csV, W)
    null = L.scan_null(csV, W, 2000, seed=W)
    i = int(ws.argmax())
    pgs = sorted({chunk_page[j] for j in range(i, i + W)})
    runs = L.runs_above(ws, tau[W])
    secs = Counter(P[chunk_page[j]]['sec'] for j in range(i, i + W))
    scan[W] = dict(max=float(ws.max()), p=float((null >= ws.max()).mean()), null95=float(np.percentile(null, 95)),
                   median=float(np.median(ws)), n_above_tau=int((ws >= tau[W]).sum()), nwin=len(ws),
                   runs=[(a, b, [P[chunk_page[a]]['id'], P[chunk_page[min(b + W - 1, len(chunk_page) - 1)]]['id']]) for a, b in runs],
                   best_pages=[P[p]['id'] for p in pgs], best_secs=dict(secs),
                   best_hand=dict(Counter(P[chunk_page[j]]['hand'] for j in range(i, i + W))),
                   best_quire=dict(Counter(P[chunk_page[j]]['quire'] for j in range(i, i + W))),
                   pG_best=float(prV[i:i + W, 2].mean()), pB_best=float(prV[i:i + W, 3].mean()))
    log('scan', W, {k: v for k, v in scan[W].items() if k != 'runs'}, 'nruns', len(runs))
res['scan'] = scan

# ---------------- window battery (non-overlapping 1200-token windows): gap, arrow, topic re-use
def battery_windows(Sx, Wtok=1200, maxn=None, seed=0):
    out = []
    for s in range(0, len(Sx) - Wtok + 1, Wtok):
        sw = Sx[s:s + Wtok]; ln = L.window_lines(sw)
        out.append(dict(start=s, gap=L.gap_ratio_tokens(ln, seed), arrow=L.arrow_lines(ln, seed=seed),
                        reuse=L.topic_reuse(sw)))
        if maxn and len(out) >= maxn: break
    return out

bat = L.load('bat_c1.json')
if bat is None:
    bat = {'ZL': battery_windows(S)}
    for name in L.LANGS:
        bat[name] = battery_windows(L.stream_of_lines(L.corpus_lines(name, maxtok=14400)))
    L.save('bat_c1.json', bat)
log('battery done')
langB = {k: np.array([w[k] for n in L.LANGS for w in bat[n]], float) for k in ('gap', 'arrow', 'reuse')}
vB = {k: np.array([w[k] for w in bat['ZL']], float) for k in ('gap', 'arrow', 'reuse')}
band = dict(gap=float(np.nanpercentile(langB['gap'], 95)), arrow=float(np.nanpercentile(langB['arrow'], 5)),
            reuse=float(np.nanpercentile(langB['reuse'], 5)))
inband = dict(gap=int((vB['gap'] <= band['gap']).sum()), arrow=int((vB['arrow'] >= band['arrow']).sum()),
              reuse=int(np.nansum(vB['reuse'] >= band['reuse'])))
# windows passing all four (classifier mean over the same 12 chunks)
cls12 = np.array([csV[w['start'] // 100: w['start'] // 100 + 12].mean() for w in bat['ZL']])
all4 = [(i, P[S[w['start']][1]]['id']) for i, w in enumerate(bat['ZL'])
        if cls12[i] >= tau[12] and w['gap'] <= band['gap'] and w['arrow'] >= band['arrow'] and w['reuse'] >= band['reuse']]
res['battery'] = dict(lang_med={k: float(np.nanmedian(v)) for k, v in langB.items()},
                      voy_med={k: float(np.nanmedian(v)) for k, v in vB.items()}, band=band, inband=inband,
                      nwin_v=len(bat['ZL']), nwin_l=len(langB['gap']), all4=all4,
                      lang_all3=int(sum(1 for n in L.LANGS for w in bat[n] if w['gap'] <= band['gap'] and w['arrow'] >= band['arrow'])))
log('battery', res['battery'])

# ---------------- groups
def group_scan(key, minc=6, nnull=2000, seed=0):
    lab = [P[chunk_page[j]][key] for j in range(len(csV))]
    groups = sorted({g for g in lab if lab.count(g) >= minc})
    lab = np.array(lab)
    def stats(lb):
        return {g: csV[lb == g].mean() for g in groups}
    obs = stats(lab)
    rng = np.random.default_rng(seed)
    mx = []; per = defaultdict(list)
    n = len(lab)
    for _ in range(nnull):      # circular shift of the label sequence (keeps contiguity and group sizes)
        k = rng.integers(1, n); lb = np.roll(lab, k)
        st = stats(lb)
        for g in groups: per[g].append(st[g])
        mx.append(max(st.values()))
    mx = np.array(mx)
    out = {}
    for g in groups:
        pv = float((mx >= obs[g]).mean())
        ch = csV[lab == g]
        best = float(L.window_scores(ch, minc).max())
        out[g] = dict(n=int((lab == g).sum()), s=float(obs[g]), p_max=pv,
                      p_lo=float((np.array(per[g]) <= obs[g]).mean()), best6=best, above_tau6=best >= tau[6])
    return out

groups = {}
for key in ('sec', 'quire', 'hand', 'lang', 'bifolio'):
    groups[key] = group_scan(key)
    top = sorted(groups[key].items(), key=lambda x: -x[1]['s'])[:4]
    log('group', key, [(g, round(d['s'], 3), d['n'], d['p_max']) for g, d in top])
res['groups'] = groups

# ---------------- controls: uniform generators
def uniform_streams():
    import v33_lib as G, v21_lib as V
    lines = [l for p in P for pa in p['paras'] for l in pa]
    tri = G.Trigram(lines)
    toks = tri.gen(len(S) + 50, random.Random(7))
    S1 = [(toks[i], pi, lk) for i, (_, pi, lk) in enumerate(S)]
    F = V.Forger(P, scope='global', pos=True, name='F3g')
    Q = F.forge(P, random.Random(7))
    S2 = L.stream_of_pages(Q)
    return {'tri': S1, 'forgeG': S2}

ctrl = {}
for nm, Sx in uniform_streams().items():
    ch = L.chunks(Sx)
    F = feats_cached('U_' + nm, ch)
    cs = L.lscore(L.probs(F))
    d = {}
    for W in (6, 12, 24):
        ws = L.window_scores(cs, W); null = L.scan_null(cs, W, 1000, seed=W)
        d[W] = dict(max=float(ws.max()), p=float((null >= ws.max()).mean()), n_above=int((ws >= tau[W]).sum()),
                    median=float(np.median(ws)))
    ctrl[nm] = d
    log('uniform', nm, d)
res['uniform'] = ctrl

# ---------------- controls: planted islands
plants = {}
vw = [w for w, _, _ in S]
for lname in ('L_msI_Lat', 'L_msG_Alem'):
    ll = L.corpus_lines(lname, maxtok=6000)
    vmap = L.translit_map([w for l in ll for w in l], vw)
    cs_base = L.lscore(L.probs(FV, exclude=(lname,)))
    for size in (600, 1200, 2400):
        for start_chunk in (60, 260):
            st = start_chunk * 100
            S2 = L.plant(S, ll, st, size, vmap)
            nc = size // 100
            ch = L.chunks(S2)[start_chunk:start_chunk + nc]
            Fp = feats_cached(f'P_{lname}_{size}_{start_chunk}', ch)
            cs = cs_base.copy(); cs[start_chunk:start_chunk + nc] = L.lscore(L.probs(Fp, exclude=(lname,)))
            d = {}
            for W in (6, 12):
                ws = L.window_scores(cs, W); null = L.scan_null(cs, W, 1000, seed=W)
                runs = L.runs_above(ws, tau[W])
                best = None; bj = 0
                true = set(range(start_chunk, start_chunk + nc))
                for a, b in runs:
                    cov = set(range(a, b + W))
                    j = len(cov & true) / len(cov | true)
                    if j > bj: bj, best = j, (a, b + W - 1)
                i = int(ws.argmax())
                d[W] = dict(max=float(ws.max()), p=float((null >= ws.max()).mean()), argmax_in=start_chunk - W < i < start_chunk + nc,
                            jacc=bj, span=best, nruns=len(runs), false_runs=sum(1 for a, b in runs if not (set(range(a, b + W)) & true)))
            plants[f'{lname}|{size}|{start_chunk}'] = d
            log('plant', lname, size, start_chunk, d)
res['plants'] = plants
L.save('cycle1.json', res)
log('done')
