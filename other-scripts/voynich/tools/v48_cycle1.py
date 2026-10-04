"""v48 cycle 1: build the typological map, calibrate it, and put the cleaned Voynich on it.

1a  60 natural-language corpora (eBible NT, Wikipedia samples, medieval MSS from ReF/CATMuS/Old Czech, Gaskell &
    Bowern historical texts, Gutenberg) -> 34 alphabet-free fingerprint features per corpus.  Nulls: every language's
    global unit shuffle and unit Markov-2 resynthesis.  Genre spread: same-language corpora of different genre.
1b  blind merge calibration: top interchange index of real letter pairs vs planted twins.
1c  plants: positional twins + bench variant + A/B rewrite on each language; blind merge + blind A/B undo;
    distance to its own clean point and rank of its own clean point among the 60.
1d  the Voynich ladder: ZL3b / IT2a x EVA / GLY / RUN / STR x E0..E3 x (none | blind A/B undo | blind merge).
"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v48_lib as L

OUT = 'v48_cycle1.txt'


def job(args):
    import hashlib
    kind, payload = args
    key = kind + '_' + (payload['name'] if kind == 'lang' else '_'.join(payload))
    fn = os.path.join(L.CK, 'c1_jobs', hashlib.md5(key.encode()).hexdigest() + '.json')
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    if os.path.exists(fn):
        return json.load(open(fn))
    r = _job(kind, payload)
    json.dump(r, open(fn, 'w'), default=float)
    print('done', key, flush=True)
    return r


def _job(kind, payload):
    if kind == 'lang':
        C = payload
        r = dict(name=C['name'], fam=C['fam'], fp=L.fp_corpus(C))
        r['shuf'] = L.fp_corpus(L.null_shuffle(C)); r['mk2'] = L.fp_corpus(L.null_markov2(C))
        pairs, _ = L.interchange(C); r['topI'] = pairs[0]
        P, tr = L.plant(C, seed=1)
        r['plant_raw'] = L.fp_corpus(P)
        pp, _ = L.interchange(P)
        tw = set(tr['twins'])
        r['plantI'] = [(I, a, b) for I, a, b in pp if (a in tw or b in tw) and (tr['twins'].get(a) == b or tr['twins'].get(b) == a)]
        M, mp, _ = L.merge_blind(P)
        r['merge_ok'] = sum(1 for k, v in mp.items() if tr['twins'].get(k) == v)
        r['merge_bad'] = len(mp) - r['merge_ok']
        N, rules = L.ab_blind(M)
        r['plant_norm'] = L.fp_corpus(N)
        r['ab_rules'] = rules; r['ab_truth'] = tr['ab']['ends']
        r['ab_hit'] = sum(1 for k, v in rules.get('end', {}).items() if tr['ab']['ends'].get(v) == k)
        # clean corpus pushed through the same blind normalisers (must not move)
        M0, mp0, _ = L.merge_blind(C); N0, rules0 = L.ab_blind(M0)
        r['clean_norm'] = L.fp_corpus(N0); r['clean_merges'] = len(mp0); r['clean_rules'] = sum(map(len, rules0.values()))
        return r
    if kind == 'voy':
        src, inv, norm, post = payload
        C = L.voynich(src, norm, inv)
        info = {}
        if post in ('M', 'MAB'):
            C, mp, top = L.merge_blind(C); info['merges'] = mp
        if post in ('AB', 'MAB'):
            C, rules = L.ab_blind(C); info['rules'] = rules
        r = dict(name=f'{src}/{inv}/{norm}/{post}', fp=L.fp_corpus(C), info=info,
                 ntok=len(L.tokens(C)), nunits=len({u for w in L.tokens(C) for u in w}))
        if post == 'none' and norm in ('E0', 'E3'):
            r['shuf'] = L.fp_corpus(L.null_shuffle(C)); r['mk2'] = L.fp_corpus(L.null_markov2(C))
        return r


def main():
    t0 = time.time()
    res = L.load('c1_results.json')
    if res is None:
        langs = L.all_languages()
        jobs = [('lang', C) for C in langs]
        for src in ('ZL3b', 'IT2a'):
            for inv in ('EVA', 'GLY', 'RUN', 'STR'):
                for norm in ('E0', 'E1', 'E2', 'E3'):
                    for post in ('none', 'AB', 'M', 'MAB'):
                        if post in ('M', 'MAB') and norm != 'E0': continue
                        jobs.append(('voy', (src, inv, norm, post)))
        with Pool(2) as pool:
            out = pool.map(job, jobs, chunksize=1)
        res = dict(lang=[r for r in out if 'fam' in r], voy=[r for r in out if 'fam' not in r])
        L.save('c1_results.json', res)
    print('computed', time.time() - t0, flush=True)
    report(res)


def report(res):
    lang = res['lang']
    names = [r['name'] for r in lang]; fams = [r['fam'] for r in lang]
    M = L.Map([r['fp'] for r in lang], names, fams)
    print('reference NN distance (median LOO)', round(M.ref, 3))
    lo = M.nn_lang / M.ref
    print('language out-scores: median %.2f  95th %.2f  max %.2f (%s)' % (np.median(lo), np.percentile(lo, 95), lo.max(), names[int(np.argmax(lo))]))
    # ---- nulls
    so = [M.place(r['shuf'])['out'] for r in lang]; mo = [M.place(r['mk2'])['out'] for r in lang]
    p95 = np.percentile(lo, 95)
    s1 = (f'shuffles: out median {np.median(so):.2f} (min {min(so):.2f}), {np.mean(np.array(so) > p95):.2f} beyond the language 95th pct {p95:.2f}; '
          f'Markov-2 resyntheses: out median {np.median(mo):.2f} (min {min(mo):.2f}), {np.mean(np.array(mo) > p95):.2f} beyond')
    print(s1)
    # ---- genre spread
    groups = {'Latin': ['eb_lat', 'wk_la', 'gk_plinyabbr', 'ms_I_Lat', 'pg_caesar'], 'Italian': ['eb_ita', 'ms_I_Ita', 'pg_dante', 'pg_manzoni'],
              'German': ['eb_deu', 'ms_G_Bav1', 'ms_G_Alem', 'ms_G_Rip', 'gk_deherb', 'pg_kafka'], 'Arabic': ['eb_arb', 'gk_quran', 'gk_avicenna'],
              'Hebrew': ['eb_heb', 'eb_hbo'], 'Occitan': ['wk_oc', 'ws_oc'], 'Czech': ['eb_ces', 'ms_C_Old'], 'Spanish': ['eb_spa', 'pg_cervantes']}
    idx = {n: i for i, n in enumerate(names)}
    D = M.dist_matrix(M.Z, M.Z)
    within, ranks = [], []
    for g, ns in groups.items():
        ns = [n for n in ns if n in idx]
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                within.append(D[idx[ns[i]], idx[ns[j]]] / M.ref)
        for n in ns:
            o = np.argsort(D[idx[n]]); o = [k for k in o if k != idx[n]]
            ranks.append(next(r for r, k in enumerate(o) if names[k] in ns) + 1)
    fam_ok = []
    for i, n in enumerate(names):
        o = [k for k in np.argsort(D[i]) if k != i]
        fam_ok.append(fams[o[0]] == fams[i])
    s2 = (f'same-language different-genre distance: median {np.median(within):.2f} ref units (range {min(within):.2f}-{max(within):.2f}); '
          f'rank of nearest same-language corpus median {np.median(ranks):.0f} (of 59); nearest neighbour same family {np.mean(fam_ok):.2f} (chance ~{np.mean([sum(f == g for g in fams) - 1 for f in fams]) / 59:.2f})')
    print(s2)
    # ---- blind merge calibration and plants
    topI = sorted([r['topI'][0] for r in lang]); plI = [x[0] for r in lang for x in r['plantI']]
    s3 = (f'interchange index: real letter pairs top per language median {np.median(topI):.2f}, max {max(topI):.2f} ({lang[int(np.argmax([r["topI"][0] for r in lang]))]["name"]} {lang[int(np.argmax([r["topI"][0] for r in lang]))]["topI"][1:]}); '
          f'planted twin pairs median {np.median(plI):.2f} (positional twins {np.median([r["plantI"][0][0] for r in lang if r["plantI"]]):.2f}); '
          f'blind merges correct {sum(r["merge_ok"] for r in lang)}/{3 * len(lang)}, false {sum(r["merge_bad"] for r in lang)}; '
          f'A/B rules recovering planted endings {sum(r["ab_hit"] for r in lang)}/{4 * len(lang)}; clean corpora: false merges {sum(r["clean_merges"] for r in lang)}, false A/B rules {sum(r["clean_rules"] for r in lang)}')
    print(s3)
    draw, dnorm, rk_raw, rk_norm, dclean = [], [], [], [], []
    for r in lang:
        for key, dl, rl in (('plant_raw', draw, rk_raw), ('plant_norm', dnorm, rk_norm)):
            p = M.place(r[key]); d = p['d']
            dl.append(d[idx[r['name']]] / M.ref)
            rl.append(int((d < d[idx[r['name']]]).sum()) + 1)
        dclean.append(M.place(r['clean_norm'])['d'][idx[r['name']]] / M.ref)
    s4 = (f'planted languages: distance to own clean point raw {np.median(draw):.2f} -> normalised {np.median(dnorm):.2f} ref units; '
          f'own clean point is nearest {np.mean(np.array(rk_raw) == 1):.2f} -> {np.mean(np.array(rk_norm) == 1):.2f} (top-3 {np.mean(np.array(rk_raw) <= 3):.2f} -> {np.mean(np.array(rk_norm) <= 3):.2f}); '
          f'clean corpora pushed through the blind normalisers move {np.median(dclean):.2f}')
    print(s4)
    # ---- Voynich
    vrows = []
    for r in res['voy']:
        p = M.place(r['fp']); fv = M.fam_vote(r['fp'])
        vrows.append((r['name'], p['out'], p['pct'], p['nn'][:3], fv[:3], r['info'], r['nunits']))
    vrows.sort(key=lambda x: x[1])
    print('\nVoynich ladder (out = NN distance / median language NN distance; pct = share of languages with a nearer NN)')
    for v in vrows:
        print(f'{v[0]:24s} out {v[1]:.2f} pct {v[2]:.2f} units {v[6]:3d} NN ' + ', '.join(f'{a}({c:.2f})' for a, b, c in v[3]) + ' | fam ' + ', '.join(f'{a}' for a, b in v[4]))
    vn = {}
    for r in res['voy']:
        if 'shuf' in r:
            vn[r['name']] = (M.place(r['shuf'])['out'], M.place(r['mk2'])['out'])
    print('Voynich nulls', {k: (round(a, 2), round(b, 2)) for k, (a, b) in vn.items()})
    # which features keep the Voynich away
    best = vrows[0][0]
    fp = next(r['fp'] for r in res['voy'] if r['name'] == best)
    z = M.z(fp)
    o = np.argsort(-np.abs(z))
    print('most extreme features of', best, [(L.FEATS[i], round(float(z[i]), 1)) for i in o[:10]])
    L.save('c1_summary.json', dict(vrows=[(v[0], v[1], v[2], v[3], v[4]) for v in vrows], ref=M.ref, p95=p95))
    # ---- rows
    raw = {v[0]: v for v in vrows}
    def g(n): return raw[n]
    ladder = ' ; '.join('%s %.2f' % (n.split("/", 1)[1], g(n)[1]) for n in [f'ZL3b/GLY/{e}/none' for e in ('E0', 'E1', 'E2', 'E3')] + ['ZL3b/GLY/E3/AB', 'ZL3b/GLY/E0/MAB'])
    L.row(OUT, 'V-48.1a', 'Typological map: 60 corpora (32 eBible NT incl. Vulgate, Kralice 1613, Gdansk 1632, Diodati, Koine TR, Sahidic Coptic, Peshitta-script Neo-Aramaic, Romani x2; Wikipedia Basque, Catalan, Occitan, Georgian, Albanian, Maltese, Latin, Armenian; Manchu (romanised); Gaskell historical Quran, Avicenna, Sanskrit, pinyin; ReF/CATMuS/Old Czech MSS 1350-1450; Gutenberg) x 34 alphabet-free features (Sukhotin/spectral V-C split and its stability, CV skeleton, word and morph length, positional entropy, affix concentration, reuse, Zipf). Out-score = NN distance / median language NN. Controls: global unit shuffle and unit Markov-2 resynthesis of every language; same-language corpora of other genres',
          f'Language out-scores median {np.median(lo):.2f}, 95th pct {p95:.2f}. {s1}. {s2}', 'Map calibrated: shuffles land nowhere; genre moves a language less than language identity (see ranks). Markov-2 texts are the hard null')
    L.row(OUT, 'V-48.1b', 'Blind normalisers calibrated on planted languages: positional twins (line-first / paragraph-first + decaying habit, like ch/sh, k/t), a neighbour-conditioned bench variant, and an A/B ending + beginning rewrite on the second half of each of 60 languages; undone by a blind interchange-index merge (THR 0.85) and a blind stem-overlap A/B learner, never told the plant',
          f'{s3}. {s4}', 'Normalisation pipeline passes if planted languages return to their clean point (see numbers)')
    vnull = ", ".join("%s %.2f/%.2f" % (k.split("/", 1)[1], a, b) for k, (a, b) in vn.items() if k.startswith("ZL3b/GLY"))
    L.row(OUT, 'V-48.1c', 'The Voynich on the map: ZL3b and IT2a x 4 unit inventories (EVA, glyph, i/e-run, stroke) x explicit ladder E0 raw, E1 k=t f=p, E2 +ch=sh, E3 +benched=plain, each with/without blind A/B undo; blind merge + A/B on raw (48 versions)',
          f'Best: {vrows[0][0]} out {vrows[0][1]:.2f} (pct {vrows[0][2]:.2f}), NN {", ".join(a for a, b, c in vrows[0][3])}; worst {vrows[-1][0]} {vrows[-1][1]:.2f}. ZL glyph ladder: {ladder}. Voynich nulls (shuf, mk2): {vnull}',
          'see cycle notes')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report(L.load('c1_results.json'))
    else:
        main()
