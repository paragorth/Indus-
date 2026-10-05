"""v65 cycle 3b: centre placement regardless of refold. For E1-E3 the physical claim is that the
named bifolio was the innermost of its quire. Share of the top 1% / 5% arrangements (seam, all words)
that put it innermost (either fold), against the prior (1 / number of real bifolios), and the bootstrap
over pages: 200 resamples of the quire's text lines (lines drawn with replacement within each page)
to get a stability share for the innermost bifolio of the single best arrangement."""
import os, sys, json, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v65_lib as L
import v65_c3 as C3

quires, _ = L.structure(); QD = dict(quires)


def inner_of(sp, ci, qn):
    o, f = sp.configs[ci]
    real = [i for i in o if not (QD[qn][i][1] is None and QD[qn][i][2] is None)]
    return real[-1]


def run(pages, meta, feat, nboot, rng):
    keys = sorted(k for k in pages if sum(len(x) for x in pages[k]) >= L.MIN_WORDS and k in meta)
    kidx = {k: i for i, k in enumerate(keys)}
    groups = [str(meta[k]['sec']) + str(meta[k]['lang']) for k in keys]
    sps = {qn: L.QuireSpace(QD[qn], kidx) for qn in C3.CENTRE}
    inner_tab = {qn: np.array([inner_of(sp, i, qn) for i in range(len(sp.configs))]) for qn, sp in sps.items()}
    def shares(W):
        r = {}
        for qn, sp in sps.items():
            sc = sp.scores(W); o = np.argsort(-sc); c = C3.CENTRE[qn]; it = inner_tab[qn]
            r[qn] = {'best_inner': int(QD[qn][it[o[0]]][1] or QD[qn][it[o[0]]][2]),
                     'top1': float((it[o[:max(1, len(o) // 100)]] == c).mean()),
                     'top5': float((it[o[:max(1, len(o) // 20)]] == c).mean()),
                     'prior': float((it == c).mean())}
        return r
    Js, _ = L.carry_matrices(pages, keys, feat=feat)
    res = shares(L.centre(Js, groups))
    if nboot:
        cnt = {qn: 0 for qn in sps}
        for _ in range(nboot):
            bp = {k: [rng.choice(pages[k]) for _ in pages[k]] if k in kidx else pages[k] for k in pages}
            # keep head/tail meaning: resample lines but keep order positions sorted by original index
            bp = {}
            for k in pages:
                idx = sorted(rng.randrange(len(pages[k])) for _ in pages[k])
                bp[k] = [pages[k][i] for i in idx]
            Jb, _ = L.carry_matrices(bp, keys, feat=feat)
            sb = shares(L.centre(Jb, groups))
            for qn in sps:
                cnt[qn] += sb[qn]['best_inner'] == (QD[qn][C3.CENTRE[qn]][1] or QD[qn][C3.CENTRE[qn]][2])
        for qn in sps:
            res[qn]['boot_inner_share'] = cnt[qn] / nboot
    return res


def main():
    rng = random.Random(6533)
    vp, vmeta = L.voynich_pages('ZL3b'); ip, imeta = L.voynich_pages('IT2a')
    isid = L.isidore_words()[5000:]; kon = [w for e in L.konrad_entries() for w in e]
    co = C3.claimed_order()
    runs = {'VOY_ZL|word': (vp, vmeta, 'word', 100), 'VOY_ZL|skel': (vp, vmeta, 'skel', 100),
            'VOY_IT|word': (ip, imeta, 'word', 100), 'VOY_IT|skel': (ip, imeta, 'skel', 100)}
    for s in range(4):
        runs[f'CAL_isid_s{s}|skel'] = (L.pour(vp, isid[s * 9000:], 'flow', 40 + s, quires, order=co), vmeta, 'skel', 0)
        runs[f'CAL_kon_s{s}|skel'] = (L.pour(vp, kon[s * 3000:] + kon[:s * 3000], 'flow', 50 + s, quires, order=co), vmeta, 'skel', 0)
    for s in range(4):
        runs[f'NULL_markov_s{s}|word'] = (L.markov_pages(vp, vmeta, 80 + s), vmeta, 'word', 0)
    out = {}
    for name, (p, m, f, nb) in runs.items():
        r = run(p, m, f, nb, rng); out[name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(L.CK, 'c3b.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
