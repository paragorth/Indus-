"""v62 cycle 1: RHYME. Do line-final word endings pair up (AA, ABAB/ABA, monorhyme runs) beyond chance,
measured as kappa at line lags 1-4 for line-final endings MINUS the same for the penultimate word
(removes page drift and neighbour-line similarity that touches every word)?

Search: N random ending definitions (glyph class merges x last k glyphs x skip s final glyphs as a
payload frame) on a training half; survivors re-scored on the held-out half and on IT2a.
Family-wise nulls: identical search on lines shuffled within page, on random re-flow of the line breaks,
and on Markov resynthesis with line effects (initial/final distributions, page drift) planted.
Controls: Regimen (rhymed hexameters), Macer (unrhymed hexameters), Dante (terza rima), Litany,
Hildegard (prose herbal) and Caesar, all in an opaque verbose code; verse re-flowed.
"""
import sys, os, json, random, pickle, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v62_lib as L

NDEF = int(os.environ.get('NDEF', 3000))
NNULL = int(os.environ.get('NNULL', 4))
A = pickle.load(open(os.path.join(L.CK, 'corpora.pkl'), 'rb'))


def split(name):
    pg = A[name]['pages']
    if A[name]['kind'] == 'voynich':
        meta = A[name]['meta']
        tr = [p for p, m in zip(pg, meta) if L.folio_num(m['folio']) % 2 == 1]
        te = [p for p, m in zip(pg, meta) if L.folio_num(m['folio']) % 2 == 0]
    else:
        h = len(pg) // 2; tr, te = pg[:h], pg[h:]
    return tr, te


def search(pages, defs, maxlag=3):
    P = L.prep_positions(pages)
    sc = []
    for d in defs:
        did, kF, kM, oF, eF = L.rhyme_score(P, d, maxlag)
        sc.append(float(np.max(did)))
    return np.array(sc)


def job(args):
    tag, pages, seed = args
    rng = random.Random(seed)
    alpha = L.alphabet(pages)
    defs = L.random_defs(alpha, NDEF, random.Random(1000 + seed))
    return tag, defs, search(pages, defs)


def main():
    out = {}
    t0 = time.time()
    # ---- natural definitions, all corpora ----
    nat = [(None, k, 0) for k in (1, 2, 3, 4)] + [(None, 2, 1)]
    out['natural'] = {}
    for name in A:
        P = L.prep_positions(A[name]['pages'])
        row = {}
        for d in nat:
            did, kF, kM, oF, eF = L.rhyme_score(P, d, 4)
            row[L.def_str(d)] = dict(did=[round(x, 3) for x in did], kF=[round(x, 3) for x in kF],
                                      kM=[round(x, 3) for x in kM], obsF1=round(float(oF[0]), 3), expF=round(float(eF[0]), 3))
        # whole word identity (refrain) at line end
        F = [[None if r is None else r[0] for r in rows] for rows in P]
        M = [[None if r is None else r[3] for r in rows] for rows in P]
        kF, *_ = L.lag_kappa(F, 4); kM, *_ = L.lag_kappa(M, 4)
        row['wholeword'] = dict(did=[round(x, 3) for x in kF - kM], kF=[round(x, 3) for x in kF])
        out['natural'][name] = row
        print(name, {k: v['did'] for k, v in row.items()}, flush=True)
    # ---- search: Voynich ZL training + nulls; controls ----
    tasks = []
    zl_tr, zl_te = split('V-ZL3b')
    tasks.append(('ZL-train', zl_tr, 1))
    for i in range(NNULL):
        r = random.Random(500 + i)
        tasks.append((f'null-shuffle-{i}', L.shuffle_lines_within_page(zl_tr, r), 1))
        tasks.append((f'null-reflow-{i}', L.reflow_page(zl_tr, random.Random(600 + i)), 1))
        tasks.append((f'null-markov-{i}', L.markov_line_generator(zl_tr, random.Random(700 + i)), 1))
    for name in ['Regimen(verse,rhymed)', 'Macer(verse,hexam)', 'Dante(verse,terza)', 'Litany(refrain)',
                 'Hildegard(prose herbal)', 'Caesar(prose)', 'Dante-reflowed']:
        tr, te = split(name)
        tasks.append((name + '|train', tr, 2))
    with Pool(2) as pool:
        res = {tag: (defs, sc) for tag, defs, sc in pool.imap_unordered(job, tasks)}
    out['search'] = {}
    for tag, (defs, sc) in res.items():
        out['search'][tag] = dict(max=round(float(sc.max()), 4), p99=round(float(np.percentile(sc, 99)), 4),
                                  mean=round(float(sc.mean()), 4), best=L.def_str(defs[int(sc.argmax())]))
        print(tag, out['search'][tag], flush=True)
    # ---- held-out: top 20 ZL-train defs on ZL even folios, IT2a odd and even ----
    defs, sc = res['ZL-train']
    top = np.argsort(-sc)[:20]
    it_tr, it_te = split('V-IT2a')
    Pte, Pit_te, Pit_tr = L.prep_positions(zl_te), L.prep_positions(it_te), L.prep_positions(it_tr)
    ho = []
    for i in top:
        d = defs[i]
        a = L.rhyme_score(Pte, d, 3)[0]; b = L.rhyme_score(Pit_te, d, 3)[0]; c = L.rhyme_score(Pit_tr, d, 3)[0]
        ho.append(dict(df=L.def_str(d), train=round(float(sc[i]), 4), zl_test=[round(x, 3) for x in a],
                       it_test=[round(x, 3) for x in b], it_train=[round(x, 3) for x in c]))
    out['heldout_top20'] = ho
    # held-out null: same top-20 procedure for each Markov null (does a generator's best def replicate?)
    # and random defs on ZL test for reference
    rdefs = L.random_defs(L.alphabet(zl_te), 300, random.Random(77))
    rs = search(zl_te, rdefs)
    out['zl_test_random_defs'] = dict(mean=round(float(rs.mean()), 4), sd=round(float(rs.std()), 4),
                                      p95=round(float(np.percentile(rs, 95)), 4))
    # controls held-out: best train def on test half
    out['controls_heldout'] = {}
    for name in ['Regimen(verse,rhymed)', 'Macer(verse,hexam)', 'Dante(verse,terza)', 'Litany(refrain)',
                 'Hildegard(prose herbal)', 'Caesar(prose)', 'Dante-reflowed']:
        defs, sc = res[name + '|train']
        tr, te = split(name)
        i = int(sc.argmax())
        out['controls_heldout'][name] = dict(df=L.def_str(defs[i]), train=round(float(sc[i]), 4),
                                             test=[round(x, 3) for x in L.rhyme_score(L.prep_positions(te), defs[i], 3)[0]])
    out['secs'] = round(time.time() - t0)
    json.dump(out, open(os.path.join(L.CK, 'cycle1.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
