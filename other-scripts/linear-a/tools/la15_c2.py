"""LA-15 cycle 2: biogeography of the Minoan word stock.
2a endemic vs migratory words (fixed-fixed curveball null on the document x word matrix);
   'confident endemics' = words common at home whose absence elsewhere is improbable by
   detection (P(miss | present) from home token frequency and the other sites' token exposure).
2b capture-recapture: Chapman two-list estimates on word types (HT vs other sites,
   tablets vs non-tablets, first vs second half of scribes) and on sign types.
2c species-area power law across sites; leave-one-site-out prediction of each site's
   word types and of its endemics.
2d nestedness (NODF) of the site x word matrix vs curveball.
2e keystone words: co-occurrence partners above the curveball null.
Controls: Linear B (all sites; and LA-sized document subsamples), LA with word tokens
shuffled across documents (word-site links broken).
"""
import sys, json, random, collections, time, math
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from la15_common import *

prng = random.Random(152)
NNULL = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def merge_site(s):
    return s if s else '?'


def matrices(docs):
    D = [d for d in docs if d['words']]
    rows = [set(d['words']) for d in D]
    sites = [merge_site(d['site']) for d in D]
    return D, rows, sites


def site_spread(rows, sites, words):
    sp = collections.defaultdict(set)
    for r, s in zip(rows, sites):
        for w in r:
            sp[w].add(s)
    return {w: len(sp[w]) for w in words}


def nodf(mat):
    """mat: list of sets (sites), universe words; NODF over rows (sites) and columns (words)."""
    def pairs_score(sets):
        sets = sorted(sets, key=len, reverse=True)
        tot, k = 0.0, 0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                k += 1
                a, b = sets[i], sets[j]
                if len(a) > len(b) and len(b) > 0:
                    tot += len(a & b) / len(b)
        return tot, k
    words = set().union(*mat)
    cols = [set(i for i, s in enumerate(mat) if w in s) for w in words]
    t1, k1 = pairs_score(mat)
    t2, k2 = pairs_score(cols)
    return 100 * (t1 + t2) / (k1 + k2)


def analyse(docs, label, nnull=NNULL, keystone=True):
    out = {}
    D, rows, sites = matrices(docs)
    inc = collections.Counter(w for r in rows for w in r)
    rec = [w for w, c in inc.items() if c >= 2]
    obs = site_spread(rows, sites, rec)
    site_tok = collections.Counter()
    for d in D:
        site_tok[merge_site(d['site'])] += len(d['words'])
    # null spreads
    null = collections.defaultdict(list)
    nodf_null, endem_null = [], []
    big_sites = [s for s, c in site_tok.items() if c >= 15]

    def site_sets(rr):
        ss = collections.defaultdict(set)
        for r, s in zip(rr, sites):
            if s in big_sites:
                ss[s] |= r
        return [ss[s] for s in big_sites]
    co_null = collections.defaultdict(list)
    kw = [w for w, c in inc.items() if c >= 3]
    kwset = set(kw)

    def cooc(rr):
        c = collections.Counter()
        for r in rr:
            ww = sorted(r & kwset)
            for i in range(len(ww)):
                for j in range(i + 1, len(ww)):
                    c[(ww[i], ww[j])] += 1
        return c
    co_obs = cooc(rows)
    nulls = []
    for k in range(nnull):
        rr = curveball(rows, 6 * len(rows), prng)
        sp = site_spread(rr, sites, rec)
        for w in rec:
            null[w].append(sp[w])
        endem_null.append(np.mean([sp[w] == 1 for w in rec]))
        nodf_null.append(nodf(site_sets(rr)))
        if keystone:
            c = cooc(rr)
            for pr in co_obs:
                co_null[pr].append(c.get(pr, 0))
    obs_endem = np.mean([obs[w] == 1 for w in rec])
    out['n_recurrent'] = len(rec)
    out['endemic_frac_obs'] = obs_endem
    out['endemic_frac_null'] = (float(np.mean(endem_null)), float(np.std(endem_null)))
    out['endemic_z'] = (obs_endem - np.mean(endem_null)) / (np.std(endem_null) + 1e-9)
    # per-word: fewer sites than null (endemic) / more (migratory)
    per = []
    for w in rec:
        nv = np.array(null[w])
        p_low = (np.sum(nv <= obs[w]) + 1) / (len(nv) + 1)
        p_high = (np.sum(nv >= obs[w]) + 1) / (len(nv) + 1)
        per.append((w, inc[w], obs[w], float(nv.mean()), p_low, p_high))
    out['migratory'] = sorted([p for p in per if p[2] >= 3], key=lambda p: (p[5], -p[2]))[:15]
    out['endemic_strict'] = sorted([p for p in per if p[2] == 1], key=lambda p: p[4])[:15]
    # detection-based confident endemics
    tok = collections.Counter(w for d in D for w in d['words'])
    home_tok = collections.defaultdict(collections.Counter)
    for d in D:
        for w in d['words']:
            home_tok[w][merge_site(d['site'])] += 1
    conf = []
    for w in rec:
        if obs[w] != 1:
            continue
        h = next(iter(home_tok[w]))
        f = tok[w] / site_tok[h]
        miss = sum(site_tok[s] for s in site_tok if s != h) * math.log(1 - min(f, 0.999))
        conf.append((w, h, tok[w], round(math.exp(miss), 4)))
    conf.sort(key=lambda x: x[3])
    out['confident_endemics'] = conf[:20]
    out['n_confident_endemics_p05'] = sum(1 for c in conf if c[3] < 0.05)
    # NODF
    ss = site_sets(rows)
    out['nodf_obs'] = nodf(ss)
    out['nodf_null'] = (float(np.mean(nodf_null)), float(np.std(nodf_null)))
    # keystones
    if keystone:
        partners = collections.Counter()
        for pr, v in co_obs.items():
            nv = np.array(co_null[pr])
            if v >= 2 and (np.sum(nv >= v) + 1) / (len(nv) + 1) <= 0.01:
                partners[pr[0]] += 1
                partners[pr[1]] += 1
        out['keystones'] = [(w, c, inc[w]) for w, c in partners.most_common(15)]
        out['n_sig_pairs'] = sum(partners.values()) // 2
    # species-area across sites (power law) and leave-one-site-out
    sty = collections.defaultdict(set)
    for d in D:
        sty[merge_site(d['site'])] |= set(d['words'])
    pts = [(site_tok[s], len(sty[s]), s) for s in site_tok if site_tok[s] >= 5]
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    z, c = np.polyfit(x, y, 1)
    out['SAR_z'] = float(z)
    loso = []
    for i, (A, S, s) in enumerate(pts):
        xx = np.delete(x, i)
        yy = np.delete(y, i)
        zz, cc = np.polyfit(xx, yy, 1)
        predS = math.exp(cc + zz * math.log(A))
        others = set().union(*[sty[t] for t in sty if t != s])
        endemic = len(sty[s] - others)
        # predicted endemics: share of types at other sites that are unique to their site
        ends = []
        for t in sty:
            if t == s or site_tok[t] < 5:
                continue
            o2 = set().union(*[sty[u] for u in sty if u not in (t, s)])
            ends.append(len(sty[t] - o2) / max(len(sty[t]), 1))
        loso.append(dict(site=s, tok=A, S=S, predS=round(predS, 1), endemic=endemic,
                         predEnd=round(np.mean(ends) * predS, 1)))
    out['loso'] = loso
    out['loso_mape_S'] = float(np.mean([abs(l['predS'] - l['S']) / l['S'] for l in loso]))
    # capture-recapture (Chapman) on word types between partitions
    def chapman(a, b):
        m = len(a & b)
        return (len(a) + 1) * (len(b) + 1) / (m + 1) - 1, len(a), len(b), m
    cr = {}
    main = site_tok.most_common(1)[0][0]
    A1 = set(w for d in D if merge_site(d['site']) == main for w in d['words'])
    A2 = set(w for d in D if merge_site(d['site']) != main for w in d['words'])
    cr['main_vs_rest'] = chapman(A1, A2)
    t1 = set(w for d in D if d['support'].lower().startswith('tablet') for w in d['words'])
    t2 = set(w for d in D if not d['support'].lower().startswith('tablet') for w in d['words'])
    if t2:
        cr['tablet_vs_other'] = chapman(t1, t2)
    half = [prng.random() < 0.5 for _ in D]
    h1 = set(w for d, h in zip(D, half) if h for w in d['words'])
    h2 = set(w for d, h in zip(D, half) if not h for w in d['words'])
    cr['random_halves'] = chapman(h1, h2)
    sg1 = set(s for d in docs if merge_site(d['site']) == main for s in d['signs'])
    sg2 = set(s for d in docs if merge_site(d['site']) != main for s in d['signs'])
    cr['signs_main_vs_rest'] = chapman(sg1, sg2)
    out['capture_recapture'] = cr
    return out


if __name__ == '__main__':
    t0 = time.time()
    LA = load_la()
    LB = load_lb()
    res = {}
    res['LA'] = analyse(LA, 'LA')
    print('LA', time.time() - t0, flush=True)
    res['LA_shuffled'] = analyse(token_shuffle(LA, prng), 'LAshuf', keystone=False)
    print('shuf', time.time() - t0, flush=True)
    # LB at LA size: random document subsamples with ~1332 word tokens, keeping site mix
    LBw = [d for d in LB if d['words']]
    subs = []
    for r in range(3):
        prng.shuffle(LBw)
        sub, tok = [], 0
        for d in LBw:
            if tok >= 1332:
                break
            sub.append(d)
            tok += len(d['words'])
        subs.append(analyse(sub, 'LBsub', nnull=max(NNULL // 2, 50), keystone=(r == 0)))
        print('lbsub', r, time.time() - t0, flush=True)
    res['LB_LAsize'] = subs
    res['LB_full'] = analyse(LB, 'LB', nnull=max(NNULL // 4, 30), keystone=False)
    print('lbfull', time.time() - t0, flush=True)
    json.dump(res, open(os.path.join(OUT, 'c2.json'), 'w'), indent=1, default=float, ensure_ascii=False)
    print('done', time.time() - t0)
