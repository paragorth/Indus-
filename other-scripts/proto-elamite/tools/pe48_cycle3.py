"""pe48 cycle 3: diffusion forecasts and outpost-specific signs.

(a) LEAVE-ONE-SITE-OUT: hold out every tablet of one outpost (PE: Yahya, Malyan, Sofalin, Sialk; Ur III
    PE-shaped draw: the four largest outposts; Linear B likewise). Fit on the rest with each survivor theta
    and score the held-out site's tablets against the travelling model (gain, nats/tablet). Nulls: the
    same on site-shuffled data (8 shuffles) -- does geography forecast an unseen site?
(b) OUTPOST ENRICHMENT: per outpost and per route (south = Malyan, Yahya, Shahr-i Sokhta; north = Sialk,
    Sofalin, Ozbaki), signs present in more tablets than the travelling expectation (Poisson-binomial
    tail with the per-tablet presence probability of the travelling model). Null: site labels permuted
    among the outpost tablets only (1,000 times) and the frequency-resampled corpus; count of signs with
    p < 0.01 real vs null.
(c) FROZEN FORECASTS for new tablets at each outpost from the survivor fits; sha256 written.
"""
import sys, json, os, random
import numpy as np
from multiprocessing import Pool
from scipy.stats import poisson
import pe48_lib as L

NSH = int(os.environ.get('NSH', 8))


def loso(dat, thetas, sites):
    out = {}
    for s in sites:
        j = dat.sites.index(s)
        te = dat.site == j
        if te.sum() == 0:
            continue
        g = [float(L.fit_eval(dat, th, ~te, te).mean()) for th in thetas]
        out[s] = dict(n=int(te.sum()), mean_gain=float(np.mean(g)), best=float(np.max(g)),
                      frac_pos=float(np.mean(np.array(g) > 0)))
    return out


def job(args):
    corpus, kind, k = args
    fn = os.path.join(L.CK, 'c3_loso_%s_%s_%d.json' % (corpus, kind, k))
    if os.path.exists(fn):
        return json.load(open(fn))
    rng = random.Random(900 + k)
    if corpus == 'PE':
        docs = L.pe_docs(); sites = ['Yahya', 'Malyan', 'Sofalin', 'Sialk']
        thetas = json.load(open(os.path.join(L.CK, 'c2_survivors.json')))[:60]
    else:
        src = L.ur3_docs() if corpus == 'UR3' else L.lb_docs()
        hub, outs = (('Umma', ['Puzriš-Dagan', 'Girsu', 'Ur', 'Nippur', 'Garšana', 'Irisagrig'])
                     if corpus == 'UR3' else ('KN', ['PY', 'TH', 'MY', 'TI', 'KH']))
        docs = L.pe_shaped(src['docs'], hub, outs, random.Random(1000 + k))
        sites = outs[:4]
        trng = np.random.default_rng(k)
        thetas = [L.rand_theta(trng) for _ in range(60)]
    if kind == 'shuf':
        lab = [d['site'] for d in docs]; rng.shuffle(lab)
        docs = [dict(d, site=s) for d, s in zip(docs, lab)]
    dat = L.Data(docs, corpus, max_vocab=2000 if corpus == 'PE' else 600)
    out = dict(corpus=corpus, kind=kind, k=k, loso=loso(dat, thetas, sites))
    json.dump(out, open(fn, 'w'))
    return out


def enrichment(dat, groups):
    """For each group (list of site names) and sign: observed tablets with the sign vs travelling
    expectation (rate fitted on all tablets). Returns dict group -> list of (sign, obs, exp, p)."""
    q = (dat.C.sum(0) + 0.5) / dat.L.sum()
    res = {}
    for gname, sites in groups.items():
        m = np.isin(dat.site, [dat.sites.index(s) for s in sites if s in dat.sites])
        p_doc = -np.expm1(-dat.L[m][:, None] * q[None, :])   # (n, S)
        exp = p_doc.sum(0); obs = dat.Y[m].sum(0)
        pv = poisson.sf(obs - 1, exp)                          # approx Poisson-binomial tail
        res[gname] = [(dat.vocab[s], int(obs[s]), float(exp[s]), float(pv[s])) for s in range(len(dat.vocab))]
    return res


GROUPS = {'Yahya': ['Yahya'], 'Malyan': ['Malyan'], 'Sofalin': ['Sofalin'], 'Sialk': ['Sialk'],
          'SOUTH': ['Malyan', 'Yahya', 'Shahr-i Sokhta'], 'NORTH': ['Sialk', 'Sofalin', 'Ozbaki'],
          'PLATEAU': ['Malyan', 'Yahya', 'Shahr-i Sokhta', 'Sialk', 'Sofalin', 'Ozbaki']}


def count_hits(enr, thr=0.01):
    return {g: sum(1 for x in v if x[3] < thr and x[1] >= 2) for g, v in enr.items()}


if __name__ == '__main__':
    jobs = [('PE', 'real', 0)] + [('PE', 'shuf', k) for k in range(NSH)]
    jobs += [('UR3', 'real', k) for k in range(2)] + [('UR3', 'shuf', k) for k in range(2)]
    jobs += [('LB', 'real', k) for k in range(2)] + [('LB', 'shuf', k) for k in range(2)]
    with Pool(2) as p:
        outs = p.map(job, jobs, chunksize=1)
    summ = {'loso': outs}
    # (b) enrichment
    docs = L.pe_docs()
    dat = L.Data(docs, 'PE', max_vocab=2000)
    enr = enrichment(dat, GROUPS)
    real_hits = count_hits(enr)
    rng = random.Random(5)
    outidx = [i for i, d in enumerate(docs) if d['site'] != 'Susa']
    null_hits = {g: [] for g in GROUPS}
    for r in range(1000):
        lab = [docs[i]['site'] for i in outidx]; rng.shuffle(lab)
        d2 = list(docs)
        for i, s in zip(outidx, lab):
            d2[i] = dict(docs[i], site=s)
        # site re-index without rebuilding the sign matrix
        si = {s: i for i, s in enumerate(dat.sites)}
        dat.site = np.array([si[d['site']] for d in d2])
        h = count_hits(enrichment(dat, {g: v for g, v in GROUPS.items() if g != 'PLATEAU'}))
        for g in h:
            null_hits[g].append(h[g])
    dat.site = np.array([dat.sites.index(d['site']) for d in docs])
    # frequency-resampled corpus
    fr_hits = []
    for r in range(20):
        rr = random.Random(700 + r)
        pool = [t for d in docs for t in d['toks']]
        fd = [dict(d, toks=[rr.choice(pool) for _ in d['toks']]) for d in docs]
        fr_hits.append(count_hits(enrichment(L.Data(fd, 'PE', max_vocab=2000), GROUPS)))
    summ['enrich'] = dict(real=real_hits,
                          null_outpost_perm={g: (float(np.mean(v)), float(np.percentile(v, 95)),
                                                 float(np.mean(np.array(v) >= real_hits[g])))
                                             for g, v in null_hits.items() if v},
                          null_freqres={g: (float(np.mean([h[g] for h in fr_hits])),
                                            float(np.max([h[g] for h in fr_hits]))) for g in GROUPS})
    top = {}
    for g, v in enr.items():
        top[g] = sorted([x for x in v if x[1] >= 2], key=lambda x: x[3])[:12]
    summ['enrich_top'] = top
    # (c) frozen forecasts
    surv = json.load(open(os.path.join(L.CK, 'c2_survivors.json')))
    allm = np.ones(len(dat.site), bool)
    rates = np.zeros((len(dat.sites), len(dat.vocab)))
    Lmed = {s: float(np.median(dat.L[dat.site == i])) for i, s in enumerate(dat.sites)}
    for th in surv[:60]:
        cls, _, q = L.fit_eval(dat, th, allm, allm, return_cls=True)
        G = L.gmat(dat, th)
        for i, s in enumerate(dat.sites):
            rates[i] += -np.expm1(-Lmed[s] * G[cls, i] * q[cls, np.arange(len(cls))])
    rates /= len(surv[:60])
    fc = {}
    for i, s in enumerate(dat.sites):
        if s == 'Susa':
            continue
        lift = rates[i] / np.maximum(rates[0] * Lmed[s] / Lmed['Susa'], 1e-9)
        order = np.argsort(-lift)
        fc[s] = [(dat.vocab[j], float(rates[i, j]), float(lift[j])) for j in order[:10] if rates[i, j] > 0.02]
    frozen = dict(note='pe48 forecasts: per-new-tablet presence probability and lift over the Susa-rate expectation '
                       'at the median tablet length of each site', forecasts=fc)
    frozen['sha256_16'] = L.sha(fc)
    json.dump(frozen, open(os.path.join(L.DATA, 'pe48_frozen_forecasts.json'), 'w'), indent=1)
    summ['frozen_sha'] = frozen['sha256_16']; summ['forecasts'] = fc
    json.dump(summ, open(os.path.join(L.CK, 'c3_summary.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != 'loso'}, indent=1, default=float)[:6000])
    for o in outs:
        print(o['corpus'], o['kind'], o['k'], json.dumps(o['loso']))
