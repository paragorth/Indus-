"""v92 cycle 3 (K4): SIMULATE THE WRITER.
Random writer programs rewrite a target's own page/line skeleton: a base layer (junction draw by previous word's
last glyph, line-initial pool, per-page lexical mood beta_lx, per-page seed vocabulary with copy-and-vary sd_p) plus an
ENTITY layer: each page has k_e page-specific names (new Voynich-shaped words made by varying the middle of a rare
real word), written at rate r_e per token with a decay phi along the page (names fade) and per-mention spelling
variation v_e. Summary statistics describe only the page-recurring frames (recurrence, confinement to one page,
front-loading, spelling likeness, rare neighbours, count scaling, number per page). Rejection ABC (closest 5%)
gives a posterior for the entity share of tokens. Calibration: real herbals/recipes/astronomy (entry pages through
the merge code + v72 surface) must show entity share above that of meaningless generators run through the same ABC."""
import os, sys, json, time, math, random
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v92_lib as L

NP_ = int(os.environ.get('V92_NP', '700'))


class Writer:
    def __init__(self, pages):
        self.pages = pages
        self.types = sorted({w for p in pages for l in p['lines'] for w in l['w']})
        self.tix = {w: i for i, w in enumerate(self.types)}
        self.NT = len(self.types)
        F = defaultdict(Counter); I = defaultdict(Counter); S = defaultdict(Counter)
        for p in pages:
            for l in p['lines']:
                I[p['sec']][self.tix[l['w'][0]]] += 1
                for w in l['w']: S[p['sec']][self.tix[w]] += 1
                for a, b in zip(l['w'], l['w'][1:]): F[(p['sec'], a[-1])][self.tix[b]] += 1
        conv = lambda c: (np.array(list(c.keys())), np.array(list(c.values()), float))
        self.F = {k: conv(v) for k, v in F.items()}; self.I = {k: conv(v) for k, v in I.items()}
        self.S = {k: conv(v) for k, v in S.items()}
        cnt = Counter(w for p in pages for l in p['lines'] for w in l['w'])
        self.rare = [w for w in self.types if 2 <= cnt[w] <= 5 and len(w) >= 4]
        self.gly = sorted({c for w in self.types for c in w})
        self.mid_gly = Counter(c for w in self.types for c in w[1:-1])

    def mutate_mid(self, w, rng):
        if len(w) < 3: return w + rng.choice(self.gly)
        j = rng.randrange(1, len(w) - 1)
        ks = list(self.mid_gly); c = rng.choice(ks)
        return w[:j] + c + w[j + 1:]

    def write(self, P, seed):
        rng = random.Random(seed); nr = np.random.default_rng(seed)
        out = []; ent_tok = 0; ntok = 0
        for p in self.pages:
            sec = p['sec']
            mood = nr.normal(0, 1, self.NT) * P['lx'] if P['lx'] > 0 else None
            ids, c = self.S[sec]
            seeds = [self.types[i] for i in nr.choice(ids, size=P['sd_n'], p=c / c.sum())] if P['sd_n'] else []
            ents = []
            for _ in range(P['k_e']):
                w = rng.choice(self.rare)
                for _ in range(1 + rng.randrange(2)): w = self.mutate_mid(w, rng)
                ents.append(w)
            nlines = len(p['lines']); nl = []
            for li, l in enumerate(p['lines']):
                n = len(l['w']); ws = []
                pos = li / max(nlines - 1, 1)
                for i in range(n):
                    ntok += 1
                    if ents and rng.random() < P['r_e'] * math.exp(-P['phi'] * pos) * (1 + P['phi']) / (1 + P['phi'] * 0.5):
                        j = rng.randrange(len(ents)); w = ents[j]
                        if rng.random() < P['v_e']: w = self.mutate_mid(w, rng)
                        ws.append(w); ent_tok += 1; continue
                    if seeds and i > 0 and rng.random() < P['sd_p']:
                        j = rng.randrange(len(seeds)); w = seeds[j]
                        if rng.random() < 0.4: w = self.mutate_mid(w, rng); seeds[j] = w
                        ws.append(w); continue
                    key = (sec, ws[-1][-1]) if ws else None
                    pool = self.F.get(key) if key else self.I.get(sec)
                    if pool is None: pool = self.S[sec]
                    ids, c = pool
                    if mood is not None:
                        wt = c * np.exp(mood[ids])
                    else:
                        wt = c
                    cs = np.cumsum(wt)
                    w = self.types[ids[min(int(np.searchsorted(cs, rng.random() * cs[-1])), len(ids) - 1)]]
                    if rng.random() < P['nov']: w = self.mutate_mid(w, rng)
                    ws.append(w)
                nl.append(dict(l, w=ws))
            out.append(dict(p, lines=nl))
        return out, ent_tok / max(ntok, 1)


def sample_P(rng):
    return dict(lx=float(rng.choice([0.0, rng.uniform(0, 1.5)])), sd_n=int(rng.choice([0, rng.integers(1, 10)])),
                sd_p=float(rng.uniform(0, 0.08)), k_e=int(rng.integers(0, 7)), r_e=float(rng.choice([0.0, rng.uniform(0, 0.06)])),
                phi=float(rng.choice([0.0, rng.uniform(0, 3)])), v_e=float(rng.uniform(0, 0.5)),
                nov=float(rng.uniform(0, 0.5)))


def ned(a, b):
    n, m = len(a), len(b)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[m] / max(n, m)


def summary(pages, seed=0):
    rng = random.Random(seed)
    f = L.frame
    cnt = Counter(f(w) for p in pages for l in p['lines'] for w in l['w'])
    nhap = sum(1 for u, c in cnt.items() if c == 1) / len(cnt)
    ev = hit = 0; conf = []; front = []; sims = []; rsim = []; nbr = []; nrec = []; logn = []; logm = []
    allu = [u for u, c in cnt.items() if 2 <= c <= 20]
    for p in pages:
        lines = [[f(w) for w in l['w']] for l in p['lines']]
        n = len(lines)
        occ = defaultdict(set); pc = Counter()
        for li, ls in enumerate(lines):
            for u in ls: occ[u].add(li); pc[u] += 1
        rec = [u for u in occ if len(occ[u]) >= 2 and 2 <= cnt[u] <= 20]
        for li, ls in enumerate(lines):
            for i, u in enumerate(ls):
                if 2 <= cnt[u] <= 20:
                    ev += 1
                    r = len(occ[u] - {li}) > 0; hit += r
                    if r and i > 0: nbr.append(2 <= cnt[ls[i - 1]] <= 20 or cnt[ls[i - 1]] == 1)
        ptok = sum(len(x) for x in lines)
        nrec.append(len(rec) / ptok)
        if rec:
            logn.append(math.log(ptok)); logm.append(math.log(max(pc[u] for u in rec)))
        for u in rec:
            conf.append(pc[u] / cnt[u] >= 0.8)
            if n >= 4: front.append(0.5 - np.mean(sorted(occ[u])) / (n - 1))
        if len(rec) >= 2:
            for _ in range(3):
                a, b = rng.sample(rec, 2); sims.append(ned(a, b))
                rsim.append(ned(rng.choice(allu), rng.choice(allu)))
    slope = float(np.polyfit(logn, logm, 1)[0]) if len(logn) > 5 else 0.0
    return dict(rec=hit / max(ev, 1), conf=float(np.mean(conf)) if conf else 0, front=float(np.mean(front)) if front else 0,
                sim=(float(np.mean(sims)) - float(np.mean(rsim))) if sims else 0, nbr=float(np.mean(nbr)) if nbr else 0,
                nrec=float(np.mean(nrec)) * 100, slope=slope, hap=nhap)


KEYS = ['rec', 'conf', 'front', 'sim', 'nbr', 'nrec', 'slope', 'hap']


def run(name):
    out_p = os.path.join(L.CK, 'c3_%s.json' % name)
    done = []
    if os.path.exists(out_p):
        done = json.load(open(out_p))['sims']
        if len(done) >= NP_: return name, 0.0
    t0 = time.time()
    pages = L.corpus(name)
    W = Writer(pages)
    obs = summary(pages)
    rng = np.random.default_rng(930 + len(done))
    sims = done
    for k in range(len(done), NP_):
        P = sample_P(rng)
        g, share = W.write(P, 1000 + k)
        sims.append(dict(P=P, share=share, s=summary(g, k)))
        if k % 50 == 49:
            json.dump(dict(name=name, obs=obs, sims=sims), open(out_p, 'w'))
    json.dump(dict(name=name, obs=obs, sims=sims, secs=time.time() - t0), open(out_p, 'w'))
    return name, time.time() - t0


NAMES = ['ZL3b', 'E_KONRAD', 'G_LX', 'E_CIRCA', 'IT2a', 'G_SEED', 'E_APIC', 'E_HYGIN', 'GC2a', 'G_SELF']

if __name__ == '__main__':
    from multiprocessing import Pool
    names = sys.argv[1:] or NAMES
    for n in names: L.corpus(n)
    with Pool(2) as P:
        for n, s in P.imap_unordered(run, names):
            print(n, '%.0fs' % s, flush=True)
