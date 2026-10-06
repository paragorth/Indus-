"""v75 cycle 3: WRITE THE DESIGNATION CODE YOURSELF. Twenty thousand random designation grammars fill the
Voynich's own page/line/paragraph skeleton (ZL3b pages, so layout is identical by construction); each grammar:
  F fields (1-6) in a fixed order kept with probability rho (else shuffled per record); each field has a
  vocabulary V_f (5-3000, log-uniform), Zipf exponent a_f, section-specific share s_f, optionality p_f;
  a page cache (probability c a token re-uses one already written on the page: one object per page);
  a doubling operator (probability d a token is written again right after itself, geometric);
  records start at every paragraph start and run on across lines.
Summary = the type-level fingerprint (13 features; symbol-level and layout features are not used).
Rejection ABC: accept the nearest 1% to a target; FIT on 7 features, PREDICT the other 6 (held-out features).
Targets: Voynich ZL3b and IT2a (paragraph text through E1c, replicate A chunks), the Voynich's own generators
(MK2, SELFCIT, WSHUF, JUNC, GSHUF), and 20 planted grammars drawn from the prior (calibration: coverage of the
true parameters, held-out feature error). Output data/v75_ckpt/c3_sims.pkl."""
import os, sys, pickle, time, random, math
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_lib as X
import v72_lib as V

TYPEF = ['ttr', 'hapax', 'zipf', 'reuse_ent', 'rep_in_ent', 'adj_rep', 'sec_mi', 'page_rec', 'direction',
         'bigram_mi', 'pos_mi', 'head_mi', 'lag2_rep']
FIT = ['ttr', 'hapax', 'zipf', 'reuse_ent', 'sec_mi', 'page_rec', 'direction']
HOLD = ['rep_in_ent', 'adj_rep', 'bigram_mi', 'pos_mi', 'head_mi', 'lag2_rep']
TI = [X.FEATS.index(f) for f in TYPEF]
SK = {}


def skeleton():
    if 'p' not in SK:
        SK['p'] = [p for p in X.extract(X.voy_surface('ZL3b'))]
    return SK['p']


def prior(rng):
    F = rng.randint(1, 6)
    fields = [dict(V=int(math.exp(rng.uniform(math.log(5), math.log(3000)))), a=rng.uniform(0.6, 1.6),
                   s=rng.random(), p=rng.uniform(0.3, 1.0) if i else 1.0) for i in range(F)]
    return dict(F=F, fields=fields, rho=rng.random(), c=rng.uniform(0, 0.5), d=rng.uniform(0, 0.12))


class Grammar:
    def __init__(self, P, secs, seed):
        self.P = P; self.rng = np.random.default_rng(seed); self.lex = {}
        self.cdf = []
        for fi, f in enumerate(P['fields']):
            r = np.arange(1, f['V'] + 1, dtype=float); w = r ** (-f['a']); self.cdf.append(np.cumsum(w) / w.sum())
        self.secs = secs

    def tok(self, fi, sec):
        f = self.P['fields'][fi]
        k = int(np.searchsorted(self.cdf[fi], self.rng.random()))
        # rank k is section-specific with probability s (its own id per section), else shared
        if (hash((fi, k)) % 1000) / 1000.0 < f['s']: return 'f%d_%s_%d' % (fi, sec, k)
        return 'f%d_%d' % (fi, k)

    def record(self, sec, cache):
        P = self.P; order = list(range(P['F']))
        if self.rng.random() > P['rho']: self.rng.shuffle(order)
        out = []
        for fi in order:
            if self.rng.random() > P['fields'][fi]['p']: continue
            if cache and self.rng.random() < P['c']: t = cache[int(self.rng.integers(len(cache)))]
            else: t = self.tok(fi, sec)
            out.append(t); cache.append(t)
            while self.rng.random() < P['d']: out.append(t)
        return out or [self.tok(0, sec)]

    def fill(self, pages):
        res = []
        for p in pages:
            cache = []; buf = []; nl = []
            for l in p['lines']:
                if l['ps']: buf = []
                ws = []
                while len(ws) < len(l['w']):
                    if not buf: buf = self.record(p['sec'], cache)
                    ws.append(buf.pop(0))
                nl.append(dict(l, w=ws))
            res.append(dict(p, lines=nl))
        return res


def summary(pages_list, seed):
    return np.array([np.array(X.fingerprint(ch, seed + i))[TI] for i, ch in enumerate(pages_list)]).mean(0)


def sim(args):
    i, seed = args
    rng = random.Random(seed); P = prior(rng)
    sk = skeleton(); chs = X.chunks(sk, 6, seed)
    g = Grammar(P, None, seed)
    filled = [g.fill(ch) for ch in chs]
    return P, summary(filled, seed)


def targets():
    T = {}
    import v75_cls as C
    for rep in ('A',):
        f, rows, F = C.load(rep)
        for key in ['voy', 'vgen:MK2', 'vgen:SELFCIT', 'vgen:WSHUF', 'vgen:JUNC', 'vgen:GSHUF']:
            for base in ('VOY_ZL3b', 'VOY_IT2a'):
                sel = [i for i, r in enumerate(rows) if r['base'] == base and (r['role'].startswith('voy') if key == 'voy' else r['role'] == key)]
                T['%s|%s' % (base, key)] = F[sel][:, TI].mean(0)
    return T


if __name__ == '__main__':
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    t0 = time.time()
    with Pool(2) as Pp:
        res = []
        for k, r in enumerate(Pp.imap_unordered(sim, [(i, 750000 + i) for i in range(NS)], chunksize=50)):
            res.append(r)
            if k % 1000 == 0: print(k, '%.0fs' % (time.time() - t0), flush=True)
        planted = Pp.map(sim, [(i, 990000 + i) for i in range(20)])
    pickle.dump(dict(sims=res, planted=planted, targets=targets(), TYPEF=TYPEF, FIT=FIT, HOLD=HOLD),
                open(os.path.join(X.CK, 'c3_sims.pkl'), 'wb'))
    print('done %.0fs' % (time.time() - t0), flush=True)
