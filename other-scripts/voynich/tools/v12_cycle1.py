"""v12 cycle 1: fixed interleaving schemes.
For every corpus: held-out link bits (CtxModel, beta 20, 2-fold folio parity) for
  lag 1..4 (fixed-lag context), parity streams (k=2) and mod-3 / mod-4 streams (per-stream tables),
  parity streams carried across line breaks (a woven continuous text keeps its streams over the break),
  glyph-class streams: stream = class of the word's FIRST glyph / LAST glyph / gallows, bipartition searched
  greedily on the training fold (objective = junction MI between consecutive same-stream words, minus the
  same on a within-line shuffle), scored on the test fold.
Every number is reported as EXCESS over the same scheme applied to the corpus with words shuffled inside
each line (2 seeds) -> what the order adds. Regularity side-metrics (lag 1 vs lag 2): repeated word-bigram
tokens and Hamming-1 neighbours per 1000 pairs, excess over the same shuffles.
Positive controls must show split gain (stream > lag-1); negatives must not."""
import sys, os, json, math, random, time
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v12_lib as V

GALL = set('ktpfKTPF')


def shuf(lines, seed):
    rng = random.Random(seed); out = []
    for l in lines:
        ws = list(l['words']); rng.shuffle(ws); out.append(dict(l, words=ws))
    return out


def jmi(lines, key, S):
    """plug-in MI(last glyph of previous same-stream word; first glyph) for class labels key(w) in S -> 1."""
    c = Counter(); a = Counter(); b = Counter(); n = 0
    for l in lines:
        last = {}
        for w in l['words']:
            s = 1 if key(w) in S else 0
            if s in last:
                c[(last[s], w[0])] += 1; a[last[s]] += 1; b[w[0]] += 1; n += 1
            last[s] = w[-1]
    return sum(v / n * math.log2(v * n / (a[x] * b[y])) for (x, y), v in c.items()) if n else 0.0


def gkey(w):
    for ch in w:
        if ch in GALL: return ch
    return '-'

KEYS = {'first': lambda w: w[0], 'last': lambda w: w[-1], 'gallows': gkey}


def class_search(train, key, seed=0):
    sh = shuf(train, 99)
    alpha = sorted({key(w) for l in train for w in l['words']})
    rng = random.Random(seed); S = set(a for a in alpha if rng.random() < 0.5)
    obj = lambda S: jmi(train, key, S) - jmi(sh, key, S)
    best = obj(S); improved = True
    while improved:
        improved = False
        for a in alpha:
            T = S ^ {a}
            if not T or len(T) == len(alpha): continue
            v = obj(T)
            if v > best + 1e-5: S, best, improved = T, v, True
    return S, best


def regularity(lines, lag):
    rep = Counter(); ham = 0; n = 0
    for l in lines:
        ws = l['words']
        for i in range(len(ws) - lag):
            rep[(ws[i], ws[i + lag])] += 1; n += 1
            ham += V.hamming1(ws[i], ws[i + lag])
    r = sum(v for v in rep.values() if v > 1)
    return 1000 * r / max(n, 1), 1000 * ham / max(n, 1)


def schemes_eval(lines):
    r = {}
    for k in (1, 2, 3, 4):
        r[f'lag{k}'] = V.cv_link(lines, lag=k)[0]
    r['par2'] = V.cv_link(lines, assign=lambda ws: [i % 2 for i in range(len(ws))])[0]
    r['mod3'] = V.cv_link(lines, assign=lambda ws: [i % 3 for i in range(len(ws))])[0]
    r['mod4'] = V.cv_link(lines, assign=lambda ws: [i % 4 for i in range(len(ws))])[0]
    r['lag1_carry'] = V.cv_link(lines, carry=True)[0]
    # parity streams carried across lines: global token parity within paragraph
    def carry_par(lines):
        out = []; t = 0
        for l in lines:
            if l.get('para_start'): t = 0
            out.append(dict(l, _off=t)); t += len(l['words'])
        return out
    cl = carry_par(lines)
    r['par2_carry'] = _cv_offset(cl)
    for k in (1, 2):
        r[f'rep{k}'], r[f'ham{k}'] = regularity(lines, k)
    return r


def _cv_offset(cl):
    """parity streams by paragraph-global token index, contexts carried across line breaks."""
    res = []
    for fold in (0, 1):
        tr, te = V.split(cl, fold)
        def pairs(ls):
            out = []; last = {}
            for l in ls:
                if l.get('para_start'): last = {}
                for i, w in enumerate(l['words']):
                    s = (l['_off'] + i) % 2; f = V.feats(w)
                    out.append((f, last.get(s, V.START), s)); last[s] = f
            return out
        res.append(V.link_bits(pairs(tr), pairs(te))[0])
    return sum(res) / 2


def run(name):
    ck = os.path.join(V.CK, f'c1_{name}.json')
    if os.path.exists(ck): return json.load(open(ck))
    t0 = time.time()
    C = V.corpus(name)
    obs = schemes_eval(C)
    nulls = [schemes_eval(shuf(C, s)) for s in (1, 2)]
    ex = {k: obs[k] - sum(n[k] for n in nulls) / 2 for k in obs}
    sd = {k: abs(nulls[0][k] - nulls[1][k]) / math.sqrt(2) for k in obs}
    # class splits: search on train fold, score test fold
    cls = {}
    for kn, key in V_KEYS():
        tot_o = tot_n = 0; sets = []
        for fold in (0, 1):
            tr, te = V.split(C, fold)
            S, trobj = class_search(tr, key)
            asg = (lambda S, key: (lambda ws: [1 if key(w) in S else 0 for w in ws]))(S, key)
            o = V.link_bits(V.ctx_pairs(tr, assign=asg), V.ctx_pairs(te, assign=asg))[0]
            trs, tes = shuf(tr, 7), shuf(te, 7)
            n = V.link_bits(V.ctx_pairs(trs, assign=asg), V.ctx_pairs(tes, assign=asg))[0]
            tot_o += o / 2; tot_n += n / 2; sets.append(sorted(S))
        cls[kn] = {'obs': tot_o, 'null': tot_n, 'excess': tot_o - tot_n, 'sets': sets}
    out = {'name': name, 'obs': obs, 'excess': ex, 'null_sd': sd, 'class': cls, 'sec': time.time() - t0}
    json.dump(out, open(ck, 'w'), indent=1)
    print(name, 'done', round(time.time() - t0), flush=True)
    return out


def V_KEYS():
    return list(KEYS.items())


if __name__ == '__main__':
    names = V.VOY + V.POS + V.NEG + ['Gloss-pair']
    with Pool(2) as p:
        R = p.map(run, names, chunksize=1)
    json.dump(R, open(os.path.join(V.CK, 'cycle1.json'), 'w'), indent=1)
    cols = ['lag1', 'lag2', 'lag3', 'lag4', 'par2', 'mod3', 'mod4', 'lag1_carry', 'par2_carry']
    print('excess link bits over within-line shuffle')
    print(f"{'corpus':14s} " + ' '.join(f'{c:>10s}' for c in cols) + '  cls-first cls-last cls-gall  rep1 rep2 ham1 ham2')
    for r in R:
        e = r['excess']
        print(f"{r['name']:14s} " + ' '.join(f'{e[c]:10.3f}' for c in cols) + '  ' +
              ' '.join(f"{r['class'][k]['excess']:8.3f}" for k in ('first', 'last', 'gallows')) +
              f"  {e['rep1']:5.1f} {e['rep2']:5.1f} {e['ham1']:5.1f} {e['ham2']:5.1f}")
