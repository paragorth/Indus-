"""la63 cycle 3: GROUP erasure. The clerk loses a whole class of words at once.

Rare words (3-4 tokens) cannot be judged one by one at Linear A size, so sets are erased together:
  LA57    : the la57 syllabic 'commodity' candidates DI-DE-RU, DA-SI-*118, *28B-NU-MA-RE, U-*325-ZA, TE-TU
  BETWEEN : the single signs between lines RA, PA, PA3, TU, ME
  FRACW   : words that stand before a fraction in >= half of their (>= 2) TRAINING numeral entries
            (defined on the training folds of each model only), totals excluded
  NOFRACW : words with >= 3 training numeral entries and no fraction (same rule, opposite side)
  HAS:x / BEG:x / END:x : multi-sign words containing / starting with / ending in syllabic sign x
  planted and Linear B control groups (see GROUPS_FOR)
For every group and model, on held-out documents: erase all its tokens (D_g), and as many random
other tokens in the same documents (R draws, D_null). excess_g = D_g - mean D_null per loss key.
The same is done for NRG random type-sets matched on token count (+-25%), drawn from multi-sign
words outside the group. diff_m = excess_g - mean excess_random; t over models. Calibration: random
set 0 played as a group against random sets 1.. gives the null distribution of t.
usage: python3 la63_groups.py CORPUS M0 M1
"""
import collections, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la63_lib as L
from la63_run import corpus, folds

R = 4
NRG = 4
LA57 = ['DI-DE-RU', 'DA-SI-*118', '*28B-NU-MA-RE', 'U-*325-ZA', 'TE-TU']
BETWEEN = ['RA', 'PA', 'PA₃', 'TU', 'ME']
TOTW = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO', 'TO-SO', 'TO-SA', 'TO-SO-DE', 'TO-SA-DE'}


def frac_groups(C, train_idx):
    st = collections.defaultdict(lambda: [0, 0])
    for i in train_idx:
        for e in C[i]['ents']:
            if e['kind'] != 'num':
                continue
            for w in e['toks']:
                if w.startswith('L:') or w.startswith('P_') or w.startswith('D_') or w in TOTW:
                    continue
                st[w][0] += 1
                st[w][1] += 1 if e['frac'] else 0
    fr = sorted(w for w, (n, f) in st.items() if n >= 2 and f / n >= 0.5)
    nf = sorted(w for w, (n, f) in st.items() if n >= 3 and f == 0)
    return fr, nf


def sign_groups(V, minf=15, top_end=12):
    multi = [w for w in V.count if '-' in w and not w.startswith('L:') and w not in TOTW]
    has, beg, end = collections.Counter(), collections.Counter(), collections.Counter()
    for w in multi:
        s = w.split('-')
        for x in set(s):
            has[x] += V.count[w]
        beg[s[0]] += V.count[w]
        end[s[-1]] += V.count[w]
    G = {}
    for x, c in has.items():
        if c >= minf:
            G['HAS:' + x] = [w for w in multi if x in w.split('-')]
    for x, c in beg.most_common(top_end):
        if c >= minf:
            G['BEG:' + x] = [w for w in multi if w.split('-')[0] == x]
    for x, c in end.most_common(top_end):
        if c >= minf:
            G['END:' + x] = [w for w in multi if w.split('-')[-1] == x]
    return G


def groups_for(name, C, V, train_idx):
    if name in ('la', 'la_nw'):
        G = {'LA57': LA57, 'BETWEEN': BETWEEN}
        G['FRACW'], G['NOFRACW'] = frac_groups(C, train_idx)
        G.update(sign_groups(V, minf=30, top_end=8))
        return G
    if name == 'dose':
        G = {'D_COM': ['D_COM10', 'D_COM20'], 'D_UNIT': ['D_UNIT10', 'D_UNIT20'], 'D_N': ['D_N10_0', 'D_N10_1', 'D_N20_0'],
             'D_COM10': ['D_COM10'], 'D_UNIT10': ['D_UNIT10'], 'D_N10': ['D_N10_0', 'D_N10_1']}
        G['FRACW'], G['NOFRACW'] = frac_groups(C, train_idx)
        return G
    if name == 'lb':
        key = L.lb_key(C)
        G = {'QUAL': sorted(w for w, r in key.items() if r == 'QUAL'),
             'PLA': sorted(w for w, r in key.items() if r == 'PLA'),
             'TRA': sorted(w for w, r in key.items() if r == 'TRA'),
             'TOT': sorted(w for w, r in key.items() if r == 'TOT')}
        per = sorted((w for w, r in key.items() if r == 'PER'), key=lambda w: -V.count[w])
        G['PER'] = per[:40]
        G['PER_RARE'] = [w for w in per if 2 <= V.count[w] <= 4][:12]
        G['FRACW'], G['NOFRACW'] = frac_groups(C, train_idx)
        G.update(sign_groups(V, minf=40, top_end=8))
        return G
    raise ValueError(name)


def run(name, m):
    fn = os.path.join(L.CK, 'grp_%s_%d.json' % (name, m))
    if os.path.exists(fn):
        return
    t0 = time.time()
    C = corpus(name)
    V = L.Vocab(C)
    E = [L.encode(d, V) for d in C]
    fo = folds(C)
    fold = m % 5
    tri = [i for i in range(len(E)) if fo[i] != fold]
    ho = [i for i in range(len(E)) if fo[i] == fold]
    arch = L.random_arch(random.Random(3000 + m))
    model = L.train([E[i] for i in tri], V, arch, seed=3000 + m)
    jobs = [(ti, vi, None) for ti in ho for vi in range(len(L.VIEWS))]
    B = {(j[0], j[1]): r for j, r in zip(jobs, L.eval_views(model, E, jobs))}
    G = groups_for(name, C, V, tri)
    rng = random.Random(77 + m)
    multi = [w for w in V.count if w in V.stoi and not w.startswith('L:')]
    ho_count = collections.Counter(V.itos[x] for ti in ho for x in E[ti]['tok'] if x >= L.NSPEC)

    def play(types):
        ids = {V.stoi[w] for w in types if w in V.stoi}
        tabs = [ti for ti in ho if ids & set(E[ti]['tok'])]
        djobs, meta, ents_of = [], [], {}
        for ti in tabs:
            e = E[ti]
            pos = [i for i, x in enumerate(e['tok']) if x in ids]
            other = [i for i, x in enumerate(e['tok']) if x not in (0, 2, 3, 4) and x not in ids]
            if not other:
                continue
            dels = [pos] + [rng.sample(other, min(len(pos), len(other))) for _ in range(R)]
            for d, dm in enumerate(dels):
                ents_of[(ti, d)] = {e['li'][p] for p in dm}
                for vi in range(len(L.VIEWS)):
                    djobs.append((ti, vi, dm)); meta.append((ti, vi, d))
        if not djobs:
            return None
        outs = L.eval_views(model, E, djobs)
        acc = {}
        for (ti, vi, d), r in zip(meta, outs):
            b = B[(ti, vi)]
            e = E[ti]
            for key, v in r.items():
                if key not in b or e['tok'][key[1]] in ids:
                    continue
                k = key[0] + '-' + ('L' if e['li'][key[1]] in ents_of[(ti, d)] else 'R')
                a = acc.setdefault(k, [0.0, 0, [0.0] * R, [0] * R])
                if d == 0:
                    a[0] += v - b[key]; a[1] += 1
                else:
                    a[2][d - 1] += v - b[key]; a[3][d - 1] += 1
        return dict(acc=acc, ntok=sum(ho_count[w] for w in types), ndoc=len({x[0] for x in meta}))

    res = {}
    for g, types in G.items():
        types = [w for w in types if w in V.stoi]
        ntok = sum(ho_count[w] for w in types)
        if ntok < 2:
            continue
        real = play(types)
        if real is None:
            continue
        rand = []
        pool = [w for w in multi if w not in types and ho_count[w] > 0]
        for _ in range(NRG):
            for _try in range(200):
                rng.shuffle(pool)
                pick, c = [], 0
                for w in pool:
                    if c >= 0.75 * ntok:
                        break
                    if c + ho_count[w] <= 1.25 * ntok:
                        pick.append(w); c += ho_count[w]
                if 0.75 * ntok <= c <= 1.25 * ntok:
                    break
            r_ = play(pick)
            if r_ is not None:
                r_['types'] = pick
                rand.append(r_)
        res[g] = dict(types=types, real=real, rand=rand)
    json.dump(dict(corpus=name, m=m, arch=arch, res=res, t=time.time() - t0), open(fn, 'w'))
    print(name, m, 'groups', len(res), '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    for m in range(int(sys.argv[2]), int(sys.argv[3])):
        run(sys.argv[1], m)
