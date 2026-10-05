"""v65 cycle 2: search every physically allowed arrangement of each quire for maximal
carry-over, held out by word class.

For each corpus, feature (word / skel), and quire with >= 8 arrangements:
  best0 = arg max seam score with word-class 0; its percentile under class 1 (held out); and
  best1 vice versa.  held-out z = sum over quires of (pct - 0.5) / sqrt(n/12).
  agree = shared directed adjacencies between best0 and best1, minus the mean for random
          arrangement pairs (stability).
  bind  = shared adjacencies between best(all words) and the binding (truth for controls).
Also the singulion model (Layfield & Davis 2026): every bifolio its own gathering, sequence of
bifolios free, for Q13 and Q20.
Null: pages shuffled within section x language (W relabelled), 30x, same pipeline.
"""
import os, sys, json, random, time, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v65_lib as L
import v65_c1 as C1


class SingulionSpace(L.QuireSpace):
    def __init__(self, bifs, kidx):
        bifs = [b for b in bifs if not (b[1] is None and b[2] is None)]
        self.bifs = bifs
        nb = len(bifs)
        self.configs = L.all_configs(nb)
        seqs = []
        for o, f in self.configs:
            s = []
            for i in o:
                s += L.seq_for([bifs[i]], (0,), (f[i],)) + [L.GAP]
            seqs.append([kidx.get(x, -1) if x != L.GAP else -1 for x in s])
        self.S = np.array(seqs, dtype=np.int32)
        self.ident = self.configs.index((tuple(range(nb)), (0,) * nb))
        a, b = self.S[:, :-1], self.S[:, 1:]
        self.valid = (a >= 0) & (b >= 0)
        self.a = np.where(self.valid, a, 0); self.b = np.where(self.valid, b, 0)


def adj_overlap(sp, i, j):
    A = set(sp.pairs(i)); B = set(sp.pairs(j))
    return len(A & B) / max(1, len(A))


def rand_overlap(sp, rng, n=200):
    m = len(sp.configs)
    return float(np.mean([adj_overlap(sp, rng.randrange(m), rng.randrange(m)) for _ in range(n)]))


def run_corpus(pages, meta, feat, cache, rng, nnull=0, models=('quire',)):
    keys = sorted(k for k in pages if sum(len(x) for x in pages[k]) >= L.MIN_WORDS and k in meta)
    kidx = {k: i for i, k in enumerate(keys)}
    groups = [str(meta[k]['sec']) + str(meta[k]['lang']) for k in keys]
    W = {}
    for cls in (None, 0, 1):
        Js, _ = L.carry_matrices(pages, keys, cls=cls, feat=feat)
        W[cls] = L.centre(Js, groups)
    spaces = []
    for qn, bifs in L.structure()[0]:
        for model in models:
            if model == 'singulion' and qn not in ('M', 'T'):
                continue
            ck = (qn, model, tuple(keys))
            sp = cache.get(ck)
            if sp is None:
                sp = L.QuireSpace(bifs, kidx) if model == 'quire' else SingulionSpace(bifs, kidx)
                sp.rand_ov = rand_overlap(sp, rng)
                cache[ck] = sp
            if len(sp.configs) >= 8:
                spaces.append((qn + ('' if model == 'quire' else '_sing'), sp))

    def evaluate(Wd):
        per = {}
        for qn, sp in spaces:
            s0, s1, sa = sp.scores(Wd[0]), sp.scores(Wd[1]), sp.scores(Wd[None])
            b0, b1, ba = int(np.argmax(s0)), int(np.argmax(s1)), int(np.argmax(sa))
            p01 = float((s1 < s1[b0]).mean() + 0.5 * (s1 == s1[b0]).mean())
            p10 = float((s0 < s0[b1]).mean() + 0.5 * (s0 == s0[b1]).mean())
            per[qn] = {'n': len(sp.configs), 'p01': p01, 'p10': p10,
                       'agree': adj_overlap(sp, b0, b1) - sp.rand_ov,
                       'bind': adj_overlap(sp, ba, sp.ident) - sp.rand_ov,
                       'bind_rank': float((sa > sa[sp.ident]).mean()),
                       'best': ba}
        q = [k for k in per if not k.endswith('_sing')]
        ps = [per[k]['p01'] for k in q] + [per[k]['p10'] for k in q]
        z = (np.mean(ps) - 0.5) / np.sqrt(1 / 12 / len(ps))
        return per, float(z), float(np.mean([per[k]['agree'] for k in q])), float(np.mean([per[k]['bind'] for k in q]))

    per, z, ag, bd = evaluate(W)
    res = {'per': per, 'heldout_z': z, 'agree': ag, 'bind': bd}
    if nnull:
        g = np.array(groups); nz, nag, nbd = [], [], []
        for _ in range(nnull):
            perm = np.arange(len(keys))
            for x_ in set(groups):
                ix = np.where(g == x_)[0]; perm[ix] = np.random.permutation(ix)
            Wn = {c: W[c][np.ix_(perm, perm)] for c in W}
            _, z_, a_, b_ = evaluate(Wn)
            nz.append(z_); nag.append(a_); nbd.append(b_)
        res['null_z'] = [float(np.mean(nz)), float(np.std(nz))]
        res['null_agree'] = [float(np.mean(nag)), float(np.std(nag))]
        res['null_bind'] = [float(np.mean(nbd)), float(np.std(nbd))]
    return res, keys


def main():
    np.random.seed(652); rng = random.Random(652)
    C = C1.corpora()
    out = {}; cache = {}
    plan = [('VOY_ZL', 'word', 30), ('VOY_ZL', 'skel', 30), ('VOY_IT', 'word', 30), ('VOY_IT', 'skel', 10),
            ('CTL_isidore_flow', 'skel', 10), ('CTL_konrad_flow', 'skel', 10), ('CTL_konrad_entry', 'skel', 0),
            ('CTL_isidore_flow', 'word', 0), ('NULL_markov', 'word', 0), ('NULL_selfcit', 'word', 0)]
    for name, feat, nnull in plan:
        t = time.time()
        pages, meta = C[name]
        r, keys = run_corpus(pages, meta, feat, cache, rng, nnull=nnull, models=('quire', 'singulion'))
        r['keys'] = [list(k) for k in keys]
        out[f'{name}|{feat}'] = r
        short = {q: (round(v['p01'], 2), round(v['p10'], 2), round(v['agree'], 2), round(v['bind'], 2)) for q, v in r['per'].items()}
        print(name, feat, 'heldout_z %.2f agree %.3f bind %.3f' % (r['heldout_z'], r['agree'], r['bind']),
              r.get('null_z'), r.get('null_agree'), r.get('null_bind'), f'{time.time()-t:.0f}s', flush=True)
        print('   ', short, flush=True)
        json.dump(out, open(os.path.join(L.CK, 'c2.json'), 'w'))


if __name__ == '__main__':
    main()
