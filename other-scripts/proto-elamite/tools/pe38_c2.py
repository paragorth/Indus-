#!/usr/bin/env python3
"""pe38 cycle 2: LEARN the type rules from the controls by massive random guessing, then test transfer.
 (a) 1,200 random rule systems (per role 1-3 random threshold rules on 12 type statistics, random weight,
     random type-checker weight lam_g in {0,1,3}, DM clustering on/off). Each is run (2 chains, 50k updates)
     on a proto-cuneiform sample at PE size (PC-A) and scored by macro recall of the known roles.
     Search null: the same runs scored against 20 label permutations (labels re-dealt among labelled types);
     each permutation's own top system is the null 'winner'.
 (b) Held-out: top 20 true-label systems, the 20 null winners and 20 median-ranked systems re-run on a
     disjoint PC sample (PC-B, other tablets) and on Ur III at PE size (other script, other words).
 (c) Supervised emission: per-role occurrence-feature profiles learned on PC-A labels (naive Bayes) used as
     the cost table, with and without the type-checker; tested on PC-B and Ur III; and Ur III -> PC.
 (d) The best transferred systems applied to PE (reported only if (b) beats its nulls).
Output: data/pe38_ckpt/c2.json
"""
import collections, json, math, os, random, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe38_common as C

NH = int(os.environ.get('NH', 1200))
STEPS = int(os.environ.get('STEPS', 50_000))
STATS = ['p_num', 'p_hdr', 'p_un', 'p_tot', 'p_cnt', 'p_cap', 'p_big', 'p_multi', 'p_alone', 'p_fin', 'p_init', 'ldocs']
NPERM = 20
_G = {}


def corpora():
    if _G:
        return _G
    PE = C.pe_docs(); n = C.ntok(PE)
    PC = C.pc_docs()
    rng = random.Random(C.seed('pe38c2-split'))
    idx = list(range(len(PC))); rng.shuffle(idx)
    half = len(idx) // 2
    A = C.sample_size([PC[i] for i in idx[:half]], n, random.Random(1))
    Bd = C.sample_size([PC[i] for i in idx[half:]], n, random.Random(2))
    U = C.sample_size(C.ur3_docs_all(), n, random.Random(C.seed('pe38c2-ur')))
    for name, docs in (('PCA', A), ('PCB', Bd), ('UR', U), ('PE', PE)):
        B = C.build(docs)
        for s in B['stats']:
            s['ldocs'] = min(1.0, math.log(s['docs']) / math.log(100))
        _G[name] = B
    return _G


def gen_hyp(rng):
    h = {'rules': {}, 'w': {}, 'lam_g': rng.choice([0.0, 1.0, 3.0]), 'alpha': rng.choice([0.0, 0.5])}
    for r in C.ROLES:
        h['rules'][r] = [(rng.choice(STATS), rng.choice(['<', '>']), round(rng.random(), 2)) for _ in range(rng.randint(1, 3))]
        h['w'][r] = round(rng.uniform(0.5, 4.0), 2)
    return h


def hyp_cost(B, h):
    Cm = np.zeros((len(B['types']), C.R))
    for t, s in enumerate(B['stats']):
        f = math.sqrt(s['n'])
        for r, rules in h['rules'].items():
            br = sum((s[st] < th) if op == '>' else (s[st] > th) for st, op, th in rules)   # rule 'stat > th' broken if below
            Cm[t, C.RI[r]] = h['w'][r] * f * br
    return Cm


def macro(B, mode, lab):
    rec = []
    for r in sorted(set(lab.values())):
        ii = [t for t, w in enumerate(B['types']) if lab.get(w) == r and B['nocc'][t] >= 3]
        if ii:
            rec.append(sum(C.ROLES[mode[t]] == r for t in ii) / len(ii))
    return float(np.mean(rec)) if rec else float('nan')


def run_h(B, h, seed, cost=None, nch=2, steps=STEPS):
    cost = hyp_cost(B, h) if cost is None else cost
    S, _ = C.run_chains(B, cost, nchains=nch, steps=steps, thin=1000, seed0=seed, alpha=h['alpha'], lam_g=h['lam_g'])
    return C.summarize(S)[0]


def perms():
    rng = random.Random(C.seed('pe38c2-perm'))
    keys = sorted(C.PC_LAB); out = []
    for _ in range(NPERM):
        v = [C.PC_LAB[k] for k in keys]; rng.shuffle(v); out.append(dict(zip(keys, v)))
    return out


def job_a(i):
    G = corpora()
    h = gen_hyp(random.Random(C.seed('pe38c2-h%d' % i)))
    mode = run_h(G['PCA'], h, 100 + i)
    return i, macro(G['PCA'], mode, C.PC_LAB), [macro(G['PCA'], mode, p) for p in PERMS]


def job_b(args):
    i, tag = args
    G = corpora()
    h = gen_hyp(random.Random(C.seed('pe38c2-h%d' % i)))
    out = {'i': i, 'tag': tag}
    for name, lab in (('PCB', C.PC_LAB), ('UR', C.UR_LAB)):
        mode = run_h(G[name], h, 500 + i, nch=3, steps=100_000)
        out[name] = macro(G[name], mode, lab)
    return out


def nb_cost(Bsrc, lab, Bdst, scale=1.0):
    """Naive-Bayes role profiles over the 8 occurrence features learned on labelled source types."""
    cnt = np.ones((C.R, C.NF, 16)) * 0.5
    of = Bsrc['ofeat'].reshape(-1, C.NF)
    for o, t in enumerate(Bsrc['otype']):
        r = lab.get(Bsrc['types'][t])
        if r:
            for k in range(C.NF):
                cnt[C.RI[r], k, of[o, k]] += 1
    lp = np.log(cnt / cnt.sum(2, keepdims=True))
    have = {C.RI[r] for r in lab.values()}
    od = Bdst['ofeat'].reshape(-1, C.NF)
    Cm = np.zeros((len(Bdst['types']), C.R))
    for o, t in enumerate(Bdst['otype']):
        for r in range(C.R):
            Cm[t, r] -= sum(lp[r, k, od[o, k]] for k in range(C.NF))
    Cm = Cm * scale
    for r in range(C.R):
        if r not in have:
            Cm[:, r] += 1e3   # roles with no labelled example cannot be learned
    return Cm


def job_c(args):
    src, dst, lam = args
    G = corpora()
    lab_s = C.PC_LAB if src.startswith('PC') else C.UR_LAB
    lab_d = C.PC_LAB if dst.startswith('PC') else C.UR_LAB
    Cm = nb_cost(G[src], lab_s, G[dst], scale=0.3)
    h = {'lam_g': lam, 'alpha': 0.0}
    mode_argmax = Cm.argmin(1)
    mode = run_h(G[dst], h, 900, cost=Cm, nch=3, steps=100_000)
    # null: source labels permuted among labelled types
    nulls = []
    keys = sorted(lab_s)
    for p in range(10):
        rng = random.Random(C.seed('pe38c2-nb%d' % p)); v = [lab_s[k] for k in keys]; rng.shuffle(v)
        Cp = nb_cost(G[src], dict(zip(keys, v)), G[dst], scale=0.3)
        nulls.append(macro(G[dst], Cp.argmin(1), lab_d))
    return dict(src=src, dst=dst, lam=lam, argmax=macro(G[dst], mode_argmax, lab_d), gibbs=macro(G[dst], mode, lab_d),
                null_argmax=nulls)


PERMS = perms()


def main():
    t = time.time()
    res = {}
    with Pool(2) as p:
        A = p.map(job_a, range(NH), chunksize=4)
        res['a'] = A
        print('a done %.0fs' % (time.time() - t), flush=True)
        tr = sorted(A, key=lambda x: -x[1])
        top = [x[0] for x in tr[:20]]
        med = [x[0] for x in tr[len(tr) // 2 - 10: len(tr) // 2 + 10]]
        nullw = []
        for j in range(NPERM):
            nullw.append(max(A, key=lambda x: x[2][j])[0])
        jobs = [(i, 'top') for i in top] + [(i, 'null') for i in nullw] + [(i, 'med') for i in med]
        res['b'] = p.map(job_b, jobs, chunksize=1)
        print('b done %.0fs' % (time.time() - t), flush=True)
        res['c'] = p.map(job_c, [('PCA', 'PCB', 0.0), ('PCA', 'PCB', 2.0), ('PCA', 'UR', 0.0), ('PCA', 'UR', 2.0),
                                  ('UR', 'PCB', 0.0), ('UR', 'PCB', 2.0)], chunksize=1)
        print('c done %.0fs' % (time.time() - t), flush=True)
    json.dump(res, open(os.path.join(C.CK, 'c2.json'), 'w'))


if __name__ == '__main__':
    main()
