"""X-4 cycle 2: lexical gaps in the stem x ending grid (Voynich v33 adapted to administrative lists).

For each corpus (same word budget, whole documents, words of >= 2 signs), 31 corpus-relative segmentation
rules (fix-1/2/3, pos 0.5-0.8, freq 10/25/50/100, harris, 20 random suffix sets) give a presence matrix of the
R commonest stems x E commonest endings. gap ratio = connectance(text) / mean connectance of 3 resyntheses by
the text's own sign-trigram chain (v33: languages 0.45-0.66, Voynich 0.95-1.10). Also vs its own independent-
word (SLOT) generator.
Controls (each through the same pipeline, ratio against THEIR own trigram resynthesis):
  TRI, SLOT, CPV  fitted generators (generator floor: should sit near 1)
  WBG             word-bigram resynthesis (keeps the real lexicon: should look like the text)
  pair            endings re-paired to stems at random over tokens (kills paradigm gaps, keeps marginals)
Budgets 3000 (R 100, E 15) and 8000 (R 200, E 25); seeds 0, 1.
"""
import os, sys, json, random, time, zlib
from collections import Counter
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x4_lib as X
import v33_lib as V

OUT = os.path.join(X.CK, 'c2'); os.makedirs(OUT, exist_ok=True)
RE = {3000: (100, 15), 8000: (200, 25)}


def toks_of(docs):
    return [w for d in docs for l in d for w in l if len(w) >= 2]


def conn(toks, rule, R, E, rng=None):
    M = V.matrix(toks, rule, R, E, pairing_shuffle=rng)
    return float(M.mean()) if M.size else float('nan'), M.shape


def tri_toks(toks, n, seed):
    T = V.Trigram([toks])
    out = []
    rng = random.Random(seed)
    while len(out) < n:
        out += [w for w in T.gen(n, rng) if len(w) >= 2]
    return out[:n]


def ratios(toks, R, E, seed, rules):
    res = {}
    base = [tri_toks(toks, len(toks), seed * 10 + i) for i in range(3)]
    sg = V.SlotGen([toks]); r = random.Random(seed + 99)
    slot = []
    while len(slot) < len(toks):
        w = sg.word(r)
        if len(w) >= 2: slot.append(w)
    for rule in rules:
        rule.fit(toks)
        c, sh = conn(toks, rule, R, E)
        cb = [conn(b, rule.fit(b), R, E)[0] for b in base]
        rule.fit(slot); cs = conn(slot, rule, R, E)[0]
        rule.fit(toks)
        res[rule.name] = dict(conn=c, shape=list(sh), tri=float(np.mean(cb)), slot=cs,
                              g_tri=c / np.mean(cb) if np.mean(cb) else float('nan'),
                              g_slot=c / cs if cs else float('nan'))
    return res


def run(job):
    name, budget, seed, cond = job
    fn = os.path.join(OUT, f'{name}_{budget}_{seed}_{cond}.json')
    if os.path.exists(fn): return job
    t0 = time.time()
    base = X.subsample(X.corpora()[name], budget, seed)
    rng = random.Random(zlib.crc32(f'{name}{budget}{seed}{cond}'.encode()))
    R, E = RE[budget]
    rules = V.rule_set(n_rand=20)
    if cond in X.GENS:
        toks = toks_of(X.GENS[cond](base, rng))
    else:
        toks = toks_of(base)
    if cond == 'pair':
        # re-pair at the token level under each rule: done inside by pairing_shuffle on the text itself
        res = {}
        for rule in rules:
            rule.fit(toks)
            c = V.matrix(toks, rule, R, E, pairing_shuffle=random.Random(seed))
            res[rule.name] = dict(conn=float(c.mean()))
        out = {'name': name, 'budget': budget, 'seed': seed, 'cond': cond, 'n': len(toks), 'rules': res}
    else:
        out = {'name': name, 'budget': budget, 'seed': seed, 'cond': cond, 'n': len(toks),
               'rules': ratios(toks, R, E, seed, rules)}
    out['sec'] = time.time() - t0
    json.dump(out, open(fn, 'w'), default=float)
    g = [v.get('g_tri') for v in out['rules'].values() if v.get('g_tri') == v.get('g_tri') and v.get('g_tri')]
    print(name, budget, seed, cond, len(toks), round(float(np.median(g)), 3) if g else '-', round(out['sec']), flush=True)
    return job


def jobs():
    C = X.corpora(); J = []
    for budget in (3000, 8000):
        for name in X.LIST + ['VOY'] + X.PROSE:
            if X.ntok(C[name]) < budget * 0.95: continue
            for seed in (0, 1):
                for cond in ['real', 'pair', 'TRI', 'SLOT', 'CPV', 'WBG']:
                    J.append((name, budget, seed, cond))
    return J


if __name__ == '__main__':
    J = jobs(); print(len(J), 'jobs', flush=True)
    with Pool(int(os.environ.get('W', '2'))) as pool:
        for _ in pool.imap_unordered(run, J): pass
