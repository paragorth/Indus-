"""v53 driver: evolve writing machines against one target.

usage: python3 v53_run.py TARGET POP GENS SEEDS
TARGET: voynich | it2a | planted | rugg | currA | currB
Writes data/v53_ckpt/run_<TARGET>.json (top programs per corpus, held-out rescoring).
"""
import sys, os, json, time, random
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v53_lib import *

TARGET, POP, GENS, SEEDS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
LOG = open(os.path.join(CK, f'run_{TARGET}.log'), 'a')


def log(s):
    LOG.write(time.strftime('%H:%M:%S ') + s + '\n'); LOG.flush()


HIDDEN = {'corpus': 'DE_rel', 'seg': 'syll', 'abbr': 'none', 'reord': 'none', 'wording': 'keep',
          'table': 'pos', 'bpe': 60, 'jit': 1, 'width': 40,
          'ops': [['PFX', 2, 'prev', 0, 3], ['LINEM', 1, 'const', 0, 3]], 'seed': 4242}


def make_targets():
    """-> (A lines, B lines) of glyph-string words for the chosen target."""
    if TARGET in ('voynich', 'currA', 'currB'):
        V = load_voynich('ZL3b')
        if TARGET != 'voynich':
            V = [l for l in V if l['lang'] == TARGET[-1]]
        return split_halves(V)
    if TARGET == 'it2a':
        return split_halves(load_voynich('IT2a'))
    if TARGET == 'planted':
        # the hidden program writes a held-out stretch of IT_med with Voynich pieces
        C = load_corpora()
        V = load_voynich('ZL3b'); A, _ = split_halves(V)
        lang, genre, lines = C['DE_rel']
        ws = words_of(lines)
        # words 12000-24000: disjoint from the search sample (0-2400) and the held-out sample (7919-10319)
        sub = {'DE_rel': (lang, genre, [ws[i:i + 9] for i in range(12000, 24000, 9)])}
        env = Env(A, sub, n_plain=12000)
        m = build(HIDDEN, env)
        out, pay, tot, gid = encode(m, max_tokens=None)
        json.dump({'hidden': HIDDEN, 'payload_share': pay / tot, 'lines': out}, open(os.path.join(CK, 'planted_truth.json'), 'w'))
        h = len(out) // 2
        return out[:h], out[h:]
    if TARGET == 'rugg':
        V = load_voynich('ZL3b'); A, _ = split_halves(V)
        tw = words_of(A)
        merges, vocab = learn_bpe(tw, 60)
        pc = piece_classes(tw, vocab)
        rng = random.Random(7)
        rows, ntab = 80, 3
        tabs = [[[(rng.choice(pc['I'][:30]) if rng.random() < 0.7 else ''),
                  (rng.choice(pc['M'][:30]) if rng.random() < 0.5 else ''),
                  rng.choice(pc['F'][:25])] for _ in range(rows)] for _ in range(ntab)]
        grilles = [(rng.randrange(1, rows), rng.randrange(1, rows)) for _ in range(12)]
        out = []
        for li in range(1200):
            tab = tabs[li % ntab]
            l = []
            for wi in range(rng.randint(6, 11)):
                r = rng.randrange(rows); g1, g2 = rng.choice(grilles)
                w = tab[r][0] + tab[(r + g1) % rows][1] + tab[(r + g2) % rows][2]
                l.append(w)
            out.append(l)
        return out[:600], out[600:]
    raise ValueError(TARGET)


def job(args):
    corpus, seed = args
    A, B = TGT
    T = target_profile(A)
    env = Env(A, CORP)
    rec = run_ga(env, T, list(CORP), POP, GENS, seed * 1000 + sum(map(ord, corpus)), log=lambda s: log(f'[{corpus} s{seed}] ' + s), fixed_corpus=corpus)
    rec.sort(key=lambda x: x[0])
    seen, top = set(), []
    for f, p, r, g in rec:
        k = json.dumps(clean(p), sort_keys=True)
        if k in seen:
            continue
        seen.add(k); top.append((f, clean(p), r, g))
        if len(top) >= 40:
            break
    # held-out: other half of the target, fresh plaintext stretch
    TB = target_profile(B)
    envB = Env(B, CORP, offset=7919)
    ho = []
    for f, p, r, g in top:
        try:
            rb = evaluate(p, envB, TB)
        except Exception as e:
            rb = {'fit': 99.0, 'd': 99.0}
        ho.append(rb)
    # random-program baseline in this corpus (same envs)
    rng = random.Random(seed + 99)
    base = []
    for _ in range(200):
        p = random_program(rng, corpus=corpus)
        try:
            base.append(evaluate(p, env, T)['fit'])
        except Exception:
            base.append(99.0)
    return {'corpus': corpus, 'seed': seed, 'n_eval': len(rec),
            'top': [{'fit': f, 'prog': p, 'res': r, 'gen': g, 'heldout': h} for (f, p, r, g), h in zip(top, ho)],
            'baseline_fits': base}


def init():
    global TGT, CORP
    CORP = {k: tuple(v) for k, v in load_corpora().items()}
    TGT = make_targets()


if __name__ == '__main__':
    t0 = time.time()
    init()
    log(f'start {TARGET} pop {POP} gens {GENS} seeds {SEEDS}; target A {sum(map(len, TGT[0]))} tokens, B {sum(map(len, TGT[1]))}')
    jobs = [(c, s) for s in range(SEEDS) for c in sorted(CORP)]
    with Pool(2, initializer=init) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump({'target': TARGET, 'pop': POP, 'gens': GENS, 'results': res}, open(os.path.join(CK, f'run_{TARGET}.json'), 'w'))
    log(f'done {time.time() - t0:.0f}s')
