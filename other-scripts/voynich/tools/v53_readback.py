"""v53 cycle 3: READ-BACK evolution.  Same program space, but fitness is the
inverse: apply each program's decoder to the TARGET and score how much of the
decoded stream is real words of the program's plaintext language, minus the
same score after the decoded letters are shuffled across words (word lengths
kept).  Held-out: the other target half, plus word-bigram hits against a
word-order shuffle.  Controls: planted (DE_rel through a hidden program) and
Rugg-style table-and-grille text.

usage: python3 v53_readback.py TARGET POP GENS
"""
import sys, os, json, time, random, math
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v53_lib import *

TARGET, POP, GENS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
LOG = open(os.path.join(CK, f'rb_{TARGET}.log'), 'a')


def log(s):
    LOG.write(time.strftime('%H:%M:%S ') + s + '\n'); LOG.flush()


def lexicon(c, abbr):
    lang, genre, lines = CORP[c]
    ws = words_of(lines)
    keep = ws[:7919] + ws[10319:]
    if c == 'DE_rel':
        keep = ws[:7919] + ws[10319:12000] + ws[24000:]   # planted stretch excluded
    V = vowels_for(lang)
    aw = [abbr_word(x, abbr, V) for x in keep]
    return set(x for x in aw if len(x) >= 2), (Counter(aw), Counter(zip(aw, aw[1:])), len(aw))


def pmi(words, wbi):
    """mean pointwise MI of consecutive decoded words (both in the corpus vocabulary), corpus bigram counts, add-0.5"""
    uni, bi, N = wbi
    tot = 0.0; n = 0
    for a, b in zip(words, words[1:]):
        if a and b and a in uni and b in uni:
            pab = (bi.get((a, b), 0) + 0.5) / (N + 0.5 * len(uni))
            tot += math.log2(pab / (uni[a] / N * uni[b] / N)); n += 1
    return tot / max(1, n)


LEX = {}


def decoded(m, lines):
    """-> list of unit lists (None = unparsed) in reading order, coverage"""
    dec, cov = decode_lines(m, lines)
    return [None if u is None else unreorder(list(u), m['p']['reord']) for u in dec], cov


def _hits(units_words, lex):
    multi = [u for u in units_words if u and len(u) >= 2]
    return sum(''.join(u) in lex for u in multi), len(multi)


def rb_score(p, env, lines, rng_seed=0, bigram=False):
    """score = coverage x (share of multi-unit decoded words that are real words
    minus the same share after the decoded units are shuffled among multi-unit words, unit counts kept)"""
    m = build(p, env)
    key = (p['corpus'], p['abbr'])
    if key not in LEX:
        LEX[key] = lexicon(*key)
    lex, wbi = LEX[key]
    uw, cov = decoded(m, lines)
    h, n = _hits(uw, lex)
    if n < 200:
        return {'score': -1.0, 'hit': 0, 'null': 0, 'cov': cov, 'n': n}
    hit = h / n
    rng = random.Random(rng_seed)
    multi = [u for u in uw if u and len(u) >= 2]
    # null: units shuffled among multi-unit words WITHIN position class (first / middle / last unit), so each
    # slot keeps its unit distribution and word lengths are kept; only the combination of units is randomised
    pools = {'I': [u[0] for u in multi], 'F': [u[-1] for u in multi], 'M': [x for u in multi for x in u[1:-1]]}

    def shuf_hit():
        for v in pools.values():
            rng.shuffle(v)
        iI, iF, iM = iter(pools['I']), iter(pools['F']), iter(pools['M'])
        hh = 0
        for u in multi:
            w = next(iI) + ''.join(next(iM) for _ in u[1:-1]) + next(iF)
            hh += w in lex
        return hh / n
    nulls = [shuf_hit() for _ in range(3)]
    null = sum(nulls) / len(nulls)
    words = [None if u is None else ''.join(u) for u in uw]
    r = {'score': (hit - null) * cov, 'hit': hit, 'null': null, 'cov': cov, 'n': n}
    if bigram:
        r['bihit'] = pmi(words, wbi)
        allw = [x for l in lines for x in l]
        bs = []
        for k in range(10):
            rng.shuffle(allw)
            it = iter(allw)
            Lw = [[next(it) for _ in l] for l in lines]
            w2, _ = decoded(m, Lw)
            bs.append(pmi([None if u is None else ''.join(u) for u in w2], wbi))
        mu = sum(bs) / len(bs); sd = (sum((x - mu) ** 2 for x in bs) / 9) ** 0.5 or 1e-6
        r['bihit_null'] = mu; r['z_bihit'] = (r['bihit'] - mu) / sd
        zs = [shuf_hit() for _ in range(20)]
        mu = sum(zs) / len(zs); sd = (sum((x - mu) ** 2 for x in zs) / 19) ** 0.5 or 1e-6
        r['z_hit'] = (hit - mu) / sd
        r['sample'] = ' '.join(w or '?' for w in words[:30])
    return r


def sub_lines(lines, ntok):
    out, n = [], 0
    for l in lines:
        out.append(l); n += len(l)
        if n >= ntok:
            break
    return out


def job(corpus):
    A, B = TGT
    env = Env(A, CORP)
    envB = Env(B, CORP, offset=7919)
    As = sub_lines(A, 3000)
    rng = random.Random(sum(map(ord, corpus)) + 17)
    pop = [random_program(rng, corpus=corpus) for _ in range(POP)]
    rec = []
    for gen in range(GENS):
        sc = []
        for p in pop:
            try:
                r = rb_score(p, env, As)
            except Exception as e:
                r = {'score': -1.0}
            sc.append((-r['score'], p, r))
        sc.sort(key=lambda x: x[0])
        rec.extend(sc)
        b = sc[0]
        log(f'[{corpus}] gen {gen} best {-b[0]:.4f} hit {b[2].get("hit", 0):.3f} null {b[2].get("null", 0):.3f} cov {b[2].get("cov", 0):.2f} {motif_key(b[1])}')
        ne = max(2, POP // 10)
        new = [json.loads(json.dumps(s[1])) for s in sc[:ne]]
        while len(new) < POP:
            t = lambda: min(rng.sample(sc[:POP // 2], 3), key=lambda x: x[0])[1]
            ch = crossover(t(), t(), rng) if rng.random() < 0.3 else t()
            ch = mutate(ch, rng)
            if rng.random() < 0.05:
                ch = random_program(rng, corpus=corpus)
            new.append(ch)
        pop = new
    rec.sort(key=lambda x: x[0])
    seen, top = set(), []
    for f, p, r in rec:
        k = json.dumps(clean(p), sort_keys=True)
        if k in seen:
            continue
        seen.add(k); top.append((p, r))
        if len(top) >= 5:
            break
    out = []
    for p, r in top:
        try:
            hb = rb_score(p, envB, B, rng_seed=5, bigram=True)
        except Exception as e:
            hb = {'score': -1.0, 'err': str(e)[:80]}
        out.append({'prog': clean(p), 'search': r, 'heldout': hb})
    # null for the search itself: best score among the first-generation random programs
    rand0 = sorted(-x[0] for x in rec[:0] or [])
    return {'corpus': corpus, 'top': out, 'n_eval': len(rec)}


def init():
    global TGT, CORP
    CORP = {k: tuple(v) for k, v in load_corpora().items()}
    sys.argv = ['v53_run.py', TARGET, '1', '1', '1']
    import importlib.util
    spec = importlib.util.spec_from_file_location('v53run', os.path.join(HERE, 'v53_run.py'))
    RUN = importlib.util.module_from_spec(spec); spec.loader.exec_module(RUN)
    RUN.TARGET = TARGET
    TGT = RUN.make_targets()


if __name__ == '__main__':
    t0 = time.time()
    init()
    log(f'start readback {TARGET} pop {POP} gens {GENS}')
    with Pool(2, initializer=init) as pool:
        res = pool.map(job, sorted(CORP), chunksize=1)
    json.dump({'target': TARGET, 'results': res}, open(os.path.join(CK, f'rb_{TARGET}.json'), 'w'))
    for r in res:
        t = r['top'][0]
        h = t['heldout']
        log(f"{r['corpus']:9s} search {t['search']['score']:.4f} heldout score {h.get('score', 0):.4f} hit {h.get('hit', 0):.3f} null {h.get('null', 0):.3f} z_hit {h.get('z_hit', 0):+.1f} bihit {h.get('bihit', 0):.4f} null {h.get('bihit_null', 0):.4f} z {h.get('z_bihit', 0):+.1f}")
    log(f'done {time.time() - t0:.0f}s')
