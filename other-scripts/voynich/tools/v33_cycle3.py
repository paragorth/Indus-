"""v33 cycle 3: every text against its OWN generators (null-free gap ratio), confound and planted controls,
fresh blocks (seed 34) and 100 new random rules (held out from cycles 1-2).

gap ratio  = connectance(text) / connectance(own glyph-trigram resynthesis), per rule; also vs own slot generator.
           < 1: the text's stem x ending grid has lexical gaps its glyph statistics would fill.
Q ratio    = raw Barber Q(text) / Q(own trigram).
Sources: Voynich ZL, IT, A, B; 7 languages; confounds: Latin in 8 letter classes (small alphabet), Latin
suspended to first 2 + last 2 letters (short words, medieval abbreviation), Hebrew (abjad, short words, in panel);
planted: Voynich trigram text with DECLENSION CLASSES (each pos-0.5 stem gets 1 of 4 classes, each class allows a
random half of the endings; disallowed words rejected) at 100% and 50% of tokens -> must give ratio < 1.
Usage: python3 v33_cycle3.py [nworkers]"""
import sys, time, zlib
from multiprocessing import Pool
from v33_lib import *

RULES = rule_set(20) + [Rule('rand', 0, seed=s) for s in range(100, 200)]
CLS = {}
for i, g in enumerate(['aeiouy', 'bp', 'cgkq', 'dt', 'fv', 'lr', 'mn', 'hsxz']):
    for c in g: CLS[c] = 'abcdefgh'[i]


def declension_text(lines, frac, seed):
    rng = random.Random(seed)
    tg = Trigram(lines)
    base = tg.gen(60000, rng)
    r = Rule('pos', 0.5).fit(base)
    ends = sorted(set(r.split(w)[1] for w in base))
    allow = {}
    out = []
    for w in base:
        s, e = r.split(w)
        if rng.random() < frac:
            if s not in allow:
                k = rng.randrange(4)
                allow[s] = k
            key = (allow[s], e)
            if key not in CLS_ALLOWED.setdefault(seed, {}):
                CLS_ALLOWED[seed][key] = rng.random() < 0.5
            if not CLS_ALLOWED[seed][key]: continue
        out.append(w)
    return [out[i:i + 10] for i in range(0, len(out), 10)]


CLS_ALLOWED = {}


def sources():
    S = {'V-ZL': voy_lines('ZL3b'), 'V-IT': voy_lines('IT2a'), 'V-A': voy_lines('ZL3b', 'A'), 'V-B': voy_lines('ZL3b', 'B')}
    for c in ('la', 'cs', 'de', 'it', 'hu', 'tr', 'he'): S['L-' + c] = lang_lines(c)
    la = S['L-la']
    S['C-la8cls'] = [[''.join(CLS.get(c, 'h') for c in w) for w in l] for l in la]
    S['C-laSusp'] = [[w if len(w) <= 4 else w[:2] + w[-2:] for w in l] for l in la]
    S['P-decl100'] = declension_text(S['V-ZL'], 1.0, 1)
    S['P-decl50'] = declension_text(S['V-ZL'], 0.5, 2)
    return S


def job(args):
    name, lines = args
    out = load(f'c3_{name}.json')
    if out: return name, 'cached'
    res = {}
    tri, slot = Trigram(lines), SlotGen(lines)
    for b, toks in enumerate(blocks(lines, seed=34)):
        rng = random.Random(1000 + b)
        texts = {'X': toks, 'T': tri.gen(NTOK, rng), 'S': slot.gen(NTOK, rng)}
        for r in RULES:
            for k, t in texts.items():
                rr = Rule(r.kind, r.param, r.seed).fit(t)
                M = matrix(t, rr)
                q = barber_Q(M, np.random.default_rng(zlib.crc32(r.name.encode())))
                res[f'{b}|{r.name}|{k}'] = [float(M.mean()), float(q), nodf(M), cscore(M), int(M.shape[0]), int(M.shape[1])]
    save(f'c3_{name}.json', res)
    return name, 'done'


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    t0 = time.time()
    S = sources()
    print('sources', {k: sum(map(len, v)) for k, v in S.items()}, flush=True)
    with Pool(nw) as P:
        for n, s in P.imap_unordered(job, sorted(S.items())):
            print(n, s, round(time.time() - t0), flush=True)
