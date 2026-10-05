"""v64 cycle 2: padding re-modelled (gap strings spelled glyph by glyph, so an alphabet
cannot hide words in memorised padding; the categorical gap model of cycle 1 let the
climber collapse coverage). Fresh random search (10,000 alphabets per corpus, spelled
gaps, ranked by sel Gfree); the survivors are hill-climbed on the selection split
and then re-tested on the held split; the climbed alphabet's section profile is
compared with random alphabets of the same size (topic test).

Hill climb (objective OBJ = Gfree, the Lullian free-combination model; cycle-1
calibration: true alphabets give Gfree ars +0.18, med -0.56, Latin -3.68):
start from the 6 best random alphabets (by sel OBJ) plus the 20 commonest single
glyphs; 20 rounds, each
proposing 60 neighbours (add / drop / swap one n-gram from the 200-n-gram pool,
extend or trim one concept by a glyph); keep the best if it improves sel OBJ.
Topic test: I(concept; section) - and the ratio to I(glyph; section) - for the
climbed alphabet vs 200 random alphabets of the same size from the same pool, on the
whole corpus; and a folio-level section-label permutation null (100 permutations).
"""
import sys, json, random, os, time
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v64_lib as V

OBJ = 'Gfree'
CORP = ['ars', 'med', 'voy', 'lat', 'voyit', 'voy_mk2', 'voy_gshuf', 'voy_sc', 'ars_mk2']


def neighbours(al, pool, rng, n):
    items = [g for g, _ in pool]
    out = []
    while len(out) < n:
        a = list(al); r = rng.random()
        if r < 0.3 and len(a) < 20:
            a.append(rng.choice(items))
        elif r < 0.5 and len(a) > 6:
            a.pop(rng.randrange(len(a)))
        elif r < 0.8:
            a[rng.randrange(len(a))] = rng.choice(items)
        else:
            i = rng.randrange(len(a)); c = a[i]
            if len(c) > 1 and rng.random() < 0.5:
                a[i] = c[1:] if rng.random() < 0.5 else c[:-1]
            else:
                ext = [g for g in items if len(g) == len(c) + 1 and (g.startswith(c) or g.endswith(c))]
                if ext:
                    a[i] = rng.choice(ext)
        a = sorted(set(a))
        if 6 <= len(a) <= 20 and a != sorted(al):
            out.append(a)
    return out


def climb(k):
    out = os.path.join(V.CK, 'c2_%s.json' % k)
    if os.path.exists(out):
        return k
    D = V.all_corpora(); C = D['corpora'][k]
    ps, ph = os.path.join(V.CK, 'c_%s_sel.txt' % k), os.path.join(V.CK, 'c_%s_held.txt' % k)
    tr, _ = V.type_counts(C, 'sel'); pool = V.ngram_pool(tr)
    rng = random.Random(sum(map(ord, k)))
    ral = [V.sample_alphabet(pool, rng) for _ in range(10000)]
    rs = V.score(ps, ral)
    order = sorted(range(len(ral)), key=lambda i: -rs[i][OBJ])
    rand_held = V.score(ph, [ral[i] for i in order[:20]])
    gl = Counter()
    for w, n in tr.items():
        for c in w:
            gl[c] += n
    starts = [ral[i] for i in order[:6]] + [sorted(g for g, _ in gl.most_common(20))]
    finals = []
    for al in starts:
        cur = sorted(al); cs = V.score(ps, [cur])[0]
        for rnd in range(20):
            nb = neighbours(cur, pool, rng, 60)
            r = V.score(ps, nb)
            b = max(range(len(nb)), key=lambda i: r[i][OBJ])
            if r[b][OBJ] > cs[OBJ]:
                cur, cs = nb[b], r[b]
        finals.append((cur, cs))
    held = V.score(ph, [a for a, _ in finals])
    best = max(range(len(finals)), key=lambda i: finals[i][1][OBJ])
    alpha = finals[best][0]
    # topic test on whole corpus
    parse = V.make_parser(alpha)
    Ic = V.section_info(C, parse, len(alpha)); Ig = V.section_info(C, parse, len(alpha), unit='glyph')
    rand = []
    for _ in range(200):
        ra = V.sample_alphabet(pool, rng, len(alpha), len(alpha))
        p2 = V.make_parser(ra)
        rand.append(V.section_info(C, p2, len(ra)))
    # folio-level label permutation
    fol_sec = {}
    for L in C:
        fol_sec.setdefault(L['fol'], L['sec'])
    fols = sorted(fol_sec); perm = []
    for _ in range(100):
        labs = [fol_sec[f] for f in fols]; rng.shuffle(labs); m = dict(zip(fols, labs))
        C2 = [dict(L, sec=m[L['fol']]) for L in C]
        perm.append(V.section_info(C2, parse, len(alpha)))
    # concept profiles per section (top concepts by log-ratio)
    from collections import Counter
    prof = {}
    tot = Counter(); bysec = {}
    for L in C:
        for w in L['words']:
            for u in parse(w):
                tot[u] += 1; bysec.setdefault(L['sec'], Counter())[u] += 1
    N = sum(tot.values())
    for s, c in bysec.items():
        n = sum(c.values())
        if n < 300:
            continue
        lr = {alpha[u]: round(((c[u] + .5) / n) / ((tot[u] + .5) / N), 2) for u in tot}
        prof[s] = sorted(lr.items(), key=lambda x: -x[1])[:4] + sorted(lr.items(), key=lambda x: x[1])[:2]
    res = {'rand_sel_best': rs[order[0]], 'rand_held_top20': rand_held,
           'rand_sel_q': [rs[order[q]][OBJ] for q in (0, 10, 100, 1000, 5000)],
           'finals': finals, 'held': held, 'best': best, 'alpha': alpha,
           'Ic': Ic, 'Ig': Ig, 'rand': rand, 'perm': perm, 'prof': prof}
    for name, codes in D['codes'].items():
        if k == name:
            res['jacc'] = V.jaccard(alpha, codes.values())
            res['jacc_start'] = [V.jaccard(a, codes.values()) for a in starts]
        res['truth_sel'] = V.score(ps, [sorted(codes.values())])[0]
        res['truth_held'] = V.score(ph, [sorted(codes.values())])[0]
        res['jacc_finals'] = [V.jaccard(a, codes.values()) for a, _ in finals]
    json.dump(res, open(out, 'w'))
    return k


if __name__ == '__main__':
    with Pool(2) as P:
        for k in P.imap_unordered(climb, CORP):
            print('done', k, time.strftime('%X'), flush=True)
