#!/usr/bin/env python3
"""LA-5 cycle 2: usage battery on administrative word TOKENS and running lines.
(a) uniqueness, hapax share, Heaps exponent, unique share vs a sign-bigram regeneration (LA-2d / S312 rule)
(b) reuse across sites: share of repeated types (>= 2 tokens) seen at >= 2 sites, against site labels permuted over tokens
(c) frame / content information split on lines: held-out unigram bits carried by word tokens vs frame tokens
    (logograms, numbers), and the bits a word saves on the NEXT frame logogram (product-code signature)
(d) word-internal vs boundary predictability: held-out bits saved by the previous sign inside words (beyond position class)
    and across a word boundary (last sign of word -> first sign of the next word on the same line)
LA = all admin docs. All populations drawn by documents, without replacement, to half the LA token count (40 draws); LB also KN-only.
PE = Proto-Elamite entry middles drawn to the same token n (usage items only).
Planted relabel code: every LA type replaced by a random ID; (a) and (b) are invariant to relabelling by construction.
Output ../data/la5_c2.json, ../data/la5_c2.txt
"""
import sys, os, json, random, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la5_common as C

ND = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rnd = random.Random(52)
LA = C.la_docs(); LB = C.lb_docs()
LBKN = [d for d in LB if d['site'] == 'KN']
NLA = len(C.words_of(LA))

def draw_docs(docs, n, r):
    idx = list(range(len(docs))); r.shuffle(idx); out = []; k = 0
    for i in idx:
        w = sum(1 for L in docs[i]['lines'] for t in L if t[0] == 'W' and len(t[1]) >= 2)
        if not w: continue
        out.append(docs[i]); k += w
        if k >= n: break
    return out

def usage(tokens, sites, r):
    cnt = collections.Counter(tokens); n = len(tokens)
    f1 = sum(1 for v in cnt.values() if v == 1)
    # bigram regeneration of the token list (one model over all tokens, same lengths)
    b = collections.defaultdict(collections.Counter)
    for t in tokens:
        p = '^'
        for a in t: b[p][a] += 1; p = a
    gen = []
    for t in tokens:
        p = '^'; s = []
        for _ in t:
            src = b[p] or b['^']; ks, ws = zip(*src.items()); a = r.choices(ks, ws)[0]; s.append(a); p = a
        gen.append(tuple(s))
    cg = collections.Counter(gen); uq = sum(1 for t in tokens if cnt[t] == 1) / n; ug = sum(1 for t in gen if cg[t] == 1) / n
    # cross-site reuse
    def reuse(st):
        by = collections.defaultdict(set)
        for t, s in zip(tokens, st): by[t].add(s)
        rep = [t for t, v in cnt.items() if v >= 2]
        return sum(1 for t in rep if len(by[t]) >= 2) / len(rep) if rep else float('nan')
    ro = reuse(sites); st = list(sites); nl = []
    for _ in range(20): r.shuffle(st); nl.append(reuse(st))
    rn = sum(nl) / len(nl)
    return dict(uniq=len(cnt) / n, hapax=f1 / n, heaps=C.heaps(tokens, r), uq_ratio=uq / ug if ug else float('nan'),
                reuse=ro, reuse_null=rn, reuse_ratio=ro / rn if rn else float('nan'))

def line_stats(docs, r):
    """(c) and (d) on running lines; docs split in halves for held-out scoring."""
    idx = list(range(len(docs))); r.shuffle(idx); halves = [[docs[i] for i in idx[::2]], [docs[i] for i in idx[1::2]]]
    bitsW = bitsF = 0.0; gains_next = []; g_int = []; g_bnd = []; nWW = 0
    for fit, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        uni = collections.Counter(); wl = collections.defaultdict(collections.Counter); lu = collections.Counter()
        c0 = collections.defaultdict(collections.Counter); c1 = collections.defaultdict(collections.Counter)
        bi = collections.Counter(); b1 = collections.defaultdict(collections.Counter)
        def toks(d):
            for L in d['lines']:
                yield [t for t in L if t[0] in ('W', 'F', 'B')]
        for d in fit:
            for L in toks(d):
                for j, t in enumerate(L):
                    if t[0] == 'B': continue
                    key = t[1] if t[0] == 'F' else ('W',) + t[1]; uni[key] += 1
                    if t[0] == 'W':
                        w = t[1]
                        for i, a in enumerate(w):
                            cl = C.pclass(i, len(w)) if len(w) > 1 else 'I'; c0[cl][a] += 1
                            if i: c1[(cl, w[i - 1])][a] += 1
                        nxt = next((u for u in L[j + 1:] if u[0] != 'W' or True), None)
                        if j + 1 < len(L) and L[j + 1][0] == 'F' and L[j + 1][1].startswith('L:'): wl[w][L[j + 1][1]] += 1
                        if j + 1 < len(L) and L[j + 1][0] == 'W':
                            bi[L[j + 1][1][0]] += 1; b1[w[-1]][L[j + 1][1][0]] += 1
                    elif t[1].startswith('L:'): lu[t[1]] += 1
        V = len(uni) + 1; N = sum(uni.values())
        Vs = len(set(a for c in c0.values() for a in c)) + 1; VL = len(lu) + 1; NL = sum(lu.values())
        for d in test:
            for L in toks(d):
                for j, t in enumerate(L):
                    if t[0] == 'B': continue
                    key = t[1] if t[0] == 'F' else ('W',) + t[1]
                    bits = -math.log2((uni[key] + 0.5) / (N + 0.5 * V))
                    if t[0] == 'W': bitsW += bits
                    else: bitsF += bits
                    if t[0] == 'W':
                        w = t[1]
                        for i in range(1, len(w)):
                            cl = C.pclass(i, len(w)); a = w[i]; N0 = sum(c0[cl].values())
                            p0 = (c0[cl][a] + 0.5) / (N0 + 0.5 * Vs); cc = c1.get((cl, w[i - 1])); n1 = sum(cc.values()) if cc else 0
                            g_int.append(math.log2(((cc[a] if cc else 0) + 5 * p0) / (n1 + 5) / p0))
                        if j + 1 < len(L) and L[j + 1][0] == 'F' and L[j + 1][1].startswith('L:'):
                            y = L[j + 1][1]; p0 = (lu[y] + 0.5) / (NL + 0.5 * VL); cc = wl.get(w); n1 = sum(cc.values()) if cc else 0
                            gains_next.append(math.log2(((cc[y] if cc else 0) + 2 * p0) / (n1 + 2) / p0))
                        if j + 1 < len(L) and L[j + 1][0] == 'W':
                            nWW += 1; a = L[j + 1][1][0]; NB = sum(bi.values())
                            p0 = (bi[a] + 0.5) / (NB + 0.5 * Vs); cc = b1.get(w[-1]); n1 = sum(cc.values()) if cc else 0
                            g_bnd.append(math.log2(((cc[a] if cc else 0) + 5 * p0) / (n1 + 5) / p0))
    m = lambda v: sum(v) / len(v) if v else float('nan')
    return dict(word_bits_share=bitsW / (bitsW + bitsF), word_to_logo_gain=m(gains_next), n_word_logo=len(gains_next),
                gain_internal=m(g_int), gain_boundary=m(g_bnd), n_boundary=nWW, int_minus_bnd=m(g_int) - m(g_bnd))

def toks_sites(docs):
    w = C.words_of(docs); return [x[2] for x in w], [x[1] for x in w]

res = {}; lines = [f'LA-5 cycle 2: LA admin word tokens (>= 2 signs) n = {NLA}; every population drawn by documents (without replacement) to n/2 = {NLA // 2} tokens, {ND} draws (half-sample spread = full-sample bootstrap spread)']
def run(name, maker):
    acc = collections.defaultdict(list)
    for d in range(ND):
        tk, st, docs = maker(d)
        for k, v in usage(tk, st, rnd).items(): acc[k].append(v)
        if docs is not None:
            for k, v in line_stats(docs, random.Random(d)).items(): acc[k].append(v)
    res[name] = {k: C.summ(v) for k, v in acc.items()}
    lines.append(f'\n{name}: ' + '  '.join(f'{k} {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]' for k, v in res[name].items()))
    print(lines[-1], flush=True)

tkA, stA = toks_sites(LA)
# LA: usage stats are fixed numbers; resampling tokens (bootstrap by document) gives the CI
HALF = NLA // 2
def la_half(d):
    r = random.Random(1000 + d); docs = draw_docs(LA, HALF, r); t, s = toks_sites(docs); return t, s, docs
run('LA_half', la_half)
def lb_draw(d, pool=LB):
    r = random.Random(2000 + d); docs = draw_docs(pool, HALF, r); t, s = toks_sites(docs); return t, s, docs
run('LB', lb_draw)
run('LB_KN', lambda d: lb_draw(d, LBKN))
PE = C.pe_tokens()
def pe_draw(d):
    r = random.Random(3000 + d); s = r.sample(PE, HALF); return [x[1] for x in s], [x[0] for x in s], None
run('PE', pe_draw)
def la_code(d):
    r = random.Random(4000 + d); docs = draw_docs(LA, HALF, r); t, s = toks_sites(docs)
    code = C.random_id_code(sorted(set(tkA)), random.Random(5)); return [code[x] for x in t], s, None
run('LA_relabel_randID', la_code)
# exact single-sample LA values
u = usage(tkA, stA, random.Random(9)); ls = line_stats(LA, random.Random(9))
lines.append('\nLA full sample: ' + '  '.join(f'{k} {v:.3f}' for k, v in {**u, **ls}.items()))
print(lines[-1])
json.dump(res, open(C.LAD + '/la5_c2.json', 'w'), indent=1)
open(C.LAD + '/la5_c2.txt', 'w').write('\n'.join(lines) + '\n')
