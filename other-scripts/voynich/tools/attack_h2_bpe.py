"""Attack on chink 2 (letter-predictability gap): if Voynich is a verbose cipher (each plain letter written as a group
of glyphs), merging the commonest glyph pairs into units (byte-pair merging, within words) should lift h2 and word
length to language values at a plausible alphabet size (25-40 units), the way it does for a known verbose cipher.
Positive control: Latin (Caesar) and Italian (Manzoni) encrypted with a fixed verbose codebook (each letter -> 1-3
symbols from 12). Negative controls: an order-3 glyph Markov text trained on Voynich (same letter statistics, no
hidden language) and a self-citation generator text. Same merge procedure on all. Output: h2, mean/SD word length
in units, at alphabet sizes 25, 30, 35, 40, 50."""
import sys, os, math, random, json
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, gen

def h_stats(words):
    seq = []
    for w in words: seq.extend(w); seq.append('_')
    n = len(seq); c1 = Counter(seq)
    c2 = Counter(zip(seq, seq[1:])); n2 = n - 1
    h12 = -sum(v/n2*math.log2(v/n2) for v in c2.values())
    hf = -sum(v/n2*math.log2(v/n2) for v in Counter(seq[:-1]).values())
    h1 = -sum(v/n*math.log2(v/n) for v in c1.values())
    L = [len(w) for w in words]; m = sum(L)/len(L); sd = (sum((x-m)**2 for x in L)/len(L))**.5
    return dict(alphabet=len(c1)-1, h1=round(h1,3), h2=round(h12-hf,3), mean=round(m,2), sd=round(sd,2))

def bpe_trajectory(words, targets=(25,30,35,40,50), max_merges=200):
    W = Counter(tuple(w) for w in words)        # word type -> count
    toks = [list(w) for w in words]
    out = {}; alpha = len({c for w in toks for c in w})
    for t in targets:
        if alpha >= t and t not in out: out[t] = h_stats(toks)
    for _ in range(max_merges):
        pc = Counter()
        for w in toks:
            for a, b in zip(w, w[1:]): pc[(a, b)] += 1
        if not pc: break
        (a, b), _n = pc.most_common(1)[0]; new = a + '+' + b
        nt = []
        for w in toks:
            i, r = 0, []
            while i < len(w):
                if i < len(w)-1 and w[i] == a and w[i+1] == b: r.append(new); i += 2
                else: r.append(w[i]); i += 1
            nt.append(r)
        toks = nt
        alpha = len({c for w in toks for c in w})
        for t in targets:
            if alpha >= t and t not in out: out[t] = h_stats(toks)
        if alpha > max(targets): break
    return out

def verbose_encrypt(words, seed=7):
    rng = random.Random(seed); syms = 'ABCDEFGHIJKL'
    letters = sorted({c for w in words for c in w}); used = set(); code = {}
    for c in letters:
        while True:
            k = ''.join(rng.choice(syms) for _ in range(rng.choice([1,2,2,3])))
            if k not in used: used.add(k); code[c] = k; break
    return [''.join(code[c] for c in w) for w in words]

N = 20000
vl = vlib.load_voynich('ZL3b', drop_uncertain=True)
vw = [''.join(vlib.glyphs(w)) for L in vl for w in L['words']][:N]
vlines = [{'words': [''.join(vlib.glyphs(w)) for w in L['words']], 'para_start': L.get('para_start'), 'para_end': L.get('para_end')} for L in vl]
res = {}
res['Voynich (glyph units)'] = {'start': h_stats([list(w) for w in vw]), 'merges': bpe_trajectory(vw)}
for key in ('Latin-Caesar', 'Italian-Manzoni'):
    lw = [w for w in vlib.words_of(vlib.load_ref(key)) if w][:N]
    res[key + ' plain'] = {'start': h_stats([list(w) for w in lw])}
    ew = verbose_encrypt(lw)
    res[key + ' verbose-encrypted'] = {'start': h_stats([list(w) for w in ew]), 'merges': bpe_trajectory(ew)}
mk = [w for L in gen.char_markov(vlines, order=3, seed=2) for w in L['words']][:N]
res['Markov-3 from Voynich (no language)'] = {'start': h_stats([list(w) for w in mk]), 'merges': bpe_trajectory(mk)}
sc = [w for L in gen.self_citation(vlines, seed=2) for w in L['words']][:N]
res['self-citation generator'] = {'start': h_stats([list(w) for w in sc]), 'merges': bpe_trajectory(sc)}
for k, v in res.items():
    print('==', k, 'start', v['start'])
    for t, s in sorted(v.get('merges', {}).items()): print(f'   alphabet>={t}: {s}')
vlib.save('attack_h2_bpe', res)
