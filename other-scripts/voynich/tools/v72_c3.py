"""v72 cycle 3: which component carries (or kills) the message-layer gain, and what the frozen payload stream is.

Part A (held-out, frozen rule; train = discovery half, test = holdout half):
  g_junc  GEN -> GEN+JBIG          gain from the junction class (last glyph of the previous payload token)
  g_id    GEN+JBIG -> +BIGJ        gain from the IDENTITY of the previous token beyond its last glyph
  g_page  GEN -> GEN+PAGE at W=20 and at W=60 (is page recurrence just longer-range self-citation?)
  corpora: ZL, IT, ZL~JUNC, ZL~MK2, planted BRU/ISI/DEU (merge, verbose) and BRU-merge~JUNC.
Part B (characterisation of the frozen payload stream on all ZL pages): alphabet, lengths, slot order,
  commonest tokens and bigrams, section-specific tokens.
Out: data/v72_ckpt/c3/<corpus>.json, data/v72_ckpt/c3_char.json
"""
import os, sys, json, time, random, math
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v72_lib as L
import v72_c2 as C2

OUT = os.path.join(L.CK, 'c3'); os.makedirs(OUT, exist_ok=True)
COMPS = L.COMPS_GEN + ['JBIG', 'BIGJ', 'BIG', 'PAGE']
MODELS = {'GEN': L.COMPS_GEN, 'GEN+JBIG': L.COMPS_GEN + ['JBIG'], 'GEN+JBIG+BIGJ': L.COMPS_GEN + ['JBIG', 'BIGJ'],
          'GEN+BIG': L.COMPS_GEN + ['BIG'], 'GEN+PAGE': L.COMPS_GEN + ['PAGE'],
          'ALL': L.COMPS_GEN + ['JBIG', 'BIGJ', 'PAGE']}


def run(name):
    fn = os.path.join(OUT, f'{name}.json')
    if os.path.exists(fn): return name
    t0 = time.time()
    fz = C2.freeze()
    S, plain, half = C2.corpus(name)
    X = L.extract(S, L.RULES[fz['rule']])
    tr = [p for p, h in zip(X, half) if h == 0]; te = [p for p, h in zip(X, half) if h == 1]
    out = dict(name=name, rule=fz['rule'], hash=fz['hash'])
    for W in (20, 60):
        r = L.ladder_generic(tr, te, COMPS, MODELS, W=W, seed=0)
        b = {k: float(v.mean()) for k, v in r.items()}
        d_id = r['GEN+JBIG'] - r['GEN+JBIG+BIGJ']
        out[f'W{W}'] = dict(bits=b, g_junc=b['GEN'] - b['GEN+JBIG'], g_id=b['GEN+JBIG'] - b['GEN+JBIG+BIGJ'],
                            g_id_z=float(d_id.mean() / (d_id.std() / math.sqrt(len(d_id)) + 1e-12)),
                            g_big=b['GEN'] - b['GEN+BIG'], g_page=b['GEN'] - b['GEN+PAGE'], g_all=b['GEN'] - b['ALL'])
    out['sec'] = time.time() - t0
    json.dump(out, open(fn, 'w'), default=float)
    w = out['W20']; w6 = out['W60']
    print(name, 'junc %.3f id %.3f (z %.1f) big %.3f page20 %.3f page60 %.3f' % (
        w['g_junc'], w['g_id'], w['g_id_z'], w['g_big'], w['g_page'], w6['g_page']), round(out['sec']), flush=True)
    return name


def slot_rigidity(toks):
    """share of ordered glyph pairs (a before b inside a word) that keep their majority order (v1 measure)."""
    c = Counter()
    for w in toks:
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                if w[i] != w[j]: c[(w[i], w[j])] += 1
    tot = agree = 0
    for (a, b), n in c.items():
        if a < b:
            m = c.get((b, a), 0); tot += n + m; agree += max(n, m)
    return agree / max(tot, 1)


def characterise():
    fn = os.path.join(L.CK, 'c3_char.json')
    if os.path.exists(fn): return
    fz = C2.freeze()
    out = {}
    for src in ('ZL3b', 'IT2a'):
        S = L.voynich(src); X = L.extract(S, L.RULES[fz['rule']])
        toks = [w for p in X for l in p['lines'] for w in l['w']]
        stoks = [w for p in S for l in p['lines'] for w in l['w']]
        tc = Counter(toks)
        lens = Counter(len(w) for w in toks)
        big = Counter((a, b) for p in X for l in p['lines'] for a, b in zip(l['w'], l['w'][1:]))
        nb = sum(big.values())
        pmi = []
        for (a, b), n in big.items():
            if n >= 8:
                e = tc[a] * tc[b] / len(toks) ** 2 * nb
                pmi.append((math.log2(n / e), n, a, b))
        pmi.sort(reverse=True)
        sec = defaultdict(Counter)
        for p in X:
            for l in p['lines']: sec[p['sec']].update(l['w'])
        spec = {}
        for s, c in sec.items():
            n = sum(c.values())
            sc = sorted(((c[w] / n) / (tc[w] / len(toks)), c[w], w) for w in c if c[w] >= 10)
            spec[s] = [(round(r, 2), k, w) for r, k, w in sc[::-1][:6]]
        # how many surface word types collapse into each payload type
        fan = defaultdict(set)
        for p, ps in zip(X, S):
            for l, ls in zip(p['lines'], ps['lines']):
                for a, b in zip(l['w'], ls['w']): fan[a].add(b)
        out[src] = dict(rule=fz['rule'], hash=fz['hash'], ntok=len(toks), types=len(tc), surface_types=len(set(stoks)),
                        alphabet=sorted({c for w in toks for c in w}),
                        glyph_freq=Counter(c for w in toks for c in w).most_common(),
                        len_dist=sorted(lens.items()), mean_len=sum(map(len, toks)) / len(toks),
                        surface_mean_len=sum(map(len, stoks)) / len(stoks),
                        rigidity=slot_rigidity(toks), top=tc.most_common(30),
                        top_share10=sum(n for _, n in tc.most_common(10)) / len(toks),
                        hapax=sum(1 for n in tc.values() if n == 1) / len(tc),
                        pmi_pairs=pmi[:25], sec_specific=spec,
                        fan_out=sorted(((len(v), k) for k, v in fan.items()), reverse=True)[:15])
    json.dump(out, open(fn, 'w'), default=float)


def arrows_ctrl():
    """where do the Voynich word arrows come from? holdout payload: real, lines shuffled within page (kills
    within-page drift), words shuffled within line (kills in-line order), paragraph-first lines removed."""
    fn = os.path.join(L.CK, 'c3_arrows.json')
    if os.path.exists(fn): return
    fz = C2.freeze(); out = {}
    for name in ('ZL', 'IT'):
        S, _, half = C2.corpus(name)
        X = [p for p, h in zip(L.extract(S, L.RULES[fz['rule']]), half) if h == 1]
        rng = random.Random(11)
        def lshuf(P):
            Q = []
            for p in P:
                ls = p['lines'][:]; rng.shuffle(ls); Q.append(dict(p, lines=ls))
            return Q
        def wshuf(P):
            return [dict(p, lines=[dict(l, w=rng.sample(l['w'], len(l['w']))) for l in p['lines']]) for p in P]
        def nops(P):
            return [dict(p, lines=[l for l in p['lines'] if not l['ps']]) for p in P]
        def rev(P):
            return [dict(p, lines=[dict(l, w=l['w'][::-1]) for l in p['lines'][::-1]]) for p in P]
        for k, f in [('real', lambda P: P), ('line_shuffled', lshuf), ('word_shuffled_in_line', wshuf),
                     ('no_para_first', nops), ('reversed', rev)]:
            out[f'{name}_{k}'] = [C2.arrows(f(X), s) for s in (0, 1, 2)]
            print(name, k, [a['surv'] for a in out[f'{name}_{k}']], [a['by'] for a in out[f'{name}_{k}']], flush=True)
    json.dump(out, open(fn, 'w'), default=float)


if __name__ == '__main__':
    characterise()
    arrows_ctrl()
    J = ['ZL', 'IT', 'ZL~JUNC', 'ZL~MK2', 'BRU-merge', 'BRU-verbose', 'ISI-merge', 'ISI-verbose', 'DEU-merge',
         'DEU-verbose', 'BRUL-merge', 'BRU-merge~JUNC', 'ISI-merge~JUNC']
    with Pool(int(os.environ.get('W', '2'))) as pool:
        for _ in pool.imap_unordered(run, J, chunksize=1): pass
