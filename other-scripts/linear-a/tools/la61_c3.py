#!/usr/bin/env python3
"""LA-61 cycle 3: the typed glossary, its stability, the la60 grammar's bits, and frozen predictions.
 3a  Glossary: fit on all GORILA administrative documents (post-1985 documents excluded), R restarts;
     and on two disjoint halves by tablet.  A word is 'stable' if its modal type holds in >= 0.75 of
     restarts AND both halves agree (where the word is fitted in the half).
 3b  The la60 grammar (role trigram x identity x numbers x arithmetic) with the glossary types added as
     roles, glossary fitted on the training half only, held-out bits vs the PRIOR reading alone and vs
     the glossary with types permuted among its words (4 splits).
 3c  Frozen predictions for the 11 post-1985 administrative documents (sha256 before scoring), then
     scored: held-out gain vs the pooled class and vs permuted types; commodity hits vs site default."""
import sys, time, copy
from la61_common import *
from la60_common import split as la60_split
from la60_model import Grammar, doc_bits

OUT = os.path.join(LOOPS, 'la61_cycle3.txt')
R_FULL = int(os.environ.get('R_FULL', 8)); R_HALF = int(os.environ.get('R_HALF', 4))
ITERS = int(os.environ.get('ITERS', 2000))


def glossary_fit(docs, rng, R, name):
    V = Vocab(docs)
    fits, _, _ = fit(V, docs, rng, R, ITERS, name)
    return V, fits, consensus(fits, V), best_asg(fits)


def run():
    t0 = time.time()
    rng = np.random.default_rng(seed('la61-c3'))
    LA = load('LA')
    gor = [d for d in LA if d['pub'] != 'post']; post = [d for d in LA if d['pub'] == 'post']
    V, fits, cons, best = glossary_fit(gor, rng, R_FULL, 'la61-c3-full')
    nscored = R_FULL * (ITERS + POLISH * V.W * (NT + V.K + V.S + 1))
    hA, hB = split_tabs(gor, 2, 'la61-c3-halves')
    half = []
    for k, h in enumerate((hA, hB)):
        Vh, fh, ch, _ = glossary_fit(h, rng, R_HALF, 'la61-c3-half%d' % k)
        nscored += R_HALF * (ITERS + POLISH * Vh.W * (NT + Vh.K + Vh.S + 1))
        half.append({Vh.words[w]: TYPES[c[0]] for w, c in enumerate(ch)})
    occ = Counter(o[0] for o in occurrences(gor))
    gl = {}
    for w, c in enumerate(cons):
        word = V.words[w]
        t = TYPES[c[0]]
        hv = [h.get(word) for h in half]
        agree = [x == t for x in hv if x is not None]
        stable = c[1] >= 0.75 and len(agree) == 2 and all(agree)
        gl[word] = dict(type=t, share=round(c[1], 2), halves=hv, stable=stable, n=occ[word],
                        com=V.coms[c[2]] if c[2] >= 0 else None, com_share=round(c[3], 2),
                        site=V.sites[c[4]] if c[4] >= 0 else None)
    # chance agreement of halves (types permuted within each half)
    both = [w for w in gl if all(h.get(w) for h in half)]
    agree_real = sum(half[0][w] == half[1][w] for w in both)
    pr = random.Random(seed('la61-c3-perm'))
    null = []
    for _ in range(2000):
        b = [half[1][w] for w in both]; pr.shuffle(b)
        null.append(sum(half[0][w] == x for w, x in zip(both, b)))
    null = np.array(null)
    # ---- 3b la60 grammar bits
    R0 = prior_reading()
    bits = dict(prior=0.0, gloss=0.0, perm=[])
    A60 = admin_docs(load_la())
    for k in range(4):
        tr, te = la60_split(A60, 'la61-c3-g%d' % k)
        trids = {d['id'] for d in tr}
        Vg, fg, cg, bg = glossary_fit([d for d in LA if d['id'] in trids], rng, 2, 'la61-c3-g%d' % k)
        nscored += 2 * (ITERS + POLISH * Vg.W * (NT + Vg.K + Vg.S + 1))
        typ = {Vg.words[w]: TYPES[c[0]] for w, c in enumerate(cg)}

        def mk(tmap):
            R = copy.deepcopy(R0)
            for w, t in tmap.items():
                if w not in R['roles']:
                    R['roles'][w] = 'G' + t
            return R
        bp = sum(doc_bits(Grammar(tr, R0, order=False), d) for d in te)
        bgl = sum(doc_bits(Grammar(tr, mk(typ), order=False), d) for d in te)
        ws = list(typ); pp = []
        for j in range(10):
            ts = [typ[w] for w in ws]; pr.shuffle(ts)
            pp.append(sum(doc_bits(Grammar(tr, mk(dict(zip(ws, ts))), order=False), d) for d in te))
        bits['prior'] += bp; bits['gloss'] += bgl; bits['perm'].append(pp)
        print('grammar split', k, round(bp), round(bgl), round(np.mean(pp)), flush=True)
    permsum = np.sum(np.array(bits['perm']), 0)
    # ---- 3c frozen predictions on post-1985 documents
    Cg = V.counts(gor); Cp = V.counts(post)
    typ_a, caf, saf = best
    preds = []
    M = np.eye(NT)[typ_a].T
    P = tpl_arrays()
    for o in occurrences(post):
        i = V.wi.get(o[0])
        if i is None:
            continue
        t = typ_a[i]
        ns = M @ Cg['slot'] + P['slot']; nsl = ns[t] / ns[t].sum()
        top = int(np.argmax(nsl))
        qc = (M @ Cg['com'] + 0.5)[t]; qc[V.ci['NONE']] = -1; qc[V.ci['OTH']] = -1
        pc = V.coms[caf[i]] if caf[i] >= 0 else V.coms[int(np.argmax(qc))]
        preds.append(dict(doc=o[6], word=o[0], type=TYPES[t], slot=PREV[top // 6] + '>' + NEXT[top % 6],
                          commodity=pc))
    frozen = dict(glossary={w: (g['type'], g['com']) for w, g in gl.items()}, predictions=preds)
    hsum = hsh(frozen)
    json.dump(dict(frozen, sha256=hsum), open(os.path.join(CK, 'c3_frozen.json'), 'w'), ensure_ascii=False)
    # score after freezing
    g_post = gain(best, Cg, Cp)
    gp = [gain((np.random.default_rng(j).permutation(typ_a), caf, saf), Cg, Cp)['core'] for j in range(100)]
    hits = 0; hsd = 0; nn = 0
    sd = defaultdict(Counter)
    for o in occurrences(gor):
        if o[4] != 'NONE':
            sd[o[5]][o[4]] += 1
    obs = {(o[6], o[0]): o for o in occurrences(post)}
    slot_hit = 0
    for p in preds:
        o = obs[(p['doc'], p['word'])]
        slot_hit += (PREV[o[1] // 6] + '>' + NEXT[o[1] % 6]) == p['slot']
        if o[4] != 'NONE':
            nn += 1; hits += p['commodity'] == o[4]
            hsd += (sd[o[5]].most_common(1)[0][0] if sd[o[5]] else None) == o[4]
    res = dict(glossary=gl, half_agree=(agree_real, len(both), float(null.mean()), float((1 + (null >= agree_real).sum()) / 2001)),
               bits=dict(prior=bits['prior'], gloss=bits['gloss'], perm=permsum.tolist()),
               post=dict(gain=g_post, perm_core=gp, n_pred=len(preds), slot_hit=slot_hit, com_hit=hits, com_sd=hsd, com_n=nn),
               sha256=hsum, n_scored=nscored, sec=time.time() - t0)
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), ensure_ascii=False, default=str)
    st = {w: g for w, g in gl.items() if g['stable']}
    tc = Counter(g['type'] for g in gl.values()); ts = Counter(g['type'] for g in st.values())
    row = ('| LA-61.3a | TYPED GLOSSARY: fit on %d GORILA administrative documents (post-1985 excluded), %d restarts, + two disjoint halves (%d restarts each); stable = modal type in >= 0.75 of restarts and both halves agree. Control: half-vs-half type agreement against types permuted within a half (2,000) | '
           '%d words typed (%s); %d stable (%s). Half-vs-half agreement %d/%d words (permuted %.1f, P %.4f) |' % (
               len(gor), R_FULL, R_HALF, len(gl), dict(sorted(tc.items())), len(st), dict(sorted(ts.items())),
               agree_real, len(both), null.mean(), res['half_agree'][3]))
    wlog(OUT, row + ' see verdict row |')
    row2 = ('| LA-61.3b | la60 GRAMMAR BITS with the glossary types as roles (glossary fitted on the training half only), 4 held-out splits; control: glossary types permuted among its words (10/split) | '
            'PRIOR alone %.0f bits; PRIOR + glossary %.0f (%+.0f); PRIOR + permuted glossary %.0f (best of 10 %.0f) |' % (
                bits['prior'], bits['gloss'], bits['gloss'] - bits['prior'], permsum.mean(), permsum.min()))
    v2 = 'glossary %s the permuted glossary by %.0f bits and %s PRIOR by %.0f bits' % (
        'beats' if bits['gloss'] < permsum.min() else 'does not beat', permsum.mean() - bits['gloss'],
        'beats' if bits['gloss'] < bits['prior'] else 'loses to', abs(bits['gloss'] - bits['prior']))
    wlog(OUT, row2 + ' ' + v2 + ' |')
    gpa = np.array(gp)
    row3 = ('| LA-61.3c | FROZEN PREDICTIONS for the %d post-1985 administrative documents (glossary fitted on GORILA only; sha256 %s, data/la61_ckpt/c3_frozen.json), then scored; control: types permuted (100) | '
            '%d predictions on glossary words; held-out gain core %.1f bits (permuted %.1f, %d/100 >= real), site %.1f; modal slot right %d/%d; commodity right %d/%d (site default %d) |' % (
                len(post), hsum[:16], len(preds), g_post['core'], gpa.mean(), int((gpa >= g_post['core']).sum()), g_post['site'],
                slot_hit, len(preds), hits, nn, hsd))
    wlog(OUT, row3 + ' - |')
    print(row); print(row2, v2); print(row3)


if __name__ == '__main__':
    run()
