"""v86 cycle 3b: dissect the line-end winner (final glyph -> m). Fixed rule tail:1:m, scored per slot and per
pressure stratum; which glyph does m replace; is the replaced glyph determined by the page (unique preimage) more
than chance; does the m-form behave in context like its preimage (previous-word profile) more than like a random
same-length page word?"""
import os, random, math
from collections import Counter, defaultdict
import v86_lib as L

RULE = (('tail', 1, 'm'),)


def strata(pages):
    """Re-label E tokens by pressure: line glyph length vs page median of non-paragraph-end lines."""
    out = {}
    for f, ls in pages.items():
        lens = sorted(sum(len(w) for w in ln['units'] if w) for ln in ls if not ln['para_end'])
        med = lens[len(lens) // 2] if lens else 0
        nl = []
        for ln in ls:
            g = sum(len(w) for w in ln['units'] if w)
            tag = 'Epe' if ln['para_end'] else ('Elong' if g >= med else 'Eshort')
            nl.append(dict(ln, cls=[(tag if c == 'E' else c) for c in ln['cls']]))
        out[f] = nl
    return out


def fixed_score(pages, target, seeds=(0, 1)):
    out = []
    for s in seeds:
        for half in L.folio_split(list(pages), s):
            P = L.prepare(pages, half, target)
            w = L.len_weights(P)
            t, c, n = L.score_rule(RULE, P, w)
            out.append((t - c, n))
    return out


def replaced_glyph(pages):
    """For unique tail:1:m recoveries at line end: which final glyph was replaced; vs final-glyph distribution of page words."""
    rep = Counter(); base = Counter(); uniq = 0; amb = 0
    for f, ls in pages.items():
        pc = Counter(w for ln in ls for w in ln['units'] if w)
        for ln in ls:
            lc = Counter(w for w in ln['units'] if w)
            for w, c in zip(ln['units'], ln['cls']):
                if not w or c != 'E' or not w.endswith('m') or len(w) < 2 or pc[w] - lc[w] > 0:
                    continue
                stem = w[:-1]
                pre = [u for u in pc if pc[u] - lc[u] > 0 and len(u) == len(w) and u[:-1] == stem and u[-1] != 'm']
                if len(pre) == 1:
                    rep[pre[0][-1]] += 1; uniq += 1
                elif pre:
                    amb += 1
                    for u in pre:
                        rep[u[-1]] += 1 / len(pre)
        for u, k in pc.items():
            if len(u) >= 2:
                base[u[-1]] += k
    return rep, base, uniq, amb


def context_test(pages, rng, nperm=200):
    """prev-word onset profile of line-end stem+m tokens vs (a) their page preimages, (b) random page words of equal length.
    Uses previous word's first two glyphs. Score = mean log-likelihood ratio."""
    prev_of = defaultdict(Counter)
    allprev = Counter()
    for f, ls in pages.items():
        for ln in ls:
            u = ln['units']
            for j in range(1, len(u)):
                if u[j] and u[j - 1]:
                    prev_of[u[j]][u[j - 1][:2]] += 1; allprev[u[j - 1][:2]] += 1
    tot = sum(allprev.values())
    def lp(word, ctx):
        c = prev_of[word]; n = sum(c.values())
        return math.log((c[ctx] + 5 * allprev[ctx] / tot) / (n + 5)) - math.log(allprev[ctx] / tot)
    real = []; rand = []
    for f, ls in pages.items():
        pc = Counter(w for ln in ls for w in ln['units'] if w)
        byl = defaultdict(list)
        for u in pc: byl[len(u)].append(u)
        for ln in ls:
            u = ln['units']
            if len(u) < 2 or not u[-1] or not u[-2] or not u[-1].endswith('m') or len(u[-1]) < 2:
                continue
            s = u[-1]; ctx = u[-2][:2]
            pre = [v for v in pc if len(v) == len(s) and v[:-1] == s[:-1] and v[-1] != 'm']
            if not pre:
                continue
            real.append(sum(lp(v, ctx) for v in pre) / len(pre))
            cand = [v for v in byl[len(s)] if v[:-1] != s[:-1] and not v.endswith('m') and v[:2] == s[:2]]
            if cand:
                rand.append(sum(lp(rng.choice(cand), ctx) for _ in range(20)) / 20)
    return sum(real) / max(len(real), 1), sum(rand) / max(len(rand), 1), len(real)


if __name__ == '__main__':
    rng = random.Random(3)
    for tr in ('ZL3b', 'IT2a'):
        VP, META = L.voynich_pages(tr)
        S = strata(VP)
        for tg in (('Elong',), ('Eshort',), ('Epe',), ('Ib',), ('B',), ('Ia',)):
            r = fixed_score(S, tg)
            print(tr, 'tail:1:m', tg[0], 'Delta per half', [round(d, 4) for d, n in r], 'n', [n for d, n in r][:2], flush=True)
        for k in range(3):
            Sh = strata(L.shuffle_positions(VP, random.Random(50 + k)))
            r = fixed_score(Sh, ('Elong', 'Eshort'))
            print(tr, 'tail:1:m POS-SHUFFLE', k, [round(d, 4) for d, n in r], flush=True)
        rep, base, uniq, amb = replaced_glyph(VP)
        tb = sum(base.values()); tr_ = sum(rep.values())
        print(tr, 'unique', uniq, 'ambiguous', amb, 'replaced glyph (share, base share):',
              [(g, round(n / tr_, 2), round(base[g] / tb, 2)) for g, n in rep.most_common(8)], flush=True)
        a, b, n = context_test(VP, rng)
        print(tr, 'context: preimage LLR %.3f vs random same-length same-onset word %.3f (n=%d)' % (a, b, n), flush=True)
        # same context test on a planted Latin control (suspension with ~ is not m; plant 'final letter -> m')
        if tr == 'ZL3b':
            lat = L.layout_like(VP, L.ref_words('Latin-Caesar', sum(len(ln['units']) for ls in VP.values() for ln in ls)))
            latp = L.plant(lat, lambda w, r: w[:-1] + 'M' if len(w) > 2 else w, classes=('E',), p=0.5, seed=4)
            latp = {f: [dict(ln, units=[(w[:-1] + 'm' if w and w.endswith('M') else w) for w in ln['units']]) for ln in ls] for f, ls in latp.items()}
            a, b, n = context_test(latp, rng)
            print('LatinPlant final->m context: preimage LLR %.3f vs random %.3f (n=%d)' % (a, b, n), flush=True)
            a, b, n = context_test(lat, rng)
            print('Latin unplanted context: preimage LLR %.3f vs random %.3f (n=%d)' % (a, b, n), flush=True)
            hab = L.plant(L.shuffle_positions(VP, random.Random(9)), lambda w, r: w[:-1] + 'm' if len(w) > 2 else w, classes=('E',), p=0.3, seed=4)
            a, b, n = context_test(hab, rng)
            print('Voynich pos-shuffled + final->m plant (context-free mark): preimage %.3f vs random %.3f (n=%d)' % (a, b, n), flush=True)
