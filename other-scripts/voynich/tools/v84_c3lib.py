"""v84 cycle 3: statistics that might tell real page content from a fitted page-mood generator.
All on line-interior tokens. Page-burst words of page P: count on P >= 2 and count / (expected from the rest of its
section + 0.5) >= 3; at most the 12 most surprising per page.

  T_far     PB_far / PB (from v84_lib.PB; within-page spread)
  T_co      distant co-return: for each pair (a, b) of burst words of one page, the number of OTHER pages (same section,
            >= 3 pages away) holding both, against mismatched pairs (a, b') with b' a burst word of a different page of
            similar total frequency; log((O_true + 1) / (O_mis + 1)), and the same for pages of OTHER sections (T_cox)
  T_form    form coherence: mean glyph-bigram Jaccard of true burst pairs minus mismatched pairs
  T_near    return profile: presence rate of a burst word on pages 1-2 away / on pages >= 10 away in the same section
            (drift makes it high; topic recurrence keeps it near 1)
  T_new     share of burst types never seen on any other page
  T_top     share of a page's burst tokens carried by its single most frequent burst word
"""
import math, random
from collections import Counter, defaultdict
import numpy as np


def _bodybg(w):
    b = w[1:] if len(w) > 1 and w[0] == 'q' else w
    return set(zip('^' + b, b + '$'))


def burst(pages, maxb=12):
    secpages = defaultdict(list); pc = []; plen = []
    for pi, p in enumerate(pages):
        c = Counter(w for l in p['lines'] for w in l['w'][1:])
        pc.append(c); plen.append(sum(c.values())); secpages[p['sec']].append(pi)
    secC = {s: sum((pc[i] for i in ix), Counter()) for s, ix in secpages.items()}
    secN = {s: sum(plen[i] for i in ix) for s, ix in secpages.items()}
    B = []
    for pi, p in enumerate(pages):
        s = p['sec']; C = secC[s]; N = secN[s] - plen[pi]; out = []
        for w, c in pc[pi].items():
            if c < 2: continue
            e = plen[pi] * (C[w] - c) / max(N, 1)
            r = c / (e + 0.5)
            if r >= 3: out.append((r, w))
        out.sort(reverse=True); B.append([w for _, w in out[:maxb]])
    return pc, plen, secpages, B


def stats(pages, seed=843):
    rng = random.Random(seed)
    pc, plen, secpages, B = burst(pages)
    pos = {}; sec = {}
    for s, ix in secpages.items():
        for k, i in enumerate(ix): pos[i] = k; sec[i] = s
    pres = defaultdict(set)
    for i, c in enumerate(pc):
        for w in c: pres[w].add(i)
    tot = Counter()
    for c in pc: tot.update(c)
    # frequency buckets of burst words for mismatching
    allb = [(i, w) for i, ws in enumerate(B) for w in ws]
    bucket = defaultdict(list)
    for i, w in allb: bucket[int(math.log2(tot[w] + 1))].append((i, w))
    O_t = O_m = X_t = X_m = 0; F_t = []; F_m = []
    for i, ws in enumerate(B):
        s = sec[i]
        for a_ix in range(len(ws)):
            for b_ix in range(a_ix + 1, len(ws)):
                a, b = ws[a_ix], ws[b_ix]
                cand = bucket[int(math.log2(tot[b] + 1))]
                for _ in range(5):
                    j, b2 = cand[rng.randrange(len(cand))]
                    if j != i and b2 != a and b2 != b: break
                else:
                    continue
                excl = {i, j}

                def co(x, y, same):
                    n = 0
                    for q in pres[x] & pres[y]:
                        if q in excl: continue
                        if same:
                            if sec[q] == s and abs(pos[q] - pos[i]) >= 3 and (sec[j] != s or abs(pos[q] - pos[j]) >= 3): n += 1
                        elif sec[q] != s: n += 1
                    return n
                O_t += co(a, b, True); O_m += co(a, b2, True)
                X_t += co(a, b, False); X_m += co(a, b2, False)
                ba, bb, bm = _bodybg(a), _bodybg(b), _bodybg(b2)
                F_t.append(len(ba & bb) / len(ba | bb)); F_m.append(len(ba & bm) / len(ba | bm))
    out = dict(T_co=math.log((O_t + 1) / (O_m + 1)), T_cox=math.log((X_t + 1) / (X_m + 1)),
               T_form=float(np.mean(F_t) - np.mean(F_m)) if F_t else 0.0, npairs=len(F_t), O_t=O_t, O_m=O_m)
    # return profile and novelty
    near = [0, 0]; far = [0, 0]; new = 0; nb = 0; top = []
    for i, ws in enumerate(B):
        s = sec[i]; nb += len(ws)
        for w in ws:
            others = pres[w] - {i}
            if not others: new += 1
            for q in secpages[s]:
                if q == i: continue
                d = abs(pos[q] - pos[i])
                if d <= 2: near[0] += q in others; near[1] += 1
                elif d >= 10: far[0] += q in others; far[1] += 1
        if ws:
            cs = [pc[i][w] for w in ws]; top.append(max(cs) / sum(cs))
    out['T_near'] = (near[0] / max(near[1], 1)) / max(far[0] / max(far[1], 1), 1e-9)
    out['T_new'] = new / max(nb, 1)
    out['T_top'] = float(np.mean(top)) if top else 0.0
    out['nburst'] = nb / len(pages)
    return out
