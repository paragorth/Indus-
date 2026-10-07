"""v86 cycle 3d: is the line-end r/l -> m recovery PAGE-specific? Score tail:1:m line-end tokens against the vocabulary of
their own page vs a random other page of the same quire x Currier language (same size class). Meaning-bearing
abbreviation of the page's own words predicts own > other."""
import random
from collections import Counter, defaultdict
import v86_lib as L


def score(VP, META, other, rng):
    groups = defaultdict(list)
    for f in VP: groups[(META[f][0], META[f][1])].append(f)
    hit = tot = 0.0
    for f, ls in VP.items():
        src = f
        if other:
            cand = [g for g in groups[(META[f][0], META[f][1])] if g != f]
            if not cand: continue
            src = rng.choice(cand)
        pc = Counter(w for ln in VP[src] for w in ln['units'] if w)
        own = Counter(w for ln in ls for w in ln['units'] if w)
        for ln in ls:
            lc = Counter(w for w in ln['units'] if w)
            s = ln['units'][-1]
            if ln['cls'][-1] != 'E' or not s or not s.endswith('m') or len(s) < 2: continue
            if (own[s] - lc[s] > 0) or pc[s] > (0 if other else lc[s]) - (0 if other else 0) and other and pc[s] > 0: continue
            pre = [u for u in pc if len(u) == len(s) and u[:-1] == s[:-1] and u[-1] != 'm' and (other or pc[u] - lc[u] > 0)]
            tot += 1; hit += (1 / len(pre)) if pre else 0
    return hit / tot, int(tot)


if __name__ == '__main__':
    for tr in ('ZL3b', 'IT2a'):
        VP, META = L.voynich_pages(tr)
        a, n = score(VP, META, False, random.Random(0))
        bs = [score(VP, META, True, random.Random(k))[0] for k in range(20)]
        bs.sort()
        print(tr, 'line-end m tokens recovered from OWN page %.3f (n=%d); from another page of the same quire x language: mean %.3f, range %.3f-%.3f (20 draws)'
              % (a, n, sum(bs) / len(bs), bs[0], bs[-1]), flush=True)


def score_mid(VP, META, other, rng, finals='rl'):
    """Baseline page stickiness: mid-line OOV tokens ending in r/l recovered by same-stem words with another final."""
    groups = defaultdict(list)
    for f in VP: groups[(META[f][0], META[f][1])].append(f)
    hit = tot = 0.0
    for f, ls in VP.items():
        src = f
        if other:
            cand = [g for g in groups[(META[f][0], META[f][1])] if g != f]
            if not cand: continue
            src = rng.choice(cand)
        pc = Counter(w for ln in VP[src] for w in ln['units'] if w)
        own = Counter(w for ln in ls for w in ln['units'] if w)
        for ln in ls:
            lc = Counter(w for w in ln['units'] if w)
            for s, c in zip(ln['units'], ln['cls']):
                if c != 'M' or not s or len(s) < 2 or s[-1] not in finals: continue
                if own[s] - lc[s] > 0 or (other and pc[s] > 0): continue
                pre = [u for u in pc if len(u) == len(s) and u[:-1] == s[:-1] and u[-1] != s[-1] and (other or pc[u] - lc[u] > 0)]
                tot += 1; hit += (1 / len(pre)) if pre else 0
    return hit / tot, int(tot)


if __name__ == '__main__':
    for tr in ('ZL3b', 'IT2a'):
        VP, META = L.voynich_pages(tr)
        a, n = score_mid(VP, META, False, random.Random(0))
        bs = sorted(score_mid(VP, META, True, random.Random(k))[0] for k in range(20))
        print(tr, 'BASELINE mid-line r/l-final OOV tokens, same-stem recovery OWN page %.3f (n=%d); other page mean %.3f, range %.3f-%.3f; ratio %.2f'
              % (a, n, sum(bs) / len(bs), bs[0], bs[-1], a / (sum(bs) / len(bs))), flush=True)
