"""v52 cycle 2: climb the field schemas. From the 8 best random schemas of cycle 1 (per corpus) a
hill-climb mutates cut rules / unit sets / N to maximise the field score FS on half A; every climbed
schema is re-tested on held-out half B. Same climb on the meta-null (VSHUF), the conditioned Markov
null (VMARK), the weak plant (PL3), and on the second transcription (VI = IT2a, own search seeds)."""
import sys, os, pickle, random, time, copy
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v52_lib as L
from v52_cycle1 import build

STEPS = int(os.environ.get('V52_STEPS', 120))


def mutate(sch, r, units):
    s = copy.deepcopy(sch); rules = [list(x) for x in s['rules']]
    op = r.random()
    if op < 0.15 and len(rules) < 4:
        rules.insert(r.randrange(len(rules) + 1), list(L.random_schema(r, units)['rules'][0]))
    elif op < 0.25 and len(rules) > 1:
        rules.pop(r.randrange(len(rules)))
    elif op < 0.35:
        s['N'] = r.choice([6, 12, 24, 48])
    else:
        i = r.randrange(len(rules)); t, a = rules[i]
        if t in ('fix', 'end'):
            if r.random() < 0.7:
                rules[i][1] = max(1, min(4, a + r.choice([-1, 1])))
            else:
                rules[i][0] = 'end' if t == 'fix' else 'fix'
        else:
            S = set(a); u = r.choice(units[:20])
            if u in S and len(S) > 1:
                S.discard(u)
            else:
                S.add(u)
            if r.random() < 0.15:
                rules[i][0] = 'run' if t == 'set' else 'set'
            rules[i][1] = tuple(sorted(S))
    s['rules'] = [tuple(x) for x in rules]
    return s


def climb(args):
    name, sch, seed = args
    C = build(name); r = random.Random(seed)
    cur = sch; best = L.score(C, cur, C['nulls'], mask=C['A'], dep=False)['FS']
    for _ in range(STEPS):
        cand = mutate(cur, r, C['top_units'])
        f = L.score(C, cand, C['nulls'], mask=C['A'], dep=False)['FS']
        if f >= best:
            cur, best = cand, f
    rA = L.score(C, cur, C['nulls'], mask=C['A'], dep=True)
    rB = L.score(C, cur, C['nulls'], mask=~C['A'], dep=True)
    return name, cur, rA, rB


if __name__ == '__main__':
    S1 = pickle.load(open(os.path.join(L.CK, 'c1_summary.pkl'), 'rb'))
    names = ['V', 'VI', 'VSHUF', 'VMARK', 'PL3', 'PL6']
    jobs = []
    for n in names:
        if n == 'VI':
            C = build('VI'); seeds = []
            for i in range(400):
                r = random.Random(1000003 * i + 17); sch = L.random_schema(r, C['top_units'])
                seeds.append((L.score(C, sch, C['nulls'], mask=C['A'], dep=False)['FS'], sch))
            seeds.sort(key=lambda x: -x[0]); starts = [x[1] for x in seeds[:8]]
        else:
            starts = sorted(S1[n]['top'], key=lambda t: -t['A']['FS'])[:8]
            starts = [t['sch'] for t in starts]
        jobs += [(n, s, 99 + k) for k, s in enumerate(starts)]
    t = time.time(); out = {n: [] for n in names}
    with Pool(2) as P:
        for name, sch, rA, rB in P.imap_unordered(climb, jobs):
            out[name].append(dict(sch=sch, A=rA, B=rB))
            print(name, round(rA['FS'], 3), round(rB['FS'], 3), round(rB['DEP'], 3), round(time.time() - t), flush=True)
    pickle.dump({n: dict(search=[(0, x['A']['FS'], x['A']['TOT']) for x in v], top=v) for n, v in out.items()},
                open(os.path.join(L.CK, 'c2_summary.pkl'), 'wb'))
    print('all done', flush=True)
