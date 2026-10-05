"""No-learning baseline: nearest-numeral-profile lexicon, known-answer precision by frequency."""
import sys, os, json, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe39_common import *  # noqa
pe = load_script('pe'); pc = load_script('pc')
gold = json.load(open(os.path.join(CK, 'gold_ur3_pc.json')))
out = {}
for SET in ['PLANT', 'PLANTSHUF', 'UR3', 'UR3SHUF']:
    for seed in range(3):
        mode = 'abs' if 'UR3' in SET else 'ncode'
        if SET.startswith('PLANT'):
            tabs = sorted({l['tab'] for l in pc}); r = random.Random(1234); r.shuffle(tabs)
            h1 = set(tabs[:len(tabs) // 2])
            A = subsample_tablets([l for l in pc if l['tab'] in h1], len(pe), seed); B = [l for l in pc if l['tab'] not in h1]
            La, Lb = 'pq', 'pc'
            cor = lambda a, b: a == b
        else:
            A = load_ur3(max_lines=len(pe), seed=seed); B = pc; La, Lb = 'ur', 'pc'
            cor = lambda a, b: (b in gold[a]) if a in gold else None
        A = numeral_tokens(A, mode); B = numeral_tokens(B, mode)
        if SET.endswith('SHUF'):
            A = shuffle_numerals(A, 1000 + seed)
        V = Vocab({La: A, Lb: B})
        P = numeral_profiles(V, {La: A, Lb: B})
        plex = profile_lexicon(V, P, La, Lb)
        fa = collections.Counter(s for l in A for s in l['signs'])
        res = []
        for s, t in plex.items():
            c = cor(s.split('|', 1)[1], t.split('|', 1)[1])
            if c is not None:
                res.append((fa[s.split('|', 1)[1]], c))
        for mn in (5, 15, 40):
            x = [c for f, c in res if f >= mn]
            print(SET, seed, 'min', mn, 'prec %d/%d' % (sum(x), len(x)))
        out['%s_%d' % (SET, seed)] = res
jdump(out, os.path.join(CK, 'profile_baseline.json'))
