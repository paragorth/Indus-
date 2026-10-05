"""v53 cycle 3 follow-up: is the best Voynich read-back (IT_herb winner) meaning or word-shape?
Apply the winning read-back programs to: Voynich ZL half B, IT2a half B, a glyph-trigram resynthesis of ZL half B
(same word shapes statistically, no message), a word-type relabelling of ZL half B (each word type replaced by a
random other type of the same length class and frequency rank +-2: keeps token frequencies, destroys which word
is which), the Rugg target and the planted target; and score the same program shape with other corpora."""
import sys, os, json, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = ['v53_readback.py', 'voynich', '1', '1']
import importlib.util
spec = importlib.util.spec_from_file_location('rb', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v53_readback.py'))
rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)
from v53_lib import *
rb.init()
CORP = rb.CORP
A, B = rb.TGT
_, IB = split_halves(load_voynich('IT2a'))
rng = random.Random(3)


def trigram_resynth(lines):
    tri = defaultdict(Counter)
    for l in lines:
        for w in l:
            s = '^^' + w + '$'
            for i in range(2, len(s)):
                tri[s[i - 2:i]][s[i]] += 1
    out = []
    for l in lines:
        nl = []
        for _ in l:
            ctx, w = '^^', ''
            while True:
                c = tri[ctx]; tot = sum(c.values()); r = rng.random() * tot
                for ch, n in c.items():
                    r -= n
                    if r <= 0: break
                if ch == '$' or len(w) > 15: break
                w += ch; ctx = ctx[1] + ch
            nl.append(w or 'o')
        out.append(nl)
    return out


def relabel(lines):
    cnt = Counter(w for l in lines for w in l)
    bylen = defaultdict(list)
    for w, n in cnt.most_common():
        bylen[min(len(w), 8)].append(w)
    mp = {}
    for L, ws in bylen.items():
        idx = list(range(len(ws)))
        for i in range(0, len(idx), 5):          # permute within blocks of 5 frequency ranks
            blk = idx[i:i + 5]; sh = blk[:]; rng.shuffle(sh)
            for a, b in zip(blk, sh):
                mp[ws[a]] = ws[b]
    return [[mp[w] for w in l] for l in lines]


R = json.load(open(os.path.join(CK, 'rb_voynich.json')))
progs = []
for r in R['results']:
    for t in r['top'][:2]:
        progs.append((t['heldout'].get('score', 0), r['corpus'], t['prog']))
progs.sort(key=lambda x: -x[0])
progs = progs[:4]
sys.argv = ['v53_run.py', 'rugg', '1', '1', '1']
spec2 = importlib.util.spec_from_file_location('run', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v53_run.py'))
RUN = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(RUN)
RUN.TARGET = 'rugg'; RB_ = RUN.make_targets()[1]
RUN.TARGET = 'planted'; PB_ = RUN.make_targets()[1]
targets = {'ZL_B': B, 'IT2a_B': IB, 'ZL_B_trigram': trigram_resynth(B), 'ZL_B_relabel': relabel(B), 'rugg_B': RB_, 'planted_B': PB_}
out = []
for hs, c, p in progs:
    for tn, T in targets.items():
        env = Env(T, CORP, offset=7919)
        sc = rb.rb_score(p, env, T, rng_seed=5)
        out.append({'prog_corpus': c, 'target': tn, 'corpus': c, **{k: sc.get(k) for k in ('score', 'hit', 'null', 'cov', 'n')}})
        print(c, tn, {k: round(v, 3) if isinstance(v, float) else v for k, v in sc.items() if k in ('score', 'hit', 'null', 'cov', 'n')}, flush=True)
    # same program shape with the other corpora on ZL_B
    env = Env(B, CORP, offset=7919)
    for c2 in sorted(CORP):
        q = dict(p); q['corpus'] = c2
        sc = rb.rb_score(q, env, B, rng_seed=5)
        out.append({'prog_corpus': c, 'target': 'ZL_B', 'corpus': c2, **{k: sc.get(k) for k in ('score', 'hit', 'null', 'cov', 'n')}})
        print('   shape of', c, 'with', c2, round(sc['score'], 3), round(sc.get('hit', 0), 3), round(sc.get('null', 0), 3), flush=True)
json.dump(out, open(os.path.join(CK, 'check.json'), 'w'))
