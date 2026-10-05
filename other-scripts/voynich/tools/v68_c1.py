"""v68 cycle 1: chant vs language as the plaintext behind the Voynich surface.

For each source, N random neume/letter codes (v68_lib.encode); each coded text is scored by
the v53 15-statistic distance to three targets: Voynich ZL (whole), a planted chant text written by a
hidden code, and a planted Latin-syllable text written by a hidden code (calibration: each planted
target must be matched best by its own source).  Feature means kept for the three traits v53 could
not reproduce (mi_lf/mi_ll line edges, mi_j junction coupling, adjrep adjacent repeats).
usage: python3 v68_c1.py N
"""
import json, os, random, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v68_lib as V
import v53_lib as L53

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = os.path.join(V.CK, 'c1_results.jsonl')

SRC = {}
T = {}


def setup():
    rng = random.Random(1)
    ch = V.chant_lines(V.chant_units())
    SRC['chant'] = ch
    SRC['chant_rewrap'] = V.rewrap(ch, 9, rng)
    SRC['chant_nshuf'] = V.neume_shuffle(ch, rng)
    SRC['chant_markov'] = V.markov_melody(ch[:40000], rng)
    SRC['LA_word'] = V.lang_lines('LA_ency')
    SRC['LA_syll'] = V.lang_lines('LA_ency', syll=True)
    SRC['IT_syll'] = V.lang_lines('IT_herb', syll=True)
    SRC['DE_word'] = V.lang_lines('DE_herb')
    vz = [l['words'] for l in L53.load_voynich('ZL3b')]
    T['ZL'] = L53.target_profile(vz)
    # planted targets: hidden codes, held-out part of the source corpus (second half)
    hidden_c = {'rep': 'absint', 'fold': False, 'group': 'one', 'trunc': 5, 'rank': True, 'p2': 0.15,
                'seed': 4242, 'sympermute': True}
    pc = V.encode(ch[len(ch) // 2: len(ch) // 2 + 3000], hidden_c)
    T['plant_chant'] = L53.target_profile(pc)
    hidden_l = {'rep': 'abs', 'fold': False, 'group': 'one', 'trunc': 0, 'rank': True, 'p2': 0.3,
                'seed': 4343, 'sympermute': True}
    la = SRC['LA_syll']
    pl = V.encode(la[len(la) // 2:], hidden_l, symmap=V.random_symmap(random.Random(9), k=26))
    T['plant_la'] = L53.target_profile(pl)


def job(args):
    src, i = args
    rng = random.Random(hash((src, i)) & 0xffffffff)
    enc = V.random_encoder(rng)
    lines = SRC[src]
    sub = V.sample_lines(lines, 9000, rng)
    symmap = None if src.startswith('chant') else V.random_symmap(rng, k=rng.choice([7, 13, 26]))
    coded = V.encode(sub, enc, symmap=symmap)
    r = V.profile(coded, 3, seed=i)
    if r is None:
        return None
    m, ps = r
    d = {t: sum(L53.distance(p, T[t])[0] for p in ps) / len(ps) for t in T}
    return {'src': src, 'i': i, 'enc': enc, 'd': d, 'f': m}


def init():
    setup()


if __name__ == '__main__':
    setup()
    json.dump({t: {'mean': T[t]['mean']} for t in T}, open(os.path.join(V.CK, 'c1_targets.json'), 'w'), indent=1)
    srcs = list(SRC)
    jobs = [(s, i) for i in range(N) for s in srcs]
    t0 = time.time()
    with Pool(2, initializer=init) as P, open(OUT, 'w') as f:
        for k, r in enumerate(P.imap_unordered(job, jobs, chunksize=8)):
            if r:
                f.write(json.dumps(r) + '\n')
            if k % 500 == 0:
                print(k, len(jobs), round(time.time() - t0), flush=True)
    print('done', time.time() - t0)
