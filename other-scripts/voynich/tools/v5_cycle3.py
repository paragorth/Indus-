"""v5 cycle 3 -- repetition with transposition (melody repeated at another pitch).

If glyphs are pitches on an ordered scale, a phrase should recur shifted by a constant step
(a_i = b_i + k).  For an ordering of the alphabet, take every n-gram of symbols inside a line
(crossing word breaks; a melody runs across syllables), its interval tuple (rank differences)
and its starting rank. TP = number of pairs of n-gram tokens with the same interval tuple but a
different start (transposed repeats); all-unison n-grams are dropped.

X1  known order: chant with its true staff order vs 300 random orderings of the same symbols.
X2  unknown order: hill-climb an ordering that maximises TP on half the units (odd), then score the
    learned ordering on the other half against 300 random orderings (z). Same search on every corpus,
    so the search's own power to manufacture transpositions is measured by the controls
    (trigram-regenerated Voynich keeps word structure; glyph-shuffled Voynich keeps line make-up).
    Chant symbols are relabelled at random before the search (the search must rediscover the scale).
Also reported: the learned Voynich order, and its agreement between the two halves / transcriptions.
"""
import sys, os, random, math
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_corpora as vc, vlib

N = 5
TOPK = 16


def encode(units, alpha):
    ai = {s: i for i, s in enumerate(alpha)}
    seqs = []
    for u in units:
        for L in u['lines']:
            s = [ai.get(x, -1) for w in L for x in w]
            seqs.append(np.array(s, dtype=np.int64))
    # windows of N symbols inside a line, all symbols in alphabet
    W = []
    for s in seqs:
        if len(s) < N:
            continue
        v = np.lib.stride_tricks.sliding_window_view(s, N)
        v = v[(v >= 0).all(1)]
        if len(v):
            W.append(v)
    return np.concatenate(W) if W else np.zeros((0, N), dtype=np.int64)


def tp_score(W, rank):
    r = rank[W]                       # windows -> ranks
    d = np.diff(r, axis=1)
    keep = (d != 0).any(1)
    r, d = r[keep], d[keep]
    base = 2 * len(rank) + 1
    key = np.zeros(len(d), dtype=np.int64)
    for j in range(d.shape[1]):
        key = key * base + (d[:, j] + len(rank))
    _, c1 = np.unique(key, return_counts=True)
    _, c2 = np.unique(key * len(rank) + r[:, 0], return_counts=True)
    return float((c1 * (c1 - 1) // 2).sum() - (c2 * (c2 - 1) // 2).sum())


def rand_z(W, rank_obs, k, rng, R=300):
    obs = tp_score(W, rank_obs)
    nulls = [tp_score(W, rng.permutation(k)) for _ in range(R)]
    return obs, float(np.mean(nulls)), float((obs - np.mean(nulls)) / (np.std(nulls) + 1e-12)), float(np.mean(np.array(nulls) >= obs))


def hill(W, k, rng, iters=3000):
    rank = rng.permutation(k); best = tp_score(W, rank)
    for it in range(iters):
        i, j = rng.choice(k, 2, replace=False)
        rank[i], rank[j] = rank[j], rank[i]
        s = tp_score(W, rank)
        if s >= best:
            best = s
        else:
            rank[i], rank[j] = rank[j], rank[i]
    return rank, best


def alphabet(units):
    c = Counter(x for u in units for L in u['lines'] for w in L for x in w)
    return [s for s, _ in c.most_common(TOPK)]


if __name__ == '__main__':
    rng = np.random.default_rng(int(os.environ.get('SEED', 5)))
    C = vc.all_corpora()
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    ITERS = int(os.environ.get('ITERS', 3000))
    res = {}
    # X1: chant with the true scale
    ch = C['chant']
    al = sorted(alphabet(ch))      # a < b < ... staff order
    W = encode(ch, al)
    o, m, z, p = rand_z(W, np.arange(len(al)), len(al), rng)
    res['X1_chant_true_order'] = {'alphabet': al, 'TP': o, 'null': m, 'z': z, 'p': p}
    print(f"X1 chant true staff order: TP={o:.0f} null={m:.0f} z={z:.1f} p={p}")
    # relabel chant symbols at random for the blind search
    perm = list(al); random.Random(3).shuffle(perm); rl = dict(zip(al, perm))
    C['chant-relabelled'] = [dict(u, lines=[[tuple(rl.get(x, x) for x in w) for w in L] for L in u['lines']]) for u in ch]
    del C['chant']
    for name, units in C.items():
        if only and name not in only:
            continue
        al = alphabet(units); k = len(al)
        A = [u for i, u in enumerate(units) if i % 2 == 0]; B = [u for i, u in enumerate(units) if i % 2 == 1]
        WA, WB = encode(A, al), encode(B, al)
        rank, best = hill(WA, k, rng, ITERS)
        o, m, z, p = rand_z(WB, rank, k, rng)
        rankB, _ = hill(WB, k, rng, ITERS)
        # agreement of learned orders between halves (Spearman, sign-free: a scale may run either way)
        rho = np.corrcoef(rank, rankB)[0, 1]
        order = [al[i] for i in np.argsort(rank)]
        res[name] = {'alphabet': al, 'learned_order': order, 'heldout_TP': o, 'null': m, 'z': z, 'p': p,
                     'half_agreement_abs_rho': float(abs(rho)), 'n_windows_B': int(len(WB))}
        print(f"{name:22s} held-out z={z:6.1f} (TP {o:.0f} vs {m:.0f}) |rho halves|={abs(rho):.2f}  order={''.join(order)}")
        if name == 'chant-relabelled':
            inv = {v: kk for kk, v in rl.items()}
            print('   recovered chant order in true letters:', ''.join(inv[x] for x in order))
            res[name]['recovered_true_letters'] = ''.join(inv[x] for x in order)
    vlib.save('v5_cycle3' + ('_' + '_'.join(only) if only else '') + os.environ.get('SEED', ''), res)
