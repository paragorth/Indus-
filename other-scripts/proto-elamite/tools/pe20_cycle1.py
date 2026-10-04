"""pe20 cycle 1: calibrate the flock-arithmetic search, then run it blind on PE.

  plant : planted herd corpora with the MDP 17,096+ missing pattern (10 replicates)
  ur3   : Ur III herd lines with known categories (u8, udu-nita2, sila4, kir11, ud5,
          masz2-nita2, masz2, asz2-gar3), full set and 20 PE-sized subsamples
  pe    : MDP 17,096+325+380 (10 herd blocks): 200,000 random assignments, Gibbs
          posterior, and nulls (numbers shuffled across entries; within sign)
Usage: python3 pe20_cycle1.py plant|ur3|pe [n_null]
"""
import sys, time
from pe20_common import *


def ur_truth_names(swap):
    out = []
    for s in UR_SIGNS:
        c, sp = UR_TRUTH[s]
        k = (0 if sp == 'sheep' else 1) ^ int(swap)
        out.append((c.split('|'), k))
    return out


def ur_correct(a):
    best = 0
    for swap in (False, True):
        n = 0
        for x, (cs, k) in zip(a, ur_truth_names(swap)):
            nm = state_name(x)
            if nm != 'X' and nm[:-1] in cs and int(nm[-1]) == k:
                n += 1
        best = max(best, n)
    return best


def ur_coarse(a):
    """Coarse recovery: adult female / male / young (any sex) / and species grouping."""
    best = 0
    for swap in (False, True):
        n = 0
        for x, (cs, k) in zip(a, ur_truth_names(swap)):
            nm = state_name(x)
            if nm == 'X':
                continue
            c = 'Y' if nm.startswith('Y') else nm[:-1]
            want = {'Y' if t.startswith('Y') else t for t in cs}
            if c in want and int(nm[-1]) == k:
                n += 1
        best = max(best, n)
    return best


def chance(fn, K, rng, n=4000):
    return float(np.mean([fn(random_assignment(K, rng)) for _ in range(n)]))


def part_plant():
    rng = np.random.default_rng(11)
    R = [r for r in pe_records() if r[0] == MAIN]
    mask = ~np.isnan(to_matrix(R, PE_SIGNS))
    res = []
    for rep in range(20):
        V = planted(mask, rng)
        sc = Scorer(V)
        v, a = maximise(sc, rng)
        marg, ab, vb = gibbs(sc, 1000, rng, init=a)
        if vb > v:
            v, a = vb, ab
        res.append({'rep': rep, 'score': v, 'best': [state_name(x) for x in a],
                    'correct': species_match(a, PLANT_TRUTH),
                    'collapsed_mode': [(['X'] + CATS)[k] for k in collapse_marg(marg).argmax(1)]})
        print(res[-1], flush=True)
    ch = chance(lambda a: species_match(a, PLANT_TRUTH), 8, rng)
    out = {'reps': res, 'mean_correct': float(np.mean([r['correct'] for r in res])),
           'chance_correct': ch}
    dump(out, os.path.join(DATA, 'pe20_cycle1_plant.json'))
    print('plant mean correct', out['mean_correct'], 'chance', ch)


def part_ur3():
    rng = np.random.default_rng(12)
    U = ur_herd_records()
    V = to_matrix(U, UR_SIGNS)
    out = {'n_records': len(U), 'source': 'CDLI Ur III Girsu + Umma, niga/dead lines dropped'}
    sc = Scorer(V)
    v, a = maximise(sc, rng, restarts=4)
    marg, ab, vb = gibbs(sc, 200, rng, init=a)
    if vb > v:
        v, a = vb, ab
    out['full'] = {'score': v, 'best': dict(zip(UR_SIGNS, [state_name(x) for x in a])),
                   'correct': ur_correct(a), 'coarse': ur_coarse(a),
                   'truth_score': sc.score([1, 2, 4, 3, 7, 8, 10, 0]),
                   'marg': {s: dict(zip(['X'] + CATS, np.round(m, 3).tolist()))
                            for s, m in zip(UR_SIGNS, collapse_marg(marg))}}
    print('full', out['full'], flush=True)
    # PE-sized subsamples: 10 records with >= 4 observed signs
    rich = [i for i in range(len(U)) if np.sum(~np.isnan(V[i])) >= 4]
    out['n_rich'] = len(rich)
    subs = []
    for rep in range(20):
        idx = rng.choice(rich, 10, replace=False)
        sc = Scorer(V[idx], lg=sc.lg if False else None)
        v, a = maximise(sc, rng)
        # null on the same subsample: numbers shuffled within sign
        nulls = [maximise(Scorer(shuffle_within_sign(V[idx], rng)), rng, restarts=3)[0]
                 for _ in range(20)]
        subs.append({'score': v, 'correct': ur_correct(a), 'coarse': ur_coarse(a),
                     'best': [state_name(x) for x in a],
                     'null_within_max': float(np.max(nulls)), 'null_within_mean': float(np.mean(nulls))})
        print(rep, subs[-1], flush=True)
    out['subsamples'] = subs
    out['sub_mean_correct'] = float(np.mean([s['correct'] for s in subs]))
    out['sub_beats_null'] = int(sum(s['score'] > s['null_within_max'] for s in subs))
    out['chance_correct'] = chance(ur_correct, 8, rng)
    out['sub_mean_coarse'] = float(np.mean([s['coarse'] for s in subs]))
    out['chance_coarse'] = chance(ur_coarse, 8, rng)
    dump(out, os.path.join(DATA, 'pe20_cycle1_ur3.json'))
    print('ur3 sub mean correct', out['sub_mean_correct'], 'chance', out['chance_correct'],
          'beats null', out['sub_beats_null'])


def part_pe(n_null=60):
    rng = np.random.default_rng(13)
    R = [r for r in pe_records() if r[0] == MAIN]
    V = to_matrix(R, PE_SIGNS)
    sc = Scorer(V)
    t = time.time()
    scores, best = random_search(sc, 200000, rng, keep=500)
    np.save(os.path.join(CK, 'c1_random_scores.npy'), scores)
    print('random 200k', time.time() - t, 'max', scores.max(), flush=True)
    margs, maxes = [], []
    for ch in range(4):
        marg, a, v = gibbs(sc, 3000, rng, init=best[ch][1])
        margs.append(marg)
        maxes.append((v, a))
    marg = np.mean(margs, 0)
    vmax, amax = max(maxes + [(best[0][0], best[0][1])], key=lambda x: x[0])
    v2, a2 = maximise(sc, rng)
    if v2 > vmax:
        vmax, amax = v2, a2
    print('PE max', vmax, [state_name(x) for x in amax], flush=True)
    nulls = {'all': [], 'within': []}
    for k in range(n_null):
        nulls['all'].append(maximise(Scorer(shuffle_all(V, rng), lg=sc.lg), rng, restarts=3)[0])
        nulls['within'].append(maximise(Scorer(shuffle_within_sign(V, rng), lg=sc.lg), rng, restarts=3)[0])
        if k % 10 == 9:
            print(k, np.max(nulls['all']), np.max(nulls['within']), flush=True)
    real3 = maximise(sc, rng, restarts=3)[0]
    out = {'records': [(r[1], [r[2][s] for s in PE_SIGNS]) for r in R],
           'max_score': vmax, 'max_assign': dict(zip(PE_SIGNS, [state_name(x) for x in amax])),
           'max_score_same_effort': real3,
           'random_pct': np.percentile(scores, [50, 90, 99, 99.9]).tolist(),
           'top20': [(v, [state_name(x) for x in a]) for v, a in best[:20]],
           'marg_full': {s: {state_name(k): round(float(m[k]), 3) for k in range(NSTATE) if m[k] > 0.02}
                         for s, m in zip(PE_SIGNS, marg)},
           'marg_collapsed': {s: dict(zip(['X'] + CATS, np.round(m, 3).tolist()))
                              for s, m in zip(PE_SIGNS, collapse_marg(marg))},
           'null_all': nulls['all'], 'null_within': nulls['within'],
           'p_all': float((1 + sum(x >= real3 for x in nulls['all'])) / (1 + n_null)),
           'p_within': float((1 + sum(x >= real3 for x in nulls['within'])) / (1 + n_null))}
    best_keep = [(v, a) for v, a in best]
    dump({'top': best_keep}, os.path.join(CK, 'c1_top_assign.json'))
    dump(out, os.path.join(DATA, 'pe20_cycle1_pe.json'))
    print('p_all', out['p_all'], 'p_within', out['p_within'])
    print(json.dumps(out['marg_collapsed'], indent=0))


if __name__ == '__main__':
    part = sys.argv[1]
    if part == 'plant':
        part_plant()
    elif part == 'ur3':
        part_ur3()
    else:
        part_pe(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
