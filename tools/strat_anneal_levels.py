"""Merge-level check for tools/strat_anneal.py: does the annealed reading hold at seq_raw and seq_strong?

For each level: (1) transfer the best seq readings (A: held-out-annealed, B: cross-city-annealed) sign by sign
(Wells numbers; signs outside that level's top 80 fall to OTHER) and score them on that level's held-out texts;
(2) a reduced anneal (A objective) and its permuted-fact control at that level; (3) the raw / length / random
baselines. Checkpointed like the main run.
Usage: python3 tools/strat_anneal_levels.py [--restarts 8] [--steps 1500] [--procs 4]
Writes data/derived/strat_anneal_levels.txt/.json.
"""
import sys, os, json, time, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import strat_anneal as sa


def transfer(best_map, S):
    a = np.full(sa.NTOP + 1, sa.LABELS.index('UNKNOWN'))
    a[sa.NTOP] = sa.OTHER
    hit = 0
    for i, w in enumerate(S['top']):
        lab = best_map.get(str(w))
        if lab is not None:
            a[i] = sa.LABELS.index(lab); hit += 1
    return a, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--restarts', type=int, default=8); ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--procs', type=int, default=4); ap.add_argument('--seed', type=int, default=11)
    A = ap.parse_args()
    t0 = time.time()
    main_json = json.load(open(os.path.join(sa.ROOT, 'data/derived/strat_anneal.json')))
    sa.CKPT['_path'] = os.path.join(sa.ROOT, 'data/derived/strat_anneal_ckpt_levels.json')
    if os.path.exists(sa.CKPT['_path']):
        old = json.load(open(sa.CKPT['_path'])); old.pop('_path', None); sa.CKPT.update(old)
    out = []; J = dict(restarts=A.restarts, steps=A.steps, levels={})
    P = lambda *a: (print(*a, flush=True), out.append(' '.join(str(x) for x in a)))
    P(f'strat_anneal_levels: reduced anneal {A.restarts} x {A.steps}; main seq run was {main_json["restarts"]} x {main_json["steps"]}')
    for level in ('seq_raw', 'seq_strong', 'seq'):
        T = sa.load(level); held = np.array([t['held'] for t in T]); big = np.array([t['big'] for t in T])
        S = sa.build(T, sa.NTOP)
        FS = sa.FactSet(T, S, big, held)
        FS.facts['region'] = sa.FactSet(T, S, ~held, held).facts.get('region')
        if FS.facts['region'] is None: del FS.facts['region']
        FSp = sa.FactSet(T, S, big, held, perm_seed=7)
        FSp.facts['region'] = sa.FactSet(T, S, ~held, held, perm_seed=7).facts.get('region')
        if FSp.facts['region'] is None: del FSp.facts['region']
        raw_s = FS.score(None, raw=True); len_s = FS.score_length_only()
        rnd = np.random.RandomState(0)
        rand = [FS.score(np.append(rnd.randint(0, sa.L, sa.NTOP), sa.OTHER)) for _ in range(50)]
        aA, hA = transfer(main_json['best_A'], S); aB, hB = transfer(main_json['best_B'], S)
        sA, dA = FS.score(aA, detail=True); sB, dB = FS.score(aB, detail=True)
        P(f'\n[{level}] texts={len(T)} held-out={held.sum()}; top-80 overlap with seq top-80: {hA}/80')
        P(f'  baselines: raw {raw_s:.3f}  length-only {len_s:.3f}  random median {np.median(rand):.3f} max {max(rand):.3f}')
        P(f'  transferred best A (seq): {sA:.3f}  per fact ' + str({f: d["gain_bits"] for f, d in dA.items()}))
        P(f'  transferred best B (seq): {sB:.3f}  per fact ' + str({f: d["gain_bits"] for f, d in dB.items()}))
        rec = dict(n_texts=len(T), n_heldout=int(held.sum()), overlap=hA, raw=raw_s, length=len_s,
                   random_median=float(np.median(rand)), random_max=float(max(rand)), transfer_A=sA, transfer_B=sB,
                   transfer_A_detail=dA, transfer_B_detail=dB)
        if level != 'seq':
            resA = sa.run_search(S, [FS], A.restarts, A.steps, A.seed, A.procs, 'A_' + level)
            resC = sa.run_search(S, [FSp], A.restarts, A.steps, A.seed + 1000, A.procs, 'C1_' + level)
            bA = np.array(resA[0][1]); sAn, dAn = FS.score(bA, detail=True); sCn = FSp.score(np.array(resC[0][1]))
            P(f'  reduced anneal at this level: best {sAn:.3f} (top range {resA[-1][0]:.3f}..{resA[0][0]:.3f}); '
              f'permuted-fact control best {sCn:.3f}; gap {sAn - sCn:+.3f}')
            P(f'  per fact: ' + str({f: d["gain_bits"] for f, d in dAn.items()}))
            # partition agreement with the seq best-A (over shared signs)
            shared = [i for i, w in enumerate(S['top']) if str(w) in main_json['best_A']]
            ref = np.array([sa.LABELS.index(main_json['best_A'][str(S['top'][i])]) for i in shared]); cur = bA[shared]
            pairs = [(i, j) for i in range(len(shared)) for j in range(i + 1, len(shared))]
            same = np.mean([(ref[i] == ref[j]) == (cur[i] == cur[j]) for i, j in pairs])
            rp = np.mean([np.mean([(ref[i] == ref[j]) == (x[i] == x[j]) for i, j in pairs])
                          for x in [np.random.RandomState(k).randint(0, sa.L, len(shared)) for k in range(20)]])
            P(f'  partition agreement with seq best A over {len(shared)} shared signs: {same:.3f} (random {rp:.3f})')
            rec.update(anneal_best=sAn, anneal_detail=dAn, anneal_scores=[r[0] for r in resA], permuted_best=sCn,
                       permuted_scores=[r[0] for r in resC], gap=sAn - sCn, partition_agreement=float(same), partition_random=float(rp),
                       best=dict(zip([str(w) for w in S['top']], [sa.LABELS[x] for x in bA[:sa.NTOP]])))
        J['levels'][level] = rec
        P(f'  elapsed {time.time() - t0:.0f}s')
    open(os.path.join(sa.ROOT, 'data/derived/strat_anneal_levels.txt'), 'w').write('\n'.join(out) + '\n')
    json.dump(J, open(os.path.join(sa.ROOT, 'data/derived/strat_anneal_levels.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
