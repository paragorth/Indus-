"""v26 EVOLVE THE SCRIBE: adversarial evolution of the generator genome.

Per corpus (V = Voynich ZL3b, LA = Latin herbal, IT = Italian herbal, PL = text written by a planted genome):
fit the scribe's tables on half the pages (split_half), then evolve genomes so that their forgeries of those
pages fool page-feature discriminators that are retrained for every genome (5-fold ridge, CV by page) AND a
growing archive of frozen discriminators trained on earlier generations' elite forgeries (adversarial memory).
Fitness (lower is better) = 0.5 * fresh ridge AUC + 0.5 * mean archive AUC + LAMBDA * genome bits.
Massive random guessing first (NRAND random genomes), then (mu + lambda) evolution with crossover and
mutation; elites are re-forged with a new seed every generation (running mean). Restarts are independent.
Checkpoints: data/v26_ckpt/evo_<corpus>_r<restart>.json (resumable per generation).
Usage: python3 v26_evo.py CORPUS RESTART [NRAND GENS LAMBDA_OFFSPRING]
"""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v26_lib import *
from multiprocessing import Pool

LAMBDA = 0.0015          # AUC per genome bit (a mechanism with a 5-bit value costs ~0.009)
MU = 12
PLANT = dict(OFF, sec=1, pos=1, pi_copy=0.15, cwin=3, w_far=2.0, edit=0.6, b_cs=1.5, b_dr=4.0, b_mg=-1.0)


def planted_corpus():
    p = os.path.join(CK, 'planted_corpus.json')
    if os.path.exists(p): return json.load(open(p))
    C = corpus('V'); S = Scribe(C)
    P = S.forge(C, PLANT, random.Random(2626))
    json.dump(P, open(p, 'w'))
    return P


def get_corpus(name):
    return planted_corpus() if name == 'PL' else corpus(name)


CSU = {'V': ('C', 'S'), 'VI': ('C', 'S'), 'PL': ('C', 'S'), 'LA': ('c', 's'), 'IT': ('c', 's')}
_W = {}


def winit(name):
    import warnings; warnings.filterwarnings('ignore')
    C = get_corpus(name); tr, te = split_half(C)
    S = Scribe(tr, CSU[name]); FZ = FastFZ(tr)
    Xr, keys = FZ.matrix(tr)
    _W.update(name=name, tr=tr, S=S, FZ=FZ, Xr=Xr, keys=keys)


def forge_feats(args):
    g, seed = args
    W = _W
    F = W['S'].forge(W['tr'], g, random.Random(seed))
    Xf, _ = W['FZ'].matrix(F, W['keys'])
    return Xf


def main():
    name = sys.argv[1]; R = int(sys.argv[2])
    NRAND = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    GENS = int(sys.argv[4]) if len(sys.argv) > 4 else 18
    LAM = int(sys.argv[5]) if len(sys.argv) > 5 else 24
    ck = f'evo_{name}_r{R}.json'
    state = load(ck)
    winit(name)
    Xr = _W['Xr']; n = len(Xr)
    fold = folds(n, 5, 100 + R)
    rng = random.Random(7000 + R * 31 + sum(map(ord, name)))
    archive = []        # list of per-fold model lists
    t0 = time.time()

    def score(Xf):
        a, _ = cv_ridge(Xr, Xf, fold)
        # an archive critic that is fooled 'too well' (AUC < 0.5) still separates: use the symmetric AUC
        arc = float(np.mean([max(x, 1 - x) for x in (archive_auc(m, Xr, Xf, fold) for m in archive[-12:])])) if archive else a
        return a, arc

    pool = Pool(2, initializer=winit, initargs=(name,))
    if state is None:
        state = dict(gen=-1, pop=[], log=[], nevals=0)
        gs = [random_genome(rng, rng.choice([0.15, 0.3, 0.5])) for _ in range(NRAND)]
        seeds = [rng.getrandbits(30) for _ in gs]
        Xs = pool.map(forge_feats, list(zip(gs, seeds)))
        ev = []
        for g, X in zip(gs, Xs):
            a, arc = score(X); ev.append(dict(g=g, auc=[a], arc=arc, bits=bits(g)))
        state['random'] = [dict(auc=e['auc'][0], bits=e['bits'], g=gstr(e['g'])) for e in ev]
        state['nevals'] += len(gs)
        ev.sort(key=lambda e: np.mean(e['auc']) + LAMBDA * e['bits'])
        state['pop'] = ev[:MU]
        # seed the archive with the best random genomes' forgeries
        for e, X in sorted(zip(ev, Xs), key=lambda t: t[0]['auc'][0])[:2]:
            pass
        state['gen'] = 0
        save(ck, state)
        print(f'{name} r{R} random {NRAND} done; best {state["pop"][0]["auc"][0]:.3f} {gstr(state["pop"][0]["g"])} '
              f'[{time.time() - t0:.0f}s]', flush=True)
    # rebuild archive lazily from stored elite genomes (re-forged; archive is not stored in the checkpoint)
    for g in state.get('arc_genomes', [])[-12:]:
        X = forge_feats((g, rng.getrandbits(30)))
        _, models = cv_ridge(Xr, X, fold, keep=True); archive.append(models)
    while state['gen'] < GENS:
        pop = state['pop']
        # offspring
        kids = []
        for _ in range(LAM):
            if rng.random() < 0.4:
                a, b = rng.sample(pop[:max(4, len(pop))], 2); c = crossover(a['g'], b['g'], rng)
                c = mutate(c, rng, 0.08)
            else:
                c = mutate(rng.choice(pop[:max(4, len(pop) // 2)])['g'], rng, 0.15)
            kids.append(c)
        allg = [e['g'] for e in pop] + kids
        seeds = [rng.getrandbits(30) for _ in allg]
        Xs = pool.map(forge_feats, list(zip(allg, seeds)))
        state['nevals'] += len(allg)
        ev = []
        for i, (g, X) in enumerate(zip(allg, Xs)):
            a, arc = score(X)
            if i < len(pop):
                e = pop[i]; e['auc'] = (e['auc'] + [a])[-4:]; e['arc'] = arc
            else:
                e = dict(g=g, auc=[a], arc=arc, bits=bits(g))
            e['fit'] = 0.5 * float(np.mean(e['auc'])) + 0.5 * e['arc'] + LAMBDA * e['bits']
            ev.append((e, X))
        ev.sort(key=lambda t: t[0]['fit'])
        state['pop'] = [e for e, _ in ev[:MU]]
        # adversary update: discriminators trained on this generation's two best forgeries join the archive
        for e, X in ev[:2]:
            _, models = cv_ridge(Xr, X, fold, keep=True); archive.append(models)
            state.setdefault('arc_genomes', []).append(e['g'])
        b = state['pop'][0]
        state['gen'] += 1
        state['log'].append(dict(gen=state['gen'], best_auc=float(np.mean(b['auc'])), best_arc=b['arc'], best_fit=b['fit'],
                                 med_auc=float(np.median([np.mean(e['auc']) for e in state['pop']])),
                                 best=gstr(b['g']), bits=b['bits'], nevals=state['nevals']))
        save(ck, state)
        print(f"{name} r{R} gen {state['gen']} best auc {np.mean(b['auc']):.3f} arc {b['arc']:.3f} bits {b['bits']:.0f} "
              f"| {gstr(b['g'])} [{time.time() - t0:.0f}s]", flush=True)
    pool.close()


if __name__ == '__main__':
    main()
