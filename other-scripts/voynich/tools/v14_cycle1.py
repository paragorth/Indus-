"""v14 cycle 1: device fit of slot-filler frequencies + slot independence.
Corpora: two planted dice corpora (positive controls), verbose Latin and verbose
Italian (negative), Voynich ZL3b paragraph text, and the Voynich with every
slot's fillers shuffled across tokens (independence calibration)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
from v14_lib import *
from multiprocessing import Pool

NNULL = 40
SIGMAS = (0.0, 0.1, 0.3)

def v7model(key):
    d = json.load(open(os.path.join(DATA, 'derived', 'v7_slotmodels.json')))[key]
    return {'order': d['order'], 'cuts': d['cuts'], 'coverage': d['coverage']}

def get_tokens(name):
    if name == 'planted_mixed':
        L, truth = planted_dice(design='mixed'); m = learn_model(L, 4, 'planted_mixed')
    elif name == 'planted_table':
        L, truth = planted_dice(design='table', seed=12); m = learn_model(L, 4, 'planted_table')
    elif name == 'latin_verbose':
        L, truth = latin_verbose(), None; m = v7model('latin_verbose_K4')
    elif name == 'italian_verbose':
        L, truth = italian_verbose(), None; m = learn_model(L, 4, 'italian_verbose')
    elif name in ('voynich_ZL3b', 'voynich_shuffled'):
        L, truth = voy_lines('ZL3b'), None; m = v7model('voynich_ZL3b_K4')
    toks = parse(L, m)
    if name == 'voynich_shuffled': toks = shuffle_slots(toks, 4, seed=5)
    return toks, m, truth

def analyse(toks, K, label, nnull=NNULL, seed=0, indep=True, strata=None):
    rng = np.random.default_rng(seed); prng = random.Random(seed)
    cnts = slot_counts(toks, K)
    out = {'label': label, 'n': len(toks), 'slots': []}
    for k in range(K):
        c = cnts[k]; N = sum(c.values())
        sc = slot_scores(c, prng)
        rec = {'nfill': len(c), 'top': c.most_common(10), 'scores': sc,
               'ident': parsimony(sc),
               'z_full': {d: zof(sc, d, 'full') for d in DEV_ORDER},
               'z_core': {d: zof(sc, d, 'core') for d in DEV_ORDER}, 'null': {}}
        for s in SIGMAS:
            zf, zc, idn = [], [], []
            for _ in range(nnull):
                cn = jitter_counts(c, s, N, rng)
                ss = slot_scores(cn, prng)
                zf.append({d: zof(ss, d, 'full') for d in DEV_ORDER})
                zc.append({d: zof(ss, d, 'core') for d in DEV_ORDER})
                idn.append(parsimony(ss))
            rec['null'][s] = {'z_full': zf, 'z_core': zc, 'ident': idn}
        out['slots'].append(rec)
    if indep:
        out['indep'] = independence(toks, K, nperm=200, strata=strata, seed=seed)
    return out

def summarise_slot(rec):
    """search-corrected percentile: share of nulls whose best device fit is at least as good."""
    res = {}
    for which in ('z_full', 'z_core'):
        real_best = min(rec[which].values())
        bestd = min(rec[which], key=rec[which].get)
        pc = {}
        for s, nl in rec['null'].items():
            nb = [min(z.values()) for z in nl[which]]
            pc[s] = (sum(1 for x in nb if x <= real_best) + 1) / (len(nb) + 1)
            pc[('med', s)] = float(np.median([x for x in nb if np.isfinite(x)])) if any(np.isfinite(nb)) else np.inf
        res[which] = (bestd, real_best, pc)
    return res

def run(name):
    if load('c1_' + name) is not None: return name
    t = time.time()
    toks, m, truth = get_tokens(name)
    r = analyse(toks, 4, name)
    r['model'] = m; r['truth'] = truth
    save('c1_' + name, r)
    print(name, 'done', round(time.time() - t), flush=True)
    return name

NAMES = ['planted_mixed', 'planted_table', 'voynich_ZL3b', 'latin_verbose', 'italian_verbose', 'voynich_shuffled']

if __name__ == '__main__':
    with Pool(2) as p:
        list(p.imap_unordered(run, NAMES))
