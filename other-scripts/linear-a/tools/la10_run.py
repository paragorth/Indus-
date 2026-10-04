#!/usr/bin/env python3
"""LA-10 driver: python3 la10_run.py TARGET SEED [POP GENS SPELLING WDIST]
TARGET: LA (Linear A admin valued types), LB (Linear B DAMOS, random subsample to LA's type count), LAshuf (LA signs
shuffled across types: keeps lengths and unigrams, destroys rules), PLANT1 / PLANT2 (vocabularies of known genomes),
LAnonHT (non-Hagia-Triada LA types; smaller n).
Writes ../data/la10/run_<TARGET>_<SPELL>_w<WDIST>_s<SEED>.json (final population, best genome, ablation) with a
checkpoint beside it.
"""
import sys, os, json, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la10_common as L
import la5_common as C

OUT = os.path.join(L.C.LAD, 'la10'); os.makedirs(OUT, exist_ok=True)

PLANTS = {
    # harmonic, suffixing, identity-OCP, open syllables, no clusters
    'PLANT1': dict(onset0_init=0.25, onset0_med=0.05, cluster=0.0, coda_med=0.1, coda_obs=0.3, coda_fin=0.5, root_len=2.2,
                   harm_copy=0.5, harm_fb=0.3, ocp_id=0.8, ocp_place=0.0, pre_p=0.0, pre_n=3, suf_p=0.5, suf_n=4,
                   aff_2syl=0.2, aff_v=0.3, reuse=0.3, fv_str=0.0),
    # prefixing, clusters and obstruent codas (LB-style echo spelling), place-OCP, final-vowel restriction, no harmony
    'PLANT2': dict(onset0_init=0.3, onset0_med=0.1, cluster=0.3, coda_med=0.35, coda_obs=0.6, coda_fin=0.8, root_len=2.5,
                   harm_copy=0.0, harm_fb=0.0, ocp_id=0.0, ocp_place=0.6, pre_p=0.45, pre_n=3, suf_p=0.05, suf_n=2,
                   aff_2syl=0.3, aff_v=0.4, reuse=0.4, fv_str=0.6),
}
def plant_genome(nm):
    d = dict(PLANTS[nm])
    for c in L.CCAT: d['cw_' + c] = 0.5
    d['cw_H'] = -0.5; d['cw_Z'] = -1.0; d['cw_Q'] = -1.0
    for v in L.VOW: d['vw_' + v] = 0.0; d['fw_' + v] = 0.0
    if nm == 'PLANT2': d['fw_O'] = 2.5; d['fw_A'] = 1.0
    if nm == 'PLANT1': d['vw_O'] = -1.5
    return d

def target_types(nm, n_la):
    la = L.la_types()
    if nm == 'LA': return la
    if nm == 'LAnonHT':
        return L.encode_types(set(w for _, s, w in C.words_of(C.la_docs()) if s != 'Haghia Triada'))
    if nm == 'LAHT':
        return L.encode_types(set(w for _, s, w in C.words_of(C.la_docs()) if s == 'Haghia Triada'))
    if nm == 'LB':
        return sorted(random.Random(4242).sample(L.lb_types(), len(la)))
    if nm == 'LAshuf':
        r = random.Random(99); pool = [a for t in la for a in t]; r.shuffle(pool); out = set(); k = 0
        for t in la: out.add(tuple(pool[k:k + len(t)])); k += len(t)
        return sorted(out)
    if nm.startswith('PLANT'):
        return L.Lang(plant_genome(nm)).vocabulary(len(la), 31337)
    raise SystemExit('unknown target')

if __name__ == '__main__':
    tg = sys.argv[1]; seed = int(sys.argv[2])
    pop = int(sys.argv[3]) if len(sys.argv) > 3 else 160
    gens = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    spell = sys.argv[5] if len(sys.argv) > 5 else 'LB'
    wd = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
    types = target_types(tg, None); n = len(types)
    full, sd = L.calib(types, random.Random(5), 40)
    L.WDIST = wd
    tag = f'{tg}_{spell}_w{wd}_s{seed}'
    ck = os.path.join(OUT, 'ckpt_' + tag + '.json')
    t0 = time.time()
    logf = open(os.path.join(OUT, 'log_' + tag + '.txt'), 'a')
    def log(s): logf.write(s + '\n'); logf.flush()
    P, F, NE, nevals, hist = L.ga(full, sd, n, seed, pop, gens, ck, spell, log=log)
    order = sorted(range(len(P)), key=lambda i: F[i] if NE[i] > 1 else F[i] + 1e3)
    elites = [i for i in order if NE[i] > 1][:8]
    # re-score the 8 elites on 12 common seeds, keep the best
    rs = [(L.evaluate(P[i], full, sd, n, [777 + k for k in range(12)], spell), i) for i in elites]
    rs.sort(); bi = rs[0][1]; best = P[bi]
    ab = L.ablate(best, full, sd, n, 12, 777, spell)
    res = dict(target=tg, seed=seed, spelling=spell, wdist=wd, n=n, nevals=nevals, secs=time.time() - t0,
               target_fp=full, target_sd=sd, best_u=best, best=L.decode(best), best_score=rs[0][0], elites_rescored=rs,
               top20=[L.decode(P[i]) for i in order[:20]], ablation=ab, hist=hist,
               best_fp=L.fingerprint(L.Lang(L.decode(best), spell).vocabulary(n, 777)))
    if tg.startswith('PLANT'):
        tru = L.encode(plant_genome(tg)); res['truth'] = plant_genome(tg)
        res['truth_score'] = L.evaluate(tru, full, sd, n, [777 + k for k in range(12)], spell)
        res['truth_ablation'] = L.ablate(tru, full, sd, n, 12, 777, spell)
    json.dump(res, open(os.path.join(OUT, 'run_' + tag + '.json'), 'w'), indent=1)
    print(tag, 'done', round(res['best_score'], 1), nevals, round(res['secs']))
