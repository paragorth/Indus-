"""R1 cycle 4: re-test every real replicated partition survivor (families a and g, cycle 1) on outside data.
  Linear A: train on Haghia Triada documents, test on all other sites.
  Proto-Elamite: train on Susa tablets, test on all other proveniences.
  Voynich: train on the ZL3b transliteration (all pages), test on the IT2a transliteration (all pages)
           -- same manuscript read twice, so this is only transcription-robustness.
Same score as stage 2 (matched-null excess, jackknife SE). Bonferroni over survivors per script.
Control: 200 fresh random partitions (k drawn from the survivors' k) through the identical test."""
import json, os, glob
import numpy as np
import r1_lib as L, r1_seq as S

out = {}
rng = np.random.default_rng(44)
for script in ['linear_a', 'proto_elamite', 'voynich']:
    name = f'c4_outside_{script}.json'
    if L.done(name):
        out[script] = json.load(open(os.path.join(L.RES, name)))
        continue
    if script == 'voynich':
        a = L.load('voynich'); b = L.load_voynich('IT2a')
        for d in a['docs']:
            d['split'] = 'B'
        for d in b['docs']:
            d['split'] = 'C'; d['id'] = 'IT2a:' + d['id']
        c = {'name': 'voynich', 'docs': a['docs'] + b['docs']}
    else:
        c = L.load(script)
        home = 'Haghia Triada' if script == 'linear_a' else 'Susa'
        for d in c['docs']:
            d['split'] = 'B' if (d['site'] or '').startswith(home) else 'C'
    enc0 = None
    surv = []
    for fam in ('a', 'g'):
        d = json.load(open(os.path.join(L.RES, f'c1_{fam}_{script}_real0.json')))
        signs = d['signs']
        for r in d['stage2']:
            if r.get('replicated') and 'assign' in r:
                surv.append((fam, r['k'], dict(zip(signs, r['assign']))))
    enc = S.Enc(c)
    sc = S.PartScorer(c, enc, ('B',), ('C',))

    def test(k, amap):
        assign = np.array([amap.get(s, int(rng.integers(0, k))) for s in enc.part])
        lab, K = S.make_lab(enc, assign, k)
        nl = [S.make_lab(enc, S.matched_null(assign, rng), k)[0] for _ in range(12)]
        ex = lambda drop: sc.score(lab, K, drop) - np.mean([sc.score(x, K, drop) for x in nl], 0)
        e = ex(None)
        return S.z_jack(e, [ex(g) for g in range(sc.G)]), float(e.sum() / sc.ntok.sum())

    thr = S.bonf_z(len(surv))
    res = [{'fam': f, 'k': k, **dict(zip(('z', 'gain'), test(k, m)))} for f, k, m in surv]
    ks = [k for _, k, _ in surv]
    ctrl = [test(k, {s: int(rng.integers(0, k)) for s in enc.part})[0] for k in rng.choice(ks, 200)]
    o = {'n_train_docs': sum(d['split'] == 'B' for d in c['docs']), 'n_test_docs': sum(d['split'] == 'C' for d in c['docs']),
         'n_survivors': len(surv), 'thr': thr,
         'n_pass': {f: sum(r['z'] > thr for r in res if r['fam'] == f) for f in ('a', 'g')},
         'n_by_fam': {f: sum(r['fam'] == f for r in res) for f in ('a', 'g')},
         'median_z': {f: float(np.median([r['z'] for r in res if r['fam'] == f])) if any(r['fam'] == f for r in res) else None for f in ('a', 'g')},
         'median_gain': {f: float(np.median([r['gain'] for r in res if r['fam'] == f])) if any(r['fam'] == f for r in res) else None for f in ('a', 'g')},
         'control_random_pass': int(sum(z > thr for z in ctrl)), 'control_random_median_z': float(np.median(ctrl)),
         'results': res}
    L.save(name, o)
    out[script] = o
    print(script, {k: v for k, v in o.items() if k != 'results'}, flush=True)
