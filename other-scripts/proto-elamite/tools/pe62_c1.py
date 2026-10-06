"""pe62 cycle 1: can the masked class be read back from what the clerk did write?
usage: python3 pe62_c1.py CORPUS MODE N_HYP
  CORPUS: PE | PC | U3   (PC and U3 subsampled to PE size by labelled entries)
  MODE:   real | lshuf<k> (global label shuffle on training tablets) | wshuf<k> (within-tablet shuffle)
          | nshuf<k> (numerals shuffled across entries, tablet system shuffled across tablets)
          | plantA<k> (implicit class carried by a random name sign) | plantB<k> (implicit class carried by x5 numbers)
          | blind (real, number groups excluded)
Writes data/pe62_ckpt/c1_<CORPUS>_<MODE>.json"""
import sys, os, json, random, copy, time, re
import numpy as np
from collections import Counter
import pe62_common as C

corpus, mode, n_hyp = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = random.Random('pe62' + corpus + mode)
t0 = time.time()
logf = open(os.path.join(C.CK, 'c1_%s_%s.log' % (corpus, mode)), 'w')


def log(m):
    logf.write('%.0fs %s\n' % (time.time() - t0, m)); logf.flush()


PE = C.build_pe()
n_lab_pe = sum(1 for t in PE for e in t['entries'] if e['label'])
if corpus == 'PE':
    T = PE
elif corpus == 'PC':
    T = C.match_size(C.build_pc(), n_lab_pe, 'pe62pc')
else:
    T = C.match_size(C.build_u3(), n_lab_pe, 'pe62u3')
T = copy.deepcopy(T)
truth_plant = None
m = re.match(r'([a-z]+?)(\d*)$', mode)
kind, rep = m.group(1), m.group(2)
prng = random.Random('pe62plant' + corpus + mode)

if kind == 'nshuf':
    allE = [e for t in T for e in t['entries']]
    pool = [(e['nums'], e['sys'], e['va'], e['vb']) for e in allE]
    prng.shuffle(pool)
    for e, p in zip(allE, pool):
        e['nums'], e['sys'], e['va'], e['vb'] = p
    ts = [t['tsys'] for t in T]
    prng.shuffle(ts)
    for t, s in zip(T, ts):
        t['tsys'] = s
if kind == 'plantA':
    occ = Counter(); tabs_of = {}
    for t in T:
        for e in t['entries']:
            if e['label']:
                for s in set(e['name']):
                    occ[s] += 1; tabs_of.setdefault(s, set()).add(t['id'])
    cand = sorted(s for s in occ if 25 <= occ[s] <= 120 and len(tabs_of[s]) >= 5)
    X = prng.choice(cand)
    truth_plant = {'sign': X, 'n': occ[X]}
    for t in T:
        for e in t['entries']:
            if e['label'] and X in e['name']:
                e['label'] = 'PLANT'
if kind == 'plantB':
    n = 0
    for t in T:
        for e in t['entries']:
            if e['label'] and prng.random() < 0.10:
                e['label'] = 'PLANT'
                e['nums'] = [[k * 5, c] for k, c in e['nums']]
                e['va'] = e['va'] + 2.32 if e['va'] > -1 else e['va']
                e['vb'] = e['vb'] + 2.32 if e['vb'] > -1 else e['vb']
                n += 1
    truth_plant = {'n': n}

tr_tabs = [i for i, t in enumerate(T) if not C.hsplit(t['id'], 'pe62ho', 0.30)]
trset = set(tr_tabs)
override = None
if kind in ('lshuf', 'wshuf'):
    override = {}
    if kind == 'lshuf':
        keys = [(ti, ei) for ti in tr_tabs for ei, e in enumerate(T[ti]['entries']) if e['label']]
        vals = [T[ti]['entries'][ei]['label'] for ti, ei in keys]
        prng.shuffle(vals)
        override = dict(zip(keys, vals))
    else:
        for ti in tr_tabs:
            keys = [(ti, ei) for ei, e in enumerate(T[ti]['entries']) if e['label']]
            vals = [T[ti]['entries'][ei]['label'] for _, ei in keys]
            prng.shuffle(vals)
            override.update(dict(zip(keys, vals)))
rows = C.featurise(T, override)
D = C.Design(rows)
classes = sorted({r[3] for r in rows if r[3]})
lab_idx = [i for i, r in enumerate(rows) if r[3]]
tr_idx = [i for i in lab_idx if rows[i][0] in trset]
ho_idx = [i for i in lab_idx if rows[i][0] not in trset]
# true held-out labels (rows built with override only touch training tablets)
y_all = [r[3] for r in rows]
allowed = [g for g in C.GROUPS if g not in C.NUMBER_GROUPS] if kind == 'blind' else C.GROUPS
log('%s %s rows %d lab %d tr %d ho %d classes %s' % (corpus, mode, len(rows), len(lab_idx), len(tr_idx),
                                                      len(ho_idx), classes))
R = C.search(D, rows, tr_idx, ho_idx, y_all, classes, n_hyp, rng, allowed=allowed, log=log)
yho = [y_all[i] for i in ho_idx]
s = C.score(R['P_ho'], yho, classes, R['prior'])
if R['P_ho150'] is not None:
    s['ho150'] = C.score(R['P_ho150'], yho, classes, R['prior'])
maj = Counter(y_all[i] for i in tr_idx).most_common(1)[0][0]
s['maj_acc'] = sum(1 for v in yho if v == maj) / len(yho)
out = {'corpus': corpus, 'mode': mode, 'n_hyp': n_hyp, 'classes': classes, 'ho': s, 'plant': truth_plant,
       'top': [(b, hp, sc) for b, hp, sc in R['top']], 'cv_bits_q': list(np.percentile(R['all_bits'], [50, 90, 99, 100])),
       'n_tr': len(tr_idx), 'n_ho': len(ho_idx), 'secs': time.time() - t0}
if truth_plant:
    k = classes.index('PLANT')
    pred = R['P_ho'].argmax(1)
    yi = np.array([classes.index(v) for v in yho])
    tp = int(((pred == k) & (yi == k)).sum())
    out['plant_rec'] = {'recall': tp / max(1, int((yi == k).sum())), 'precision': tp / max(1, int((pred == k).sum())),
                        'base': float((yi == k).mean()), 'n_ho': int((yi == k).sum()),
                        'mean_p_true': float(R['P_ho'][yi == k, k].mean()) if (yi == k).any() else None,
                        'mean_p_other': float(R['P_ho'][yi != k, k].mean())}
# group usage among top hypotheses
out['group_use'] = dict(Counter(g for _, hp, _ in R['top'] for g in hp['groups']))
if kind in ('real', 'blind'):
    out['P_ho'] = R['P_ho'].round(4).tolist()
    out['ho_rows'] = [(rows[i][0], rows[i][1]) for i in ho_idx]
C.jdump(out, os.path.join(C.CK, 'c1_%s_%s.json' % (corpus, mode)))
log('done %s' % json.dumps(s))
