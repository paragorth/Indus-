"""S-DARK-34 cycle 2: determinism table for OBJECT-FACT targets (emblem class, material, boss class, shape class)
on seals, plus rule extraction for the strongest target of cycles 1-2 (depth-3 tree, rules in words, each rule's
held-out precision with a < 3/n bound for zero exceptions).
Usage: python3 tools/dark_loop34_c2.py <seq_raw|seq_strong|seq_all>"""
import sys, json
sys.argv = [sys.argv[0], '2'] + sys.argv[1:]
from dark_loop34 import *
from sklearn.tree import export_text

seals = [o for o in OBJ if o['ot'] == 'SEAL']
train = [o for o in seals if o['big']]
sites = [o for o in seals if not o['big']]
im77 = [o for o in NEW if o['ot'] == 'SEAL']
log = open(SP + f'loop34_c2_{LV}.txt', 'w')
def P(s): print(s); log.write(s + '\n'); log.flush()
P(f'# S-DARK-34 cycle 2, level {LV}: train seals {len(train)}, held-out site seals {len(sites)}, IM77-only seals {len(im77)}')
for f in ('emblem', 'material', 'boss', 'shape'):
    P(f'  train coverage {f}: {sum(1 for o in train if o[f] is not None)}; classes {collections.Counter(o[f] for o in train if o[f] is not None).most_common(8)}')
rows = {}
for t in ('emblem', 'material', 'boss', 'shape'):
    P(f'--- target {t}')
    tests = {'sites': sites, 'im77': im77} if t == 'emblem' else {'sites': sites}
    r_text, _ = evaluate(t, train, tests, use_facts=False); P('TEXT-ONLY  ' + fmt(r_text))
    r_full, _ = evaluate(t, train, {'sites': sites}, use_facts=True); P('TEXT+FACTS ' + fmt(r_full))
    r_ck, _ = evaluate(t, train, tests, use_facts=False, plant='checksum', report_feats=False); P('PLANT-CK   ' + fmt(r_ck))
    r_cs, _ = evaluate(t, train, tests, use_facts=False, plant='classsign', report_feats=False); P('PLANT-CS   ' + fmt(r_cs))
    r_sh, _ = evaluate(t, train, tests, use_facts=False, shuffle=True, report_feats=False); P('SHUFFLED   ' + fmt(r_sh))
    rows[t] = dict(text=r_text, full=r_full, checksum=r_ck, classsign=r_cs, shuffled=r_sh)
json.dump(rows, open(SP + f'loop34_c2_{LV}.json', 'w'), indent=1, default=str)

# ---------------- rule extraction for the strongest sign target (last_sign / closer) and for emblem
def rules(tname, use_facts, depth=3, min_leaf=15):
    P(f'=== rules for {tname} (depth {depth} tree, min leaf {min_leaf}, facts={use_facts})')
    tr = [o for o in train if TARGETS[tname][1](o)]
    te = {'sites': [o for o in sites if TARGETS[tname][1](o)], 'im77': [o for o in (NEW if tname != 'emblem' else im77) if TARGETS[tname][1](o)]}
    ytr, etr = labels_for(tr, tname)
    ytes = {k: labels_for(v, tname) for k, v in te.items()}
    ytr, others = reduce_labels(ytr, [ytes[k][0] for k in te]); ytes = {k: (others[i], ytes[k][1]) for i, k in enumerate(te)}
    V = Vec(tr, tname, use_facts); Xtr = V.matrix(tr, etr)
    T = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=0).fit(Xtr, ytr)
    P(export_text(T, feature_names=V.names, show_weights=True, max_depth=depth))
    leaf_tr = T.apply(Xtr); pred_tr = T.predict(Xtr)
    for k, objs in te.items():
        if not objs: continue
        y, e = ytes[k]; X = V.matrix(objs, e); leaf = T.apply(X); pred = T.predict(X)
        P(f'  hold-out {k}: n={len(y)} acc={acc(pred, y):.3f} modal={acc([collections.Counter(ytr).most_common(1)[0][0]]*len(y), y):.3f}')
        for lf in sorted(set(leaf_tr)):
            mtr = leaf_tr == lf; ptr = float(np.mean(pred_tr[mtr] == np.array(ytr)[mtr]))
            m = leaf == lf
            if m.sum() == 0 or ptr < 0.85: continue
            n = int(m.sum()); hits = int(np.sum(pred[m] == np.array(y)[m])); exc = n - hits
            bound = f'exceptions 0 of {n} (< 3/{n} = {3/n:.2f})' if exc == 0 else f'exceptions {exc} of {n}'
            P(f'    leaf {lf}: predicts {pred_tr[mtr][0]}; train {int(mtr.sum())} objs precision {ptr:.2f}; {k}: {bound}')
rules('last_sign', False); rules('closer', False); rules('emblem', False); rules('emblem', True)
