"""pe62 cycle 2: fill in what the clerk left out, and see what changes.
usage: python3 pe62_c2.py CORPUS
  1. Retrain the top-15 ensembles of cycle 1 (with numbers: c1_<C>_real; number-blind: c1_<C>_blind) on ALL labelled
     entries; impute a class for every unmarked entry (name-only, bare number, class sign inside).
  2. Number-system / numeral consistency (non-circular: number-blind imputation vs the written numerals).
     bits/entry gained when the observed numeral key (system + log-size bin) is predicted through the imputed class,
     P(key|class) fitted on labelled entries.  Nulls: imputation rows permuted across all unmarked entries (1000x)
     and within tablets (1000x).  Calibration: the same score on masked labelled held-out entries.
  3. Totals: on tablets whose numerals are all ambiguous (N01/N14/N45/N34), choose capacity vs count by the
     blind-imputed class mix; count totals that close, vs the other choice and vs a coin.
  4. Tablet system prediction: CAPT vs CNTT from written classes only vs written + blind-imputed classes
     (tablet-grouped CV logistic).  Planted-null: imputed rows permuted.
Writes data/pe62_ckpt/c2_<CORPUS>.json"""
import sys, os, json, random, math, copy
import numpy as np
from collections import Counter, defaultdict
from fractions import Fraction as Fr
import pe62_common as C
import pe59_lib as L59

corpus = sys.argv[1]
rng = np.random.default_rng(62)
PE = C.build_pe()
n_lab_pe = sum(1 for t in PE for e in t['entries'] if e['label'])
T = PE if corpus == 'PE' else (C.match_size(C.build_pc(), n_lab_pe, 'pe62pc') if corpus == 'PC'
                               else C.match_size(C.build_u3(), n_lab_pe, 'pe62u3'))
rows = C.featurise(T)
D = C.Design(rows)
lab = [i for i, r in enumerate(rows) if r[3]]
unl = [i for i, r in enumerate(rows) if not r[3]]
y = [rows[i][3] for i in lab]
out = {'corpus': corpus}
imp = {}
for mode in ('real', 'blind'):
    c1 = json.load(open(os.path.join(C.CK, 'c1_%s_%s.json' % (corpus, mode))))
    classes = c1['classes']
    P = C.ensemble_predict(D, c1['top'], lab, y, unl, classes)
    imp[mode] = P
    out['imp_' + mode] = {'classes': classes, 'dist': dict(Counter(classes[k] for k in P.argmax(1))),
                          'conf_mean': float(P.max(1).mean()),
                          'n_conf80': int((P.max(1) >= 0.8).sum())}
classes = out['imp_real']['classes']


# ------------------------------------------------------------------ 2. numeral consistency
def key(e):
    return '%s|%d' % (e['sys'], C.bin_(e['va'], 1.5))


E = lambda i: T[rows[i][0]]['entries'][rows[i][1]]  # noqa: E731
keys_lab = [key(E(i)) for i in lab]
allkeys = sorted(set(keys_lab) | {key(E(i)) for i in unl})
K = {k: j for j, k in enumerate(allkeys)}
M = np.full((len(classes), len(allkeys)), 0.5)
for i, k in zip(lab, keys_lab):
    M[classes.index(rows[i][3]), K[k]] += 1
M /= M.sum(1, keepdims=True)
marg = np.full(len(allkeys), 0.5)
for k in keys_lab:
    marg[K[k]] += 1
marg /= marg.sum()


def gain(P, idx):
    kk = np.array([K[key(E(i))] for i in idx])
    pk = (P * M[:, kk].T).sum(1)
    return float((np.log2(pk) - np.log2(marg[kk])).mean())


def perm_nulls(P, idx, n=1000):
    g_all, g_tab = [], []
    tab = np.array([rows[i][0] for i in idx])
    groups = defaultdict(list)
    for j, t in enumerate(tab):
        groups[t].append(j)
    for _ in range(n):
        g_all.append(gain(P[rng.permutation(len(idx))], idx))
        perm = np.arange(len(idx))
        for g in groups.values():
            if len(g) > 1:
                perm[g] = rng.permutation(g)
        g_tab.append(gain(P[perm], idx))
    return g_all, g_tab


res2 = {}
Pb = imp['blind']
for sub in ('NAME', 'BARE', 'IN', 'ALL'):
    sel = [j for j, i in enumerate(unl) if sub == 'ALL' or rows[i][4] == sub]
    sel = [j for j in sel if E(unl[j])['sys'] != 'NONE']
    if len(sel) < 20:
        continue
    idx = [unl[j] for j in sel]
    g = gain(Pb[sel], idx)
    na, nt = perm_nulls(Pb[sel], idx, 500)
    res2[sub] = {'n': len(sel), 'gain_bits': g, 'null_all_mean': float(np.mean(na)), 'p_all': float(np.mean(np.array(na) >= g)),
                 'null_tab_mean': float(np.mean(nt)), 'p_tab': float(np.mean(np.array(nt) >= g))}
# calibration: masked labelled held-out (cycle-1 blind P_ho)
c1b = json.load(open(os.path.join(C.CK, 'c1_%s_blind.json' % corpus)))
rid = {(r[0], r[1]): i for i, r in enumerate(rows)}
hidx = [rid[tuple(x)] for x in c1b['ho_rows']]
Pho = np.array(c1b['P_ho'])
keep = [j for j, i in enumerate(hidx) if E(i)['sys'] != 'NONE']
hidx2 = [hidx[j] for j in keep]
g = gain(Pho[keep], hidx2)
na, nt = perm_nulls(Pho[keep], hidx2, 500)
# oracle: the true class of the masked entries
Ptrue = np.zeros((len(hidx2), len(classes)))
for j, i in enumerate(hidx2):
    Ptrue[j, classes.index(rows[i][3])] = 1
res2['LAB_heldout'] = {'n': len(hidx2), 'gain_bits': g, 'null_all_mean': float(np.mean(na)), 'p_all': float(np.mean(np.array(na) >= g)),
                       'null_tab_mean': float(np.mean(nt)), 'p_tab': float(np.mean(np.array(nt) >= g)),
                       'oracle_gain': gain(Ptrue, hidx2)}
out['numeral_consistency'] = res2


# ------------------------------------------------------------------ 3. totals on ambiguous tablets (PE only)
if corpus == 'PE':
    cap, cnt = L59.pe_maps()
    mapsC = {'cap': cap, 'dec': cnt['dec2'], 'sex': cnt['sex2']}
    capcls = {'MEASURED', 'ALLOT', 'FRACLINE'}
    uidx = {(rows[i][0], rows[i][1]): j for j, i in enumerate(unl)}
    tl = []
    for ti, t in enumerate(T):
        if t['tsys'] != 'AMBT' or not t['totals'] or len(t['totals']) != 1:
            continue
        tot = t['totals'][0]['nums']
        ents = [e['nums'] for e in t['entries'] if e['nums'] and e['nums'][0][1] != 'n']
        if len(ents) < 2 or len(ents) != len(t['entries']) or not tot or tot[0][1] == 'n':
            continue
        if not all(c in L59.AMB for n in ents for _, c in n) or not all(c in L59.AMB for _, c in tot):
            continue
        close = {}
        for nm, m in mapsC.items():
            vals = [L59.value(n, m) for n in ents]
            tv = L59.value(tot, m)
            close[nm] = (None not in vals and tv is not None and sum(vals) == tv)
        if close['cap'] == close['dec'] and close['dec'] == close['sex']:
            diff = False
        else:
            diff = True
        # class mix: written + blind-imputed
        pc = 0.0; n = 0
        for ei, e in enumerate(t['entries']):
            if e['label']:
                pc += 1.0 if e['label'] in capcls else 0.0
            else:
                p = Pb[uidx[(ti, ei)]]
                pc += sum(p[classes.index(c)] for c in capcls if c in classes)
            n += 1
        wr = [e['label'] for e in t['entries'] if e['label']]
        tl.append({'id': t['id'], 'close': close, 'diff': diff, 'cap_share': pc / n,
                   'written_cap_share': (sum(1 for v in wr if v in capcls) / len(wr)) if wr else None,
                   'n_unmarked': sum(1 for e in t['entries'] if not e['label'])})
    dif = [x for x in tl if x['diff']]
    choice = [(x['close']['cap'] if x['cap_share'] >= 0.5 else (x['close']['dec'] or x['close']['sex'])) for x in dif]
    other = [(x['close']['cap'] if x['cap_share'] < 0.5 else (x['close']['dec'] or x['close']['sex'])) for x in dif]
    wchoice = [(x['close']['cap'] if (x['written_cap_share'] or 0) >= 0.5 else (x['close']['dec'] or x['close']['sex']))
               for x in dif]
    coin = []
    for _ in range(2000):
        coin.append(sum((x['close']['cap'] if rng.random() < 0.5 else (x['close']['dec'] or x['close']['sex'])) for x in dif))
    out['totals'] = {'n_amb_tabs_with_total': len(tl), 'n_where_system_matters': len(dif),
                     'close_imputed_choice': int(sum(choice)), 'close_other_choice': int(sum(other)),
                     'close_written_only_choice': int(sum(wchoice)),
                     'close_any': int(sum(1 for x in dif if any(x['close'].values()))),
                     'coin_mean': float(np.mean(coin)), 'p_coin': float(np.mean(np.array(coin) >= sum(choice))),
                     'tablets': dif}

# ------------------------------------------------------------------ 4. tablet system from classes
from sklearn.linear_model import LogisticRegression  # noqa: E402


def tab_feats(Puse, use_imp):
    X, Y, G = [], [], []
    uidx = {(rows[i][0], rows[i][1]): j for j, i in enumerate(unl)}
    for ti, t in enumerate(T):
        if t['tsys'] not in ('CAPT', 'CNTT'):
            continue
        v = np.zeros(len(classes) + 1)
        for ei, e in enumerate(t['entries']):
            if e['label']:
                v[classes.index(e['label'])] += 1
            elif use_imp:
                v[:-1] += Puse[uidx[(ti, ei)]]
            else:
                v[-1] += 1
        v = v / max(1, v.sum())
        X.append(v); Y.append(t['tsys']); G.append(C.hash_fold(t['id'], 5))
    return np.array(X), np.array(Y), np.array(G)


def cv_acc(X, Y, G):
    pred = np.empty(len(Y), dtype=object)
    for k in range(5):
        m = LogisticRegression(C=1.0, max_iter=500).fit(X[G != k], Y[G != k])
        pred[G == k] = m.predict(X[G == k])
    return float((pred == Y).mean())


Xw, Yw, Gw = tab_feats(Pb, False)
Xi, Yi, Gi = tab_feats(Pb, True)
nulls = []
for _ in range(200):
    Xn, _, _ = tab_feats(Pb[rng.permutation(len(unl))], True)
    nulls.append(cv_acc(Xn, Yi, Gi))
out['tsys_pred'] = {'n_tabs': len(Yw), 'maj': float(Counter(Yw).most_common(1)[0][1] / len(Yw)),
                    'written_only': cv_acc(Xw, Yw, Gw), 'written_plus_blind_imputed': cv_acc(Xi, Yi, Gi),
                    'perm_null_mean': float(np.mean(nulls)), 'perm_null_p': float(np.mean(np.array(nulls) >= cv_acc(Xi, Yi, Gi)))}

# ------------------------------------------------------------------ imputation table
tab = []
for j, i in enumerate(unl):
    ti, ei = rows[i][0], rows[i][1]
    e = T[ti]['entries'][ei]
    pr, pb = imp['real'][j], imp['blind'][j]
    tab.append({'tab': T[ti]['id'], 'line': ei, 'kind': rows[i][4], 'signs': e['raw'], 'nums': e['nums'],
                'imp': classes[int(pr.argmax())], 'conf': round(float(pr.max()), 3),
                'imp_blind': classes[int(pb.argmax())], 'conf_blind': round(float(pb.max()), 3)})
out['imputation'] = tab
C.jdump(out, os.path.join(C.CK, 'c2_%s.json' % corpus))
print(json.dumps({k: v for k, v in out.items() if k not in ('imputation', 'totals')}, indent=1, default=str)[:4000])
if 'totals' in out:
    print({k: v for k, v in out['totals'].items() if k != 'tablets'})
