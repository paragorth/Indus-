"""pe62 cycle 3: which name-string signs carry the class when the class sign is missing?
usage: python3 pe62_c3.py CORPUS [plant]
For every name sign s (>= 8 labelled occurrences on >= 4 tablets):
  W (within-tablet): are entries with s of class k more often than other labelled entries on the SAME tablets?
     G statistic vs expected from each tablet's class shares; null = labels permuted within tablets (400x).
  X (cross-tablet): do tablets carrying s have a class mix unlike tablets of the same size?  one vote per tablet;
     null = random tablet sets of the same size drawn within the same museum batch (400x).
  Replication: signs found on DEV tablets (hash split) are re-tested on HOLD tablets in the same direction.
  Outside check: for unmarked entries carrying a W-carrier of a capacity class (MEASURED/ALLOT/FRACLINE) vs a count
     class, compare the share of capacity numerals with other unmarked entries on the same tablets.
  plant: a random name sign X is made an implicit class carrier (its labelled entries relabelled to the
     majority class of the other entries... no: to a random other class), to check W finds it.
Writes data/pe62_ckpt/c3_<CORPUS>[_plant].json"""
import sys, os, json, random, math
import numpy as np
from collections import Counter, defaultdict
import pe62_common as C

corpus = sys.argv[1]
plant = len(sys.argv) > 2
rng = np.random.default_rng(6203)
PE = C.build_pe()
n_lab_pe = sum(1 for t in PE for e in t['entries'] if e['label'])
T = PE if corpus == 'PE' else (C.match_size(C.build_pc(), n_lab_pe, 'pe62pc') if corpus == 'PC'
                               else C.match_size(C.build_u3(), n_lab_pe, 'pe62u3'))
classes = sorted({e['label'] for t in T for e in t['entries'] if e['label']})
CAPC = {'MEASURED', 'ALLOT', 'FRACLINE', 'GRAIN', 'PRODUCT', 'OIL'}
plant_info = None
if plant:
    import copy
    T = copy.deepcopy(T)
    occ = Counter(); tb = defaultdict(set)
    for t in T:
        for e in t['entries']:
            if e['label']:
                for s in set(e['name']):
                    occ[s] += 1; tb[s].add(t['id'])
    pr = random.Random('pe62c3plant' + corpus)
    cand = sorted(s for s in occ if 12 <= occ[s] <= 60 and len(tb[s]) >= 5)
    X = pr.choice(cand)
    k = pr.choice(classes)
    n = 0
    for t in T:
        for e in t['entries']:
            if e['label'] and X in e['name'] and pr.random() < 0.7:
                e['label'] = k; n += 1
    plant_info = {'sign': X, 'class': k, 'n_relabelled': n, 'occ': occ[X]}


def analyse(tabs, min_occ=8, min_tabs=4, nperm=400):
    # labelled entries per tablet
    L = []
    for ti, t in enumerate(tabs):
        for e in t['entries']:
            if e['label']:
                L.append((ti, classes.index(e['label']), set(e['name'])))
    if not L:
        return {}
    tabi = np.array([x[0] for x in L])
    yi = np.array([x[1] for x in L])
    occ = defaultdict(list)
    for j, (ti, _, ns) in enumerate(L):
        for s in ns:
            occ[s].append(j)
    signs = [s for s, js in occ.items() if len(js) >= min_occ and len({tabi[j] for j in js}) >= min_tabs]
    groups = defaultdict(list)
    for j, ti in enumerate(tabi):
        groups[ti].append(j)
    groups = {k: np.array(v) for k, v in groups.items()}
    # expected shares per tablet
    share = {}
    for ti, js in groups.items():
        c = np.bincount(yi[js], minlength=len(classes)).astype(float)
        share[ti] = c / c.sum()

    def Wstat(y, js):
        O = np.bincount(y[js], minlength=len(classes)).astype(float)
        Ex = np.zeros(len(classes))
        for ti, cnt in Counter(tabi[js]).items():
            c = np.bincount(y[groups[ti]], minlength=len(classes)).astype(float)
            Ex += cnt * c / c.sum()
        g = 2 * float(sum(o * math.log(o / e) for o, e in zip(O, Ex) if o > 0 and e > 0))
        return g, O, Ex

    perms = []
    for _ in range(nperm):
        yp = yi.copy()
        for ti, js in groups.items():
            if len(js) > 1:
                yp[js] = yp[rng.permutation(js)]
        perms.append(yp)
    out = {}
    # tablet class mix for X
    tmix = {ti: np.bincount(yi[js], minlength=len(classes)).astype(float) for ti, js in groups.items()}
    batch_of = {ti: tabs[ti]['batch'] for ti in groups}
    by_batch = defaultdict(list)
    for ti, b in batch_of.items():
        by_batch[b].append(ti)
    gmix = sum(tmix.values()); gmix = gmix / gmix.sum()
    for s in signs:
        js = np.array(occ[s])
        g, O, Ex = Wstat(yi, js)
        null = [Wstat(yp, js)[0] for yp in perms]
        pW = (1 + sum(1 for v in null if v >= g)) / (1 + nperm)
        top = int(np.argmax((O + 0.5) / (Ex + 0.5)))
        # X: tablets with s, class mix (votes = each tablet's normalised mix)
        tset = sorted({int(tabi[j]) for j in js})
        def xstat(ts):
            v = sum(tmix[t] / tmix[t].sum() for t in ts) / len(ts)
            return float(np.abs(v - gmix).sum()), v
        xs, xv = xstat(tset)
        xn = []
        for _ in range(nperm):
            pick = []
            for t in tset:
                pool = by_batch[batch_of[t]]
                pick.append(pool[rng.integers(len(pool))])
            xn.append(xstat(pick)[0])
        pX = (1 + sum(1 for v in xn if v >= xs)) / (1 + nperm)
        out[s] = {'n': len(js), 'tabs': len(tset), 'W_G': g, 'pW': pW, 'W_top': classes[top],
                  'W_OE': float((O[top] + 0.5) / (Ex[top] + 0.5)), 'O': O.tolist(), 'E': [round(x, 2) for x in Ex],
                  'X_dev': xs, 'pX': pX, 'X_top': classes[int(np.argmax(xv - gmix))]}
    return out


def bh(ps):
    ps = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(ps); q = {}
    prev = 1.0
    for r in range(m, 0, -1):
        s, p = ps[r - 1]
        prev = min(prev, p * m / r)
        q[s] = prev
    return q


res = {'corpus': corpus, 'classes': classes, 'plant': plant_info}
full = analyse(T)
qW = bh({s: v['pW'] for s, v in full.items()})
qX = bh({s: v['pX'] for s, v in full.items()})
for s in full:
    full[s]['qW'] = qW[s]; full[s]['qX'] = qX[s]
res['full'] = full
dev = [t for t in T if not C.hsplit(t['id'], 'pe62ho', 0.30)]
ho = [t for t in T if C.hsplit(t['id'], 'pe62ho', 0.30)]
A = analyse(dev)
B = analyse(ho, min_occ=3, min_tabs=2)
rep = {}
for s, v in A.items():
    if v['pW'] <= 0.05 and s in B:
        rep[s] = {'dev_top': v['W_top'], 'dev_OE': v['W_OE'], 'ho_top': B[s]['W_top'], 'ho_OE_devclass':
                  float((B[s]['O'][classes.index(v['W_top'])] + 0.5) / (B[s]['E'][classes.index(v['W_top'])] + 0.5)),
                  'ho_pW': B[s]['pW']}
res['replication'] = rep
nA = sum(1 for v in A.values() if v['pW'] <= 0.05)
res['rep_summary'] = {'n_signs_dev': len(A), 'dev_sig': nA, 'testable_on_ho': len(rep),
                      'same_top_on_ho': sum(1 for v in rep.values() if v['ho_top'] == v['dev_top']),
                      'OE_gt1_on_ho': sum(1 for v in rep.values() if v['ho_OE_devclass'] > 1),
                      'expected_same_top_chance': 1.0 / len(classes)}
# outside check: capacity numerals among unmarked entries carrying capacity-class carriers
car = {s: v['W_top'] for s, v in full.items() if v['pW'] <= 0.05}
diffs = []
for t in T:
    U = [e for e in t['entries'] if not e['label'] and e['sys'] != 'NONE']
    if len(U) < 2:
        continue
    for e in U:
        cs = [car[s] for s in e['name'] if s in car]
        if not cs:
            continue
        capc = sum(1 for c in cs if c in CAPC) > len(cs) / 2
        others = [o for o in U if o is not e and not any(s in car for s in o['name'])]
        if not others:
            continue
        iscap = e['sys'] == 'CAP'
        base = np.mean([o['sys'] == 'CAP' for o in others])
        diffs.append((capc, iscap - base))
cap_d = [d for c, d in diffs if c]
cnt_d = [d for c, d in diffs if not c]
res['outside'] = {'n_cap_carrier_entries': len(cap_d), 'n_cnt_carrier_entries': len(cnt_d),
                  'cap_excess': float(np.mean(cap_d)) if cap_d else None,
                  'cnt_excess': float(np.mean(cnt_d)) if cnt_d else None}
fn = os.path.join(C.CK, 'c3_%s%s.json' % (corpus, '_plant' if plant else ''))
C.jdump(res, fn)
sig = sorted(((v['qW'], s) for s, v in full.items() if v['qW'] <= 0.10))
print(corpus, 'plant' if plant else '', 'signs tested', len(full), 'W q<=0.10:', len(sig),
      'pW<=0.05:', sum(1 for v in full.values() if v['pW'] <= 0.05),
      'X q<=0.10:', sum(1 for v in full.values() if v['qX'] <= 0.10))
for q, s in sig[:30]:
    v = full[s]
    print('  %-14s n%3d tabs%3d %s OE %.2f qW %.3f | X %s qX %.3f' % (s, v['n'], v['tabs'], v['W_top'], v['W_OE'], q,
                                                                    v['X_top'], v['qX']))
print('rep', res['rep_summary'])
print('outside', res['outside'])
if plant_info:
    print('plant', plant_info, full.get(plant_info['sign']))
