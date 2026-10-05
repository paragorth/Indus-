"""v56 cycle 4: pointer rules under alternative bindings. A pointer system addresses the ORIGINAL page order.
If the present binding is disturbed (bifolios re-nested or flipped inside quires, as proposed from the physical
quires), an arithmetic pointer rule should appear only under (or near) the true order. We rerun the cycle-1
search under K re-nested bifolio orders and under K random page relabellings (null), and compare the held-out
z of the best rules. A planted pointer system addressed in a hidden bifolio order is the control: the search must
score its hidden order above the current one and above the relabelled nulls."""
import sys, os, json, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v56_lib as L
from multiprocessing import Pool

TAG = sys.argv[1] if len(sys.argv) > 1 else 'c4'
NCFG = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 2400
K = int(sys.argv[4]) if len(sys.argv) > 4 else 3


def bifolio_order(C, seed):
    """re-nest and flip bifolios inside each quire; quire order and pages within a leaf are kept."""
    rng = random.Random(seed)
    pages = C['pages']
    quires = []
    for pi, p in enumerate(pages):
        if not quires or quires[-1][0] != p['quire']: quires.append((p['quire'], []))
        quires[-1][1].append(pi)
    order = []
    for q, pis in quires:
        leaves = []
        for pi in pis:
            f = pages[pi]['folio']
            if not leaves or leaves[-1][0] != f: leaves.append((f, []))
            leaves[-1][1].append(pi)
        n = len(leaves)
        bif = [(leaves[i], leaves[n - 1 - i]) for i in range(n // 2)]
        mid = [leaves[n // 2]] if n % 2 else []
        rng.shuffle(bif)
        bif = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in bif]
        seq = [a for a, b in bif] + mid + [b for a, b in bif][::-1]
        for f, ps in seq: order += ps
    return order


def plant_in_order(Z, order, seed=11):
    """planted pointers whose addresses count pages in a hidden bifolio order."""
    Zo = dict(Z, pages=[Z['pages'][k] for k in order])
    P = L.plant_pointers(Zo, L.build_R, seed=seed)
    inv = {k: j for j, k in enumerate(order)}
    back = [None] * len(order)
    for j, k in enumerate(order): back[k] = P['pages'][j]
    return dict(Z, pages=back, name=Z['name'] + '_plantedhidden', digits=P['digits'])


def job(arg):
    name, kind, k = arg
    if name == 'ZL': C = L.voynich('ZL3b')
    elif name == 'IT2a': C = L.voynich('IT2a')
    elif name == 'PLH':
        Z = L.voynich('ZL3b'); hidden = bifolio_order(Z, 999); C = plant_in_order(Z, hidden)
    if kind == 'cur': order = list(range(len(C['pages'])))
    elif kind == 'hidden': order = bifolio_order(C, 999)
    elif kind == 'bif': order = bifolio_order(C, 100 + k)
    else:
        order = list(range(len(C['pages']))); random.Random(500 + k).shuffle(order)
    E = L.build_R(C, order=order)
    Co = dict(C, pages=E['pages'])
    rows, E, TT, ne, cov = L.search_grid(Co, seed=0, time_budget=BUDGET, E=E, max_cfg=NCFG, mode_filter=('abs_clip',),
                                         sel_kind='fixed', starts=6, hitk=10, focused=True)
    S = L.summarise(rows)
    out = dict(name=name, kind=kind, k=k, n_eval=ne, summary=S, coverage=cov, digits=C.get('digits'), rows=rows,
               top=sorted(rows, key=lambda r: -r['z_tr'])[:10])
    json.dump(out, open(os.path.join(L.CK, '%s_%s_%s_%d.json' % (TAG, name, kind, k)), 'w'))
    return '%s %s %d top_te %.2f max %.2f' % (name, kind, k, S['top_te_mean'], S['top_te_max'])


if __name__ == '__main__':
    which = sys.argv[5].split(',') if len(sys.argv) > 5 else ['PLH', 'ZL']
    jobs = [('PLH', 'hidden', 0), ('PLH', 'cur', 0), ('ZL', 'bif', 0), ('ZL', 'bif', 1), ('ZL', 'relab', 0),
            ('ZL', 'relab', 1)]
    with Pool(2) as P:
        for r in P.imap_unordered(job, jobs): print(r, flush=True)
