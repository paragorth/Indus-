"""v58 cycle 3: MASSIVE ORDER GUESSING. If the Voynich leaves were misbound, the folio order hides the
source's shape. For the herbal pages and the stars paragraphs, search thousands of leaf orders
(leaves permuted within their quire; recto/verso kept together; paragraphs kept within their page)
for the order that best aligns with each candidate's entry lengths. The SAME search with the
same budget is run on block-shuffled Voynich sequences (null) and on a PLANTED sequence (an encoded
real source whose leaves were permuted within quire-sized groups; positive control).

Score: local length alignment at the folio-order best scale c (fixed during the search).
Search: R random orders + hill climbing by leaf swaps inside quires.
"""
import sys, os, json, random, time
from collections import defaultdict, OrderedDict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v58_lib as L

PAR = dict(lam=1.0, s=0.30, pm=0.30, g=0.5)
OUT = os.path.join(L.CK, 'cycle3.json')
RAND = int(os.environ.get('RAND', 300)); CLIMB = int(os.environ.get('CLIMB', 1200)); NREP = int(os.environ.get('NREP', 6))

def leaf_structure(section):
    """list of leaves [(quire, leafno, [unit lengths in order])] in folio order for a section."""
    P = L.voynich_units()
    if section == 'herbal_pages':
        P = [p for p in P if p['illus'] == 'H']
        pages = OrderedDict()
        for p in P: pages.setdefault(p['folio'], [p['quire'], 0]); pages[p['folio']][1] += p['w']
        leaves = OrderedDict()
        for f, (q, w) in pages.items():
            leaves.setdefault((q, L.fnum(f)[0]), []).append(w)
    else:
        code = {'stars_paras': 'S', 'bio_paras': 'B'}[section]
        P = [p for p in P if p['illus'] == code]
        leaves = OrderedDict()
        for p in P: leaves.setdefault((p['quire'], L.fnum(p['folio'])[0]), []).append(p['w'])
    return [(k[0], k[1], v) for k, v in leaves.items()]

def flatten(leaves, order):
    return [w for i in order for w in leaves[i][2]]

def search(leaves, y, c, rng):
    """random orders within quire + hill climbing on leaf swaps within quire. Returns best score, order."""
    byq = defaultdict(list)
    for i, l in enumerate(leaves): byq[l[0]].append(i)
    qs = [q for q in byq if len(byq[q]) > 1]
    base = list(range(len(leaves)))
    def score(order):
        return L.align(flatten(leaves, order), y, PAR, cgrid=np.array([c]))[0]
    def rand_order():
        o = []
        for q in sorted(byq, key=lambda q: min(byq[q])):
            idx = byq[q][:]; rng.shuffle(idx); o += idx
        return o
    best_o, best = base, score(base)
    for _ in range(RAND):
        o = rand_order(); v = score(o)
        if v > best: best, best_o = v, o
    cur, cv = best_o[:], best
    pos = {q: [k for k, i in enumerate(cur) if leaves[i][0] == q] for q in qs}
    for _ in range(CLIMB):
        q = rng.choice(qs); a, b = rng.sample(pos[q], 2)
        cur[a], cur[b] = cur[b], cur[a]
        v = score(cur)
        if v >= cv: cv = v
        else: cur[a], cur[b] = cur[b], cur[a]
    if cv > best: best, best_o = cv, cur[:]
    return best, best_o

def block_shuffle_leaves(leaves, rng, b=4):
    """null: unit lengths block-shuffled across the section, re-poured into the same leaf/unit slots."""
    flat = [w for l in leaves for w in l[2]]
    sh = L.block_shuffle(flat, rng, b=8)
    out, k = [], 0
    for q, n, ws in leaves:
        out.append((q, n, sh[k:k + len(ws)])); k += len(ws)
    return out

def planted(leaves, src_units, rng):
    """positive control: encoded real source poured into the Voynich leaf/unit slots in source order,
    then leaves permuted within their quire (simulated misbinding)."""
    enc = L.encode_lengths(src_units, rng)
    n = sum(len(l[2]) for l in leaves)
    st = rng.randrange(max(1, len(enc) - n)); enc = enc[st:st + n]
    while len(enc) < n: enc.append(enc[-1])
    out, k = [], 0
    for q, nn, ws in leaves:
        out.append((q, nn, enc[k:k + len(ws)])); k += len(ws)
    byq = defaultdict(list)
    for i, l in enumerate(out): byq[l[0]].append(i)
    perm = []
    for q in sorted(byq, key=lambda q: min(byq[q])):
        idx = byq[q][:]; rng.shuffle(idx); perm += idx
    return [out[i] for i in perm]

def main():
    T = json.load(open(os.path.join(L.CK, 'texts.json')))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rng = random.Random(583)
    jobs = json.loads(os.environ.get('JOBS', '[]'))
    for sec, cand in jobs:
        leaves = leaf_structure(sec)
        y = [u['w'] for u in T[cand]['units']]
        x0 = flatten(leaves, range(len(leaves)))
        s0, c = L.align(x0, y, PAR)
        key = sec + '|' + cand
        if key in res: continue
        t0 = time.time()
        real, order = search(leaves, y, c, rng)
        nulls = []
        for r in range(NREP):
            lv = block_shuffle_leaves(leaves, rng)
            s0n, cn = L.align(flatten(lv, range(len(lv))), y, PAR)
            nulls.append(search(lv, y, cn, rng)[0])
        z, p = L.zp(real, nulls)
        res[key] = dict(folio_order=round(s0, 2), c=c, searched=round(real, 2), null_searched=[round(v, 2) for v in nulls],
                        z=round(float(z), 2), p=round(float(p), 3),
                        order=[[leaves[i][0], leaves[i][1]] for i in order], secs=round(time.time() - t0))
        print(key, {k: v for k, v in res[key].items() if k != 'order'}, flush=True)
        json.dump(res, open(OUT, 'w'))
    # planted positive controls
    for sec, src, tgt in json.loads(os.environ.get('PLANT', '[]')):
        key = 'PLANT|%s|%s|%s' % (sec, src, tgt)
        if key in res: continue
        leaves = leaf_structure(sec)
        pl = planted(leaves, T[src]['units'], rng)
        y = [u['w'] for u in T[tgt]['units']]
        s0, c = L.align(flatten(pl, range(len(pl))), y, PAR)
        real, order = search(pl, y, c, rng)
        nulls = []
        for r in range(NREP):
            lv = block_shuffle_leaves(pl, rng)
            s0n, cn = L.align(flatten(lv, range(len(lv))), y, PAR)
            nulls.append(search(lv, y, cn, rng)[0])
        z, p = L.zp(real, nulls)
        res[key] = dict(misbound_order=round(s0, 2), searched=round(real, 2), null_searched=[round(v, 2) for v in nulls], z=round(float(z), 2), p=round(float(p), 3))
        print(key, res[key], flush=True)
        json.dump(res, open(OUT, 'w'))

if __name__ == '__main__':
    main()
