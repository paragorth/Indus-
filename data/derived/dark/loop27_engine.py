"""Loop 27: COMPLETE THE W<->M BRIDGE FROM ALIGNED OBJECT PAIRS.
Core = the loop 24 aligner (tools/dark24_align.py: Wells seq_raw + lost tokens vs IM77 text, bridge-aware edit distance,
site-aware, accepted if cost <= 1 + 0.15 L, type/field-symbol compatible, unique best).
Cycle 1  candidate table: every aligned sign-to-sign position whose (W, M) pair is not in the bridge; confidence =
         Wilson 90% lower bound of the W sign's consistency (k co-alignments / n aligned positions of W), gated by a
         binomial null (M drawn at its background frequency) and compared with the same statistic on wrong-site pairs.
Cycle 2  validation: hold out 20% of the existing bridge entries (5 folds), re-align, recover them; precision/recall by
         confidence threshold; pick the threshold with >= 95% precision.
Cycle 3  apply: proposals at that threshold -> bridge_proposals.json; re-run loop 21's classification of the 690
         indeterminate IM77 texts with the completed bridge.
Cycle 4  loop 21 tests (frame, lot, nesting, names, reuse) on the enlarged new set.
Usage: python3 loop27_engine.py <1|2|3|4>
"""
import sys, os, json, math, random, collections, statistics as st, datetime
ROOT = '/home/user/Indus-/'; HERE = ROOT + 'data/derived/dark/'
sys.path.insert(0, ROOT + 'tools')
import dark24_align as A

CYCLE = sys.argv[1] if len(sys.argv) > 1 else '1'
OUT = open(HERE + f'loop27_cycle{CYCLE}.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()

BRIDGE = {w: set(v) for w, v in A.bridge.items()}          # non-empty entries only (143)
def covered(corr): return set(m for v in corr.values() for m in v)

# ------------------------------------------------------------------ statistics
def wilson_lb(k, n, z=1.2816):   # 90% one-sided lower bound
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d)
def binom_sf(k, n, p):   # P(X >= k)
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

def coalign(acc, corr):
    """count aligned sign-to-sign positions. Returns W->Counter(M), M->Counter(W), examples, and per-W text sets."""
    co = collections.defaultdict(collections.Counter); coM = collections.defaultdict(collections.Counter)
    ex = collections.defaultdict(list); texts = collections.defaultdict(set)
    seen = set()
    for r in acc:
        key = (tuple(r['wells']['full']), tuple(r['im']['seq']))
        if key in seen: continue          # identical duplicate text pairs counted once
        seen.add(key)
        for op in r['ops']:
            if op[0] == 'S' and op[1] and op[2]:
                w, m = op[1], op[2]
                co[w][m] += 1; coM[m][w] += 1; texts[(w, m)].add(key)
                if len(ex[(w, m)]) < 4: ex[(w, m)].append(f"{r['cisi']}={r['text_no']}")
    return co, coM, ex, texts

def candidates(acc, corr, co=None):
    """candidate (W, M) mappings not in corr, with k, n_W, n_M, consistency, Wilson lb, binomial null P."""
    co, coM, ex, texts = coalign(acc, corr)
    totM = sum(sum(c.values()) for c in coM.values())
    pM = {m: sum(c.values()) / totM for m, c in coM.items()}
    rows = []
    for w, c in co.items():
        n = sum(c.values())
        for m, k in c.items():
            if m in corr.get(w, set()): continue
            nM = sum(coM[m].values())
            row = dict(W=w, M=m, k=k, nW=n, nM=nM, consW=k / n, consM=k / nM, lb=wilson_lb(k, n),
                       P=binom_sf(k, n, pM[m]), ntexts=len(texts[(w, m)]), ex=ex[(w, m)],
                       W_bridged=(w in corr), M_covered=(m in covered(corr)),
                       bridge_W=sorted(corr.get(w, ())), bridge_M=sorted(x for x, v in corr.items() if m in v))
            rows.append(row)
    return rows, co, coM

def kind(row):
    if not row['W_bridged'] and not row['M_covered']: return 'A new'
    if not row['W_bridged'] and row['M_covered']: return 'B Wells-split'      # W unbridged -> M that another W already covers
    if row['W_bridged'] and not row['M_covered']: return 'C Mahadevan-split'  # bridged W also read as an uncovered M
    return 'D disagreement'                                                     # both known, pair not in bridge

def run_pass(corr, null=False, seed=0):
    res = A.run(corr, null=null, seed=seed)
    return [r for r in res if A.accept(r)[0]]

def learn(rows, thr, pgate=0.01, kmin=2):
    """proposals at threshold: for each W take every M with lb >= thr (allows one W -> several M, flagged)."""
    out = collections.defaultdict(set)
    for r in rows:
        if r['lb'] >= thr and r['P'] < pgate and r['k'] >= kmin: out[r['W']].add(r['M'])
    return out

def fmt(r):
    return (f"W{r['W']:<4d} -> M{r['M']:<4d} k={r['k']:3d} nW={r['nW']:3d} nM={r['nM']:3d} consW={r['consW']:.2f} consM={r['consM']:.2f} "
            f"lb={r['lb']:.2f} P={r['P']:.1e} texts={r['ntexts']} {kind(r):16s} bridgeW={r['bridge_W']} bridgeM={r['bridge_M']} e.g. {', '.join(r['ex'][:3])}")

# ------------------------------------------------------------------ cycle 1
def cycle1():
    P(f'# LOOP 27 cycle 1: candidate W<->M mappings from aligned object pairs ({datetime.datetime.now().isoformat(timespec="minutes")})')
    corr = {w: set(v) for w, v in BRIDGE.items()}
    P(f'bridge: {len(corr)} Wells signs -> {len(covered(corr))} M signs. Wells inventory 713 signs (571 unbridged, 17% of tokens); IM77 418 signs (279 uncovered, 13% of tokens).')
    acc = run_pass(corr)
    P(f'pass 1 (bridge only): accepted aligned object pairs {len(acc)} (cost 0: {sum(r["cost"] == 0 for r in acc)})')
    rows, co, coM = candidates(acc, corr)
    P(f'aligned sign-to-sign positions {sum(sum(c.values()) for c in co.values())}; candidate (W, M) pairs not in bridge: {len(rows)}; by kind {dict(collections.Counter(kind(r) for r in rows))}')
    # null: wrong-site pairs
    nacc = run_pass(corr, null=True, seed=1)
    nrows, nco, _ = candidates(nacc, corr)
    P(f'wrong-site null: accepted pairs {len(nacc)}; candidate pairs {len(nrows)}; aligned positions {sum(sum(c.values()) for c in nco.values())}')
    P('\n## confidence distribution (Wilson 90% lower bound of consW, k >= 2, P < 0.01): home vs wrong-site null')
    P('  threshold  home pairs  null pairs  null share of home')
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        h = sum(1 for r in rows if r['lb'] >= thr and r['P'] < 0.01 and r['k'] >= 2)
        n_ = sum(1 for r in nrows if r['lb'] >= thr and r['P'] < 0.01 and r['k'] >= 2)
        P(f'  {thr:.1f}        {h:5d}       {n_:5d}       {n_ / h if h else float("nan"):.3f}')
    P('\n## pass 2: add pairs with lb >= 0.5 (provisional), re-align, recount')
    prov = learn(rows, 0.5)
    corr2 = {w: set(v) for w, v in corr.items()}
    for w, ms in prov.items(): corr2.setdefault(w, set()).update(ms)
    acc2 = run_pass(corr2)
    P(f'provisional additions {sum(len(v) for v in prov.values())} on {len(prov)} Wells signs; pass 2 accepted pairs {len(acc2)} (cost 0: {sum(r["cost"] == 0 for r in acc2)})')
    rows2, co2, coM2 = candidates(acc2, corr)      # evaluate against the ORIGINAL bridge so all candidates are listed
    P(f'pass 2 candidate pairs not in the original bridge: {len(rows2)}; by kind {dict(collections.Counter(kind(r) for r in rows2))}')
    rows2.sort(key=lambda r: (-r['lb'], -r['k']))
    P('\n## all candidates with k >= 2, sorted by confidence (lb)')
    for r in rows2:
        if r['k'] >= 2: P('  ' + fmt(r))
    P('\n## uncovered M signs (>= 5 IM77 tokens) and their best Wells partner')
    mcnt = collections.Counter(m for t in A.IM for m in t['seq'] if m)
    cov = covered(corr)
    for m, c in sorted(mcnt.items(), key=lambda z: -z[1]):
        if m in cov or m == 0 or c < 5: continue
        best = coM2.get(m, collections.Counter()).most_common(3)
        P(f'  M{m:<4d} tokens {c:4d} aligned {sum(coM2.get(m, collections.Counter()).values()):3d}  partners {[(f"W{w}", k) for w, k in best]}')
    P('\n## unbridged Wells signs (>= 5 tokens in seq_raw) and their best M partner')
    wcnt = collections.Counter(s for w in A.W for s in w['seq'])
    for w, c in sorted(wcnt.items(), key=lambda z: -z[1]):
        if w in corr or c < 5: continue
        best = co2.get(w, collections.Counter()).most_common(3)
        P(f'  W{w:<4d} tokens {c:4d} aligned {sum(co2.get(w, collections.Counter()).values()):3d}  partners {[(f"M{m}", k) for m, k in best]}')
    json.dump(dict(rows=rows2, null_rows=nrows, provisional={str(w): sorted(v) for w, v in prov.items()}), open(HERE + 'loop27_candidates.json', 'w'))
    P(f'\nwritten loop27_candidates.json ({len(rows2)} candidate rows, {len(nrows)} null rows)')

# ------------------------------------------------------------------ cycle 2: hold-out validation
def cycle2():
    P(f'# LOOP 27 cycle 2: hold-out validation of the alignment-based bridge recovery ({datetime.datetime.now().isoformat(timespec="minutes")})')
    rnd = random.Random(27)
    ws = sorted(BRIDGE); rnd.shuffle(ws)
    folds = [ws[i::5] for i in range(5)]
    allres = []   # (W, true set, top M, lb, k, nW, P, correct)
    for fi, held in enumerate(folds):
        corr = {w: set(v) for w, v in BRIDGE.items() if w not in held}
        acc = run_pass(corr)
        rows, co, coM = candidates(acc, corr)
        by = collections.defaultdict(list)
        for r in rows:
            if r['W'] in held: by[r['W']].append(r)
        rec = 0
        for w in held:
            cand = sorted(by.get(w, []), key=lambda r: (-r['lb'], -r['k']))
            if not cand:
                allres.append(dict(fold=fi, W=w, true=sorted(BRIDGE[w]), top=None, lb=0, k=0, nW=0, P=1, correct=False, aligned=False)); continue
            top = cand[0]
            ok = top['M'] in BRIDGE[w]
            # also: every M at lb>=thr for one-to-many entries
            allres.append(dict(fold=fi, W=w, true=sorted(BRIDGE[w]), top=top['M'], lb=top['lb'], k=top['k'], nW=top['nW'], P=top['P'], correct=ok, aligned=True,
                               second=(cand[1]['M'], round(cand[1]['lb'], 2), cand[1]['k']) if len(cand) > 1 else None))
            rec += ok
        P(f'fold {fi}: held out {len(held)} bridge entries; accepted pairs {len(acc)}; top candidate correct for {rec}, aligned at all {sum(1 for w in held if by.get(w))}')
    P('\n## per held-out sign (W, true M set, recovered top M, lb, k/nW, P, verdict)')
    for r in sorted(allres, key=lambda r: -r['lb']):
        P(f"  W{r['W']:<4d} true {str(r['true']):28s} top {str(r['top']):5s} lb={r['lb']:.2f} k={r['k']}/{r['nW']} P={r['P']:.1e} {'OK ' if r['correct'] else 'BAD'} second={r.get('second')}")
    P('\n## precision / recall of the top candidate by confidence threshold (gate P < 0.01, k >= 2); n held out = %d, of which aligned %d' % (len(allres), sum(r['aligned'] for r in allres)))
    P('  thr   proposed  correct  precision  recall(all held out)  recall(aligned only)')
    best = None
    nrows = json.load(open(HERE + 'loop27_candidates.json'))['null_rows']
    for thr in [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]:
        prop = [r for r in allres if r['aligned'] and r['lb'] >= thr and r['P'] < 0.01 and r['k'] >= 2]
        cor = sum(r['correct'] for r in prop)
        prec = cor / len(prop) if prop else float('nan')
        nnull = sum(1 for r in nrows if r['lb'] >= thr and r['P'] < 0.01 and r['k'] >= 2)
        P(f'  {thr:.2f}  {len(prop):6d}   {cor:6d}   {prec:.3f}      {cor / len(allres):.3f}                 {cor / sum(r["aligned"] for r in allres):.3f}   wrong-site null pairs passing: {nnull}')
        if prop and prec >= 0.95 and nnull == 0 and best is None: best = thr
    P(f'\nlowest threshold with hold-out precision >= 0.95 AND zero wrong-site null pairs: {best}')
    bad = [r for r in allres if r['aligned'] and not r['correct'] and r['lb'] >= (best or 0.5)]
    P('errors above that threshold: ' + '; '.join(f"W{r['W']} true {r['true']} got M{r['top']} (lb {r['lb']:.2f}, k {r['k']}/{r['nW']})" for r in bad))
    json.dump(dict(results=allres, threshold=best), open(HERE + 'loop27_validation.json', 'w'))

# ------------------------------------------------------------------ cycle 3: apply + loop 21 reclassification
def load_eng(tag):
    """import loop21_engine without clobbering its cycle files; redirect its output to our file."""
    import importlib
    sys.argv = ['loop21_engine.py', tag]
    eng = importlib.import_module('loop21_engine')
    try: eng.OUT.close(); os.remove(HERE + f'loop21_cycle{tag}.txt')
    except OSError: pass
    eng.OUT = OUT
    return eng

def completed_bridge(thr):
    cand = json.load(open(HERE + 'loop27_candidates.json'))
    rows = cand['rows']
    prop = {}
    for r in rows:
        if r['lb'] >= thr and r['P'] < 0.01 and r['k'] >= 2:
            prop.setdefault(r['W'], []).append(r)
    return rows, prop

def cycle3():
    val = json.load(open(HERE + 'loop27_validation.json')); thr = val['threshold']
    P(f'# LOOP 27 cycle 3: apply the completed bridge (threshold lb >= {thr}) and re-run loop 21 classification ({datetime.datetime.now().isoformat(timespec="minutes")})')
    rows, prop = completed_bridge(thr)
    # second pass: with proposals added, re-align and look for further candidates that only appear once the first batch is in
    corr = {w: set(v) for w, v in BRIDGE.items()}
    for w, rs in prop.items(): corr.setdefault(w, set()).update(r['M'] for r in rs)
    acc = run_pass(corr)
    rows3, co3, coM3 = candidates(acc, corr)
    extra = {}
    for r in rows3:
        if r['lb'] >= thr and r['P'] < 0.01 and r['k'] >= 2: extra.setdefault(r['W'], []).append(r)
    P(f'proposals from cycle 1 table: {sum(len(v) for v in prop.values())} mappings on {len(prop)} Wells signs; after re-alignment with them, further mappings at the same threshold: {sum(len(v) for v in extra.values())} on {len(extra)} signs')
    for w, rs in extra.items():
        for r in rs: P('  extra: ' + fmt(r))
        prop.setdefault(w, []).extend(rs)
    # classify proposals
    P('\n## proposed bridge entries by kind')
    kinds = collections.defaultdict(list)
    for w, rs in prop.items():
        for r in rs: kinds[kind(r)].append(r)
    for k_ in sorted(kinds):
        P(f'\n### {k_}: {len(kinds[k_])} mappings')
        for r in sorted(kinds[k_], key=lambda r: (-r['lb'], -r['k'])): P('  ' + fmt(r))
    # one-to-many on either side
    P('\n## one-to-many (allograph-level facts, not errors)')
    wm = collections.defaultdict(set); mw = collections.defaultdict(set)
    for w, v in BRIDGE.items():
        for m in v: wm[w].add(m); mw[m].add(w)
    for w, rs in prop.items():
        for r in rs: wm[w].add(r['M']); mw[r['M']].add(w)
    newW = set(prop)
    newM = set(r['M'] for rs in prop.values() for r in rs)
    P('  Wells sign -> several M signs (Mahadevan splits): ' + '; '.join(f'W{w}={sorted(wm[w])}' for w in sorted(wm) if len(wm[w]) > 1 and (w in newW or wm[w] & newM)))
    P('  M sign <- several Wells signs (Wells splits / Mahadevan lumps): ' + '; '.join(f'M{m}={sorted(mw[m])}' for m in sorted(mw) if len(mw[m]) > 1 and (m in newM or mw[m] & newW)))
    # write proposals
    out = dict(source='loop27 (aligned Wells/IM77 object pairs; dark24 aligner)', threshold=dict(wilson_lb=thr, P_gate=0.01, k_min=2),
               validation='5-fold hold-out of existing bridge entries; see loop27_cycle2.txt',
               proposals=[dict(W=r['W'], M=r['M'], k=r['k'], nW=r['nW'], nM=r['nM'], consistency_W=round(r['consW'], 3), consistency_M=round(r['consM'], 3),
                               confidence_lb=round(r['lb'], 3), P_null=r['P'], kind=kind(r), bridge_W=r['bridge_W'], bridge_M=r['bridge_M'], examples=r['ex'])
                          for w, rs in sorted(prop.items()) for r in sorted(rs, key=lambda r: -r['lb'])],
               corrections=[dict(W=798, old=[216], new=[53], from_='S-DARK-24.1'), dict(W=806, old=[387], new=[389], from_='S-DARK-24.1')])
    json.dump(out, open(HERE + 'bridge_proposals.json', 'w'), indent=1)
    P(f'\nwritten bridge_proposals.json: {len(out["proposals"])} mappings on {len(prop)} Wells signs')
    # ---- loop 21 reclassification
    eng = load_eng('x27')
    base = {t['id']: t['status'] for t in eng.classify('seq_raw')}
    newBR = {w: set(v) for w, v in eng.BR.items()}
    for w, rs in prop.items(): newBR.setdefault(w, set()).update(r['M'] for r in rs)
    # loop 24 corrections
    newBR[798] = {53}; newBR[806] = {389}
    eng.BR = newBR; eng.MCOV = set(m for v in newBR.values() for m in v)
    T = eng.classify('seq_raw')
    P(f'\n## loop 21 reclassification: bridge {len(BRIDGE)} -> {len(newBR)} Wells signs, M coverage {len(covered(BRIDGE))} -> {len(eng.MCOV)} signs')
    comp = [t for t in T if t['status'] != 'damaged']
    P(f'status with completed bridge: {dict(collections.Counter(t["status"] for t in comp))} (loop 21: exact 1685, near 157, indet 690, none 210)')
    trans = collections.Counter((base[t['id']], t['status']) for t in T)
    P('transitions (old -> new): ' + '; '.join(f'{a}->{b}: {n}' for (a, b), n in sorted(trans.items()) if a != b))
    P('the 690 indeterminate: ' + str({b: n for (a, b), n in trans.items() if a == 'indet'}))
    NEW = [t for t in comp if t['status'] == 'none']; OLDNEW = set(eng_sets()['new'])
    added = [t for t in NEW if (t['id'][0], t['id'][1]) not in OLDNEW]
    lost = [tid for tid in OLDNEW if tid not in {(t['id'][0], t['id'][1]) for t in NEW}]
    P(f'certainly-new texts: {len(NEW)} (loop 21: 210); added {len(added)}, dropped {len(lost)} (now exact/near thanks to a new mapping: {collections.Counter(next(t["status"] for t in T if (t["id"][0], t["id"][1]) == tid) for tid in lost)})')
    P(f'site mix new: {dict(collections.Counter(t["site"] for t in NEW))}; object types {dict(collections.Counter(t["otype"] for t in NEW))}; mean length {st.mean(len(t["seq"]) for t in NEW):.2f}')
    P(f'site mix of the ADDED texts: {dict(collections.Counter(t["site"] for t in added))}; types {dict(collections.Counter(t["otype"] for t in added))}; lengths {dict(sorted(collections.Counter(len(t["seq"]) for t in added).items()))}')
    ind = [t for t in comp if t['status'] == 'indet']
    P(f'remaining indeterminate {len(ind)}: unbridged M signs they contain (top 25): {collections.Counter(m for t in ind for m in t["seq"] if m not in eng.MCOV).most_common(25)}')
    rnd = random.Random(27)
    n4, em, emax, nm, nmax = eng.match_control(T, 'seq_raw', rnd, 10)
    P(f'matcher false-positive control with completed bridge (texts >= 4 shuffled, n={n4}, 10x): exact {em / n4:.1%} (max {emax}) near {nm / n4:.1%} (loop 21: 0.3% / 1.1%)')
    P('\n### the added certainly-new texts (text_no:side site object seq in M numbers)')
    for t in sorted(added, key=lambda t: t['id']): P(f'  {t["id"][0]}:{t["id"][1]} {t["site"]:12s} {t["otype"]:12s} {t["seq"]}')
    json.dump({'new': [t['id'] for t in NEW], 'added': [t['id'] for t in added], 'strict': [t['id'] for t in comp if t['status'] != 'exact'],
               'overlap': [t['id'] for t in comp if t['status'] == 'exact'], 'bridge': {str(w): sorted(v) for w, v in newBR.items()}},
              open(HERE + 'loop27_sets.json', 'w'))

def eng_sets():
    s = json.load(open(HERE + 'loop21_cycle1_sets.json'))
    return {k: set(tuple(x) for x in v) for k, v in s.items()}

# ------------------------------------------------------------------ cycle 4: tests on the enlarged new set
def cycle4():
    P(f'# LOOP 27 cycle 4: loop 21 tests on the enlarged certainly-new set ({datetime.datetime.now().isoformat(timespec="minutes")})')
    S = json.load(open(HERE + 'loop27_sets.json'))
    eng = load_eng('x27')
    eng.BR = {int(w): set(v) for w, v in S['bridge'].items()}; eng.MCOV = set(m for v in eng.BR.values() for m in v)
    T = eng.classify('seq_raw'); comp = [t for t in T if t['status'] != 'damaged']
    NEW = [t for t in comp if t['status'] == 'none']; OVER = [t for t in comp if t['status'] == 'exact']
    addset = set(tuple(x) for x in S['added']); ADDED = [t for t in NEW if tuple(t['id']) in addset]
    rnd = random.Random(2027)
    for lab, X in (('ENLARGED new set', NEW), ('ADDED texts only (unlocked by the bridge completion)', ADDED)):
        eng.describe(X, lab)
        eng.test_frame(X, lab, rnd, 200); eng.test_fish(X, lab); eng.test_lot(X, lab, rnd, 2000)
        eng.test_names(X, lab, rnd); eng.test_quantity(X, lab, rnd, 100)
        eng.test_nesting(X, X, lab + ' (hosts = same set)', rnd, 50)
        eng.test_nesting(X, comp, lab + ' shorts, hosts = all complete IM77', rnd, 50)
        eng.test_reuse(X, OVER, lab, rnd, 100)
    OVERM = eng.length_matched(OVER, NEW, rnd)
    eng.describe(OVERM, 'OVERLAP length-matched (3 per new text)')
    eng.test_frame(OVERM, 'OVERLAP length-matched', rnd, 100); eng.test_lot(OVERM, 'OVERLAP length-matched', rnd, 1000)
    eng.test_names(OVERM, 'OVERLAP length-matched', rnd); eng.test_nesting(OVERM, OVERM, 'OVERLAP length-matched', rnd, 30)

if __name__ == '__main__':
    {'1': cycle1, '2': cycle2, '3': cycle3, '4': cycle4}[CYCLE]()
