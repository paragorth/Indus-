"""v33 cycle 2: where does the Voynich fall among languages and generators?

Per rule: 6-feature ecology vector per block (conn, signed log z of NODF, C, Q, stem-degree CV, ending-degree Gini),
standardised over the reference blocks. Classification of a query block = nearest reference CORPUS (mean distance
to its blocks), query's own corpus excluded. Families: FUS (la cs de it), AGG (hu tr), ROOT (he), GEN (slot, tri,
table on Voynich or Latin source; v26 minimal genome).
Calibration: leave-one-corpus-out family accuracy of the reference blocks themselves (can the profile tell a
language from a generator at all?), per rule and with all rules concatenated; bootstrap over rules for stability.
"""
from v33_report1 import *

FAM = {'L-la': 'FUS', 'L-cs': 'FUS', 'L-de': 'FUS', 'L-it': 'FUS', 'L-hu': 'AGG', 'L-tr': 'AGG', 'L-he': 'ROOT'}


def fam(c):
    return FAM.get(c, 'GEN' if c.startswith('G-') else c)


def vecs(D, rules):
    names = sorted(D)
    X = np.array([np.concatenate([fvec(D[n][r]) for r in rules]) for n in names])
    return names, X


def classify(D, rules, refs, queries):
    names, X = vecs(D, rules)
    ix = {n: i for i, n in enumerate(names)}
    R = np.array([ix[n] for n in names if corp(n) in refs])
    mu, sd = X[R].mean(0), X[R].std(0) + 1e-9
    Z = (X - mu) / sd
    out = {}
    for q in queries:
        qc = corp(q)
        best = {}
        for c in refs:
            if c == qc: continue
            m = [ix[n] for n in names if corp(n) == c]
            best[c] = float(np.mean([np.linalg.norm(Z[ix[q]] - Z[j]) for j in m]))
        o = sorted(best, key=best.get)
        out[q] = dict(nearest=o[0], fam=fam(o[0]), order=o[:4], d=[round(best[c], 2) for c in o[:4]])
    return out


if __name__ == '__main__':
    D = loadall()
    cs = sorted(set(corp(n) for n in D))
    refs = [c for c in cs if c.startswith(('L-', 'G-'))]
    queries = [n for n in D if corp(n) in ('V-ZL', 'V-IT', 'V-A', 'V-B') or n.startswith('P-')]
    res = {}
    # A. calibration: reference blocks, leave-own-corpus-out
    refblocks = [n for n in D if corp(n) in refs]
    cal_all = classify(D, COMMON, refs, refblocks)
    acc_all = np.mean([cal_all[n]['fam'] == fam(corp(n)) for n in refblocks])
    lang_vs_gen = np.mean([(cal_all[n]['fam'] == 'GEN') == (fam(corp(n)) == 'GEN') for n in refblocks])
    per_rule_acc = {r: float(np.mean([(v['fam'] == 'GEN') == (fam(corp(n)) == 'GEN') for n, v in classify(D, [r], refs, refblocks).items()])) for r in COMMON}
    res['cal'] = dict(all_rules_family_acc=float(acc_all), all_rules_lang_vs_gen_acc=float(lang_vs_gen),
                      per_rule_lang_vs_gen=per_rule_acc, detail={n: cal_all[n] for n in refblocks})
    print('CAL concatenated: family acc', round(acc_all, 2), 'lang-vs-gen acc', round(lang_vs_gen, 2))
    print('CAL per rule lang-vs-gen acc: median', round(float(np.median(list(per_rule_acc.values()))), 2),
          'min', round(min(per_rule_acc.values()), 2), 'max', round(max(per_rule_acc.values()), 2))
    for n in refblocks: print('   ', n, '->', cal_all[n]['order'][:3], cal_all[n]['d'][:3])
    # B. Voynich and controls, concatenated and per rule
    allq = classify(D, COMMON, refs, queries)
    per = {r: classify(D, [r], refs, queries) for r in COMMON}
    res['query_all'] = allq
    res['query_per_rule'] = {q: Counter(per[r][q]['fam'] for r in COMMON) for q in queries}
    res['query_per_rule_corpus'] = {q: Counter(per[r][q]['nearest'] for r in COMMON).most_common(5) for q in queries}
    for q in sorted(queries):
        print('Q', q, 'concat ->', allq[q]['order'], allq[q]['d'], '| per-rule fam votes', dict(res['query_per_rule'][q]),
              '| top corpora', res['query_per_rule_corpus'][q][:3])
    # C. bootstrap over rules (concatenated classifier on resampled rule sets)
    rng = random.Random(33)
    boot = defaultdict(Counter)
    for b in range(200):
        rs = [rng.choice(COMMON) for _ in COMMON]
        o = classify(D, rs, refs, queries)
        for q in queries: boot[q][o[q]['fam']] += 1; boot[q + ':c'][o[q]['nearest']] += 1
    res['boot'] = {k: dict(v) for k, v in boot.items()}
    for q in sorted(queries): print('BOOT', q, dict(boot[q]), boot[q + ':c'].most_common(3))
    # D. Voynich-alphabet slot rules (only Voynich-alphabet corpora): V vs Voynich-source generators and plants
    slot = [f'slot-{c}' for c in (3, 4, 5, 6)]
    Dv = {n: D[n] for n in D if all(s in D[n] for s in slot)}
    vrefs = sorted(set(corp(n) for n in Dv if n.startswith('G-')))
    vq = [n for n in Dv if not n.startswith('G-')]
    so = classify(Dv, slot, vrefs, vq)
    res['slot'] = so
    for q in sorted(vq): print('SLOT', q, so[q]['order'], so[q]['d'])
    save('c2.json', res)
