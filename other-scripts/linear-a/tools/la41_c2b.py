#!/usr/bin/env python3
"""LA-41 cycle 2b: (i) number-only families against a strict null (amounts shuffled WITHIN each
document, so each list keeps its own amount set; only order can make a family); (ii) the Linear B
known draft -> fair copy control (PY Eo -> En, Es 644 summary, Jn copies) edit profile; (iii) the
LB edit profile with formula series removed (pairs whose matched words occur on > 8 documents)."""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la41_common import *
from la41_c2 import elem_rates, rate, boot, null_sameword, null_positional, load_fams

OUT = os.path.join(HERE, '..', 'loops', 'la41_cycle2.txt')


def lcs_pairs(X, Y):
    n, m = len(X), len(Y)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            H[i + 1][j + 1] = H[i][j] + 1 if X[i][1] == Y[j][1] else max(H[i][j + 1], H[i + 1][j])
    return H[n][m]


def num_scan(docs, seqs):
    by = defaultdict(list)
    for d in seqs: by[docs[d]['site']].append(d)
    res = []
    for s, L in by.items():
        for p in range(len(L)):
            for q in range(p + 1, len(L)):
                a, b = L[p], L[q]
                k = lcs_pairs(seqs[a], seqs[b])
                if k >= 4 and k / min(len(seqs[a]), len(seqs[b])) >= 0.5: res.append((a, b, k))
    return res


def main():
    rng = random.Random(4142)
    rows = []
    LA = la_docs()
    seqs = {d: [(i, it['num']) for i, it in enumerate(LA[d]['items']) if it['num'] is not None and (it['val'] or 0) >= 2] for d in LA}
    seqs = {d: s for d, s in seqs.items() if len(s) >= 4}
    real = num_scan(LA, seqs)
    nul = []
    for r in range(300):
        sh = {}
        for d, s in seqs.items():
            nums = [n for _, n in s]; rng.shuffle(nums); sh[d] = [(i, n) for (i, _), n in zip(s, nums)]
        nul.append(len(num_scan(LA, sh)))
    nm = sum(nul) / len(nul); p = (1 + sum(1 for x in nul if x >= len(real))) / (len(nul) + 1)
    rows.append(f"| LA-41.2d | LA number-only families, strict null: amounts shuffled WITHIN each list (own amount set kept; only order can align), 300 reps. | Real {len(real)} pairs vs null mean {nm:.2f} (max {max(nul)}), P {p:.4f}. | {'ORDER of amounts is copied' if p < 0.05 else 'KILLED: number-only families are shared amount sets (round numbers), not copied order'} |")

    LB = lb_docs()
    fams = load_fams('LB')
    SD = site_df(LB)

    def maxdf(a, b, mp):
        A = LB[a]['items']
        return max(SD[LB[a]['site']][1][A[i]['w']] for i, _ in mp)
    sets = {
        'known draft/copy (PY Eo-En, Es, Jn)': [(a, b) for a, b in fams if any(k in a and k2 in b or k in b and k2 in a for k, k2 in (('Eo', 'En'), ('Es', 'Es'), ('Jn', 'Jn')))],
        'non-formula (matched words on <= 8 docs)': [],
    }
    for a, b in fams:
        mp = set_match(LB[a]['items'], LB[b]['items'])
        if mp and maxdf(a, b, mp) <= 8: sets['non-formula (matched words on <= 8 docs)'].append((a, b))
    for nm_, F in sets.items():
        fl = [(a, b, set_match(LB[a]['items'], LB[b]['items'])) for a, b in F]
        R = elem_rates(LB, fl)
        ops = Counter(); subs = []
        for a, b, mp in fl:
            o, s, am = catalogue(LB[a]['items'], LB[b]['items'], mp)
            o['order_swaps'] = sum(1 for p_ in range(len(mp)) for q in range(p_ + 1, len(mp)) if mp[q][1] < mp[p_][1])
            ops.update(o); subs += ['%s->%s (%s/%s)' % (e[2], e[3], '-'.join(x), '-'.join(y)) for x, y, e in s if e[0] == 'sub']
        line = []
        for k, nm2 in (('w', 'spelling'), ('l', 'logogram'), ('a', 'amount')):
            r, s_, n = rate(R, k)
            ci = boot(R, k, rng) if n else (float('nan'),) * 2
            line.append(f"{nm2} same {s_}/{n} = {r:.2f} [{ci[0]:.2f}-{ci[1]:.2f}]")
        rows.append(f"| LA-41.2e | LB control, {nm_}: {len(F)} robust family pairs; same catalogue. | {'; '.join(line)}. Ops: {', '.join(f'{k} {v}' for k, v in sorted(ops.items()))}. Substitutions: {'; '.join(subs[:25])}{' ...' if len(subs) > 25 else ''} | see verdict |")
    for r in rows: wlog(OUT, r)
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
