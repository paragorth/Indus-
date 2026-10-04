"""pe5 cycle 4: the herd-office family -- a real multi-tablet ledger?

Family = tablets that name herds with |M362+X| compounds (found by cycle 2/3 block links).
(a) Herd x tablet table: herd key = base of the |M362+X| compound (variants stripped);
    value of the herd's head line (D3 counting; only N01/N14 occur, so value-set free).
(b) Does a herd's size in the herd account MDP 17,096+ (P008294) predict its head-line
    number in the column-ration lists MDP 17,085 (P008283) and 17,097 (P008295)?
    Spearman rho over shared herds; null = herd labels permuted, 20,000x.
    Negative control: same test against 20 random tablets' first-column numbers matched by
    row count (no herd link expected).
(c) Column totals of the two ration lists (reverse lines) vs sums of their columns.
(d) Ordering: in which ration list is each shared herd larger (sign test).
"""
import json, os, random, re
from collections import defaultdict
from pe5_common import *  # noqa

VS = VSETS['D3']


def val(nums):
    if not nums or any(not isinstance(n, int) for n, _ in nums):
        return None
    if not {c for _, c in nums} <= {'N01', 'N14'}:
        return None
    return sum(n * VS[c] for n, c in nums)


def herd_key(signs):
    for s in signs:
        if s.startswith('|M362+'):
            k = base(s).strip('|').split('+')[1]
            if k != 'X':
                extra = [base(x) for x in signs if x != s]
                return k + ('/' + '/'.join(extra) if extra and k == 'M059' else '')
    if len(signs) == 2 and base(signs[1]) == 'M362':
        return base(signs[0]) + '+M362'
    if len(signs) == 2 and base(signs[0]) == 'M362':
        return base(signs[1]) + '+M362'
    return None


def herds(t):
    out = {}
    for l in t['lines']:
        k = herd_key(l['signs'])
        v = val(l['numerals'])
        if k and v is not None:
            k = k.replace('/M001+M379', '')       # M059 herd written with or without |M001+M379|
            out.setdefault(k, v)
    return out


def spearman(x, y):
    def rk(a):
        s = sorted(range(len(a)), key=lambda i: a[i])
        r = [0.0] * len(a)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and a[s[j + 1]] == a[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rk(x), rk(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
    return num / den if den else 0.0


def perm_p(x, y, rng, reps=20000):
    r0 = spearman(x, y)
    yy = list(y)
    c = 0
    for _ in range(reps):
        rng.shuffle(yy)
        c += spearman(x, yy) >= r0
    return r0, (1 + c) / (1 + reps)


def main():
    rng = random.Random(54)
    T = load()
    m = {t['id']: t for t in T}
    H = {p: herds(m[p]) for p in ('P008294', 'P008389', 'P008283', 'P008295', 'P008905')}
    out = {'table': H}
    for p, h in H.items():
        print(p, m[p]['designation'], h)
    acc = H['P008294']
    res = {}
    for p in ('P008283', 'P008295'):
        shared = sorted(set(acc) & set(H[p]))
        x = [acc[k] for k in shared]
        y = [H[p][k] for k in shared]
        r, pv = perm_p(x, y, rng)
        res[p] = {'shared': shared, 'herd_size': x, 'ration': y, 'rho': round(r, 3), 'p': round(pv, 4),
                  'ratio_sum': round(sum(y) / sum(x), 3)}
        print('herd size vs', p, res[p])
    # pooled (both lists)
    xs, ys = [], []
    for p in ('P008283', 'P008295'):
        xs += res[p]['herd_size']
        ys += res[p]['ration']
    r, pv = perm_p(xs, ys, rng)
    res['pooled'] = {'n': len(xs), 'rho': round(r, 3), 'p': round(pv, 4)}
    print('pooled', res['pooled'])
    # negative control: P008294 herd sizes vs other tablets' column-of-numbers (first n counting lines)
    neg = []
    k = len(res['P008283']['herd_size'])
    cands = [t for t in T if t['id'] not in H]
    for _ in range(200):
        t = rng.choice(cands)
        vals = [val(l['numerals']) for l in t['lines'] if l['surface'] == 'obverse']
        vals = [v for v in vals if v is not None]
        if len(vals) < k:
            continue
        neg.append(spearman(res['P008283']['herd_size'], vals[:k]))
        if len(neg) >= 100:
            break
    res['neg_random_tablets'] = {'n': len(neg), 'mean_rho': round(sum(neg) / len(neg), 3),
                                 'share_ge_obs': round(sum(r >= res['pooled']['rho'] for r in neg) / len(neg), 3)}
    print('neg', res['neg_random_tablets'])
    # (c) column totals
    cols = {}
    for p in ('P008283', 'P008295'):
        colsum = defaultdict(int)
        rows = set()
        for l in m[p]['lines']:
            if l['surface'] != 'obverse':
                continue
            mm = re.match(r'(\d+)\.([a-g])', l['label'])
            v = val(l['numerals'])
            if mm:
                rows.add(mm.group(1))
                if v is not None:
                    colsum[mm.group(2)] += v
        rev = [(l['label'], l['raw']) for l in m[p]['lines'] if l['surface'] == 'reverse']
        cols[p] = {'rows': len(rows), 'obverse_column_sums_preserved': dict(colsum), 'reverse': rev}
        print(p, cols[p])
    # (d) ordering between the two ration lists
    sh = sorted(set(H['P008283']) & set(H['P008295']))
    d = [(k, H['P008283'][k], H['P008295'][k]) for k in sh]
    print('283 vs 295', d)
    out.update({'herd_vs_ration': res, 'columns': cols, 'ration_pairs': d})
    json.dump(out, open(os.path.join(PEDATA, 'pe5_cycle4.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
