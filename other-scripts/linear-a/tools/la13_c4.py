#!/usr/bin/env python3
"""LA-13 cycle 4: scribes as dialects (idiolect spelling conventions inside one archive).
Cycles 1-3 lacked power because sites share few words. Scribes of one archive share names and terms, so a scribe's
private spelling convention (sign X where colleagues write Y) should leave clean one-sign pairs. Groups = attributed
scribes with >= 15 word tokens (lineara.xyz scribe field); unattributed documents are dropped.
Null: scribe labels permuted over the attributed documents (and within site). Positive control: 3 rules planted in the
most connected scribe (rate 1.0 and 0.5). Negative: attributed docs split at random into two pseudo-scribes."""
import sys, os, json, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la13_common import *
from la13_c1 import plant, pct
from la5_common import ADMIN
import json as _j

def scribe_units(minlen=2, ht_only=False):
    C = _j.load(open(os.path.join(HERE, '..', 'data', 'corpus.json')))
    sc = {c['id']: (c.get('scribe') or '', c['site']) for c in C}
    out = []
    for u in la_units(minlen):
        s, site = sc.get(u[0], ('', ''))
        if not s: continue
        if ht_only and site != 'Haghia Triada': continue
        out.append((u[0], s, site, u[3]))   # slot 2 = site (used for within-site stratified shuffles)
    cnt = collections.Counter()
    for u in out: cnt[u[1]] += len(u[3])
    return [u for u in out if cnt[u[1]] >= 15]

def run(units, rnd, nperm, say, tag):
    T = types_by_group(units)
    allg = collections.Counter(w for g in T for w in T[g])
    S, P = summary_stats(T)
    R = rule_support(P, False)
    ranked = sorted(R.items(), key=lambda kv: -kv[1][0])
    nul = {'plain': [], 'site': []}
    for k in range(nperm):
        for mode in nul:
            labs = shuffle_labels(units, rnd, mode == 'site')
            s2, _ = summary_stats(types_by_group(units, labs)); nul[mode].append(s2)
    say(f'{tag}: groups {len(T)} types {sum(len(v) for v in T.values())} shared(>=2 groups) {sum(1 for c in allg.values() if c >= 2)} | real {S}')
    for mode, v in nul.items():
        for key in ('any_max', 'any_n3', 'npairs'):
            xs = [x[key] for x in v]
            say(f'   null {mode} {key}: real {S[key]} mean {sum(xs)/len(xs):.2f} 95% {pct(xs,0.95)} P(>=) {sum(1 for x in xs if x >= S[key])/len(xs):.3f} P(<=) {sum(1 for x in xs if x <= S[key])/len(xs):.3f}')
    for r, v in ranked[:8]:
        fw = sum(1 for x in nul['plain'] if x['any_max'] >= v[0]) / nperm
        say('   ', r, 'k', v[0], 'FWER-P', f'{fw:.3f}', [('-'.join(a), '-'.join(b)) for a, b in v[2][:6]])
    return S, nul, ranked

def main():
    out = open(os.path.join(OUT, 'c4_report.txt'), 'w')
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()
    rnd = random.Random(44)
    for ht in (False, True):
        U = scribe_units(ht_only=ht)
        run(U, rnd, 300, say, 'HT scribes' if ht else 'all scribes')
    # planted control on the HT scribes
    U = scribe_units(ht_only=True)
    big = collections.Counter()
    for u in U: big[u[1]] += len(u[3])
    tgt = big.most_common(1)[0][0]
    sc = collections.Counter(s for u in U if u[1] == tgt for w in u[3] for s in w)
    cand = [s for s, c in sc.items() if 4 <= c <= 20 and not s.startswith('*')]
    hits = []
    for rate in (1.0, 0.5):
        for rep in range(3):
            xs = rnd.sample(cand, 3); ys = rnd.sample([s for s in sc if s not in xs and not s.startswith('*')], 3)
            rules = list(zip(xs, ys))
            U2 = plant(U, tgt, rules, rate, rnd)
            T = types_by_group(U2); S, P = summary_stats(T); R = rule_support(P, False)
            mx = []
            for k in range(100):
                labs = shuffle_labels(U2, rnd, False); mx.append(summary_stats(types_by_group(U2, labs))[0]['any_max'])
            thr = pct(mx, 0.95)
            got = []
            for x, y in rules:
                k = max([v[0] for r, v in R.items() if (r[2], r[3]) in ((x, y), (y, x))] or [0])
                got.append((x, y, k))
            n = sum(1 for g in got if g[2] > thr)
            hits.append(n)
            say(f'planted in {tgt} rate {rate} rep {rep}: {got} null95 {thr} recovered {n}/3')
    say('planted total recovered', sum(hits), '/', 3 * len(hits))
    # negative: random 2-way split of attributed HT docs
    for rep in range(3):
        U3 = [(u[0], rnd.choice(('S1', 'S2')), u[2], u[3]) for u in U]
        T = types_by_group(U3); S, P = summary_stats(T)
        mx = [summary_stats(types_by_group(U3, shuffle_labels(U3, rnd, False)))[0]['any_max'] for _ in range(100)]
        say(f'random split {rep}: real max {S["any_max"]} null95 {pct(mx,0.95)}')
    out.close()

if __name__ == '__main__':
    main()
