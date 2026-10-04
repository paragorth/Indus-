"""v24 cycle 4: (a) how many genuine corrections would separate Latin-like repair from a copied generator and
from a rule-repairing generator? (word-ratio signature, bootstrap over each control's sites);
(b) the 20 after-only Voynich corrections (ZL corr? notes): is the corrected word an ordinary word?
Compared with Latin corrected words and with random tokens of the same length (token frequency excl. self).
"""
import os, sys, json, pickle, math, random
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v24_lib as L
import v24_data as D
from v24_cycle1 import load


def word_signature(lines, sites):
    M = L.Model(lines)
    bw, nw = [], []
    for s in sites:
        if s.kind not in ('sub', 'ins', 'del') or not s.before:
            continue
        pool, pw = L.null_pool(s, M)
        if not pool:
            continue
        cnt = lambda w: max(M.vocab[w] - (1 if w == s.after else 0), 0)
        bw.append(1.0 if cnt(s.before) else 0.0)
        nw.append(float(sum(p * (1 if cnt(w) else 0) for w, p in zip(pool, pw))))
    return np.array(bw), np.array(nw)


def zstat(bw, nw):
    return (bw.sum() - nw.sum()) / math.sqrt((nw * (1 - nw)).sum() + 1e-9)


def main():
    rng = np.random.default_rng(0)
    V = load('voy.pkl'); PL = load('plaoul.pkl')
    pg_lines, pg_sites, _ = D.planted_generator()
    pc_lines, pc_sites, _ = D.planted_copy()
    vocabL = Counter(w for Lw in PL['lines'] for w in Lw)
    lat = [s for s in PL['sites'] if not (s.kind == 'ins' and s.pos == len(s.after) - 1 and s.after[-1] in 'abc' and vocabL[s.after] <= 1)]
    sig = {'latin': word_signature(PL['lines'], lat), 'copy': word_signature(pc_lines, pc_sites), 'gen': word_signature(pg_lines, pg_sites),
           'ambig': word_signature(V['corpus'], V['T2'])}
    out = {k: dict(n=len(v[0]), ratio=float(v[0].mean() / v[1].mean())) for k, v in sig.items()}
    print({k: (v['n'], round(v['ratio'], 2)) for k, v in out.items()})
    # power: ratio statistic at size n; separation = P(stat_A > 95th pct of stat_B)
    def boot(k, n, R=3000):
        bw, nw = sig[k]
        idx = rng.integers(len(bw), size=(R, n))
        return bw[idx].sum(1) / np.maximum(nw[idx].sum(1), 1e-9)
    sep = {}
    for n in (4, 8, 16, 24, 32, 48, 64, 96, 128):
        a, c, g, t = boot('latin', n), boot('copy', n), boot('gen', n), boot('ambig', n)
        sep[n] = dict(latin_vs_copy=float((a > np.percentile(c, 95)).mean()), copy_vs_gen=float((c > np.percentile(g, 95)).mean()),
                      latin_vs_gen=float((a > np.percentile(g, 95)).mean()), ambig_vs_copy=float((t > np.percentile(c, 95)).mean()))
        print(n, {k: round(v, 2) for k, v in sep[n].items()}, flush=True)
    out['separation'] = sep
    # (b) after-only Voynich corrections
    M = L.Model(V['corpus'])
    lens = Counter(len(w) for Lw in V['corpus'] for w in Lw)
    toks = [w for Lw in V['corpus'] for w in Lw]
    rows = []
    for d in V['T1after']:
        a = d['after']
        if '?' in a or '@' in a or not a:
            continue
        w = L.U(a)
        rows.append((d['locus'], a, max(M.vocab[w] - 1, 0)))
    for s in V['T1']:
        rows.append((s.meta['locus'], ''.join(s.after), max(M.vocab[s.after] - 1, 0)))
    obs_word = np.mean([r[2] > 0 for r in rows])
    # null: random tokens with the same length distribution
    bylen = {}
    for w in toks:
        bylen.setdefault(len(w), []).append(w)
    nulls = []
    for _ in range(4000):
        v = []
        for r in rows:
            Lw = len(L.U(r[1])); pool = bylen.get(Lw) or toks
            w = pool[rng.integers(len(pool))]
            v.append(M.vocab[w] - 1 > 0)
        nulls.append(np.mean(v))
    nulls = np.array(nulls)
    out['voy_after'] = dict(n=len(rows), word=float(obs_word), null=float(nulls.mean()), p_low=float(((nulls <= obs_word).sum() + 1) / 4001), rows=rows)
    print('Voynich corrected words (after-state) that occur elsewhere: %d/%d = %.2f; same-length random tokens %.2f; p(low) %.4f' % (
        sum(r[2] > 0 for r in rows), len(rows), obs_word, nulls.mean(), out['voy_after']['p_low']))
    print(rows)
    # Latin: corrected (after) words that occur elsewhere, all 1-1 replacements incl. multi-letter
    Ml = Counter(w for Lw in PL['lines'] for w in Lw)
    la = [Ml[s.after] - 1 > 0 for s in lat]
    ltoks = [w for Lw in PL['lines'][:3000] for w in Lw]
    lnull = np.mean([Ml[w] - 1 > 0 for w in ltoks])
    out['latin_after'] = dict(n=len(la), word=float(np.mean(la)), token_rate=float(lnull))
    print('Latin corrected words occurring elsewhere %.2f (n %d) vs random token %.2f' % (np.mean(la), len(la), lnull))
    json.dump(out, open(os.path.join(L.CK, 'c4_results.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
