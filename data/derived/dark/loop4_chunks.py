"""Loop 4 cycle 2: direct hidden-lexicon test. If texts (or the NAME residue) are built from multi-sign WORDS under a wrong
sign-level segmentation, then sign chunks (pairs, triples) should recur across sites more than sign-by-sign chaining
predicts. Statistic: number of distinct chunks seen in BOTH the fit sites (MD+Harappa) and a held-out site set.
Nulls: (A) bigram chaining: fit and test regenerated from the bigram model of the fit set, same lengths (100x);
       (B) unigram shuffle: signs permuted across texts within each set (100x).
Slots: all signs; NAME residue only (S310 parser); frame-only for contrast. Levels seq_raw/seq_strong/seq_all.
Held-out sets: all other named sites; Dholavira; Lothal; Kalibangan; and the MD<->Harappa swap (fit Harappa, test MD).
Arrows fired = levels x slots x chunk sizes x held-out sets; report Bonferroni-corrected P."""
import sys, random, collections, json
sys.path.insert(0, '/home/user/Indus-/data/derived/dark'); sys.path.insert(0, '/home/user/Indus-')
import loop4_seg as L
CYCLE = sys.argv[1] if len(sys.argv) > 1 else '2'
OUT = open(f'/home/user/Indus-/data/derived/dark/loop4_cycle{CYCLE}.txt', 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
rng = random.Random(2)
REPS = 100

def texts_by_site(level):
    seen = set(); out = collections.defaultdict(list)
    for r in L.C:
        s = r[level]
        if not s or r['site'] == 'Unknown': continue
        key = (r['site'], tuple(s))
        if key in seen: continue
        seen.add(key); out[r['site']].append([int(a) for a in s])
    return out
def segments(texts, QUAL, slot):
    """slot='all': whole texts; 'name': NAME runs; 'frame': non-NAME runs"""
    if slot == 'all': return [t for t in texts]
    out = []
    for t in texts:
        lab = L.parse(t, QUAL); cur = []
        for a, l in zip(t, lab):
            keep = (l == 'NAME') if slot == 'name' else (l != 'NAME')
            if keep: cur.append(a)
            else:
                if cur: out.append(cur); cur = []
        if cur: out.append(cur)
    return out
def chunks(segs, n):
    return {tuple(s[i:i + n]) for s in segs for i in range(len(s) - n + 1)}
def shared(fit, test, n): return len(chunks(fit, n) & chunks(test, n))
def null_bigram(fit, test, n, reps):
    m = L.Bigram([[ (a,) for a in s] for s in fit]); out = []
    for _ in range(reps):
        f2 = [[u[0] for u in m.sample(len(s), rng)] for s in fit]; t2 = [[u[0] for u in m.sample(len(s), rng)] for s in test]
        out.append(shared(f2, t2, n))
    return sorted(out)
def null_shuffle(fit, test, n, reps):
    out = []
    for _ in range(reps):
        def sh(segs):
            pool = [a for s in segs for a in s]; rng.shuffle(pool); k = 0; res = []
            for s in segs: res.append(pool[k:k + len(s)]); k += len(s)
            return res
        out.append(shared(sh(fit), sh(test), n))
    return sorted(out)
def pval(obs, null): return (sum(1 for x in null if x >= obs) + 1) / (len(null) + 1)

if __name__ == '__main__':
    import datetime
    P(f'# loop4 cycle {CYCLE}: cross-site chunk recurrence vs bigram chaining ({datetime.datetime.now().isoformat(timespec="minutes")})')
    rows = []
    for level in ('seq_raw', 'seq_strong', 'seq_all'):
        by = texts_by_site(level)
        splits = {'other sites': ({'Mohenjo-daro', 'Harappa'}, None), 'Dholavira': ({'Mohenjo-daro', 'Harappa'}, {'Dholavira'}),
                  'Lothal': ({'Mohenjo-daro', 'Harappa'}, {'Lothal'}), 'Kalibangan': ({'Mohenjo-daro', 'Harappa'}, {'Kalibangan'}),
                  'MD->Harappa swap': ({'Harappa'}, {'Mohenjo-daro'})}
        for sname, (fs, ts) in splits.items():
            fit = [t for s in fs for t in by[s]]; test = [t for s, v in by.items() if (s in ts if ts else s not in fs) for t in v]
            QUAL = L.learn_qual(fit)
            for slot in ('all', 'name', 'frame'):
                F = segments(fit, QUAL, slot); T = segments(test, QUAL, slot)
                for n in (2, 3):
                    o = shared(F, T, n)
                    if o < 3 and n == 3 and slot == 'frame': continue
                    nb = null_bigram(F, T, n, REPS); ns = null_shuffle(F, T, n, REPS)
                    rows.append(dict(level=level, split=sname, slot=slot, n=n, obs=o, big_med=nb[len(nb)//2], big_max=nb[-1], p_big=pval(o, nb),
                                     shuf_med=ns[len(ns)//2], p_shuf=pval(o, ns), nfit=len(F), ntest=len(T)))
                    r = rows[-1]
                    P(f'  {level:10s} {sname:17s} {slot:5s} n={n} shared={o:4d}  bigram-null med {r["big_med"]:4d} max {r["big_max"]:4d} P={r["p_big"]:.3f}'
                      f'  shuffle-null med {r["shuf_med"]:4d} P={r["p_shuf"]:.3f}  (fit segs {len(F)}, test segs {len(T)})')
    json.dump(rows, open(f'/home/user/Indus-/data/derived/dark/loop4_cycle{CYCLE}_chunks.json', 'w'))
    k = len(rows); P(f'\narrows fired: {k}; Bonferroni threshold P < {0.05 / k:.4f}')
    sig = [r for r in rows if r['p_big'] < 0.05 / k]
    P('rows beating the bigram null after correction: ' + (', '.join(f'{r["level"]}/{r["split"]}/{r["slot"]}/n={r["n"]} ({r["obs"]} vs {r["big_med"]})' for r in sig) or 'none'))
    above = [r for r in rows if r['obs'] > r['big_med']]; below = [r for r in rows if r['obs'] < r['big_med']]
    P(f'rows above bigram median: {len(above)}, below: {len(below)}; NAME-slot rows above: {sum(1 for r in above if r["slot"]=="name")} of {sum(1 for r in rows if r["slot"]=="name")}')
    for slot in ('all', 'name', 'frame'):
        rs = [r for r in rows if r['slot'] == slot and r['big_med']]
        P(f'  {slot}: mean obs/bigram-median ratio {sum(r["obs"]/r["big_med"] for r in rs)/len(rs):.3f}; mean obs/shuffle-median {sum(r["obs"]/max(r["shuf_med"],1) for r in rs)/len(rs):.3f}')
