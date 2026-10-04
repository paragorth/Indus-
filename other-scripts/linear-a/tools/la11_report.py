#!/usr/bin/env python3
"""LA-11 report: from run_<tag>.jsonl -> gap matrix, column-normalised scores, calibration and the Linear A verdict.

G[X][T] = mean bits/symbol of X's shuffles under T  -  bits/symbol of X under T   (structure absorbed by T's best map)
z[X][T] = (G[X][T] - mean_ref) / sd_ref, reference = held-out samples of all real languages NOT in T's family
          (and not X itself): how much better T absorbs X than it absorbs unrelated real words.
"""
import os, sys, json, math, random, collections, statistics as st
import la11_common as c

def load(tag):
    rows = [json.loads(l) for l in open(os.path.join(c.D, f'run_{tag}.jsonl'))]
    rows = [r for r in rows if 'bits' in r]
    B = collections.defaultdict(lambda: collections.defaultdict(dict))
    maps = {}
    for r in rows:
        B[r['input']][r['target']][r['var']] = r['bits']
        if r.get('map'): maps[(r['input'], r['target'])] = r['map']
    return B, maps

def fam(code):
    return c.load_lang(code)['family']

def famgroup(f):
    """coarse family for the 'relative present' tests (IE sub-branches merged, Semitic etc. merged)."""
    return f.split('-')[0]

def analyse(tag, out=sys.stdout, quiet=False):
    B, maps = load(tag)
    targets = sorted({t for X in B for t in B[X]})
    inputs = sorted(B)
    G = {}; SD = {}
    for X in inputs:
        for T in targets:
            d = B[X].get(T, {})
            if 0 not in d or len(d) < 2: continue
            sh = [v for k, v in d.items() if k > 0]
            G[X, T] = st.mean(sh) - d[0]
            SD[X, T] = st.pstdev(sh) if len(sh) > 1 else float('nan')
    langX = [X for X in inputs if X.startswith('X:')]
    def z(X, T):
        ref = [G[Y, T] for Y in langX if (Y, T) in G and Y != X and famgroup(fam(Y[2:])) != famgroup(fam(T))]
        if (X, T) not in G or len(ref) < 5: return float('nan')
        return (G[X, T] - st.mean(ref)) / st.pstdev(ref)
    Z = {(X, T): z(X, T) for X in inputs for T in targets}
    P = lambda *a: print(*a, file=out)
    res = dict(tag=tag, targets=targets)

    # --- calibration on real languages
    self_rank = []; fam_hit = []; fam_chance = []; orphan_max = []; rel_rank = []
    for X in langX:
        code = X[2:]
        sc = sorted(((Z[X, T], T) for T in targets if not math.isnan(Z[X, T])), reverse=True)
        if not sc: continue
        order = [T for _, T in sc]
        if code in order: self_rank.append(order.index(code) + 1)
        others = [T for T in order if T != code]
        fg = famgroup(fam(code))
        rel = [T for T in others if famgroup(fam(T)) == fg]
        if rel:
            fam_hit.append(famgroup(fam(others[0])) == fg)
            fam_chance.append(len(rel) / len(others))
            rel_rank.append(min(others.index(T) for T in rel) + 1)
        unrel = [Z[X, T] for T in others if famgroup(fam(T)) != fg]
        orphan_max.append(max(unrel))
    res['self_rank'] = self_rank; res['fam_hit'] = sum(fam_hit); res['fam_n'] = len(fam_hit); res['fam_chance'] = sum(fam_chance)
    res['orphan_max'] = sorted(orphan_max)
    P(f'== {tag}: {len(inputs)} inputs x {len(targets)} targets')
    P(f'self-identification: top-1 {sum(r == 1 for r in self_rank)}/{len(self_rank)}, top-3 {sum(r <= 3 for r in self_rank)}/{len(self_rank)}, '
      f'median rank {st.median(self_rank)} of {len(targets)} (chance median {(len(targets) + 1) / 2})')
    P(f'relative (self excluded): best target in own family {sum(fam_hit)}/{len(fam_hit)} vs chance {sum(fam_chance):.1f}; '
      f'median best-relative rank {st.median(rel_rank)}')
    P(f'orphan max z (best unrelated language, real inputs): median {st.median(orphan_max):.2f}, 90th pct {sorted(orphan_max)[int(0.9 * len(orphan_max))]:.2f}, max {max(orphan_max):.2f}')

    def show(X, k=8):
        sc = sorted(((Z[X, T], T) for T in targets if not math.isnan(Z[X, T])), reverse=True)
        raw = sorted(((G[X, T], T) for T in targets if (X, T) in G), reverse=True)
        sig = [G[X, T] / SD[X, T] for T in targets if (X, T) in G and SD[X, T] > 0]
        P(f'{X:8s} top z: ' + ', '.join(f'{T} {v:+.2f}' for v, T in sc[:k]) + f' | bottom: ' + ', '.join(f'{T} {v:+.2f}' for v, T in sc[-3:]))
        P(f'{"":8s} raw gap top: ' + ', '.join(f'{T} {v:.3f}' for v, T in raw[:5]) + f' | mean gap {st.mean(v for v, _ in raw):.3f}; gap/sd_shuf median {st.median(sig):.1f}')
        return sc
    out_rows = {}
    for X in ['LA', 'LB0', 'LB1', 'LB2', 'MK0', 'MK1', 'MK2', 'MK3']:
        if X in B: out_rows[X] = show(X)
    # Greek rank for LB
    for X in ['LB0', 'LB1', 'LB2']:
        if X in out_rows:
            order = [T for _, T in out_rows[X]]
            P(f'{X}: grc rank {order.index("grc") + 1 if "grc" in order else None}, ell rank {order.index("ell") + 1 if "ell" in order else None} of {len(order)}')
    res['LB_grc_rank'] = [[T for _, T in out_rows[X]].index('grc') + 1 for X in ['LB0', 'LB1', 'LB2'] if X in out_rows]
    res['LB_ell_rank'] = [[T for _, T in out_rows[X]].index('ell') + 1 for X in ['LB0', 'LB1', 'LB2'] if X in out_rows]
    # LA vs nulls
    la_max = out_rows['LA'][0][0]; la_top = out_rows['LA'][0][1]
    mk_max = [out_rows[X][0][0] for X in out_rows if X.startswith('MK')]
    p_orph = (1 + sum(v >= la_max for v in orphan_max)) / (1 + len(orphan_max))
    P(f'LA max z {la_max:.2f} ({la_top}); orphan-real-language max z exceed it in {sum(v >= la_max for v in orphan_max)}/{len(orphan_max)} (p={p_orph:.2f}); Markov max z {[round(v, 2) for v in mk_max]}')
    lb_max = [out_rows[X][0][0] for X in out_rows if X.startswith('LB')]
    P(f'LB max z {[round(v, 2) for v in lb_max]} (tops {[out_rows[X][0][1] for X in out_rows if X.startswith("LB")]})')
    res.update(LA_top=[(T, round(v, 3)) for v, T in out_rows['LA'][:10]], LA_bottom=[(T, round(v, 3)) for v, T in out_rows['LA'][-5:]],
               LA_max=la_max, p_orphan=p_orph, MK_max=mk_max, LB_max=lb_max)
    # family means: LA and LB, permutation of family labels over targets
    def fam_test(X, nperm=5000, rnd=random.Random(5), fine=False):
        fams = {T: (fam(T) if fine else famgroup(fam(T))) for T in targets}
        vals = {T: Z[X, T] for T in targets if not math.isnan(Z[X, T])}
        groups = collections.defaultdict(list)
        for T, v in vals.items(): groups[fams[T]].append(v)
        obs = {g: st.mean(v) for g, v in groups.items()}
        Ts = list(vals); labels = [fams[T] for T in Ts]; sizes = collections.Counter(labels)
        perm_max = collections.defaultdict(int)
        for _ in range(nperm):
            rnd.shuffle(labels)
            g2 = collections.defaultdict(list)
            for T, l in zip(Ts, labels): g2[l].append(vals[T])
            for g in obs:
                if st.mean(g2[g]) >= obs[g]: perm_max[g] += 1
        return sorted(((obs[g], g, len(groups[g]), (perm_max[g] + 1) / (nperm + 1)) for g in obs), reverse=True)
    for X in ['LA', 'LB0', 'LB1', 'LB2']:
      for fine in (False, True):
        ft = fam_test(X, 2000, fine=fine)
        P(f'{X} {"branch" if fine else "family"} means (z, n targets, one-sided perm p): ' + '; '.join(f'{g} {m:+.2f} (n{n}, p{p:.3f})' for m, g, n, p in ft[:6]))
        res[f'{X}_{"branch" if fine else "family"}'] = [(g, round(m, 3), n, round(p, 4)) for m, g, n, p in ft]
    # convergence check
    sp = [r for r in (json.loads(l) for l in open(os.path.join(c.D, f'run_{tag}.jsonl'))) if 'spread' in r]
    P(f'restart spread (bits/sym): median {st.median(r["spread"] for r in sp):.4f}; gap sd over shuffles median {st.median(v for v in SD.values() if not math.isnan(v)):.4f}')
    # LA best map under top language (opaque, for the record only)
    m = maps.get(('LA', la_top))
    if m: P(f'LA best map under {la_top} (signs -> syllables, top 12 LA signs): ' + ', '.join(f'{k}->{v}' for k, v in list(m.items())[:12]))
    json.dump(res, open(os.path.join(c.D, f'report_{tag}.json'), 'w'), indent=1)
    return res

if __name__ == '__main__':
    for tag in sys.argv[1:]: analyse(tag)
