#!/usr/bin/env python3
"""LA-3 cycle 1: the libation formula as a natural experiment.

1. Seeds (data-driven): word types of >= 3 signs written on >= 2 distinct
   non-administrative objects (stone/metal/clay vessels, stone objects, inked,
   architecture...).  No published list of "formula words" is used.
2. Each seed is scanned against the sign stream of every non-administrative
   object (dividers removed, so mis-split words are caught) by semi-global
   alignment (free end gaps in the stream; match +2, mismatch -1, gap -1).
   A hit needs matches >= max(3, ceil(0.6 * len(seed))) and no other seed
   claiming the same stretch with a better score.
   Calibration: the same scan on administrative texts (tablets, nodules...)
   and on sign-shuffled non-administrative streams (false-hit rate).
3. Every hit is aligned to its seed (multiple alignment = star alignment
   on the seed).  Substitution columns give sign alternations X~Y.
4. Systematic vs idiosyncratic: a pair X~Y is systematic if it occurs in
   >= 2 different families.  Null: every observed substitution event keeps
   its seed sign X, the variant Y is redrawn from the non-admin sign
   frequency (Y != X); 5000 runs.
5. Geography: per alternation column, do attestations from the same site
   (and same region) agree in variant more than when variants are permuted
   among the attestations of that column?  Pooled statistic, 5000 runs.

Inputs: data/corpus.json.  Output: data/la3_formula.json
"""
import json, os, random, math
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
random.seed(3)
NR = 5000

ADM = {'Tablet', 'Nodule', 'Roundel', 'Sealing', 'Lames (short thin tablet)',
       '3-sided bar', '4-sided bar', 'Label'}

# Approximate site longitudes (deg E), public gazetteer values, used only to
# bin sites into regions.  Region bins: W < 24.8 <= C < 25.35 <= E-C < 25.9 <= E.
LON = {'Iouktas': 25.12, 'Knossos': 25.16, 'Troullos': 25.15, 'Prassa': 25.20,
       'Poros Herakleiou': 25.15, 'Arkhalkhori': 25.27, 'Platanos': 24.97,
       'Kophinas': 25.02, 'Syme': 25.47, 'Psykhro': 25.46, 'Palaikastro': 26.26,
       'Zakros': 26.26, 'Petras': 26.13, 'Vrysinas': 24.47, 'Apodoulou': 24.68,
       'Nerokurou': 24.04, 'Kythera': 22.99, 'Haghia Triada': 24.79,
       'Phaistos': 24.81, 'Malia': 25.49, 'Khania': 24.02, 'Fourni': 25.15,
       'Skoteino Cave': 25.27, 'Kannia': 24.83, 'Tylissos': 25.02,
       'Larani': 24.95, 'Sitia': 26.10, 'Kardamoutsa': 25.0}

def region(site):
    x = LON.get(site)
    if x is None: return 'other'
    if x < 24.8: return 'W'
    if x < 25.35: return 'C'
    if x < 25.9: return 'EC'
    return 'E'

def load():
    return json.load(open(os.path.join(D, 'corpus.json')))

def streams(recs):
    out = []
    for r in recs:
        s = []
        for t in r['tokens']:
            if t['t'] == 'word': s.extend(t['s'])
        if s: out.append((r, s))
    return out

def semi_global(seed, stream):
    """Align all of seed to a substring of stream. Returns best (score, matches, cols, start, end).
    cols: list of (seed_sign or None, stream_sign or None)."""
    n, m = len(seed), len(stream)
    NEG = -10**9
    S = [[0] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        S[i][0] = -i; B[i][0] = 'U'
    for j in range(1, m + 1):
        S[0][j] = 0; B[0][j] = 'S'          # free leading stream
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = S[i-1][j-1] + (2 if seed[i-1] == stream[j-1] else -1)
            u = S[i-1][j] - 1               # seed sign deleted in variant
            l = S[i][j-1] - 1               # extra sign inserted in variant
            best = max(d, u, l)
            S[i][j] = best
            B[i][j] = 'D' if best == d else ('U' if best == u else 'L')
    best_sc = max(S[n]); j_end = max(j for j in range(m + 1) if S[n][j] == best_sc)
    i, j = n, j_end
    cols = []
    while i > 0:
        b = B[i][j]
        if b == 'D': cols.append((seed[i-1], stream[j-1])); i -= 1; j -= 1
        elif b == 'U': cols.append((seed[i-1], None)); i -= 1
        elif b == 'L': cols.append((None, stream[j-1])); j -= 1
        else: break
    cols.reverse()
    # trim inserted signs at the edges (they are free context, not part of the word)
    while cols and cols[0][0] is None: cols.pop(0);
    while cols and cols[-1][0] is None: cols.pop()
    matches = sum(1 for a, b in cols if a is not None and a == b)
    return S[n][j_end], matches, cols, j, j_end

def need(seed):
    return max(3, math.ceil(0.6 * len(seed)))

def accept(seed, sc, mt):
    # strict: >= 60% of the seed matched AND score >= len(seed)
    # (e.g. a 6-sign seed allows 2 substitutions, or 1 substitution + 1 indel)
    return mt >= need(seed) and sc >= len(seed)

def scan(seeds, objs, raw=False):
    hits = []
    for r, s in objs:
        cand = []
        for k, seed in enumerate(seeds):
            # allow several hits per object: repeat on the stream with the hit masked
            st = list(s)
            for _ in range(3):
                sc, mt, cols, a, b = semi_global(seed, st)
                if not accept(seed, sc, mt): break
                cand.append((sc / len(seed), k, a, b, cols))
                for q in range(a, b): st[q] = '#'
        # resolve overlaps: best normalized score first
        cand.sort(key=lambda x: -x[0])
        used = set(); keep = []
        for c in cand:
            span = set(range(c[2], c[3]))
            if len(span & used) > len(span) / 2: continue
            used |= span; keep.append(c)
        for sc, k, a, b, cols in keep:
            if raw: hits.append((r['id'], a, b)); continue
            hits.append({'obj': r['id'], 'site': r['site'], 'support': r['support'],
                         'seed': k, 'score': sc, 'cols': cols,
                         'form': '-'.join(x for _, x in cols if x)})
    return hits

def main():
    recs = load()
    nonadm = [r for r in recs if r['support'] not in ADM]
    adm = [r for r in recs if r['support'] in ADM]
    # ---- seeds, tier 1: exact types on >= 2 non-admin objects
    occ = defaultdict(set)
    for r in nonadm:
        for t in r['tokens']:
            if t['t'] == 'word' and len(t['s']) >= 3: occ[tuple(t['s'])].add(r['id'])
    na_obj = streams(nonadm); ad_obj = streams(adm)
    # tier 2: any non-admin type of >= 3 signs whose strict scan hits >= 2 objects;
    # greedy: most-attested candidates first, a candidate is dropped if half or more
    # of its hits fall on stretches already claimed by an accepted seed.
    cands = sorted(occ, key=lambda k: (-len(occ[k]), -len(k)))
    seeds = []; claimed = defaultdict(set); tier = {}
    for c in cands:
        c = list(c)
        hs = scan([c], na_obj, raw=True)
        objs = {h[0] for h in hs}
        if len(objs) < 2: continue
        new = [h for h in hs if len(set(range(h[1], h[2])) & claimed[h[0]]) < (h[2] - h[1]) / 2]
        if len({h[0] for h in new}) < 2: continue
        seeds.append(c); tier['-'.join(c)] = 1 if len(occ[tuple(c)]) >= 2 else 2
        for o, a, b in hs: claimed[o] |= set(range(a, b))
    rep = ['LA-3 cycle 1: libation formula alignment', '',
           'Seeds (tier 1 = exact type on >=2 objects; tier 2 = single type whose strict variants hit >=2 objects): ' + ', '.join('-'.join(s) + ' (t%d)' % tier['-'.join(s)] for s in seeds)]
    # ---- scan
    hits = scan(seeds, na_obj)
    ad_hits = scan(seeds, ad_obj)
    # shuffled-stream control
    pool = [x for _, s in na_obj for x in s]
    sh_counts = []
    for it in range(20):
        random.shuffle(pool); k = 0; sh = []
        for r, s in na_obj:
            sh.append((r, pool[k:k+len(s)])); k += len(s)
        sh_counts.append(len(scan(seeds, sh)))
    rep.append('Hits on non-admin objects: %d on %d objects; on admin texts: %d (%d texts, rate %.4f/text vs %.3f non-admin); shuffled non-admin streams: mean %.1f (20 runs)'
               % (len(hits), len({h['obj'] for h in hits}), len(ad_hits), len(ad_obj), len(ad_hits)/len(ad_obj),
                  len(hits)/len(na_obj), sum(sh_counts)/20))
    rep.append('Admin hits (false-positive check): ' + '; '.join('%s %s~%s' % (h['obj'], '-'.join(seeds[h['seed']]), h['form']) for h in ad_hits))
    # ---- families and alignment table
    fam = defaultdict(list)
    for h in hits: fam[h['seed']].append(h)
    rep.append('')
    rep.append('Families (star alignment on seed; "." = deleted, "+X" = inserted):')
    events = []   # (family, column, seed sign X, variant sign Y, obj, site)
    columns = defaultdict(list)  # (family, column) -> [(obj, site, variant)]
    for k in sorted(fam):
        seed = seeds[k]
        rep.append('  [%s] n=%d' % ('-'.join(seed), len(fam[k])))
        for h in fam[k]:
            ci = -1; row = []
            for a, b in h['cols']:
                if a is None: row.append('+' + b); continue
                ci += 1
                if b is None: row.append('.'); columns[(k, ci)].append((h['obj'], h['site'], '.')); continue
                row.append(b if a == b else b.lower() if b.isalpha() else b + '!')
                columns[(k, ci)].append((h['obj'], h['site'], b))
                if a != b: events.append((k, ci, a, b, h['obj'], h['site']))
            rep.append('    %-12s %-16s %s' % (h['obj'], h['site'], ' '.join(row)))
    # ---- alternations
    pairs = defaultdict(set); pair_ev = Counter()
    for k, ci, a, b, o, s in events:
        p = tuple(sorted((a, b))); pairs[p].add(k); pair_ev[p] += 1
    rep.append('')
    rep.append('Substitution events: %d; distinct pairs: %d' % (len(events), len(pairs)))
    for p, fs in sorted(pairs.items(), key=lambda x: (-len(x[1]), -pair_ev[x[0]])):
        rep.append('  %s~%s  events %d  families %d (%s)' % (p[0], p[1], pair_ev[p], len(fs), ', '.join('-'.join(seeds[f]) for f in sorted(fs))))
    obs_sys = sum(1 for fs in pairs.values() if len(fs) >= 2)
    # null: redraw the variant sign from non-admin sign frequency
    freq = Counter(pool); signs = list(freq); wts = [freq[s] for s in signs]
    null = []
    for it in range(NR):
        pp = defaultdict(set)
        for k, ci, a, b, o, s in events:
            while True:
                y = random.choices(signs, wts)[0]
                if y != a: break
            pp[tuple(sorted((a, y)))].add(k)
        null.append(sum(1 for fs in pp.values() if len(fs) >= 2))
    mu = sum(null) / NR; sd = (sum((x - mu) ** 2 for x in null) / NR) ** .5
    p = sum(1 for x in null if x >= obs_sys) / NR
    rep.append('Systematic pairs (same pair in >=2 families): observed %d, null %.2f +- %.2f, P(>=obs) = %.4f' % (obs_sys, mu, sd, p))
    # same statistic counting each (family,column) once (repeated copies of one slip do not count twice)
    # ---- geography
    def agree_stat(cols_, key):
        tot = 0
        for c, lst in cols_.items():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if key(lst[i]) == key(lst[j]) and lst[i][0] != lst[j][0]:
                        tot += (lst[i][2] == lst[j][2])
        return tot
    var_cols = {c: l for c, l in columns.items() if len({v for _, _, v in l}) >= 2 and len(l) >= 3}
    rep.append('')
    rep.append('Geography: %d variable columns with >=3 attestations' % len(var_cols))
    for key_name, key in (('site', lambda x: x[1]), ('region', lambda x: region(x[1]))):
        obs = agree_stat(var_cols, key)
        nl = []
        for it in range(NR):
            perm = {}
            for c, l in var_cols.items():
                vs = [v for _, _, v in l]; random.shuffle(vs)
                perm[c] = [(o, s, v) for (o, s, _), v in zip(l, vs)]
            nl.append(agree_stat(perm, key))
        mu2 = sum(nl) / NR; p2 = sum(1 for x in nl if x >= obs) / NR
        pairs_tot = sum(1 for c, l in var_cols.items() for i in range(len(l)) for j in range(i+1, len(l))
                        if key(l[i]) == key(l[j]) and l[i][0] != l[j][0])
        rep.append('  same-%s pairs agreeing: %d of %d, null %.2f, P = %.4f' % (key_name, obs, pairs_tot, mu2, p2))
    for c, l in sorted(var_cols.items()):
        rep.append('    %s col %d (%s): %s' % ('-'.join(seeds[c[0]]), c[1], seeds[c[0]][c[1]],
                   '; '.join('%s/%s:%s' % (o, region(s), v) for o, s, v in l)))
    out = {'seeds': ['-'.join(s) for s in seeds], 'hits': hits, 'admin_hits': ad_hits,
           'events': events, 'pairs': {'~'.join(p): sorted(fs) for p, fs in pairs.items()},
           'systematic_obs': obs_sys, 'systematic_null': [mu, sd, p]}
    json.dump(out, open(os.path.join(D, 'la3_formula.json'), 'w'), ensure_ascii=False, indent=1)
    open(os.path.join(D, 'la3_formula_report.txt'), 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))

if __name__ == '__main__':
    main()
