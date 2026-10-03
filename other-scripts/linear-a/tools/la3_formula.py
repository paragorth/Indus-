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

MATCH_FRAC = float(os.environ.get('LA3_MATCH_FRAC', '0.6'))
SCORE_FRAC = float(os.environ.get('LA3_SCORE_FRAC', '1.0'))
TAG = os.environ.get('LA3_TAG', '')

def need(seed):
    return max(3, math.ceil(MATCH_FRAC * len(seed)))

def accept(seed, sc, mt):
    # strict: >= 60% of the seed matched AND score >= len(seed)
    # (e.g. a 6-sign seed allows 2 substitutions, or 1 substitution + 1 indel)
    return mt >= need(seed) and sc >= SCORE_FRAC * len(seed)

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
        if any('-' + '-'.join(c) + '-' in '-' + '-'.join(f) + '-' for f in seeds): continue  # sub-form of a seed
        hs = scan([c], na_obj, raw=True)
        objs = {h[0] for h in hs}
        if len(objs) < 2: continue
        new = [h for h in hs if len(set(range(h[1], h[2])) & claimed[h[0]]) < (h[2] - h[1]) / 2]
        if len({h[0] for h in new}) < 2: continue
        seeds.append(c); tier['-'.join(c)] = 1 if len(occ[tuple(c)]) >= 2 else 2
        for o, a, b in hs: claimed[o] |= set(range(a, b))
    rep = ['LA-3 cycle 1: libation formula alignment (match frac %.2f, score frac %.2f)' % (MATCH_FRAC, SCORE_FRAC), '',
           'Seeds (tier 1 = exact type on >=2 objects; tier 2 = single type whose strict variants hit >=2 objects): ' + ', '.join('-'.join(s) + ' (t%d)' % tier['-'.join(s)] for s in seeds)]
    # ---- scan
    hits = scan(seeds, na_obj)
    # merge seeds that are strict variants of each other (either direction) into one family;
    # the earlier (more attested) seed is the reference and member hits are re-aligned to it
    ref = list(range(len(seeds)))
    for j in range(len(seeds)):
        for i in range(j):
            if ref[i] != i: continue
            ok = False
            for x, y in ((seeds[i], seeds[j]), (seeds[j], seeds[i])):
                sc, mt, cols, a, b = semi_global(x, y)
                ok |= accept(x, sc, mt)
            if ok: ref[j] = i; break
    merged_note = ['%s -> %s' % ('-'.join(seeds[j]), '-'.join(seeds[ref[j]])) for j in range(len(seeds)) if ref[j] != j]
    for h in hits:
        if ref[h['seed']] != h['seed']:
            k = ref[h['seed']]
            sc, mt, cols, a, b = semi_global(seeds[k], h['form'].split('-'))
            h['seed'] = k; h['cols'] = cols; h['form'] = '-'.join(x for _, x in cols if x)
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
    rep.append('Merged seed variants: ' + '; '.join(merged_note))
    rep.append('Admin hits (false-positive check): ' + '; '.join('%s %s~%s' % (h['obj'], '-'.join(seeds[h['seed']]), h['form']) for h in ad_hits))
    # ---- families and alignment table
    fam = defaultdict(list)
    for h in hits: fam[h['seed']].append(h)
    # core = tier-1 families with >= 3 attestations; formula objects carry a core hit;
    # formula families = families with >= 2 hits on formula objects
    core = {k for k in fam if tier['-'.join(seeds[k])] == 1 and len(fam[k]) >= 3}
    fobj = {h['obj'] for h in hits if h['seed'] in core}
    formula = {k for k in fam if sum(1 for h in fam[k] if h['obj'] in fobj) >= 2}
    rep.append('Core families: %s; formula objects: %d; formula families: %s'
               % (', '.join('-'.join(seeds[k]) for k in sorted(core)), len(fobj), ', '.join('-'.join(seeds[k]) for k in sorted(formula))))
    rep.append('')
    rep.append('Families (star alignment on seed; lower case = substituted, "." = deleted, "+X" = inserted; * = formula family):')
    events = []   # dict(fam, col, kind, a, b, obj, site)
    columns = defaultdict(list)  # (family, column) -> [(obj, site, variant)]
    for k in sorted(fam):
        seed = seeds[k]
        rep.append('  [%s]%s n=%d' % ('-'.join(seed), '*' if k in formula else '', len(fam[k])))
        for h in fam[k]:
            ci = -1; row = []
            for a, b in h['cols']:
                if a is None:
                    row.append('+' + b)
                    events.append(dict(fam=k, col=ci + .5, kind='ins', a=None, b=b, obj=h['obj'], site=h['site']))
                    continue
                ci += 1
                if b is None:
                    row.append('.'); columns[(k, ci)].append((h['obj'], h['site'], '.'))
                    events.append(dict(fam=k, col=ci, kind='del', a=a, b=None, obj=h['obj'], site=h['site']))
                    continue
                row.append(b if a == b else b.lower() if b.isalpha() else b + '!')
                columns[(k, ci)].append((h['obj'], h['site'], b))
                if a != b: events.append(dict(fam=k, col=ci, kind='sub', a=a, b=b, obj=h['obj'], site=h['site']))
            rep.append('    %-12s %-16s %-14s %s' % (h['obj'], h['site'], h['support'][:14], ' '.join(row)))
    # ---- alternations
    freq = Counter(pool); signs = list(freq); wts = [freq[x] for x in signs]
    def sys_test(evs, label):
        subs = [e for e in evs if e['kind'] == 'sub']
        pairs = defaultdict(set); pev = Counter()
        for e in subs:
            p = tuple(sorted((e['a'], e['b']))); pairs[p].add(e['fam']); pev[p] += 1
        rep.append('')
        rep.append('[%s] substitution events: %d; distinct pairs: %d; deletions %d; insertions %d'
                   % (label, len(subs), len(pairs), sum(e['kind'] == 'del' for e in evs), sum(e['kind'] == 'ins' for e in evs)))
        for p, fs in sorted(pairs.items(), key=lambda x: (-len(x[1]), -pev[x[0]])):
            rep.append('  %s~%s  events %d  families %d (%s)' % (p[0], p[1], pev[p], len(fs), ', '.join('-'.join(seeds[f]) for f in sorted(fs))))
        obs = sum(1 for fs in pairs.values() if len(fs) >= 2)
        obs_ev = sum(1 for e in subs if len(pairs[tuple(sorted((e['a'], e['b'])))]) >= 2)
        null = []; null_ev = []
        for it in range(NR):
            pp = defaultdict(set); keys = []
            for e in subs:
                while True:
                    y = random.choices(signs, wts)[0]
                    if y != e['a']: break
                q = tuple(sorted((e['a'], y))); pp[q].add(e['fam']); keys.append(q)
            null.append(sum(1 for fs in pp.values() if len(fs) >= 2))
            null_ev.append(sum(1 for q in keys if len(pp[q]) >= 2))
        mu = sum(null) / NR; sd = (sum((x - mu) ** 2 for x in null) / NR) ** .5
        pv = sum(1 for x in null if x >= obs) / NR
        mu_e = sum(null_ev) / NR; pv_e = sum(1 for x in null_ev if x >= obs_ev) / NR
        rep.append('  systematic pairs (>=2 families): obs %d, null %.2f +- %.2f, P = %.4f; events in systematic pairs: obs %d, null %.2f, P = %.4f'
                   % (obs, mu, sd, pv, obs_ev, mu_e, pv_e))
        return pairs, (obs, mu, sd, pv)
    pairs_f, st_f = sys_test([e for e in events if e['fam'] in formula], 'formula families')
    pairs_a, st_a = sys_test(events, 'all families')
    # ---- geography
    fev = [e for e in events if e['fam'] in formula]
    att = defaultdict(set)      # (fam) -> set of (obj, site) attestations
    for h in hits:
        if h['seed'] in formula: att[h['seed']].add((h['obj'], h['site']))
    # (a) deviation rate by region: share of formula attestations with any deviation
    dev = {(e['fam'], e['obj']) for e in fev}
    rows = [(k, o, s, (k, o) in dev) for k in att for o, s in att[k]]
    byreg = defaultdict(lambda: [0, 0])
    for k, o, s, d in rows: byreg[region(s)][0] += d; byreg[region(s)][1] += 1
    rep.append('')
    rep.append('Geography (formula families; regions by longitude W/C/EC/E):')
    rep.append('  deviating attestations by region: ' + ', '.join('%s %d/%d' % (r_, v[0], v[1]) for r_, v in sorted(byreg.items())))
    def chi(rows_):
        tab = defaultdict(lambda: [0, 0])
        for k, o, s, d in rows_: tab[region(s)][d] += 1
        n = len(rows_); dtot = sum(r_[3] for r_ in rows_); x = 0
        for r_, (a0, a1) in tab.items():
            t = a0 + a1
            for obs_, exp in ((a1, t * dtot / n), (a0, t * (n - dtot) / n)):
                if exp > 0: x += (obs_ - exp) ** 2 / exp
        return x
    x0 = chi(rows); ds = [r_[3] for r_ in rows]; nl = []
    for it in range(NR):
        random.shuffle(ds); nl.append(chi([(k, o, s, d) for (k, o, s, _), d in zip(rows, ds)]))
    rep.append('  chi2 region x deviation = %.2f, permutation P = %.4f' % (x0, sum(1 for x in nl if x >= x0) / NR))
    # by site, Iouktas vs rest
    io = [r_ for r_ in rows if r_[2] == 'Iouktas']; rest = [r_ for r_ in rows if r_[2] != 'Iouktas']
    rep.append('  Iouktas %d/%d deviating vs other sites %d/%d' % (sum(r_[3] for r_ in io), len(io), sum(r_[3] for r_ in rest), len(rest)))
    # (b) shared deviants: the same non-consensus variant (same family, column, kind, sign) on >= 2 objects
    key = lambda e: (e['fam'], e['col'], e['kind'], e['b'])
    grp = defaultdict(list)
    for e in fev: grp[key(e)].append(e)
    shared = {g: l for g, l in grp.items() if len({e['obj'] for e in l}) >= 2}
    rep.append('  shared deviant variants: ' + '; '.join('%s col %s %s %s: %s' % ('-'.join(seeds[g[0]]), g[1], g[2], g[3] or '', ','.join('%s/%s' % (e['obj'], region(e['site'])) for e in l)) for g, l in shared.items()))
    def same_pairs(groups, keyf):
        t = 0; n = 0
        for l in groups:
            for i in range(len(l)):
                for j in range(i + 1, len(l)):
                    if l[i]['obj'] != l[j]['obj']:
                        n += 1; t += keyf(l[i]['site']) == keyf(l[j]['site'])
        return t, n
    for kn, kf in (('site', lambda x: x), ('region', region)):
        o_, n_ = same_pairs(shared.values(), kf)
        # null: the deviant variants keep their counts but are reassigned to random attestations of the same family
        nl = []
        for it in range(NR):
            gl = []
            for g, l in shared.items():
                pool_ = list(att[g[0]]); pick = random.sample(pool_, min(len(l), len(pool_)))
                gl.append([{'obj': o, 'site': s_} for o, s_ in pick])
            nl.append(same_pairs(gl, kf)[0])
        rep.append('  shared-deviant pairs from same %s: %d of %d, null %.2f, P = %.4f' % (kn, o_, n_, sum(nl) / NR, sum(1 for x in nl if x >= o_) / NR))
    out = {'seeds': ['-'.join(s) for s in seeds], 'formula': sorted(formula), 'core': sorted(core),
           'hits': hits, 'admin_hits': ad_hits, 'events': events,
           'pairs_formula': {'~'.join(p): sorted(fs) for p, fs in pairs_f.items()},
           'pairs_all': {'~'.join(p): sorted(fs) for p, fs in pairs_a.items()},
           'systematic_formula': st_f, 'systematic_all': st_a}
    json.dump(out, open(os.path.join(D, 'la3_formula%s.json' % TAG), 'w'), ensure_ascii=False, indent=1)
    open(os.path.join(D, 'la3_formula%s_report.txt' % TAG), 'w').write('\n'.join(rep) + '\n')
    print('\n'.join(rep))

if __name__ == '__main__':
    main()
