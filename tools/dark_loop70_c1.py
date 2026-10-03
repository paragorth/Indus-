"""S-DARK-70 cycle 1: extraction of every multi-line text + split rate of each candidate word unit vs three break nulls.
usage: python3 tools/dark_loop70_c1.py [nrep]
writes data/derived/dark/loop70_c1_<src>_<level>.txt and loop70_cycle1.txt (row)."""
import sys, collections, random, math, json
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop70 import *

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
rnd = random.Random(70)
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

def unit_sets_at_level(level):
    """map the W-number unit constants through the merge map (identity for seq_raw)."""
    m = MAP[level]; f = lambda x: m.get(x, x)
    import dark_loop70 as D
    D.FROZEN_SET = {frozenset((f(a), f(b))) for a, b in FROZEN}
    D.S303 = {(f(a), f(b)) for a, b in S303}
    D.HEADS = {f(x) for x in HEADS}; D.OPEN = {f(x) for x in OPEN}; D.FISH = {f(x) for x in FISH}
    D.CL[:] = sorted({f(x) for x in CL})

def prepare(src, level):
    objs = load_wells() if src == 'wells' else load_im77()
    texts = []
    for o in objs:
        seq, br = flat(o)
        if src == 'wells': seq = lv(seq, level)
        if len(seq) < 2: continue
        texts.append(dict(o=o, seq=seq, br=br, wseq=(seq if src == 'wells' else wells_view_of_im77(seq))))
    single = [t['wseq'] for t in texts if not t['br'] and t['o']['complete'] and 0 not in t['seq']]
    QUAL = build_qual(single); LOY = loyal_pairs(single)
    for t in texts:
        t['lab'] = parse(t['wseq'], QUAL)
        t['U'] = gap_units(t['wseq'], t['lab'], QUAL, LOY)
    return texts, QUAL, LOY

def finality_tables(single):
    fin = collections.Counter(); ini = collections.Counter(); tok = collections.Counter()
    for s in single:
        for i, a in enumerate(s):
            tok[a] += 1
            if i == len(s) - 1: fin[a] += 1
            if i == 0: ini[a] += 1
    F = {a: fin[a] / tok[a] for a in tok}; I = {a: ini[a] / tok[a] for a in tok}
    return F, I, tok

def valid_gaps(t):
    return [g for g in range(len(t['seq']) - 1) if 'UNREAD' not in t['U'][g]]

def space_break(t, k, by='width'):
    """gap closest to the k-th of (nb+1) equal shares of total glyph width (or sign count)."""
    seq = t['seq']; n = len(seq)
    w = [(width(a) if by == 'width' else 1.0) or MEDW for a in seq]
    tot = sum(w); cum = []; c = 0
    for g in range(n - 1): c += w[g]; cum.append(c)
    vg = valid_gaps(t)
    if not vg: return []
    nb = len(t['br']); out = []
    for j in range(1, nb + 1):
        target = tot * j / (nb + 1)
        out.append(min(vg, key=lambda g: abs(cum[g] - target)))
    return out

def run(src, level, texts, QUAL, LOY, tag):
    wrapped = [t for t in texts if t['br'] and valid_gaps(t)]
    single = [t['wseq'] for t in texts if not t['br'] and t['o']['complete'] and 0 not in t['seq']]
    F, I, TOK = finality_tables(single)
    P(f'== {tag}: {len(wrapped)} multi-line texts with >= 1 usable gap; {len(single)} complete single-line texts (for finality tables)')
    classes = ['opener_phrase', 'opener_phrase_fixed', 'qual_head', 'qual_head_loyal60', 'qual_head_loyal80', 'qual_head_S303', 'head_suffix',
               'frozen_pair', 'fish_word', 'fish_arrow', 'num_item', 'ANY_UNIT', 'name_name']
    # observed
    obs = collections.Counter(); occ = collections.Counter(); ntext_with = collections.Counter()
    for t in wrapped:
        for g in valid_gaps(t):
            for c in t['U'][g]: occ[c] += 1
        seen = set()
        for g in t['br']:
            if 'UNREAD' in t['U'][g]: continue
            for c in t['U'][g]: obs[c] += 1; seen.add(c)
        for c in seen: ntext_with[c] += 1
    nbreaks = sum(1 for t in wrapped for g in t['br'] if 'UNREAD' not in t['U'][g])
    P(f'usable breaks {nbreaks}; gaps {sum(len(valid_gaps(t)) for t in wrapped)}')
    # finality of pre-break sign, initiality of post-break sign
    def fin_stats(pairs):
        f = [F.get(a, 0.0) for a, b in pairs]; i = [I.get(b, 0.0) for a, b in pairs]
        return (sum(f) / len(f) if f else float('nan'), sum(i) / len(i) if i else float('nan'))
    obs_pairs = [(t['wseq'][g], t['wseq'][g + 1]) for t in wrapped for g in t['br'] if 'UNREAD' not in t['U'][g]]
    obsF, obsI = fin_stats(obs_pairs)
    # empirical relative break positions (for the position-matched null)
    relpos = [(g + 1) / len(t['seq']) for t in wrapped for g in t['br']]
    # nulls
    def draw(t, kind):
        vg = valid_gaps(t); nb = len(t['br'])
        if kind == 'uniform': return rnd.sample(vg, min(nb, len(vg)))
        if kind == 'width': return space_break(t, nb, 'width')
        if kind == 'count': return space_break(t, nb, 'count')
        if kind == 'posmatch':
            out = []
            for _ in range(nb):
                r = rnd.choice(relpos); g = round(r * len(t['seq'])) - 1
                out.append(min(vg, key=lambda x: abs(x - g)))
            return out
    res = {}
    for kind in ['uniform', 'width', 'count', 'posmatch']:
        reps = min(NREP, 1) if kind in ('width', 'count') else NREP
        sims = collections.defaultdict(list); simF = []; simI = []
        for r in range(reps):
            cnt = collections.Counter(); pairs = []
            for t in wrapped:
                for g in draw(t, kind):
                    for c in t['U'][g]: cnt[c] += 1
                    pairs.append((t['wseq'][g], t['wseq'][g + 1]))
            for c in classes: sims[c].append(cnt[c])
            f, i = fin_stats(pairs); simF.append(f); simI.append(i)
        res[kind] = dict(sims=sims, F=simF, I=simI)
    P(f"{'unit class':22s} {'gaps':>5s} {'texts':>5s} {'split':>5s} {'rate':>6s} {'95%CI':>13s} | {'uniform E':>9s} {'P<=':>6s} | {'posmatch E':>10s} {'P<=':>6s} | {'width-mid':>9s} {'count-mid':>9s}")
    rows = {}
    for c in classes:
        k = obs[c]; n = occ[c]; lo, hi = wilson(k, n)
        u = res['uniform']['sims'][c]; pm = res['posmatch']['sims'][c]
        Eu = sum(u) / len(u); Ep = sum(pm) / len(pm)
        Pu = sum(1 for x in u if x <= k) / len(u); Pp = sum(1 for x in pm if x <= k) / len(pm)
        Pu_hi = sum(1 for x in u if x >= k) / len(u); Pp_hi = sum(1 for x in pm if x >= k) / len(pm)
        wm = res['width']['sims'][c][0]; cm = res['count']['sims'][c][0]
        rows[c] = dict(gaps=n, texts=ntext_with[c], split=k, rate=k / n if n else None, ci=(lo, hi), E_uniform=Eu, P_lo_uniform=Pu, P_hi_uniform=Pu_hi,
                       E_posmatch=Ep, P_lo_posmatch=Pp, P_hi_posmatch=Pp_hi, width_mid=wm, count_mid=cm)
        P(f'{c:22s} {n:5d} {ntext_with[c]:5d} {k:5d} {k / n if n else float("nan"):6.3f} [{lo:5.3f},{hi:5.3f}] | {Eu:9.2f} {Pu:6.3f} | {Ep:10.2f} {Pp:6.3f} | {wm:9d} {cm:9d}'
          + ('   (P>= %.3f / %.3f)' % (Pu_hi, Pp_hi) if c == 'name_name' else ''))
    # finality
    for kind in ['uniform', 'posmatch']:
        sf = res[kind]['F']; si = res[kind]['I']
        Pf = sum(1 for x in sf if x >= obsF) / len(sf); Pi = sum(1 for x in si if x >= obsI) / len(si)
        P(f'pre-break sign finality {obsF:.3f} vs {kind} null {sum(sf) / len(sf):.3f} (P>= {Pf:.3f}); post-break sign initiality {obsI:.3f} vs {sum(si) / len(si):.3f} (P>= {Pi:.3f})')
    wf = res['width']['F'][0]; wi = res['width']['I'][0]
    P(f'width-midpoint break would give finality {wf:.3f} / initiality {wi:.3f}; count-midpoint {res["count"]["F"][0]:.3f} / {res["count"]["I"][0]:.3f}')
    # where do breaks fall: relative position histogram and sign-count of the first-read line
    hist = collections.Counter(round(r, 1) for r in relpos)
    P('relative break position (signs before break / n), histogram:', sorted(hist.items()))
    firstlen = collections.Counter(len(t['o']['segs'][0]) for t in wrapped); lastlen = collections.Counter(len(t['o']['segs'][-1]) for t in wrapped)
    P('first-read line length:', sorted(firstlen.items()), ' last line length:', sorted(lastlen.items()))
    # which pre-break signs
    pre = collections.Counter(a for a, b in obs_pairs); post = collections.Counter(b for a, b in obs_pairs)
    P('commonest pre-break signs:', pre.most_common(12)); P('commonest post-break signs:', post.most_common(12))
    # the split units themselves
    splits = collections.Counter()
    for t in wrapped:
        for g in t['br']:
            if 'ANY_UNIT' in t['U'][g]: splits[(t['wseq'][g], t['wseq'][g + 1], tuple(sorted(t['U'][g] - {'ANY_UNIT'})))] += 1
    P('units actually split by a break:', splits.most_common(30))
    return dict(rows=rows, nwrapped=len(wrapped), nbreaks=nbreaks, obsF=obsF, obsI=obsI,
                nullF={k: sum(res[k]['F']) / len(res[k]['F']) for k in res}, nullI={k: sum(res[k]['I']) / len(res[k]['I']) for k in res})

def extraction_report():
    W = load_wells(); M = load_im77()
    P('== (1) EXTRACTION')
    wl = [o for o in W if len(o['segs']) > 1]
    P(f"Wells: {len(W)} recorded faces, {len(wl)} with a '/' line break on one face ({sum(len(o['segs']) - 1 for o in wl)} breaks; "
      f"{sum(1 for o in wl if len(o['segs']) > 2)} with 2 breaks); complete & undamaged {sum(1 for o in wl if o['complete'])}; "
      f"no unread sign {sum(1 for o in wl if all(0 not in s for s in o['segs']))}")
    P('  by object type:', sorted(collections.Counter(o['type'] for o in wl).items(), key=lambda x: -x[1]))
    P('  by site:', sorted(collections.Counter(o['site'] for o in wl).items(), key=lambda x: -x[1]))
    P('  by direction field:', sorted(collections.Counter(o['dir'] for o in wl).items(), key=lambda x: -x[1]))
    objs = collections.defaultdict(list)
    for o in W: objs[o['obj']].append(o)
    ms = {k: v for k, v in objs.items() if len(v) > 1}
    P(f'Wells multi-sided objects (id n.k, SEPARATE texts per S-DARK-37): {len(ms)} objects, {sum(len(v) for v in ms.values())} faces; by type:',
      sorted(collections.Counter(v[0]['type'].split(":")[0] for v in ms.values()).items(), key=lambda x: -x[1]), '; by site:',
      sorted(collections.Counter(v[0]['site'] for v in ms.values()).items(), key=lambda x: -x[1])[:8])
    P(f"  Harappa tablets with >= 2 faces: {sum(1 for v in ms.values() if v[0]['site'] == 'Harappa' and v[0]['type'].startswith('TAB'))}; "
      f"Wells Harappa tablets with a '/' on one face: {sum(1 for o in wl if o['site'] == 'Harappa' and o['type'].startswith('TAB'))}")
    ml = [o for o in M if len(o['segs']) > 1]
    P(f"IM77: {len(M)} sides, {len(ml)} multi-line sides ({sum(len(o['segs']) - 1 for o in ml)} breaks; {sum(1 for o in ml if len(o['segs']) > 2)} with 3 lines); "
      f"no unread sign {sum(1 for o in ml if o['complete'])}; second line a single sign {sum(1 for o in ml if len(o['segs'][1]) == 1)}")
    P('  by object type:', sorted(collections.Counter(o['type'] for o in ml).items(), key=lambda x: -x[1]))
    P('  by site:', sorted(collections.Counter(o['site'] for o in ml).items(), key=lambda x: -x[1]))
    P('  line directions:', sorted(collections.Counter(d for o in ml for d in o['dirs']).items(), key=lambda x: -x[1]))
    msi = collections.defaultdict(list)
    for o in M: msi[o['obj']].append(o)
    P(f'IM77 multi-sided texts: {sum(1 for v in msi.values() if len(v) > 1)}')
    P('Wells break positions (signs read before the break / n) by text length n:')
    bl = collections.defaultdict(list)
    for o in wl:
        seq, br = flat(o)
        for g in sorted(br): bl[len(seq)].append(g + 1)
    for n in sorted(bl): P(f'  n={n:2d}: {sorted(collections.Counter(bl[n]).items())}')
    bl = collections.defaultdict(list)
    for o in ml:
        seq, br = flat(o)
        for g in sorted(br): bl[len(seq)].append(g + 1)
    P('IM77 break positions by text length n:')
    for n in sorted(bl): P(f'  n={n:2d}: {sorted(collections.Counter(bl[n]).items())}')

if __name__ == '__main__':
    extraction_report()
    ALL = {}
    for src, level in [('wells', 'seq_raw'), ('wells', 'seq_strong'), ('wells', 'seq_all'), ('im77', 'seq_raw')]:
        unit_sets_at_level(level if src == 'wells' else 'seq_raw')
        texts, QUAL, LOY = prepare(src, level)
        P(f'\n#### {src} {level}: qualifier sets (60% cover) {{h: sorted(q)}} =', {h: sorted(q) for h, q in QUAL.items()})
        P(f'loyal(>=80%, n>=5) qualifier->head pairs: {sorted(LOY)}')
        ALL[f'{src}_{level}'] = run(src, level, texts, QUAL, LOY, f'{src} {level}')
        if src == 'wells':
            # held-out sites and home separately (seq_raw only to save time)
            if level == 'seq_raw':
                for name, keep in [('home MD+Harappa', lambda o: o['site'] in BIG), ('held-out sites', lambda o: o['site'] not in BIG),
                                   ('seals only', lambda o: o['ot'] == 'seal'), ('complete, no unread sign', lambda o: o['complete'])]:
                    sub = [t for t in texts if keep(t['o']) or not t['br']]
                    ALL[f'wells_{name}'] = run(src, level, sub, QUAL, LOY, f'wells seq_raw {name}')
        else:
            for name, keep in [('home MD+Harappa', lambda o: o['site'] in ('Mohenjodaro', 'Harappa')), ('held-out sites', lambda o: o['site'] not in ('Mohenjodaro', 'Harappa')),
                               ('second line > 1 sign', lambda o: len(o['segs']) > 1 and all(len(s) > 1 for s in o['segs']))]:
                sub = [t for t in texts if keep(t['o']) or not t['br']]
                ALL[f'im77_{name}'] = run(src, level, sub, QUAL, LOY, f'im77 {name}')
    json.dump(ALL, open(OUT + 'loop70_c1.json', 'w'), indent=1, default=str)
    open(OUT + 'loop70_c1_log.txt', 'w').write('\n'.join(LOG) + '\n')
