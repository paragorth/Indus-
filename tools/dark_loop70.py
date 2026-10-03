#!/usr/bin/env python3
"""Loop 70: do scribes' line breaks fall at word boundaries? (see dark_loop70_common.py for data/units)

Cycle 1: rate of breaks INSIDE each candidate unit vs three nulls (uniform random interior cut of the same
         text; ink-midpoint cut; ink-balance-matched cut), Wells canonical + order-agnostic at three merge
         levels, IM77 known order + agnostic, home (MD+HP) vs other sites. Controls: Ur III seal legends
         (word-internal pairs; true line breaks vs random re-breaks), planted Indus breaks (random vs
         unit-avoiding, same sizes as the Wells texts) for power.
Usage: python3 tools/dark_loop70.py 1
"""
import sys, json, random, collections, itertools
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop70_common import *

rng = random.Random(70)
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def ur3_legends():
    out = []
    for line in open(OUT + 'loop61_corpora/ur3_legend_fields.jsonl'):
        d = json.loads(line)
        segs, wordpos = [], []
        for ln in d['lines']:
            seg = []; wp = []
            for wi, w in enumerate(ln.split()):
                sg = [x for x in w.split('-') if x]
                for si, s in enumerate(sg):
                    seg.append(s); wp.append(si > 0)   # True = this sign continues the previous sign's word
            if seg:
                segs.append(tuple(seg)); wordpos.append(wp)
        if len(segs) >= 2:
            out.append(dict(segs=segs, wp=wordpos))
    return out


def ur3_sets(legs):
    inw = collections.Counter(); tot = collections.Counter()
    big = collections.Counter(); lc = collections.Counter(); rc = collections.Counter(); tw = collections.Counter()
    for L in legs:
        T = sum(L['segs'], ()); WP = sum(L['wp'], [])
        # within-line adjacency word status (line breaks are not used: cross-line pairs are excluded here
        # only for the WORD label; they are rare and would bias the label toward 'not in word')
        for seg, wp in zip(L['segs'], L['wp']):
            for i in range(1, len(seg)):
                p = (seg[i - 1], seg[i]); tot[p] += 1; inw[p] += wp[i]
        for p in set(zip(T, T[1:])):
            tw[p] += 1
        for a, b in zip(T, T[1:]):
            big[(a, b)] += 1; lc[a] += 1; rc[b] += 1
    word = {p for p in tot if inw[p] / tot[p] >= 0.8}
    frozen = {p for p, n in tw.items() if n >= 5 and big[p] / lc[p[0]] >= 0.4 and big[p] / rc[p[1]] >= 0.4}
    return dict(QH=set(), OPEN=set(), NUM=set(), FISH=word, FROZEN=frozen, UNIT=word | frozen), set()


def planted(one, sets, frame, W, sizes, n, avoid, reps=200):
    """Indus single-line texts of the right length, broken at random (avoid=False) or only at positions
    not inside a UNIT pair (avoid=True); returns O/E (uniform) and P_low for UNIT over reps."""
    bylen = collections.defaultdict(list)
    for t in one:
        bylen[len(t['seq'])].append(t['seq'])
    oes, sig = [], 0
    for _ in range(reps):
        texts = []
        for k in rng.sample(sizes, min(n, len(sizes))):
            pool = bylen.get(sum(k), [])
            if not pool:
                continue
            s = rng.choice(pool)
            cuts = list(range(1, len(s)))
            if avoid:
                ok = [c for c in cuts if (s[c - 1], s[c]) not in sets['UNIT']]
                cuts = ok or cuts
            c = rng.choice(cuts)
            texts.append(dict(segs=[s[:c], s[c:]]))
        r = category_test(texts, 'known', sets, frame, W, cats=['UNIT'])['UNIT']
        oes.append(r['uniform']['OE']); sig += r['uniform']['p_low'] < 0.05
    return float(np.median(oes)), sig / reps


def cycle1():
    W = widths()
    br = bridge(True)
    P('LOOP 70 cycle 1: break rate inside candidate units vs random / ink-midpoint / ink-balance nulls')
    res = {}
    for lv in LEVELS:
        one, multi = wells(lv)
        sets, frame = unit_sets(one)
        if lv == 'seq_raw':
            P(f'Wells: {len(one)} single-line complete texts (deduplicated), {len(multi)} multi-segment texts '
              f'({collections.Counter(len(t["segs"]) for t in multi)}), home {sum(is_home(t["site"]) for t in multi)}')
            P('Unit set sizes: ' + ', '.join(f'{c} {len(sets[c])}' for c in ('QH', 'OPEN', 'NUM', 'FISH', 'FROZEN', 'UNIT')))
            P('FROZEN pairs: ' + ' '.join(f'{a}-{b}' for a, b in sorted(sets['FROZEN'])))
            P('Observed Wells junctions (canonical order, reading order: last of segment | first of next) with category:')
            jc = collections.Counter()
            for t in multi:
                for p in junctions_known(t['segs']):
                    jc[(p, '/'.join(pair_cat(p, sets, frame)))] += 1
            P('  ' + '; '.join(f'{a}|{b} [{c}] x{n}' for ((a, b), c), n in jc.most_common()))
            P('Segment sizes: ' + str(collections.Counter(tuple(len(s) for s in t['segs']) for t in multi).most_common()))
        for mode in ('canon', 'agnostic'):
            for sub, f in (('all', lambda t: True), ('home', lambda t: is_home(t['site'])), ('other', lambda t: not is_home(t['site']))):
                tx = [t for t in multi if f(t)]
                r = category_test(tx, mode, sets, frame, W)
                res[f'wells|{lv}|{mode}|{sub}'] = r
                P(f'-- Wells {lv} {mode} {sub} (n={len(tx)})')
                for c in CATS:
                    P('   ' + fmt_row(c, r[c]))
    # IM77
    one, multi = im77()
    smap = lambda w: br.get(w, [])
    sets, frame = unit_sets(one, smap)
    P(f'IM77: {len(one)} one-line sides, {len(multi)} multi-line sides (lines: {collections.Counter(len(t["segs"]) for t in multi)}); '
      f'home {sum(is_home(t["site"]) for t in multi)}. Unit sizes: ' + ', '.join(f'{c} {len(sets[c])}' for c in ('QH', 'OPEN', 'NUM', 'FISH', 'FROZEN', 'UNIT')))
    jc = collections.Counter()
    for t in multi:
        for p in junctions_known(t['segs']):
            jc[(p, '/'.join(pair_cat(p, sets, frame)))] += 1
    P('IM77 junctions (line i last | line i+1 first): ' + '; '.join(f'{a}|{b} [{c}] x{n}' for ((a, b), c), n in jc.most_common(60)))
    for mode in ('known', 'agnostic'):
        for sub, f in (('all', lambda t: True), ('home', lambda t: is_home(t['site'])), ('other', lambda t: not is_home(t['site']))):
            tx = [t for t in multi if f(t)]
            r = category_test(tx, mode, sets, frame, {})
            res[f'im77|{mode}|{sub}'] = r
            P(f'-- IM77 {mode} {sub} (n={len(tx)})  [widths: M signs have no font width, all 0.52 -> midpoint = sign-count midpoint]')
            for c in CATS:
                P('   ' + fmt_row(c, r[c]))
    # Ur III positive control
    legs = ur3_legends()
    usets, uframe = ur3_sets(legs)
    P(f'Ur III control: {len(legs)} distinct legends with >= 2 lines; WORD pairs {len(usets["FISH"])} (label FISH below = WORD), FROZEN {len(usets["FROZEN"])}')
    sample = rng.sample(legs, 400)
    for mode in ('known', 'agnostic'):
        r = category_test(sample, mode, usets, uframe, {}, cats=['FISH', 'FROZEN', 'UNIT', 'BOUND'])
        res[f'ur3|{mode}'] = r
        for c in ('FISH', 'FROZEN', 'UNIT', 'BOUND'):
            P(f'   Ur III {mode} ' + fmt_row('WORD' if c == 'FISH' else c, r[c]))
    # Ur III random re-break (negative control)
    rb = []
    for L in sample:
        T = sum(L['segs'], ()); k = len(L['segs'])
        cuts = sorted(rng.sample(range(1, len(T)), k - 1)) if len(T) > k - 1 else None
        if not cuts:
            continue
        b = [0] + cuts + [len(T)]
        rb.append(dict(segs=[T[b[i]:b[i + 1]] for i in range(k)]))
    r = category_test(rb, 'known', usets, uframe, {}, cats=['FISH', 'UNIT'])
    P('   Ur III RANDOM re-break ' + fmt_row('WORD', r['FISH'])); P('   Ur III RANDOM re-break ' + fmt_row('UNIT', r['UNIT']))
    res['ur3|rebreak'] = r
    # Planted Indus power
    one, multi = wells('seq_raw'); sets, frame = unit_sets(one)
    sizes = [tuple(len(s) for s in t['segs']) for t in multi if len(t['segs']) == 2]
    for avoid in (False, True):
        med, power = planted(one, sets, frame, W, sizes, len(sizes), avoid)
        P(f'Planted Indus breaks ({"unit-avoiding" if avoid else "random"}; n={len(sizes)} single-line texts with the Wells size mix): '
          f'median O/E(UNIT, uniform) {med:.2f}; share of 200 replicates with P_low < 0.05: {power:.2f}')
        res[f'planted|avoid={avoid}'] = dict(median_OE=med, power=power)
    json.dump(res, open(OUT + 'loop70_cycle1.json', 'w'), indent=0, default=str)
    open(OUT + 'loop70_cycle1_log.txt', 'w').write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    for c in sys.argv[1:] or ['1']:
        {'1': cycle1}.get(c, lambda: globals()['cycle' + c]())()
