"""Extras for attack_linetype: (a) split-half check: is there true line-level mode beyond the page?
(b) 5-class templates (q, a, c=ch/sh-initial, y=other ending -y, o) vs within-line and interior shuffles.
(c) T2 position effect by Currier language. Results merged into data/results/attack_linetype.json under 'extra'."""
import sys, os, random, json, math
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, gen
from attack_linetype import cls_voy, score, templ_stats, t2, cls_ita, cls_lat, pseudo_meta, with_paras

def cls5(w):
    c = cls_voy(w)
    if c != 'o': return c
    if w.startswith(('ch', 'sh')): return 'c'
    if w.endswith('y'): return 'y'
    return 'o'

def pearson(a, b):
    n = len(a); ma = sum(a)/n; mb = sum(b)/n
    return sum((x-ma)*(y-mb) for x, y in zip(a, b)) / math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))

def splithalf(lines, cf, rng):
    """Score odd and even word positions of each line separately, demean each by page (using other lines' halves),
    correlate. >0 means lines carry their own mode beyond the page mean. Null: swap lines within page (should be ~0)."""
    L = [l for l in lines if len(l['words']) >= 6]
    def halves(l):
        c = [cf(w) for w in l["words"]]; h = len(c) // 2; return score(c[:h]), score(c[len(c) - h:])  # first vs second half (odd/even was confounded by adjacent-word agreement)
    H = [halves(l) for l in L]
    bypg = defaultdict(list)
    for i, l in enumerate(L): bypg[l['folio']].append(i)
    a, b = [], []
    for p, ix in bypg.items():
        if len(ix) < 3: continue
        sa = sum(H[i][0] for i in ix); sb = sum(H[i][1] for i in ix); n = len(ix)
        for i in ix:  # leave-one-out page mean
            a.append(H[i][0] - (sa - H[i][0])/(n-1)); b.append(H[i][1] - (sb - H[i][1])/(n-1))
    r_raw = pearson([h[0] for h in H], [h[1] for h in H])
    r_dm = pearson(a, b)
    # null: pair odd-half of a line with even-half of another line on same page
    nl = []
    for _ in range(200):
        a2, b2 = [], []
        for p, ix in bypg.items():
            if len(ix) < 3: continue
            sh = ix[:]; rng.shuffle(sh); n = len(ix)
            sa = sum(H[i][0] for i in ix); sb = sum(H[i][1] for i in ix)
            for i, j in zip(ix, sh):
                a2.append(H[i][0] - (sa - H[i][0])/(n-1)); b2.append(H[j][1] - (sb - H[j][1])/(n-1))
        nl.append(pearson(a2, b2))
    m = sum(nl)/len(nl); sd = (sum((x-m)**2 for x in nl)/len(nl))**.5
    return {'n_lines': len(L), 'r_halves_raw': round(r_raw, 3), 'r_halves_page_demeaned': round(r_dm, 3),
            'null_mean': round(m, 3), 'z': round((r_dm - m)/sd, 1)}

def templ5(lines, cf, rng):
    seqs = [[cf(w) for w in l['words']] for l in lines]
    H, top, allc, n = templ_stats(seqs)
    res = {'H': round(H, 3), 'top20': round(top, 4)}
    exp = defaultdict(float); R = 60
    for name in ('within_line_shuffle', 'interior_shuffle'):
        hs, ts = [], []
        for _ in range(R):
            if name == 'within_line_shuffle': sh = [rng.sample(s, len(s)) for s in seqs]
            else: sh = [s if len(s) < 3 else [s[0]] + rng.sample(s[1:-1], len(s)-2) + [s[-1]] for s in seqs]
            h, t, c, _ = templ_stats(sh); hs.append(h); ts.append(t)
            if name == 'interior_shuffle':
                for k, v in c.items(): exp[k] += v / R
        mh = sum(hs)/R; sdh = (sum((x-mh)**2 for x in hs)/R)**.5; mt = sum(ts)/R; sdt = (sum((x-mt)**2 for x in ts)/R)**.5
        res['vs_' + name] = {'H_deficit_bits': round(mh - H, 3), 'z_H': round((mh - H)/sdh, 1), 'top20_null': round(mt, 4), 'z_top20': round((top-mt)/sdt, 1)}
    res['top_templates_obs_exp_interior'] = [[k, v, round(exp[k], 1)] for k, v in allc.most_common(10)]
    enr = sorted([(k, v, exp[k]) for k, v in allc.items() if v >= 6], key=lambda t: -(t[1]-t[2])/math.sqrt(t[2]+1))
    res['most_enriched_vs_interior_shuffle'] = [[k, v, round(e, 1)] for k, v, e in enr[:8]]
    return res

out = {}
T = {}
for n in ('ZL3b', 'IT2a'):
    T[n] = (with_paras(vlib.load_voynich(n, drop_uncertain=True)), cls_voy)
T['selfcit(ZL3b)'] = (with_paras(gen.self_citation(T['ZL3b'][0], seed=1)), cls_voy)
for k, c in (('Italian-Manzoni', cls_ita), ('Latin-Caesar', cls_lat)):
    T[k] = (with_paras(pseudo_meta(vlib.load_ref(k, max_words=35000))), c)
for name, (lines, cf) in T.items():
    rng = random.Random(11); r = {'splithalf_line_mode_beyond_page': splithalf(lines, cf, rng)}
    if name in ('ZL3b', 'IT2a', 'selfcit(ZL3b)'):
        r['templates_5class'] = templ5(lines, cls5, rng)
    if name in ('ZL3b', 'IT2a'):
        for lg in ('A', 'B'):
            t = t2([l for l in lines if l['lang'] == lg], cf, rng)
            r['T2_lang' + lg] = {'n_mixed': t['n_mixed_lines'], 'all': [t['all_positions']['obs'], t['all_positions']['z']], 'interior': [t['interior_only']['obs'], t['interior_only']['z']]}
    out[name] = r; print(name, json.dumps(r), flush=True)
p = os.path.join(vlib.RES, 'attack_linetype.json'); d = json.load(open(p)); d['extra'] = out
json.dump(d, open(p, 'w'), indent=1, ensure_ascii=False)
