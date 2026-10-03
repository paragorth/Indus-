#!/usr/bin/env python3
"""Cycle 1: do labels track the class/position of what they label?  (text-and-layout features only)

A. Role tests inside mixed pages, cross-page pairs only, null = permute role within page.
   pharma: jar (Lc) vs fragment (Lf); bio: nymph (Ln) vs tube (Lt); zodiac: inner vs outer ring.
   Negative control: role = parity of locus number. Power check: inject a shared 2-glyph suffix
   into a fraction of one role's labels.
B. Zodiac clock: within page-ring, correlation of label similarity with angular closeness;
   null = random circular rotation and full permutation of labels within ring.
C. Parallel zodiac pages: are labels at the same ring + clock position on different pages
   more alike than labels at different positions?  null = random rotation of each ring.
D. Pharma jar-groups: are a jar and its fragments more alike than labels of other groups
   on the same page?  null = permute group among labels within page.
E. Label words in running text of the same page / section vs frequency-matched text words.
"""
import json, math, os, random, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v2_labels_lib import load, sims, SIMNAMES, perm_within, pval, zscore, ROOT

R = 1000
rng = random.Random(408)
labs, text = load()
out = {}


def role_stat(sub, roles, S):
    same = [0.0] * 3; ns = 0
    diff = [0.0] * 3; nd = 0
    n = len(sub)
    for i in range(n):
        for j in range(i + 1, n):
            if sub[i]['folio'] == sub[j]['folio']:
                continue
            s = S[i][j]
            if roles[i] == roles[j]:
                ns += 1
                for k in range(3): same[k] += s[k]
            else:
                nd += 1
                for k in range(3): diff[k] += s[k]
    return [same[k] / ns - diff[k] / nd for k in range(3)]


def simmat(sub):
    n = len(sub)
    S = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            S[i][j] = S[j][i] = sims(sub[i]['g'], sub[j]['g'])
    return S


def role_test(name, sub, rolef, inject=None):
    sub = [dict(l) for l in sub]
    for l in sub:
        l['role'] = rolef(l)
    sub = [l for l in sub if l['role'] is not None]
    if inject:
        frac, role = inject
        r2 = random.Random(7)
        for l in sub:
            if l['role'] == role and r2.random() < frac:
                l['g'] = l['g'][:-2] + ['q', 'q']   # impossible-in-text marker suffix
    S = simmat(sub)
    roles = [l['role'] for l in sub]
    obs = role_stat(sub, roles, S)
    null = [role_stat(sub, perm_within('folio', sub, 'role', rng), S) for _ in range(R // 5)]
    res = {'n': len(sub), 'roles': dict(Counter(roles)), 'pages': len({l['folio'] for l in sub})}
    for k, nm in enumerate(SIMNAMES):
        nk = [x[k] for x in null]
        res[nm] = {'obs': round(obs[k], 4), 'z': round(zscore(obs[k], nk), 2), 'p': round(pval(obs[k], nk), 4)}
    # simple descriptive: mean length and initial glyph by role
    byr = defaultdict(list)
    for l in sub:
        byr[l['role']].append(l)
    res['desc'] = {r: {'len': round(sum(len(l['g']) for l in v) / len(v), 2),
                       'first_o': round(sum(l['g'][:1] == ['o'] for l in v) / len(v), 2),
                       'top_first2': Counter(''.join(l['g'][:2]) for l in v).most_common(4)}
                   for r, v in byr.items()}
    out[name] = res
    print(name, json.dumps(res))


ph = [l for l in labs if l['cls'] in ('jar', 'frag')]
bio = [l for l in labs if l['cls'] in ('nymph', 'tube')]
zod = [l for l in labs if l['cls'] == 'zodiac' and l['ring'] in ('inner', 'outer')]
role_test('A_pharma_jar_vs_frag', ph, lambda l: l['cls'])
role_test('A_pharma_parity_negctl', ph, lambda l: l['n'] % 2)
role_test('A_pharma_power_inject30', ph, lambda l: l['cls'], inject=(0.3, 'jar'))
role_test('A_bio_nymph_vs_tube', [l for l in bio if l['folio'] in
          {x['folio'] for x in bio if x['cls'] == 'tube'} & {x['folio'] for x in bio if x['cls'] == 'nymph'}],
          lambda l: l['cls'])
role_test('A_zodiac_inner_vs_outer', [l for l in zod if l['folio'] in
          {x['folio'] for x in zod if x['ring'] == 'inner'}], lambda l: l['ring'])
role_test('A_zodiac_parity_negctl', zod, lambda l: l['n'] % 2)

# ---------- B. zodiac clock, within ring ----------
rings = defaultdict(list)
for l in labs:
    if l['cls'] == 'zodiac' and l['clock'] is not None and l['ring'] in ('inner', 'outer', 'middle', 'top'):
        rings[(l['folio'], l['ring'])].append(l)
rings = {k: sorted(v, key=lambda l: l['clock']) for k, v in rings.items() if len(v) >= 5}


def angd(a, b):
    d = abs(a - b) % 12
    return min(d, 12 - d)


def clock_stat(assign):
    """assign: dict ring -> list of glyph lists placed at the ring's clock positions.
    Returns mean edit-sim of circularly adjacent pairs minus mean sim of all pairs in ring."""
    adj = []; allp = []
    for k, gl in assign.items():
        n = len(gl)
        for i in range(n):
            adj.append(sims(gl[i], gl[(i + 1) % n])[0])
            for j in range(i + 1, n):
                allp.append(sims(gl[i], gl[j])[0])
    return sum(adj) / len(adj) - sum(allp) / len(allp)


obs_assign = {k: [l['g'] for l in v] for k, v in rings.items()}
obsB = clock_stat(obs_assign)
nullB = []
for _ in range(R // 2):
    a = {}
    for k, gl in obs_assign.items():
        g = gl[:]; rng.shuffle(g); a[k] = g
    nullB.append(clock_stat(a))
out['B_zodiac_adjacent'] = {'rings': len(rings), 'labels': sum(map(len, rings.values())),
                            'obs': round(obsB, 4), 'z': round(zscore(obsB, nullB), 2), 'p': round(pval(obsB, nullB), 4)}
print('B', out['B_zodiac_adjacent'])

# ---------- C. parallel positions across zodiac pages ----------
def parallel_stat(assign_clock):
    """assign_clock: ring-key -> list of (clock, glyphs). Pairs from different pages,
    same ring name: mean sim when |clock diff| <= 0.75h minus when > 2h."""
    keys = list(assign_clock)
    near = [0.0] * 3; nn = 0; far = [0.0] * 3; nf = 0
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ka, kb = keys[a], keys[b]
            if ka[0] == kb[0] or ka[1] != kb[1]:
                continue
            for ca, ga in assign_clock[ka]:
                for cb, gb in assign_clock[kb]:
                    d = angd(ca, cb)
                    if d <= 0.75 or d > 2:
                        s = sims(ga, gb)
                        if d <= 0.75:
                            nn += 1
                            for t in range(3): near[t] += s[t]
                        else:
                            nf += 1
                            for t in range(3): far[t] += s[t]
    return [near[t] / nn - far[t] / nf for t in range(3)]


clk = {k: [(l['clock'], l['g']) for l in v] for k, v in rings.items()}
obsC = parallel_stat(clk)
nullC = []
for _ in range(R // 10):
    a = {}
    for k, v in clk.items():
        sh = rng.random() * 12
        a[k] = [((c + sh) % 12, g) for c, g in v]
    nullC.append(parallel_stat(a))
out['C_zodiac_parallel_position'] = {nm: {'obs': round(obsC[t], 4), 'z': round(zscore(obsC[t], [x[t] for x in nullC]), 2),
                                          'p': round(pval(obsC[t], [x[t] for x in nullC]), 4)} for t, nm in enumerate(SIMNAMES)}
print('C', out['C_zodiac_parallel_position'])

# ---------- D. pharma jar-group cohesion ----------
def group_stat(sub, grp):
    w = []; b = []
    for i in range(len(sub)):
        for j in range(i + 1, len(sub)):
            if sub[i]['folio'] != sub[j]['folio']:
                continue
            s = sims(sub[i]['g'], sub[j]['g'])[0]
            (w if grp[i] == grp[j] else b).append(s)
    return sum(w) / len(w) - sum(b) / len(b)


phg = [l for l in ph if l['group'] is not None and l['group'] >= 0]
obsD = group_stat(phg, [l['group'] for l in phg])
nullD = [group_stat(phg, perm_within('folio', phg, 'group', rng)) for _ in range(R)]
out['D_pharma_group_cohesion'] = {'n': len(phg), 'obs': round(obsD, 4), 'z': round(zscore(obsD, nullD), 2),
                                  'p': round(pval(obsD, nullD), 4)}
print('D', out['D_pharma_group_cohesion'])

# ---------- E. label words in running text ----------
tf = Counter(w for ws in text.values() for w in ws)
page_tf = {p: Counter(ws) for p, ws in text.items()}
sec_of = {l['folio']: l['illus'] for l in labs}
recs = json.load(open(os.path.join(ROOT, 'data', 'derived', 'ZL3b_lines.json')))
for r in recs:
    sec_of.setdefault(r['folio'], r['illus'])
sec_tf = defaultdict(Counter)
for p, ws in text.items():
    sec_tf[sec_of.get(p)].update(ws)
byfreq = defaultdict(list)
for w, f in tf.items():
    if '?' not in w:
        byfreq[f].append(w)
freqs = sorted(byfreq)


def matched_pool(f):
    if len(byfreq[f]) >= 20:
        return byfreq[f]
    pool = []
    for g in freqs:
        if 0.8 * f <= g <= 1.25 * f:
            pool += byfreq[g]
    return pool or byfreq[f]


items = []
for l in labs:
    for w in l['words']:
        if '?' in w or tf[w] == 0:
            continue
        items.append((w, l['folio'], sec_of.get(l['folio']), tf[w], l['cls']))
nolab = sum(1 for l in labs for w in l['words'] if '?' not in w)


def e_stats(sel):
    obs_p = sum(page_tf.get(p, {}).get(w, 0) for w, p, s, f, c in sel)
    obs_s = sum(sec_tf[s].get(w, 0) for w, p, s, f, c in sel)
    nul_p, nul_s = [], []
    pools = [matched_pool(f) for w, p, s, f, c in sel]
    for _ in range(R):
        tp = ts = 0
        for (w, p, s, f, c), pool in zip(sel, pools):
            m = rng.choice(pool)
            tp += page_tf.get(p, {}).get(m, 0)
            ts += sec_tf[s].get(m, 0)
        nul_p.append(tp); nul_s.append(ts)
    mp = sum(nul_p) / R; ms = sum(nul_s) / R
    tot = sum(f for w, p, s, f, c in sel)
    return {'n_tokens': len(sel), 'textfreq_total': tot,
            'same_page_obs': obs_p, 'same_page_null_mean': round(mp, 1), 'ratio_page': round(obs_p / mp, 2) if mp else None,
            'z_page': round(zscore(obs_p, nul_p), 2), 'p_page': round(pval(obs_p, nul_p), 4),
            'same_section_obs': obs_s, 'same_section_null_mean': round(ms, 1), 'ratio_section': round(obs_s / ms, 2) if ms else None,
            'z_section': round(zscore(obs_s, nul_s), 2), 'p_section': round(pval(obs_s, nul_s), 4)}


E = {'label_word_tokens': nolab, 'in_text': len(items)}
E['all'] = e_stats(items)
for c in ('zodiac', 'frag', 'jar', 'cosmo', 'star', 'nymph', 'tube', 'other', 'astro'):
    sel = [x for x in items if x[4] == c]
    if len(sel) >= 10:
        E[c] = e_stats(sel)
# exclude the 'trivial' frequent words (text freq > 100) as a robustness check
E['all_freq_le100'] = e_stats([x for x in items if x[3] <= 100])
out['E_label_in_text'] = E
for k, v in E.items():
    print('E', k, v)
json.dump(out, open(os.path.join(ROOT, 'data', 'results', 'v2_cycle1.json'), 'w'), indent=1)
