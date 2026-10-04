"""v6 cycle 4: dissect the left-edge vertical link found in cycle 3 (first glyph of line r predicts first glyph of line r+1).
(a) nulls that keep position-in-paragraph: lines at the same position index swapped between paragraphs of the same
    folio, and of the same Currier language + section; plus within-paragraph lag profile (1..4).
(b) repeat vs transition: share of pairs with the SAME first glyph (obs/null), and MI on pairs with different glyphs only.
(c) which glyph pairs carry it (obs/expected under the within-paragraph null).
(d) column specificity: first glyph of the word in column c (0..3) of line r vs column c of line r+1; the line-wrap pair
    (last glyph of line r -> first glyph of line r+1); second glyph of the line.
Controls: letter acrostic (positive, verbose Latin), Latin prose (negative), and a 'scribal persistence' generator
(prose whose first glyph is copied from the line above with p = 0.3; positive for persistence, not for acrostic).
Paragraph-first lines excluded throughout (they carry the gallows rule)."""
import sys, os, json, random
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from v6_gridlib import voynich_paragraphs, shapes_of, mi, units_voy, units_plain
from v6_cycle3 import letter_acrostic, prose, encode_grid, latin_words

REPS = int(os.environ.get('REPS', 200))

def voy_meta(name):
    """paragraphs with folio, lang, illus for the cross-paragraph nulls."""
    lines = vlib.load_voynich(name, drop_uncertain=True); paras, cur, fol = [], None, None
    for L in lines:
        if L['para_start'] or L['folio'] != fol:
            if cur and len(cur['lines']) >= 3: paras.append(cur)
            cur = {'folio': L['folio'], 'key': (L.get('lang'), L.get('illus')), 'lines': []}
        fol = L['folio']; cur['lines'].append(list(L['words']))
        if L.get('para_end'):
            if len(cur['lines']) >= 3: paras.append(cur)
            cur = {'folio': L['folio'], 'key': (L.get('lang'), L.get('illus')), 'lines': []}
    if cur and len(cur['lines']) >= 3: paras.append(cur)
    return paras

def feat(l, U, kind, c=0):
    if kind == 'fg': return U(l[0])[0]
    if kind == 'g2': u = U(l[0]); return u[1] if len(u) > 1 else '#'
    if kind == 'col': return U(l[c])[0] if len(l) > c else None
    if kind == 'lg': return U(l[-1])[-1]

def pairs(paras, U, kind='fg', lag=1, c=0, wrap=False):
    P = []
    for p in paras:
        q = p[1:]
        for i in range(len(q) - lag):
            if wrap: P.append((U(q[i][-1])[-1], U(q[i + 1][0])[0])); continue
            a, b = feat(q[i], U, kind, c), feat(q[i + lag], U, kind, c)
            if a is not None and b is not None: P.append((a, b))
    return P

def offdiag_mi(P): return mi([x for x in P if x[0] != x[1]])
def same_rate(P): return sum(a == b for a, b in P) / len(P)

def perm_within(paras, rng): return [[p[0]] + rng.sample(p[1:], len(p) - 1) for p in paras]

def perm_samepos(paras, groups, rng):
    """swap lines with the same position index among paragraphs of the same group."""
    new = [list(p) for p in paras]
    for g in groups.values():
        maxlen = max(len(paras[i]) for i in g)
        for r in range(1, maxlen):
            idx = [i for i in g if len(paras[i]) > r]
            src = [paras[i][r] for i in idx]; rng.shuffle(src)
            for i, l in zip(idx, src): new[i][r] = l
    return new

def zstat(o, xs):
    m = sum(xs) / len(xs); sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** .5
    return {'obs': o, 'null': m, 'ex': o - m, 'z': (o - m) / sd if sd else 0.0}

def analyse(paras, U, groups=None):
    R = {}
    nulls = {'within-para': lambda rng: perm_within(paras, rng)}
    if groups:
        for gname, G in groups.items():
            nulls['samepos-' + gname] = (lambda G: lambda rng: perm_samepos(paras, G, rng))(G)
    obsP = pairs(paras, U)
    for nn, f in nulls.items():
        sims = [f(random.Random(s)) for s in range(REPS)]
        sP = [pairs(q, U) for q in sims]
        R['MI ' + nn] = zstat(mi(obsP), [mi(x) for x in sP])
        R['offdiagMI ' + nn] = zstat(offdiag_mi(obsP), [offdiag_mi(x) for x in sP])
        R['same-rate ' + nn] = zstat(same_rate(obsP), [same_rate(x) for x in sP])
        if nn == 'within-para':
            for lag in (2, 3, 4):
                R['MI lag%d within-para' % lag] = zstat(mi(pairs(paras, U, lag=lag)), [mi(pairs(q, U, lag=lag)) for q in sims[:60]])
            for c in (1, 2, 3):
                R['MI column%d within-para' % c] = zstat(mi(pairs(paras, U, 'col', c=c)), [mi(pairs(q, U, 'col', c=c)) for q in sims[:60]])
            R['MI second-glyph within-para'] = zstat(mi(pairs(paras, U, 'g2')), [mi(pairs(q, U, 'g2')) for q in sims[:60]])
            R['MI wrap last->first within-para'] = zstat(mi(pairs(paras, U, wrap=True)), [mi(pairs(q, U, wrap=True)) for q in sims[:60]])
            # glyph-pair table
            oc = Counter(obsP); nc = Counter(x for q in sP for x in q)
            tab = []
            for k, v in oc.items():
                e = nc[k] / len(sP)
                if v >= 15: tab.append((round(v / e, 2), k[0] + '>' + k[1], v, round(e, 1)))
            tab.sort(reverse=True); R['top pairs'] = tab[:12]; R['bottom pairs'] = sorted(tab)[:8]
    return R

def persistence(shapes, cover, p=0.3, seed=3):
    rng = random.Random(seed); g = prose(shapes, cover); out = []
    for para in g:
        q = [list(para[0])]
        for l in para[1:]:
            l = list(l)
            if rng.random() < p:
                prev0 = q[-1][0][0]; l[0] = prev0 + l[0][1:]
            q.append(l)
        out.append(q)
    return out

if __name__ == '__main__':
    res = {}
    for name in ('ZL3b', 'IT2a'):
        vm = voy_meta(name); paras = [p['lines'] for p in vm]
        G1, G2 = defaultdict(list), defaultdict(list)
        for i, p in enumerate(vm): G1[p['folio']].append(i); G2[p['key']].append(i)
        res['Voynich-' + name] = analyse(paras, units_voy, {'folio': dict(G1), 'lang+illus': dict(G2)})
    vz = [p for p in voynich_paragraphs('ZL3b') if len(p) >= 3]; shapes = shapes_of(vz)
    caes = latin_words('Latin-Caesar'); desc = latin_words('Latin-Descartes')
    res['Latin letter-acrostic (pos)'] = analyse(encode_grid(letter_acrostic(shapes, caes, [c for w in desc for c in w]), None), units_plain)
    res['Latin prose (neg)'] = analyse(encode_grid(prose(shapes, caes), None), units_plain)
    res['Latin persistence p=.3'] = analyse(encode_grid(persistence(shapes, caes), None), units_plain)
    for n, R in res.items():
        print('\n==', n)
        for k, v in R.items():
            if isinstance(v, dict): print('  %-34s obs %.4f null %.4f excess %+.4f z %6.2f' % (k, v['obs'], v['null'], v['ex'], v['z']))
            else: print('  %-34s %s' % (k, v))
    json.dump(res, open(os.path.join(vlib.RES, 'v6_cycle4.json'), 'w'), indent=0)
