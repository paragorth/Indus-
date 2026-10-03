"""N6-N9: what decides a line's mode, and is a line a fixed template?
T1 nested variance of line mode (section > page > paragraph > line) + factor effects (illus, lang, hand, quire).
T2 positions of q-words and a-words inside mixed lines vs within-line shuffle.
T3 class templates (q/a/o per word) vs within-line shuffle, interior shuffle (first+last fixed) and Markov-3.
T4 special first/last-word vocabularies beyond first/last glyph; words before -m line ends.
T5 same recipes on Italian, Latin, Dante (pseudo pages) and self_citation output.
Run: python3 tools/attack_linetype.py  -> data/results/attack_linetype.json"""
import sys, os, random, math
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, gen

NPERM = 300

# ---------- word classes ----------
def cls_voy(w):
    if w.startswith('q'): return 'q'
    if 'ai' in w or 'ar' in w: return 'a'
    return 'o'

def cls_ita(w):  # Italian: masc/plural-ish endings vs fem endings (an agreement-driven split)
    if w.endswith(('o', 'i')): return 'q'
    if w.endswith(('a', 'e')): return 'a'
    return 'o'

def cls_lat(w):  # Latin: -us/-um/-o vs -a/-ae/-am/-is
    if w.endswith(('us', 'um', 'o')): return 'q'
    if w.endswith(('a', 'ae', 'am', 'is', 'as')): return 'a'
    return 'o'

def score(cs):
    return (cs.count('q') - cs.count('a')) / len(cs)

# ---------- texts ----------
def pseudo_meta(lines, per_page=20, pages_per_sec=10):
    out, para = [], 0
    for i, L in enumerate(lines):
        L = dict(L)
        if L.get('para_start') and i: para += 1
        pg = i // per_page
        L['folio'] = 'p%d' % pg; L['illus'] = 's%d' % (pg // pages_per_sec)
        L['lang'] = L['hand'] = L['quire'] = None
        out.append(L)
    return out

def with_paras(lines):
    """Attach a paragraph id (page-local), using para_start."""
    out, pid, prevf = [], -1, None
    for L in lines:
        L = dict(L)
        if L.get('para_start') or L['folio'] != prevf: pid += 1
        L['para'] = pid; prevf = L['folio']; out.append(L)
    return out

def build_texts():
    T = {}
    for n in ('ZL3b', 'IT2a'):
        T[n] = (with_paras(vlib.load_voynich(n, drop_uncertain=True)), cls_voy)
    T['selfcit(ZL3b)'] = (with_paras(gen.self_citation(T['ZL3b'][0], seed=1)), cls_voy)
    for k, c in (('Italian-Manzoni', cls_ita), ('Italian-Dante', cls_ita), ('Latin-Caesar', cls_lat)):
        T[k] = (with_paras(pseudo_meta(vlib.load_ref(k, max_words=35000))), c)
    return T

# ---------- T1 ----------
def ss_levels(x, sec, page, para):
    n = len(x); g = sum(x) / n
    def means(keys):
        s, c = defaultdict(float), Counter()
        for v, k in zip(x, keys): s[k] += v; c[k] += 1
        return {k: s[k] / c[k] for k in c}
    ms, mp, mq = means(sec), means(page), means(para)
    tot = sum((v - g) ** 2 for v in x)
    S = sum((ms[a] - g) ** 2 for a in sec) / tot
    P = sum((mp[b] - ms[a]) ** 2 for a, b in zip(sec, page)) / tot
    Q = sum((mq[c] - mp[b]) ** 2 for b, c in zip(page, para)) / tot
    return S, P, Q, 1 - S - P - Q

def eta2(x, keys):
    n = len(x); g = sum(x) / n
    s, c = defaultdict(float), Counter()
    for v, k in zip(x, keys): s[k] += v; c[k] += 1
    return sum(c[k] * (s[k] / c[k] - g) ** 2 for k in c) / sum((v - g) ** 2 for v in x)

def zstat(o, null):
    m = sum(null) / len(null); sd = (sum((v - m) ** 2 for v in null) / len(null)) ** .5 or 1e-9
    p = (1 + sum(v >= o for v in null)) / (1 + len(null))
    return {'obs': round(o, 4), 'null': round(m, 4), 'z': round((o - m) / sd, 1), 'p': round(p, 4)}

def t1(lines, cf, rng):
    L = [l for l in lines if len(l['words']) >= 3]
    x = [score([cf(w) for w in l['words']]) for l in L]
    sec = [l['illus'] for l in L]; page = [l['folio'] for l in L]; para = [l['para'] for l in L]
    obs = ss_levels(x, sec, page, para)
    res = {'n_lines': len(L)}
    # section null: shuffle section labels across whole pages
    pages = sorted(set(page)); psec = {p: s for p, s in zip(page, sec)}
    # page null: within section, shuffle lines across pages (page and para labels move together with slots)
    idx_by_sec = defaultdict(list)
    for i, s in enumerate(sec): idx_by_sec[s].append(i)
    idx_by_page = defaultdict(list)
    for i, p in enumerate(page): idx_by_page[p].append(i)
    nS, nP, nQ = [], [], []
    for _ in range(NPERM):
        lab = [psec[p] for p in pages]; rng.shuffle(lab); m = dict(zip(pages, lab))
        nS.append(ss_levels(x, [m[p] for p in page], page, para)[0])
        y = list(x)
        for s, ix in idx_by_sec.items():
            v = [x[i] for i in ix]; rng.shuffle(v)
            for i, vv in zip(ix, v): y[i] = vv
        nP.append(ss_levels(y, sec, page, para)[1])
        y = list(x)
        for p, ix in idx_by_page.items():
            v = [x[i] for i in ix]; rng.shuffle(v)
            for i, vv in zip(ix, v): y[i] = vv
        nQ.append(ss_levels(y, sec, page, para)[2])
    res['section_share'] = zstat(obs[0], nS)
    res['page_within_section_share'] = zstat(obs[1], nP)
    res['para_within_page_share'] = zstat(obs[2], nQ)
    res['line_residual_share'] = round(obs[3], 4)
    # factor effects (labels permuted across pages); within-language for illus and hand
    fac = {}
    for f in ('illus', 'lang', 'hand', 'quire'):
        keys = [l[f] for l in L]
        if all(k is None for k in keys): continue
        pl = {}
        for l in L: pl[l['folio']] = l[f]
        o = eta2(x, keys); nl = []
        for _ in range(NPERM):
            lab = [pl[p] for p in pages]; rng.shuffle(lab); m = dict(zip(pages, lab))
            nl.append(eta2(x, [m[p] for p in page]))
        d = zstat(o, nl)
        mm = defaultdict(list)
        for v, k in zip(x, keys): mm[k].append(v)
        d['mean_score_by_level'] = {str(k): [round(sum(v) / len(v), 3), len(v)] for k, v in sorted(mm.items(), key=lambda t: str(t[0]))}
        fac[f] = d
        if f in ('illus', 'hand'):
            for lg in ('A', 'B'):
                sub = [i for i, l in enumerate(L) if l['lang'] == lg]
                if len(sub) < 100 or all(L[i][f] is None for i in sub): continue
                xs = [x[i] for i in sub]; ks = [L[i][f] for i in sub]; ps = [page[i] for i in sub]
                pgs = sorted(set(ps)); o = eta2(xs, ks); nl = []
                for _ in range(NPERM):
                    lab = [pl[p] for p in pgs]; rng.shuffle(lab); m = dict(zip(pgs, lab))
                    nl.append(eta2(xs, [m[p] for p in ps]))
                fac[f + '_within_lang' + lg] = zstat(o, nl)
    res['factors_eta2'] = fac
    return res

# ---------- T2 ----------
def t2(lines, cf, rng):
    mixed = [[cf(w) for w in l['words']] for l in lines if len(l['words']) >= 4]
    mixed = [c for c in mixed if 'q' in c and 'a' in c]
    def stat(LL, interior):
        sq, nq, sa, na = 0, 0, 0, 0
        prof = {'q': [0] * 5, 'a': [0] * 5}
        for c in LL:
            n = len(c)
            rng_i = range(1, n - 1) if interior else range(n)
            for i in rng_i:
                r = i / (n - 1)
                if c[i] == 'q': sq += r; nq += 1; prof['q'][min(4, int(r * 5))] += 1
                elif c[i] == 'a': sa += r; na += 1; prof['a'][min(4, int(r * 5))] += 1
        if not nq or not na: return 0.0, prof
        return sq / nq - sa / na, prof
    out = {'n_mixed_lines': len(mixed)}
    for interior in (False, True):
        o, prof = stat(mixed, interior); nl = []
        for _ in range(NPERM):
            sh = []
            for c in mixed:
                c = list(c)
                if interior:
                    mid = c[1:-1]; rng.shuffle(mid); c = [c[0]] + mid + [c[-1]]
                else: rng.shuffle(c)
                sh.append(c)
            nl.append(stat(sh, interior)[0])
        d = zstat(o, nl)
        m = sum(nl) / len(nl); sd = (sum((v - m) ** 2 for v in nl) / len(nl)) ** .5
        d['z_two_sided_note'] = 'positive = q-words later than a-words'
        d['profile_q_by_fifth'] = [round(v / max(1, sum(prof['q'])), 3) for v in prof['q']]
        d['profile_a_by_fifth'] = [round(v / max(1, sum(prof['a'])), 3) for v in prof['a']]
        out['interior_only' if interior else 'all_positions'] = d
    return out

# ---------- T3 ----------
def templ_stats(seqs):
    """Template entropy conditional on length (bits/line) and top-20 coverage, lengths 5-10."""
    by = defaultdict(Counter)
    for s in seqs:
        if 5 <= len(s) <= 10: by[len(s)][''.join(s)] += 1
    n = sum(sum(c.values()) for c in by.values())
    H = 0.0
    for c in by.values():
        t = sum(c.values())
        H += t / n * -sum(v / t * math.log2(v / t) for v in c.values())
    allc = Counter()
    for c in by.values(): allc.update(c)
    top = sum(v for _, v in allc.most_common(20)) / n
    return H, top, allc, n

def markov3(seqs, rng):
    tr = defaultdict(Counter)
    for s in seqs:
        st = ('^', '^', '^')
        for c in s: tr[st][c] += 1; st = st[1:] + (c,)
    out = []
    for s in seqs:
        st = ('^', '^', '^'); g = []
        for _ in s:
            k, v = zip(*tr[st].items()); c = rng.choices(k, v)[0]; g.append(c); st = st[1:] + (c,)
        out.append(g)
    return out

def t3(lines, cf, rng):
    seqs = [[cf(w) for w in l['words']] for l in lines]
    H, top, allc, n = templ_stats(seqs)
    res = {'n_lines_len5_10': n, 'H_template_given_len': round(H, 3), 'top20_coverage': round(top, 4)}
    nulls = {'within_line_shuffle': [], 'interior_shuffle': [], 'markov3': []}
    exp = defaultdict(float)
    for _ in range(NPERM // 3):
        sh = [rng.sample(s, len(s)) for s in seqs]
        h, t, c, _ = templ_stats(sh); nulls['within_line_shuffle'].append((h, t))
        for k, v in c.items(): exp[k] += v / (NPERM // 3)
        sh = [s if len(s) < 3 else [s[0]] + rng.sample(s[1:-1], len(s) - 2) + [s[-1]] for s in seqs]
        h, t, _, _ = templ_stats(sh); nulls['interior_shuffle'].append((h, t))
        h, t, _, _ = templ_stats(markov3(seqs, rng)); nulls['markov3'].append((h, t))
    for k, v in nulls.items():
        hs = [a for a, _ in v]; ts = [b for _, b in v]
        mh = sum(hs) / len(hs); sdh = (sum((a - mh) ** 2 for a in hs) / len(hs)) ** .5 or 1e-9
        mt = sum(ts) / len(ts); sdt = (sum((a - mt) ** 2 for a in ts) / len(ts)) ** .5 or 1e-9
        res['vs_' + k] = {'H_null': round(mh, 3), 'H_deficit_bits': round(mh - H, 3), 'z_H_lower': round((mh - H) / sdh, 1),
                          'top20_null': round(mt, 4), 'z_top20': round((top - mt) / sdt, 1)}
    res['top_templates'] = [[k, v, round(exp[k], 1), round(v / max(exp[k], .1), 2)] for k, v in allc.most_common(12)]
    enr = [(k, v, exp[k]) for k, v in allc.items() if v >= 8]
    enr.sort(key=lambda t: -(t[1] - t[2]) / math.sqrt(t[2] + 1))
    res['most_enriched_vs_shuffle'] = [[k, v, round(e, 1)] for k, v, e in enr[:8]]
    return res

# ---------- T4 ----------
def strat_mi(items):
    """items: (stratum, poslabel, word). Sum over strata of n_s * MI(pos; word | s) / N, in bits."""
    by = defaultdict(list)
    for s, p, w in items: by[s].append((p, w))
    N = len(items); tot = 0.0
    for s, L in by.items():
        n = len(L); cp = Counter(p for p, _ in L); cw = Counter(w for _, w in L); cj = Counter(L)
        tot += sum(v / N * math.log2(v * n / (cp[p] * cw[w])) for (p, w), v in cj.items())
    return tot

def t4(lines, cf, rng, glyphmode):
    G = vlib.glyphs if glyphmode else list
    res = {}
    for pos in ('first', 'last'):
        items = []
        for l in lines:
            ws = l['words']
            if len(ws) < 3: continue
            for i, w in enumerate(ws):
                lab = (i == 0) if pos == 'first' else (i == len(ws) - 1)
                g = G(w); key = g[0] if pos == 'first' else g[-1]
                items.append((key, lab, w))
        o = strat_mi(items); nl = []
        by = defaultdict(list)
        for k, (s, p, w) in enumerate(items): by[s].append(k)
        for _ in range(max(50, NPERM // 6)):
            labs = [p for _, p, _ in items]
            for s, ix in by.items():
                v = [labs[i] for i in ix]; rng.shuffle(v)
                for i, vv in zip(ix, v): labs[i] = vv
            nl.append(strat_mi([(s, p, w) for (s, _, w), p in zip(items, labs)]))
        d = zstat(o, nl)
        # most over-represented words at this position relative to same-key interior
        cpos, call = Counter(), Counter(); keyn = Counter(); keypos = Counter()
        for s, p, w in items:
            call[(s, w)] += 1; keyn[s] += 1
            if p: cpos[(s, w)] += 1; keypos[s] += 1
        ex = []
        for (s, w), v in cpos.items():
            e = call[(s, w)] * keypos[s] / keyn[s]
            if v >= 6: ex.append((w, v, round(e, 1), (v - e) / math.sqrt(e + 1)))
        ex.sort(key=lambda t: -t[3])
        d['over_represented_beyond_glyph'] = [[w, v, e] for w, v, e, _ in ex[:8]]
        d['class_share_at_pos'] = {}
        c = Counter(cf(w) for s, p, w in items if p); t = sum(c.values())
        ci = Counter(cf(w) for s, p, w in items if not p); ti = sum(ci.values())
        d['class_share_at_pos'] = {k: [round(c[k] / t, 3), round(ci[k] / ti, 3)] for k in 'qao'}
        res[pos + '_word_MI_beyond_' + ('first' if pos == 'first' else 'last') + '_glyph'] = d
    if glyphmode:
        # lines ending in -m: class of the word before vs other lines' penultimate words; null within-line shuffle of non-last words
        L = [l for l in lines if len(l['words']) >= 4]
        def pen_share(LL):
            m = [cf(l[-2]) for l in LL if l[-1].endswith('m')]
            o = [cf(l[-2]) for l in LL if not l[-1].endswith('m')]
            return {k: (m.count(k) / len(m), o.count(k) / len(o)) for k in 'qao'}, len(m)
        ws = [l['words'] for l in L]
        ob, nm = pen_share(ws)
        nl = defaultdict(list)
        for _ in range(NPERM // 3):
            sh = [rng.sample(w[:-1], len(w) - 1) + [w[-1]] for w in ws]
            s, _ = pen_share(sh)
            for k in 'qao': nl[k].append(s[k][0])
        d = {'n_m_lines': nm}
        for k in 'qao':
            z = zstat(ob[k][0], nl[k]); z['other_lines_penult_share'] = round(ob[k][1], 3)
            d['penult_' + k] = z
        mm = [score([cf(w) for w in l['words'][:-1]]) for l in L if l['words'][-1].endswith('m')]
        mo = [score([cf(w) for w in l['words'][:-1]]) for l in L if not l['words'][-1].endswith('m')]
        d['line_score_m_end_vs_other'] = [round(sum(mm) / len(mm), 3), round(sum(mo) / len(mo), 3)]
        res['m_line_end'] = d
    return res

def main():
    T = build_texts(); out = {}
    for name, (lines, cf) in T.items():
        rng = random.Random(7)
        print('==', name, len(lines), 'lines', flush=True)
        r = {'T1_variance': t1(lines, cf, rng)}
        print(' T1 done', flush=True)
        r['T2_positions'] = t2(lines, cf, rng)
        r['T3_templates'] = t3(lines, cf, rng)
        r['T4_edges'] = t4(lines, cf, rng, glyphmode=name in ('ZL3b', 'IT2a', 'selfcit(ZL3b)'))
        out[name] = r
        import json; print(json.dumps(r)[:3000], flush=True)
    out['_notes'] = {'classes_voynich': 'q = q-initial; a = contains ai or ar (not q-initial); o = other',
                     'classes_italian': 'q = ends -o/-i; a = ends -a/-e; o = other', 'classes_latin': 'q = -us/-um/-o; a = -a/-ae/-am/-is/-as',
                     'score': '(n_q - n_a)/n per line', 'pseudo_pages': 'reference texts: 20 typeset lines per page, 10 pages per section',
                     'nperm': NPERM, 'lines': 'paragraph lines only, words with uncertain glyphs dropped'}
    vlib.save('attack_linetype', out)

if __name__ == '__main__':
    main()
