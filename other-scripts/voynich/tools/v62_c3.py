"""v62 cycle 3: REFRAINS. Line-level patterns that recur at FIXED intervals on a page.

Signature of a line at a position (first word, last word, second word as interior comparator, or the
whole-line skeleton) under thousands of random definitions (whole word, last/first k glyphs, glyph class
merges, word-length skeletons). Statistic: number of equal consecutive recurrence gaps (A..A..A with the
same spacing, gap <= 8) against within-page line shuffles (exact null per definition, numba), as excess
per 100 lines. Edge score = excess(edge) - excess(second word). Also within-page concentration of the
line-final word (one refrain word per page) against a Markov generator with planted line-final
distributions and page drift.
Controls: litany laid out as versicle / response lines (period-2 refrain), merged litany (period 1),
Dante (rhyme period 2 and 3), Regimen, Macer, prose. Survivors: held-out pages and IT2a.
"""
import sys, os, json, random, pickle, time
import numpy as np
import numba
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v62_lib as L

NDEF = int(os.environ.get('NDEF', 800))
NSH = int(os.environ.get('NSH', 40))
A = pickle.load(open(os.path.join(L.CK, 'corpora.pkl'), 'rb'))


@numba.njit(cache=True)
def eqgaps(lab, starts, ends):
    c = 0
    for p in range(len(starts)):
        s, e = starts[p], ends[p]
        for i in range(s, e):
            if lab[i] < 0: continue
            j = -1
            for x in range(i + 1, min(e, i + 9)):
                if lab[x] == lab[i]:
                    j = x; break
            if j < 0: continue
            g = j - i
            if j + g < e and lab[j + g] == lab[i]:
                ok = True
                for x in range(j + 1, j + g):
                    if lab[x] == lab[i]:
                        ok = False; break
                if ok: c += 1
    return c


@numba.njit(cache=True)
def shuf_null(lab, starts, ends, nsh, seed):
    np.random.seed(seed)
    out = np.zeros(nsh)
    w = lab.copy()
    for k in range(nsh):
        for p in range(len(starts)):
            s, e = starts[p], ends[p]
            for i in range(e - 1, s, -1):
                j = s + np.random.randint(0, i - s + 1)
                t = w[i]; w[i] = w[j]; w[j] = t
        out[k] = eqgaps(w, starts, ends)
    return out


def litany_vr():
    """litany with versicle and response on separate lines (period-2 refrain)."""
    t = open(os.path.join(L.CK, 'src', 'Litaniae_Sanctorum.wiki'), encoding='utf-8').read()
    import re
    t = re.sub(r'\{\{[^}]*\}\}', '', t); t = re.sub(r'<[^>]+>', '', t)
    out = []
    for l in t.split('\n'):
        l = l.strip().lstrip(':').strip()
        if not l: continue
        if re.match(r'^[IVX]+ [A-Z]', l): out.append(None); continue
        if l[:2] in ('R.', 'V.'): l = l[2:]
        ws = L.norm_latin(l)
        if ws: out.append(ws)
    tab = L.verbose_table(7)
    return [L.encode_lines(pg, tab) for pg in L.pages_from_verse(out)]


def sig_fn(d):
    kind = d[0]
    if kind == 'word':
        return lambda w: w
    if kind == 'end':
        _, m, k = d
        return lambda w: tuple(m.get(x, -1) for x in w[-k:]) if m else tuple(w[-k:])
    if kind == 'beg':
        _, m, k = d
        return lambda w: tuple(m.get(x, -1) for x in w[:k]) if m else tuple(w[:k])
    if kind == 'len':
        _, cuts = d
        return lambda w: int(np.searchsorted(cuts, len(w)))
    raise ValueError


def make_defs(alpha, n, rng):
    defs = [('word',), ('end', None, 1), ('end', None, 2), ('end', None, 3), ('beg', None, 1), ('beg', None, 2)]
    while len(defs) < n:
        r = rng.random()
        if r < 0.45:
            c = rng.choice([2, 3, 4, 6, 8]); m = {a: rng.randrange(c) for a in alpha}
            defs.append(('end', m, rng.choice([1, 2, 3])))
        elif r < 0.85:
            c = rng.choice([2, 3, 4, 6, 8]); m = {a: rng.randrange(c) for a in alpha}
            defs.append(('beg', m, rng.choice([1, 2, 3])))
        else:
            defs.append(('len', np.array(sorted(rng.sample(range(2, 10), rng.choice([1, 2])))) + 0.5))
    return defs


def dstr(d):
    if d[0] == 'word': return 'whole word'
    if d[0] == 'len': return f'wordlen cuts {list(d[1])}'
    m = d[1]
    return f"{d[0]} k{d[2]} " + ('full' if m is None else f'{len(set(m.values()))}cls')


POS = {'first': lambda l: l[0], 'last': lambda l: l[-1], 'second': lambda l: l[1]}


def labels(pages, f, pos):
    lab, starts, ends, vocab = [], [], [], {}
    for pg in pages:
        starts.append(len(lab))
        for l in pg:
            if len(l) < 3 or str(l[0][0]).startswith('?') and pos == 'first':
                lab.append(-1); continue
            w = POS[pos](l)
            if str(w[0]).startswith('?'):
                lab.append(-1); continue
            v = f(w)
            lab.append(vocab.setdefault(v, len(vocab)))
        ends.append(len(lab))
    return np.array(lab, np.int64), np.array(starts, np.int64), np.array(ends, np.int64)


def skel_labels(pages, f):
    lab, starts, ends, vocab = [], [], [], {}
    for pg in pages:
        starts.append(len(lab))
        for l in pg:
            v = tuple(f(w) for w in l)
            lab.append(vocab.setdefault(v, len(vocab)) if len(l) >= 3 else -1)
        ends.append(len(lab))
    return np.array(lab, np.int64), np.array(starts, np.int64), np.array(ends, np.int64)


def excess(pages, f, pos, seed):
    lab, s, e = labels(pages, f, pos) if pos != 'skeleton' else skel_labels(pages, f)
    o = eqgaps(lab, s, e); nl = shuf_null(lab, s, e, NSH, seed)
    n = max(1, (lab >= 0).sum())
    return 100 * (o - nl.mean()) / n, (o - nl.mean()) / (nl.std() + 1e-9)


def scan(pages, defs, seed):
    rows = []
    for i, d in enumerate(defs):
        f = sig_fn(d)
        r = {}
        for pos in ('first', 'last', 'second'):
            r[pos] = excess(pages, f, pos, seed + i)
        if d[0] in ('len', 'beg', 'end') and d[1] is not None or d[0] == 'len':
            r['skeleton'] = excess(pages, f, 'skeleton', seed + i)
        rows.append(r)
    return rows


def summarize(rows):
    last = np.array([r['last'][0] - r['second'][0] for r in rows])
    first = np.array([r['first'][0] - r['second'][0] for r in rows])
    skel = np.array([r['skeleton'][1] for r in rows if 'skeleton' in r])
    zl = np.array([r['last'][1] for r in rows]); zf = np.array([r['first'][1] for r in rows])
    return dict(last_did_max=round(float(last.max()), 3), last_did_mean=round(float(last.mean()), 3),
                first_did_max=round(float(first.max()), 3), first_did_mean=round(float(first.mean()), 3),
                z_last_max=round(float(zl.max()), 2), z_first_max=round(float(zf.max()), 2),
                z_skel_max=round(float(skel.max()), 2) if len(skel) else None,
                argmax_last=int(last.argmax()), argmax_first=int(first.argmax()), argmax_skel=int(np.argmax([r['skeleton'][1] if 'skeleton' in r else -99 for r in rows])))


def final_concentration(pages):
    """mean within-page Simpson index of the line-final word and of the second word."""
    sf, ss = [], []
    for pg in pages:
        ls = [l for l in pg if len(l) >= 3]
        if len(ls) < 5: continue
        for pos, acc in ((-1, sf), (1, ss)):
            c = {}
            for l in ls: c[l[pos]] = c.get(l[pos], 0) + 1
            n = len(ls); acc.append(sum(v * (v - 1) for v in c.values()) / (n * (n - 1)))
    return float(np.mean(sf)), float(np.mean(ss))


def split(name):
    pg = A[name]['pages']
    if A[name]['kind'] == 'voynich':
        meta = A[name]['meta']
        return ([p for p, m in zip(pg, meta) if L.folio_num(m['folio']) % 2 == 1],
                [p for p, m in zip(pg, meta) if L.folio_num(m['folio']) % 2 == 0])
    h = len(pg) // 2
    return pg[:h], pg[h:]


def main():
    t0 = time.time()
    A['Litany-VR(period2)'] = dict(pages=litany_vr(), kind='litany')
    out = {'conc': {}, 'scan': {}, 'heldout': {}}
    rng = random.Random(3)
    # concentration of line-final words vs Markov generator
    for name in A:
        f, s = final_concentration(A[name]['pages'])
        g = [final_concentration(L.markov_line_generator(A[name]['pages'], random.Random(40 + i))) for i in range(3)]
        out['conc'][name] = dict(final=round(f, 4), second=round(s, 4), gen_final=round(float(np.mean([x[0] for x in g])), 4),
                                 gen_second=round(float(np.mean([x[1] for x in g])), 4))
        print('conc', name, out['conc'][name], flush=True)
    zl_tr, zl_te = split('V-ZL3b'); it_tr, it_te = split('V-IT2a')
    alpha = L.alphabet(A['V-ZL3b']['pages'])
    defs = make_defs(alpha, NDEF, random.Random(5))
    sets = [('ZL-train', zl_tr, defs), ('null-markov', L.markov_line_generator(zl_tr, random.Random(8)), defs),
            ('null-shuffle', L.shuffle_lines_within_page(zl_tr, random.Random(8)), defs)]
    for name in ['Litany(refrain)', 'Litany-VR(period2)', 'Dante(verse,terza)', 'Regimen(verse,rhymed)',
                 'Macer(verse,hexam)', 'Hildegard(prose herbal)', 'Caesar(prose)']:
        a2 = L.alphabet(A[name]['pages'])
        sets.append((name, A[name]['pages'], make_defs(a2, NDEF // 4, random.Random(6))))
    for tag, pages, dd in sets:
        rows = scan(pages, dd, 100)
        sm = summarize(rows)
        sm['best_last'] = dstr(dd[sm['argmax_last']]); sm['best_first'] = dstr(dd[sm['argmax_first']])
        sm['best_skel'] = dstr(dd[sm['argmax_skel']])
        out['scan'][tag] = sm
        print(tag, sm, flush=True)
        if tag == 'ZL-train':
            zrows = rows
    # held-out: top 10 defs for last-edge and first-edge on ZL test and IT2a test
    for edge in ('last', 'first'):
        sc = np.array([r[edge][0] - r['second'][0] for r in zrows])
        top = np.argsort(-sc)[:10]
        ho = []
        for i in top:
            f = sig_fn(defs[i])
            rz = {p: excess(zl_te, f, p, 900)[0] for p in (edge, 'second')}
            ri = {p: excess(it_te, f, p, 901)[0] for p in (edge, 'second')}
            ho.append(dict(df=dstr(defs[i]), train=round(float(sc[i]), 3), zl_test=round(rz[edge] - rz['second'], 3),
                           it_test=round(ri[edge] - ri['second'], 3)))
        out['heldout'][edge] = ho
        print(edge, ho, flush=True)
    sk = np.array([r['skeleton'][1] if 'skeleton' in r else -99 for r in zrows])
    top = np.argsort(-sk)[:10]
    out['heldout']['skeleton'] = [dict(df=dstr(defs[i]), train_z=round(float(sk[i]), 2),
                                       zl_test_z=round(float(excess(zl_te, sig_fn(defs[i]), 'skeleton', 902)[1]), 2),
                                       it_test_z=round(float(excess(it_te, sig_fn(defs[i]), 'skeleton', 903)[1]), 2)) for i in top]
    print('skel', out['heldout']['skeleton'], flush=True)
    out['secs'] = round(time.time() - t0)
    json.dump(out, open(os.path.join(L.CK, 'cycle3.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
