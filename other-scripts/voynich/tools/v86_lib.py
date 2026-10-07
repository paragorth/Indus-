"""v86: 'the scribe under pressure leaks the code'.

Squeezed positions (line end, before a drawing intrusion) treated as natural compression experiments.
Random compression rules f (word -> squeezed form) are scored by how well they RECOVER out-of-vocabulary
squeezed tokens from the unsqueezed vocabulary of the same page (other lines), against length-matched
mid-line out-of-vocabulary tokens (which no compression produced).

Corpus format used here: list of pages; page = list of lines; line = dict(units=[str|None], cls=[str],
para_end=bool). Each word is a string of single-character units (Voynich glyph units via vlib.glyphs).
Classes: E line-final, Ib before intrusion, Ia after intrusion, B line-initial, M mid-line, X other.
"""
import os, sys, json, random, math, hashlib
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vlib
CK = os.path.join(vlib.DATA, 'v86_ckpt')
os.makedirs(CK, exist_ok=True)

UNIT = {'cth': 'T', 'ckh': 'K', 'cph': 'P', 'cfh': 'F', 'ch': 'C', 'sh': 'S'}


def vunits(w):
    return ''.join(vlib.glyphs(w))


# ---------------------------------------------------------------- corpora
def voynich_pages(name='ZL3b'):
    import v83_parse
    mode = os.environ.get('VOY_MODE', 'glyph')
    pages = defaultdict(list)
    meta = {}
    for r in v83_parse.records(name):
        if r['ltype'] != 'P':
            continue
        ws = [w if v83_parse.keep_word(r, j, mode) else None for j, w in enumerate(r['words'])]
        if not any(ws):
            continue
        n = len(ws)
        seps = r['seps']
        cls = []
        for j in range(n):
            before_intr = j < len(seps) and seps[j] in '-~'
            after_intr = j > 0 and seps[j - 1] in '-~'
            if n >= 2 and j == n - 1:
                c = 'E'
            elif before_intr:
                c = 'Ib'
            elif n >= 2 and j == 0:
                c = 'B'
            elif after_intr:
                c = 'Ia'
            elif 0 < j < n - 1:
                c = 'M'
            else:
                c = 'X'
            cls.append(c)
        units = [vunits(w) if w and '?' not in w else None for w in ws]
        pages[r['folio']].append(dict(units=units, cls=cls, para_end=r['para_end'], para_start=r['para_start']))
        meta[r['folio']] = (r['quire'], r['lang'], r['illus'], r['hand'])
    return dict(pages), meta


def ref_words(key, n):
    L = vlib.load_ref(key, max_words=n + 2000, skip_frac=0.05)
    return [w for l in L for w in l['words']][:n]


def layout_like(vpages, words):
    """Lay a running text into the Voynich page/line/class template (same words per line, same intrusions)."""
    it = iter(words)
    out = {}
    for f, lines in vpages.items():
        nl = []
        for ln in lines:
            u = []
            for _ in ln['units']:
                try:
                    u.append(next(it))
                except StopIteration:
                    return out
            nl.append(dict(units=u, cls=list(ln['cls']), para_end=ln['para_end'], para_start=ln['para_start']))
        out[f] = nl
    return out


def shuffle_positions(pages, rng):
    """Kill control: permute words inside each line (class labels stay with the slots)."""
    out = {}
    for f, lines in pages.items():
        nl = []
        for ln in lines:
            u = list(ln['units']); rng.shuffle(u)
            nl.append(dict(ln, units=u))
        out[f] = nl
    return out


def shuffle_lines_within_page(pages, rng):
    """Second kill control: words pooled over the page and re-dealt into the same slots."""
    out = {}
    for f, lines in pages.items():
        pool = [w for ln in lines for w in ln['units']]
        rng.shuffle(pool)
        k = 0; nl = []
        for ln in lines:
            m = len(ln['units']); nl.append(dict(ln, units=pool[k:k + m])); k += m
        out[f] = nl
    return out


# ---------------------------------------------------------------- planted squeezing
def lat_abbrev(w, rng):
    """Medieval-style abbreviation (suspension, contraction, nasal and -us marks)."""
    V = set('aeiouy')
    if len(w) < 4:
        return w
    r = rng.random()
    if w.endswith('us') and r < 0.3:
        return w[:-2] + '9'
    if w[-1] in 'mn' and r < 0.5:
        return w[:-1] + '~'
    if r < 0.75:   # contraction: drop vowels inside, keep first and last letter, tilde
        core = w[0] + ''.join(c for c in w[1:-1] if c not in V) + w[-1]
        return core + '~'
    return w[:3] + '~'   # suspension


def plant(pages, fn, classes=('E', 'Ib'), p=0.5, seed=0, para_end_exempt=True):
    rng = random.Random(seed)
    out = {}
    for f, lines in pages.items():
        nl = []
        for ln in lines:
            u = list(ln['units'])
            for j, c in enumerate(ln['cls']):
                if c in classes and u[j] and rng.random() < p:
                    if para_end_exempt and ln['para_end'] and c == 'E':
                        continue      # the last line of a paragraph is not under pressure
                    u[j] = fn(u[j], rng)
            nl.append(dict(ln, units=u))
        out[f] = nl
    return out


def make_det_compress(alphabet, seed):
    """A deterministic meaning-preserving compression on Voynich units: drop a fixed glyph set, cap length."""
    rng = random.Random(seed)
    D = set(rng.sample(sorted(alphabet), 3))
    k = rng.choice([3, 4])

    def fn(w, _r):
        v = w[0] + ''.join(c for c in w[1:] if c not in D)
        return v[:k] if len(v) > k else v
    return fn, D, k


def make_habit(alphabet):
    A = sorted(alphabet)

    def fn(w, r):
        w = list(w)
        for _ in range(r.choice([1, 2])):
            op = r.random()
            i = r.randrange(len(w))
            if op < 0.4 and len(w) > 1:
                del w[i]
            elif op < 0.8:
                w[i] = r.choice(A)
            else:
                w.insert(i, r.choice(A))
        return ''.join(w)
    return fn


# ---------------------------------------------------------------- random rules
def random_rule(rng, alphabet, marks=()):
    A = sorted(alphabet)
    def one():
        t = rng.choice(['drop', 'drop', 'trunc', 'contract', 'subst', 'tail', 'merge', 'droplast'])
        if t == 'drop':
            return ('drop', ''.join(sorted(rng.sample(A, rng.randint(1, 4)))), rng.random() < 0.5)
        if t == 'trunc':
            return ('trunc', rng.randint(1, 5), rng.choice([''] * 3 + A))
        if t == 'contract':
            return ('contract', rng.randint(1, 3), rng.randint(0, 2), rng.choice([''] * 2 + A))
        if t == 'subst':
            k = rng.randint(1, 3)
            return ('subst', tuple((rng.choice(A), rng.choice(A)) for _ in range(k)))
        if t == 'tail':
            return ('tail', rng.randint(1, 3), rng.choice(A))
        if t == 'merge':
            return ('merge', rng.choice(A) + rng.choice(A), rng.choice(A))
        return ('droplast', rng.randint(1, 2))
    ops = [one() for _ in range(rng.choice([1, 1, 2, 2, 3]))]
    return tuple(ops)


def apply_op(op, w):
    t = op[0]
    if t == 'drop':
        D = op[1]
        if op[2]:
            return w[:1] + ''.join(c for c in w[1:] if c not in D)
        return ''.join(c for c in w if c not in D)
    if t == 'trunc':
        return w[:op[1]] + op[2] if len(w) > op[1] else w
    if t == 'contract':
        a, b, m = op[1], op[2], op[3]
        if len(w) > a + b + 1:
            return w[:a] + m + (w[-b:] if b else '')
        return w
    if t == 'subst':
        for x, y in op[1]:
            w = w.replace(x, y)
        return w
    if t == 'tail':
        n, x = op[1], op[2]
        return w[:-n] + x if len(w) > n + 1 else w
    if t == 'merge':
        return w.replace(op[1], op[2])
    if t == 'droplast':
        return w[:-op[1]] if len(w) > op[1] + 1 else w
    raise ValueError(t)


def apply_rule(rule, w):
    for op in rule:
        w = apply_op(op, w)
    return w


def rule_str(rule):
    return ' > '.join(':'.join(str(x) if not isinstance(x, tuple) else ','.join(a + b for a, b in x) for x in op)
                      for op in rule)


# ---------------------------------------------------------------- scoring
def prepare(pages, folios, target_classes, control_class='M', exclude_para_end=False):
    """For each page: vocabulary counter over all (kept) words; per token: (word, line-local counter).
    Target tokens = class in target_classes and OOV w.r.t. the page's OTHER lines.
    Control tokens = class M, OOV w.r.t. other lines."""
    P = []
    for f in folios:
        lines = pages.get(f)
        if not lines:
            continue
        pc = Counter(w for ln in lines for w in ln['units'] if w)
        tgt, ctl = [], []
        for ln in lines:
            lc = Counter(w for w in ln['units'] if w)
            for w, c in zip(ln['units'], ln['cls']):
                if not w:
                    continue
                if pc[w] - lc[w] > 0:
                    continue   # in vocabulary of other lines: not a compression candidate
                if c in target_classes and not (exclude_para_end and ln['para_end'] and c == 'E'):
                    tgt.append((w, lc))
                elif c == control_class:
                    ctl.append((w, lc))
        P.append((f, pc, tgt, ctl))
    return P


def len_weights(P):
    """Weights for control tokens so their length distribution matches the targets'."""
    lt = Counter(min(len(w), 8) for _, _, tg, _ in P for w, _ in tg)
    lc = Counter(min(len(w), 8) for _, _, _, ct in P for w, _ in ct)
    nt = sum(lt.values()); nc = sum(lc.values())
    return {L: (lt[L] / nt) / (lc[L] / nc) if lc[L] else 0.0 for L in lc}


def score_rule(rule, P, wts, cache=None):
    """Recoverability R = mean over tokens of 1/|preimages| (0 if none). Returns (R_tgt, R_ctl_lenmatched, n_t)."""
    if cache is None:
        cache = {}
    st = 0.0; nt = 0; sc = 0.0; wc = 0.0
    for f, pc, tgt, ctl in P:
        inv = defaultdict(list)
        for u in pc:
            v = cache.get(u)
            if v is None:
                v = apply_rule(rule, u); cache[u] = v
            if v != u:
                inv[v].append(u)
        for w, lc in tgt:
            pre = [u for u in inv.get(w, ()) if pc[u] - lc[u] > 0]
            st += (1.0 / len(pre)) if pre else 0.0
            nt += 1
        for w, lc in ctl:
            ww = wts.get(min(len(w), 8), 0.0)
            if ww == 0:
                continue
            pre = [u for u in inv.get(w, ()) if pc[u] - lc[u] > 0]
            sc += ww * ((1.0 / len(pre)) if pre else 0.0)
            wc += ww
    return st / max(nt, 1), sc / max(wc, 1e-9), nt


def alphabet_of(pages):
    return set(c for ls in pages.values() for ln in ls for w in ln['units'] if w for c in w)


def folio_split(folios, seed=0):
    r = random.Random(seed)
    f = sorted(folios); r.shuffle(f)
    h = len(f) // 2
    return sorted(f[:h]), sorted(f[h:])


def search(pages, rules, target=('E', 'Ib'), seed=0, top=20, exclude_para_end=False):
    """Score all rules on discovery folios, re-score top survivors on held-out folios."""
    disc, held = folio_split(list(pages), seed)
    Pd = prepare(pages, disc, target, exclude_para_end=exclude_para_end)
    Ph = prepare(pages, held, target, exclude_para_end=exclude_para_end)
    wd, wh = len_weights(Pd), len_weights(Ph)
    res = []
    for i, r in enumerate(rules):
        rt, rc, nt = score_rule(r, Pd, wd)
        res.append((rt - rc, rt, rc, i))
    res.sort(reverse=True)
    surv = []
    for d, rt, rc, i in res[:top]:
        ht, hc, nh = score_rule(rules[i], Ph, wh)
        surv.append(dict(i=i, rule=rule_str(rules[i]), disc_delta=d, disc_t=rt, disc_c=rc, held_delta=ht - hc,
                         held_t=ht, held_c=hc))
    n_t = sum(len(t) for _, _, t, _ in Pd); n_c = sum(len(c) for _, _, _, c in Pd)
    return dict(best_disc=res[0][0], mean_disc=sum(x[0] for x in res) / len(res),
                surv=surv, held_top=surv[0]['held_delta'], held_top5=sum(s['held_delta'] for s in surv[:5]) / 5,
                n_target_oov=n_t, n_ctl_oov=n_c)


# ---------------------------------------------------------------- rule-free recoverability (subsequence)
def is_subseq(s, u):
    it = iter(u)
    return all(c in it for c in s)


def subseq_recovery(pages, folios, target, exclude_para_end=False):
    """Share of OOV target tokens that are a proper subsequence of some other-line page word (unique weight),
    vs length-matched OOV mid-line tokens; plus the glyphs deleted in the unique cases."""
    P = prepare(pages, folios, target, exclude_para_end=exclude_para_end)
    w = len_weights(P)
    dels_t = Counter(); kept_t = Counter(); dels_c = Counter(); kept_c = Counter()
    st = nt = 0.0; sc = wc = 0.0
    for f, pc, tgt, ctl in P:
        voc = list(pc)
        for grp, isT in ((tgt, True), (ctl, False)):
            for s, lc in grp:
                ww = 1.0 if isT else w.get(min(len(s), 8), 0.0)
                if ww == 0:
                    continue
                pre = [u for u in voc if len(u) > len(s) and pc[u] - lc[u] > 0 and is_subseq(s, u)]
                val = 1.0 / len(pre) if pre else 0.0
                if isT:
                    st += val; nt += 1
                else:
                    sc += ww * val; wc += ww
                if len(pre) == 1:
                    u = pre[0]
                    # greedy alignment: which glyphs of u are dropped
                    j = 0; dl = Counter(); kp = Counter()
                    for c in u:
                        if j < len(s) and s[j] == c:
                            kp[c] += 1; j += 1
                        else:
                            dl[c] += 1
                    (dels_t if isT else dels_c).update({k: v * ww for k, v in dl.items()})
                    (kept_t if isT else kept_c).update({k: v * ww for k, v in kp.items()})
    return dict(R_t=st / max(nt, 1), R_c=sc / max(wc, 1e-9), n_t=nt, dels_t=dels_t, kept_t=kept_t,
                dels_c=dels_c, kept_c=kept_c)


def save(name, obj):
    json.dump(obj, open(os.path.join(CK, name), 'w'), default=lambda o: dict(o) if isinstance(o, Counter) else str(o))


def subseq1(s, u):
    """s is recoverable from u: s is a proper subsequence of u, or s minus ONE glyph (an added mark) is.
    Returns (ok, mark) with mark the removed glyph or ''."""
    if len(u) > len(s) and is_subseq(s, u):
        return True, ''
    if len(s) >= 2:
        for i in range(len(s)):
            t = s[:i] + s[i + 1:]
            if len(u) > len(t) and is_subseq(t, u):
                return True, s[i]
    return False, ''


def subseq_recovery1(pages, folios, target, exclude_para_end=False, boot=200, seed=0):
    P = prepare(pages, folios, target, exclude_para_end=exclude_para_end)
    w = len_weights(P)
    per_page = []
    agg = dict(dels_t=Counter(), kept_t=Counter(), dels_c=Counter(), kept_c=Counter(), mark_t=Counter(), mark_c=Counter())
    for f, pc, tgt, ctl in P:
        voc = list(pc)
        st = nt = sc = wc = 0.0
        for grp, isT in ((tgt, True), (ctl, False)):
            for s, lc in grp:
                ww = 1.0 if isT else w.get(min(len(s), 8), 0.0)
                if ww == 0:
                    continue
                pre = []
                for u in voc:
                    if pc[u] - lc[u] <= 0:
                        continue
                    ok, mk = subseq1(s, u)
                    if ok:
                        pre.append((u, mk))
                val = 1.0 / len(pre) if pre else 0.0
                if isT:
                    st += val; nt += 1
                else:
                    sc += ww * val; wc += ww
                if len(pre) == 1:
                    u, mk = pre[0]
                    ss = s.replace(mk, '', 1) if mk else s
                    j = 0; dl = Counter(); kp = Counter()
                    for c in u:
                        if j < len(ss) and ss[j] == c:
                            kp[c] += 1; j += 1
                        else:
                            dl[c] += 1
                    k = 't' if isT else 'c'
                    agg['dels_' + k].update({a: b * ww for a, b in dl.items()})
                    agg['kept_' + k].update({a: b * ww for a, b in kp.items()})
                    if mk:
                        agg['mark_' + k][mk] += ww
        per_page.append((st, nt, sc, wc))
    def stat(pp):
        T = sum(x[0] for x in pp) / max(sum(x[1] for x in pp), 1)
        C = sum(x[2] for x in pp) / max(sum(x[3] for x in pp), 1e-9)
        return T, C
    T, C = stat(per_page)
    rng = random.Random(seed)
    bs = []
    for _ in range(boot):
        pp = [rng.choice(per_page) for _ in per_page]
        t, c = stat(pp); bs.append(t - c)
    bs.sort()
    return dict(R_t=T, R_c=C, delta=T - C, ci=(bs[int(0.025 * boot)], bs[int(0.975 * boot)]),
                n_t=sum(x[1] for x in per_page), **agg)


def protect_profile(r, minn=5):
    """keep-rate of each glyph in target unique alignments minus keep-rate in control alignments."""
    out = {}
    for g in set(r['kept_t']) | set(r['dels_t']):
        kt, dt = r['kept_t'][g], r['dels_t'][g]
        kc, dc = r['kept_c'][g], r['dels_c'][g]
        if kt + dt < minn or kc + dc < minn:
            continue
        out[g] = (kt / (kt + dt), kc / (kc + dc), kt + dt)
    return out
