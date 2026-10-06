#!/usr/bin/env python3
"""LA-61 cycle 2: the puzzles.
 2b  ARITHMETIC: a global hypothesis gives every word / sign / logogram that labels an entry inside a
     KU-RO section one of COUNT / EXCL (number not part of the total) / DED (number subtracted), plus a
     scope rule (reset at every total, or running from the document start).  Score = sections that
     close minus an MDL cost per non-COUNT role.  Massive random sampling + hill-climbing.
     Controls: (i) DECOY totals - every section's total replaced by its entry sum +- an offset drawn
     from the real failing sections' discrepancies (so nothing closes by default): how many does the
     same search close?  (ii) leave-one-section-out: roles fitted without a section, does it close?
 2a  SINGLE SIGNS BETWEEN LINES (RA PA PA3 TU ME *318): every role hypothesis in {SGL (la60 default),
     HDR heading, COM commodity marker, TOT sub-total (its number predicted as the running sum), TRX
     heading word} for the six signs, scored by the la60 grammar's held-out bits (selection on 4 splits,
     evaluation on 4 other splits).  Control: 24 decoy sets of 6 frequency-matched single signs through
     the identical search.
usage: la61_c2.py arith | signs"""
import sys, time, copy
from la61_common import *
from la60_common import sections, close_test, _num_after, split as la60_split
from la60_model import Grammar, doc_bits

OUT = os.path.join(LOOPS, 'la61_cycle2.txt')


# ------------------------------------------------------------------ 2b arithmetic
def la_sections():
    """Sections ending in a TOT word (PRIOR roles), entries with their label words."""
    A = admin_docs(load_la()); R = prior_reading(); roles = R['roles']
    out = []
    for d in A:
        toks = d['toks']
        ents = []; allents = []; labels = []
        skip = set()
        for i, t in enumerate(toks):
            if t[0] == 'NL':
                labels = []; continue
            if t[0] == 'W' and roles.get(t[1]) in ('TOT', 'RES'):
                j = _num_after(toks, i)
                if j is not None:
                    skip.add(j)
                    if roles.get(t[1]) == 'TOT':
                        out.append(dict(doc=d['id'], site=d['site'], total=toks[j][1], tfr=toks[j][2],
                                        ents=list(ents), allents=list(allents)))
                ents = []; labels = []
                continue
            if t[0] in ('W', 'L'):
                labels.append(t[1])
            if t[0] == 'N' and i not in skip:
                e = (t[1], t[2], tuple(labels) if labels else ('<bare>',))
                ents.append(e); allents.append(e)
                labels = []
    return [s for s in out if len(s['ents']) >= 2]


def closes(sec, role, scope, total=None):
    ents = sec['ents'] if scope == 0 else sec['allents']
    s = 0; nfr = 0
    for v, fr, labs in ents:
        r = 'COUNT'
        for l in labs:
            x = role.get(l, 'COUNT')
            if x == 'DED':
                r = 'DED'; break
            if x == 'EXCL':
                r = 'EXCL'
        if r == 'COUNT':
            s += v; nfr += bool(fr)
        elif r == 'DED':
            s -= v; nfr += bool(fr)
    tv = sec['total'] if total is None else total
    return close_test(tv, s, 0, nfr)


def score_h(secs, role, scope, lam, totals=None):
    c = sum(closes(s, role, scope, None if totals is None else totals[k]) for k, s in enumerate(secs))
    return c - lam * sum(1 for v in role.values() if v != 'COUNT'), c


def search_arith(secs, rng, nrand, nclimb, iters, lam, totals=None):
    words = sorted({l for s in secs for e in s['allents'] for l in e[2]})
    best = (-1e9, None, None, 0)
    for _ in range(nrand):
        role = {w: rng.choice(['COUNT', 'COUNT', 'COUNT', 'COUNT', 'EXCL', 'DED']) for w in words}
        role = {w: r for w, r in role.items() if r != 'COUNT'}
        sc = rng.randrange(2)
        v, c = score_h(secs, role, sc, lam, totals)
        if v > best[0]:
            best = (v, dict(role), sc, c)
    for r in range(nclimb):
        role = {}; sc = rng.randrange(2)
        cur, c = score_h(secs, role, sc, lam, totals)
        for it in range(iters):
            w = rng.choice(words); old = role.get(w, 'COUNT')
            new = rng.choice([x for x in ('COUNT', 'EXCL', 'DED') if x != old])
            flip_scope = rng.random() < 0.05
            if new == 'COUNT':
                role.pop(w, None)
            else:
                role[w] = new
            sc2 = 1 - sc if flip_scope else sc
            v, c2 = score_h(secs, role, sc2, lam, totals)
            if v >= cur:
                cur, c, sc = v, c2, sc2
            else:
                if old == 'COUNT':
                    role.pop(w, None)
                else:
                    role[w] = old
        if cur > best[0]:
            best = (cur, dict(role), sc, c)
    return best, words


def run_arith():
    t0 = time.time()
    secs = la_sections()
    rng = random.Random(seed('la61-c2-arith'))
    base = sum(closes(s, {}, 0) for s in secs)
    fail = [s for s in secs if not closes(s, {}, 0)]
    disc = [abs(s['total'] - sum(e[0] for e in s['ents'])) for s in fail]
    NR, NC, IT, LAM = 20000, 40, 3000, 0.5
    best, words = search_arith(secs, rng, NR, NC, IT, LAM)
    nscored = NR + NC * IT
    closed_by = [s['doc'] for s in secs if closes(s, best[1], best[2]) and not closes(s, {}, 0)]
    broke = [s['doc'] for s in secs if not closes(s, best[1], best[2]) and closes(s, {}, 0)]
    # decoy control: all totals replaced by sum +- an offset from the real discrepancies
    dec = []
    for k in range(20):
        r2 = random.Random(seed('la61-decoy-%d' % k))
        tots = []
        for s in secs:
            sm = sum(e[0] for e in s['ents'])
            d = r2.choice(disc) or 1
            tv = sm + d if (r2.random() < 0.5 or sm - d < 1) else sm - d
            tots.append(tv)
        base_d = sum(closes(s, {}, 0, tots[i]) for i, s in enumerate(secs))
        b2, _ = search_arith(secs, r2, NR // 4, NC // 4, IT, LAM, tots)
        nscored += NR // 4 + NC // 4 * IT
        dec.append((b2[3] - base_d, len(b2[1])))
        print('decoy', k, b2[3], base_d, len(b2[1]), flush=True)
    # leave-one-section-out
    loo = 0; loo_fail_closed = 0
    for k, s in enumerate(secs):
        rest = secs[:k] + secs[k + 1:]
        b3, _ = search_arith(rest, random.Random(seed('la61-loo-%d' % k)), 2000, 6, 1500, LAM)
        nscored += 2000 + 6 * 1500
        ok = closes(s, b3[1], b3[2])
        loo += ok
        if ok and not closes(s, {}, 0):
            loo_fail_closed += 1
    gains = np.array([g for g, _ in dec])
    res = dict(n_sec=len(secs), base=base, best_close=best[3], best_roles=best[1], scope=best[2],
               closed_by=closed_by, broke=broke, decoy=dec, loo=loo, loo_fail_closed=loo_fail_closed,
               n_scored=nscored, sec=time.time() - t0)
    json.dump(res, open(os.path.join(CK, 'c2_arith.json'), 'w'), ensure_ascii=False)
    row = ('| LA-61.2b | ARITHMETIC PUZZLE: %d KU-RO sections with >= 2 entries (%d close as written). Global hypothesis = COUNT / EXCL / DED for each of %d label words, signs and logograms + scope (reset at each total / running from document start); '
           'score = closing sections - %.1f per non-COUNT role (MDL); %d random hypotheses + %d hill-climbs x %d steps (%s scored in all). Controls: 20 DECOY copies (every total = entry sum +- a real discrepancy, nothing closes as written) through the same search; leave-one-section-out | '
           'best: %d close (+%d) with %d roles, scope %s: %s; newly closed %s; broken %s. DECOY: the same search closes +%.1f (range %d-%d) with %.1f roles. Leave-one-out: %d / %d sections close with roles fitted without them; %d of the %d failing sections closed out of sample |' % (
               len(secs), base, len(words), LAM, NR, NC, IT, format(nscored, ','), best[3], best[3] - base, len(best[1]),
               ['reset', 'running'][best[2]], ', '.join('%s %s' % kv for kv in sorted(best[1].items())), closed_by, broke,
               gains.mean(), gains.min(), gains.max(), np.mean([n for _, n in dec]), loo, len(secs), loo_fail_closed, len(fail)))
    verdict = ('real gain %d vs decoy %.1f: %s' % (best[3] - base, gains.mean(),
               'roles close MORE real failing totals than decoys (candidate)' if best[3] - base > gains.max() else
               'no better than decoys: the flexible arithmetic closes invented totals as easily; the 15 failures are not explained (damage / other-side totals remain the simplest account)'))
    wlog(OUT, row + ' ' + verdict + ' |')
    print(row, verdict)


# ------------------------------------------------------------------ 2a single signs
TARGET = ['RA', 'PA', 'PA₃', 'TU', 'ME', '*318']
SROLES = ['SGL', 'HDR', 'COM', 'TOT', 'TRX']


def grammar_bits(R, splits):
    return sum(sum(doc_bits(Grammar(tr, R, order=False), d) for d in te) for tr, te in splits)


def with_roles(R0, targets, roles):
    R = copy.deepcopy(R0)
    for w, r in zip(targets, roles):
        if r == 'SGL':
            R['roles'].pop(w, None)
        else:
            R['roles'][w] = r
    return R


def search_signs(R0, targets, sel, rng, nrand=40, sweeps=2):
    """Random role vectors, then coordinate ascent from the best; objective = held-out bits on the
    selection splits (lower is better)."""
    cache = {}

    def f(rv):
        k = tuple(rv)
        if k not in cache:
            cache[k] = grammar_bits(with_roles(R0, targets, rv), sel)
        return cache[k]
    default = ['SGL'] * len(targets)
    best = (f(default), default)
    for _ in range(nrand):
        rv = [rng.choice(SROLES) for _ in targets]
        b = f(rv)
        if b < best[0]:
            best = (b, rv)
    cur = list(best[1]); cb = best[0]
    for _ in range(sweeps):
        for i in range(len(targets)):
            for r in SROLES:
                rv = list(cur); rv[i] = r
                b = f(rv)
                if b < cb - 1e-9:
                    cb, cur = b, rv
    return cur, cb, f(default), len(cache)


def run_signs():
    t0 = time.time()
    A = admin_docs(load_la()); R0 = prior_reading()
    for w in TARGET:
        R0['roles'].pop(w, None)
    sel = [la60_split(A, 'la61-sel-%d' % k) for k in range(4)]
    ev = [la60_split(A, 'la61-ev-%d' % k) for k in range(4)]
    rng = random.Random(seed('la61-c2-signs'))
    cur, cb, c0, n = search_signs(R0, TARGET, sel, rng)
    ev_best = grammar_bits(with_roles(R0, TARGET, cur), ev); ev_def = grammar_bits(with_roles(R0, TARGET, ['SGL'] * 6), ev)
    per = {}
    for i, w in enumerate(TARGET):                       # each sign's own role, others default
        rv = ['SGL'] * 6; rv[i] = cur[i]
        per[w] = (cur[i], ev_def - grammar_bits(with_roles(R0, TARGET, rv), ev))
    # decoys: frequency-matched single signs not in the scaffold
    cnt = Counter(t[1] for d in A for t in d['toks'] if t[0] == 'W' and '-' not in t[1])
    pool = [w for w, c in cnt.items() if w not in R0['roles'] and w not in TARGET and c >= 3]
    tf = sorted(cnt[w] for w in TARGET)
    dec = []
    nscored = n
    for k in range(24):
        r2 = random.Random(seed('la61-sdec-%d' % k))
        ds = []
        for f_ in tf:
            cand = [w for w in pool if w not in ds and 0.5 * f_ <= cnt[w] <= 2 * f_] or [w for w in pool if w not in ds]
            ds.append(r2.choice(cand))
        Rk = copy.deepcopy(R0)
        dflt = [Rk['roles'].get(w, 'SGL') for w in ds]
        cur2, _, _, n2 = search_signs(Rk, ds, sel, r2, nrand=20, sweeps=1)
        e_b = grammar_bits(with_roles(Rk, ds, cur2), ev); e_d = grammar_bits(with_roles(Rk, ds, ['SGL'] * 6), ev)
        dec.append((ds, cur2, e_d - e_b)); nscored += n2
        print('decoy', k, ds, cur2, round(e_d - e_b, 1), flush=True)
    g = ev_def - ev_best
    dg = np.array([x[2] for x in dec])
    res = dict(target=TARGET, roles=cur, sel_gain=c0 - cb, ev_gain=g, per=per, decoys=dec, n_scored=nscored,
               sec=time.time() - t0)
    json.dump(res, open(os.path.join(CK, 'c2_signs.json'), 'w'), ensure_ascii=False)
    row = ('| LA-61.2a | SINGLE SIGNS BETWEEN LINES: roles for RA PA PA3 TU ME *318 in {SGL default, HDR, COM, TOT (number predicted as running sum), TRX}; '
           '%d role vectors scored by the la60 grammar (held-out bits; selection on 4 splits, evaluation on 4 other splits). Control: 24 decoy sets of 6 frequency-matched single signs, same search | '
           'chosen roles %s; selection gain %.1f bits, EVALUATION gain %.1f bits (per sign: %s). Decoys: evaluation gain %.1f +- %.1f (max %.1f; %d/24 >= real) |' % (
               nscored, ' '.join('%s=%s' % x for x in zip(TARGET, cur)), c0 - cb, g,
               '; '.join('%s %s %+.1f' % (w, r, x) for w, (r, x) in per.items()), dg.mean(), dg.std(), dg.max(), int((dg >= g).sum())))
    verdict = ('real gain beats %d/24 decoys: %s' % (int((dg < g).sum()),
               'the roles carry information beyond what any single sign gains from a role (candidate)' if (dg >= g).sum() <= 1 else
               'not distinguishable from what arbitrary single signs gain; no role for these signs is supported'))
    wlog(OUT, row + ' ' + verdict + ' |')
    print(row, verdict)


if __name__ == '__main__':
    {'arith': run_arith, 'signs': run_signs}[sys.argv[1]]()
