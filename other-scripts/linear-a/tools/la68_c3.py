#!/usr/bin/env python3
"""la68 cycle 3: shorthand for SOMEONE ELSE's words. If receipts were sealed or written by an issuing
office elsewhere (HT nodules carry impressions of rings also used at Zakros, Sklavokambos, Gournia), the
sealing signs may abbreviate the words of another site's ledgers, or of the religious (libation) texts.

Abbreviation sets (rows): HT receipts, KH receipts, all other LA receipts pooled.
Word sources (columns): tablets of HT, KH, ZA, PH, KN, other sites pooled, and non-tablet inscriptions
(stone/metal/clay vessels etc.).
For each row x column and each named position rule (11 options x 5 filters x 3 weights) the score is the
cycle-1 bits gain over the source's any-position sign frequency; nulls: within-word shuffle of the source
words (kills position; B) and source words replaced by frequency-matched random LA word types (N2).
Family-wise max over rules and cells; the LB control runs the same matrix on KN, PY, TH, MY.
"""
import sys, os, json, collections, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la68_common as C

NNULL = int(os.environ.get('NNULL', 100))
SEED = int(os.environ.get('SEED', 683))


def la_sources():
    d = json.load(open(os.path.join(C.DATA, 'corpus.json')))
    rows = {'R:HT': collections.Counter(), 'R:KH': collections.Counter(), 'R:other': collections.Counter()}
    cols = collections.defaultdict(list)
    short = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Zakros': 'ZA', 'Phaistos': 'PH', 'Knossos': 'KN'}
    for r in d:
        sh = short.get(r['site'], 'other')
        if r['support'] in C.LA_R:
            key = 'R:' + (sh if sh in ('HT', 'KH') else 'other')
            for t in r['tokens']:
                if t['t'] == 'word' and len(t['s']) == 1:
                    rows[key][t['s'][0]] += 1
            continue
        col = 'T:' + sh if r['support'] in C.LA_T else 'X:inscr'
        toks = r['tokens']
        li = True
        for i, t in enumerate(toks):
            if t['t'] == 'nl':
                li = True
                continue
            if t['t'] != 'word':
                continue
            amt = None
            for u in toks[i + 1:i + 4]:
                if u['t'] == 'num':
                    amt = u['v']
                    break
                if u['t'] in ('word', 'nl'):
                    break
            if len(t['s']) >= 2:
                cols[col].append(dict(signs=tuple(t['s']), amt=amt, li=li, doc=r['id'], dep=''))
            li = False
    return rows, dict(cols)


def lb_sources():
    U = C.load_lb()
    rows = {'R:' + s: U[s]['abbr'] for s in ('KN', 'PY', 'TH', 'MY')}
    cols = {'T:' + s: U[s]['words'] for s in ('KN', 'PY', 'TH', 'MY')}
    return rows, cols


def build(rows, cols):
    """units = (row, col) pairs; Engine per unit-name."""
    units = {}
    for r, a in rows.items():
        for c, ws in cols.items():
            units[r + '>' + c] = dict(abbr=collections.Counter(a), abbr_docs=[], words=ws, deps={})
    return units


RULES = list(C.named_rules().items())


def sweep(units):
    names = sorted(units)
    E = C.Engine(units, names)
    M = np.zeros((len(RULES), len(names)))
    for i, (_, r) in enumerate(RULES):
        for j, s in enumerate(names):
            M[i, j] = E.score_unit(s, E.predicted(s, r))
    return names, M


_AW = {}


def allwords(script):
    if script not in _AW:
        c = collections.Counter()
        rows, cols = la_sources() if script == 'LA' else lb_sources()
        for ws in cols.values():
            for w in ws:
                c[w['signs']] += 1
        _AW[script] = c
    return _AW[script]


def job(args):
    script, kind, k = args
    rng = np.random.default_rng(SEED * 1009 + k + (0 if kind == 'B' else 77777) + (0 if script == 'LA' else 5555))
    rows, cols = la_sources() if script == 'LA' else lb_sources()
    if kind == 'B':
        for c in cols:
            for w in cols[c]:
                sg = list(w['signs'])
                rng.shuffle(sg)
                w['signs'] = tuple(sg)
    elif kind == 'N2':
        aw = allwords(script)
        band = lambda f: 0 if f <= 1 else 1 if f == 2 else 2 if f <= 4 else 3 if f <= 8 else 4
        byb = collections.defaultdict(list)
        for w, f in aw.items():
            byb[band(f)].append(w)
        for c in cols:
            mp = {}
            for w in cols[c]:
                t = w['signs']
                if t not in mp:
                    cand = byb[band(aw[t])]
                    mp[t] = cand[rng.integers(len(cand))]
                w['signs'] = mp[t]
    names, M = sweep(build(rows, cols))
    return script, kind, k, names, M


def main():
    t0 = time.time()
    res = {}
    jobs = [(s, 'real', 0) for s in ('LA', 'LB')] + [(s, kd, k) for s in ('LA', 'LB') for kd in ('B', 'N2') for k in range(NNULL)]
    with Pool(2) as P:
        out = P.map(job, jobs, chunksize=2)
    for script in ('LA', 'LB'):
        real = [o for o in out if o[0] == script and o[1] == 'real'][0]
        names, Mr = real[3], real[4]
        rows, cols = la_sources() if script == 'LA' else lb_sources()
        d = dict(cells=names, sizes={r: int(sum(a.values())) for r, a in rows.items()},
                 col_sizes={c: len(w) for c, w in cols.items()})
        for kind in ('B', 'N2'):
            NM = np.array([o[4] for o in out if o[0] == script and o[1] == kind])   # [nnull, rules, cells]
            mu, sd = NM.mean(0), NM.std(0) + 1e-9
            Z = (Mr - mu) / sd
            Zn = (NM - mu) / sd
            fw_null = Zn.reshape(len(NM), -1).max(1)
            cell = {}
            for j, nm in enumerate(names):
                i = int(Z[:, j].argmax())
                cell_null = Zn[:, :, j].max(1)
                cell[nm] = dict(best=RULES[i][0], z=float(Z[i, j]), real=float(Mr[i, j]),
                                P_cell=float((1 + (cell_null >= Z[i, j]).sum()) / (1 + len(NM))),
                                P_fw=float((1 + (fw_null >= Z[i, j]).sum()) / (1 + len(NM))),
                                p1_all_token_z=float(Z[[n for n, _ in RULES].index('p1|all|token'), j]),
                                any_all_token_real=float(Mr[[n for n, _ in RULES].index('any|all|token'), j]))
            d[kind] = cell
            # diagonal vs off-diagonal: is the own-site cell best?
            print(script, kind, flush=True)
            for nm in names:
                c = cell[nm]
                print('  ', nm, c['best'], round(c['z'], 2), 'Pcell', round(c['P_cell'], 3), 'Pfw', round(c['P_fw'], 3),
                      'p1 z', round(c['p1_all_token_z'], 2), flush=True)
        res[script] = d
    res['secs'] = time.time() - t0
    json.dump(res, open(os.path.join(C.CK, f'c3_seed{SEED}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
