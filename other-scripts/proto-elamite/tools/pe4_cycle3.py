"""pe4 cycle 3: outside check.  If middles describe the counted item (age, sex,
quality...), their signs should predict the size of the number written beside
them (young animals and fattened animals come in other lot sizes; better cloth
in smaller numbers), WITHIN a tablet.  Names should predict amounts weakly.

Statistic per corpus (records with a value, tablets with >= 3 records):
  y = log(value), demeaned within tablet;
  S = sum over tokens with >= 10 records on >= 4 tablets of n_t * mean_t(y)^2,
      divided by sum(y^2)  (share of within-tablet variance carried by tokens);
  null: y permuted within tablet (1,000x); report S / null mean, p, and the
  number of single tokens with p < 0.01 (vs expected).
Corpora: PE_SDB, PE_C (no class sign; middle = all signs), PE_fin (middle
before a class sign, counted and capacity apart), Ur III herd (attribute
lines, word tokens), Ur III textiles (attribute lines), Ur III herd lines whose
tail is one non-attribute word (persons, gods: name control).
Run: python3 pe4_cycle3.py -> data/pe4_cycle3.json
"""
import json, math, os, random, statistics as st, sys
from collections import Counter, defaultdict
from pe4_common import *

NPERM = int(os.environ.get('PE4_P', 1000))
OUT = os.path.join(PEDATA, 'pe4_cycle3.json')


def prep(recs, tok):
    recs = [r for r in recs if r['n'] and r['n'] > 0 and tok(r)]
    byt = defaultdict(list)
    for r in recs:
        byt[r['t']].append(r)
    out = []
    for t, L in byt.items():
        if len(L) < 3:
            continue
        ys = [math.log(r['n']) for r in L]
        m = sum(ys) / len(ys)
        for r, y in zip(L, ys):
            out.append((t, set(tok(r)), y - m))
    return out


def stat(rows, ys, toks):
    tot = sum(y * y for y in ys) or 1e-9
    S = 0.0
    per = {}
    for tk, idx in toks.items():
        s = sum(ys[i] for i in idx)
        v = s * s / len(idx)
        per[tk] = s / len(idx)
        S += v
    return S / tot, per


def run(name, rows, rng):
    tabs = defaultdict(set)
    occ = defaultdict(list)
    for i, (t, ts, y) in enumerate(rows):
        for tk in ts:
            occ[tk].append(i)
            tabs[tk].add(t)
    toks = {tk: idx for tk, idx in occ.items() if len(idx) >= 10 and len(tabs[tk]) >= 4}
    if not toks:
        return None
    ys = [y for _, _, y in rows]
    obs, per = stat(rows, ys, toks)
    groups = defaultdict(list)
    for i, (t, _, _) in enumerate(rows):
        groups[t].append(i)
    null, pern = [], defaultdict(list)
    for _ in range(NPERM):
        yp = ys[:]
        for t, idx in groups.items():
            vals = [ys[i] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                yp[i] = v
        s, pp = stat(rows, yp, toks)
        null.append(s)
        for tk, v in pp.items():
            pern[tk].append(abs(v))
    m = st.mean(null)
    sig = {tk: (1 + sum(x >= abs(per[tk]) for x in pern[tk])) / (NPERM + 1) for tk in toks}
    nsig = sum(p < 0.01 for p in sig.values())
    top = sorted(toks, key=lambda tk: sig[tk])[:8]
    r = {'records': len(rows), 'tablets': len(groups), 'tokens': len(toks), 'S': obs, 'null': m,
         'ratio': obs / m, 'p': (1 + sum(x >= obs for x in null)) / (NPERM + 1),
         'nsig01': nsig, 'exp01': 0.01 * len(toks),
         'top': [(tk, round(per[tk], 2), sig[tk], len(toks[tk])) for tk in top]}
    print('%-12s rec %5d tab %4d tokens %3d  S %.3f null %.3f ratio %.2f p %.4f  sig %d/%d (exp %.1f)  top %s'
          % (name, r['records'], r['tablets'], r['tokens'], obs, m, r['ratio'], r['p'], nsig, len(toks),
             r['exp01'], r['top'][:4]), flush=True)
    return r


def main():
    rng = random.Random(4)
    R = pe_records()
    C = controls()
    herd_c = clean_ledger(C['herd'])
    voc = {w for r in herd_c for w in r['words']}
    herd_pn = [r for r in C['herd'] if len(r['words']) == 1 and r['words'][0] not in voc
               and not NUMW.match(r['words'][0])]
    tex_c = clean_ledger([dict(r, head=(r['words'][0] if r['words'] else '-')) for r in C['textile']],
                         min_tab=10, min_heads=2)
    mid = lambda r: tuple(r['attr'])
    sets = {
        'PE_SDB': prep([r for r in R if r['head'] == '-' and r['sys'] == 'SDB'], mid),
        'PE_SDB_m2': prep([r for r in R if r['head'] == '-' and r['sys'] == 'SDB' and len(r['attr']) >= 2], mid),
        'PE_C': prep([r for r in R if r['head'] == '-' and r['sys'] == 'C'], mid),
        'PE_fin_SDB': prep([r for r in R if r['head'] != '-' and r['sys'] == 'SDB'], mid),
        'PE_fin_C': prep([r for r in R if r['head'] != '-' and r['sys'] == 'C'], mid),
        'herd_attr': prep(herd_c, lambda r: tuple(r['words']) + (r['head'],)),
        'herd_attr_noh': prep(herd_c, lambda r: tuple(r['words'])),
        'tex_attr': prep(tex_c, lambda r: tuple(r['words'])),
        'herd_names': prep(herd_pn, lambda r: tuple(r['words'])),
        'herd_names_h': prep(herd_pn, lambda r: tuple(r['words']) + (r['head'],)),
    }
    res = {}
    for k, rows in sets.items():
        res[k] = run(k, rows, rng)
    json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
