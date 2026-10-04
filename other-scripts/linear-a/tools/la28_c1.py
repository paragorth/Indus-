#!/usr/bin/env python3
"""la28 cycle 1: do receipt terms reappear on same-site tablets more than chance?

Statistic M = number of distinct (site, receipt term) types whose term is written on
at least one tablet of the same site (E = exact term, B = ligature base).
Nulls: N1 site labels permuted among receipt documents; N2 random receipts (each term
replaced by a string of the same sign length drawn from the pooled receipt sign
distribution; logograms from the pooled receipt logograms).
References: same statistic for other (non-receipt, non-tablet) documents and for a
random half of the tablets ('tablet->tablet' ceiling of plain regional vocabulary).
Planted control: copy a fraction f of receipt types into k random same-site tablets.
Linear B control: W- sealings / nodules vs tablets at KN, PY, TH, MY, MI.
Deposit test (HT): receipt types on the Portico 11 / Room 13 tablets vs random HT
tablet sets of the same size.
"""
import random, sys, json, collections, math, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la28_common as C

R = random.Random(28)


def tabsets(docs, use_base):
    ts = collections.defaultdict(collections.Counter)
    for d in docs:
        if d['cls'] == 'T':
            for t in C.doc_terms(d, use_base):
                ts[d['site']][t] += 1
    return ts


def stat(rdocs, sites, ts, use_base):
    types = set()
    for d, s in zip(rdocs, sites):
        for t in C.doc_terms(d, use_base):
            types.add((s, t))
    hit = sum(1 for s, t in types if ts[s][t] > 0)
    return hit, len(types)


def n1(rdocs, ts, use_base, n=2000):
    sites = [d['site'] for d in rdocs]
    obs = stat(rdocs, sites, ts, use_base)
    vals = []
    for _ in range(n):
        s2 = sites[:]
        R.shuffle(s2)
        vals.append(stat(rdocs, s2, ts, use_base)[0])
    return obs, vals


def zp(obs, vals):
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** .5 or 1e-9
    p = (1 + sum(1 for v in vals if v >= obs)) / (1 + len(vals))
    return m, sd, (obs - m) / sd, p


def split_sign(t):
    k, v = t.split(':', 1)
    return k, (v.split('-') if k == 'W' else [v])


def n2(rdocs, ts, use_base, n=500):
    signs, logos = collections.Counter(), collections.Counter()
    for d in rdocs:
        for it in d['items']:
            for t in it['terms']:
                k, ss = split_sign(t)
                (signs if k == 'W' else logos).update(ss)
    sp, sw = zip(*signs.items())
    lp, lw = zip(*logos.items()) if logos else ((), ())
    obs = stat(rdocs, [d['site'] for d in rdocs], ts, use_base)[0]
    vals = []
    for _ in range(n):
        fake = []
        for d in rdocs:
            items = []
            for it in d['items']:
                tt = []
                for t in it['terms']:
                    k, ss = split_sign(t)
                    if k == 'W':
                        tt.append('W:' + '-'.join(R.choices(sp, sw, k=len(ss))))
                    else:
                        tt.append('L:' + R.choices(lp, lw)[0])
                items.append(dict(terms=tt, nums=it['nums']))
            fake.append(dict(site=d['site'], items=items))
        vals.append(stat(fake, [d['site'] for d in fake], ts, use_base)[0])
    return obs, vals


def analyse(docs, label, out, nperm=2000):
    tsites = {d['site'] for d in docs if d['cls'] == 'T'}
    res = {}
    for use_base in (False, True):
        tag = 'B' if use_base else 'E'
        ts = tabsets(docs, use_base)
        rd = [d for d in docs if d['cls'] == 'R' and d['site'] in tsites and C.doc_terms(d)]
        (h, ntyp), vals = n1(rd, ts, use_base, nperm)
        m, sd, z, p = zp(h, vals)
        o2, v2 = n2(rd, ts, use_base, 300)
        m2, sd2, z2, p2 = zp(o2, v2)
        # other documents (regional-vocabulary baseline)
        od = [d for d in docs if d['cls'] == 'O' and d['site'] in tsites and C.doc_terms(d)]
        if len(od) >= 10 and len({d['site'] for d in od}) > 1:
            (ho, no), vo = n1(od, ts, use_base, 1000)
            mo, sdo, zo, po = zp(ho, vo)
            oline = f'other docs {ho}/{no} vs N1 {mo:.1f} (z {zo:.1f}, P {po:.3f})'
        else:
            oline, zo = 'other docs: too few', float('nan')
        # tablet->tablet: random half of tablets as pseudo-receipts against the other half
        zt = []
        for rep in range(5):
            tabs = [d for d in docs if d['cls'] == 'T' and C.doc_terms(d)]
            R.shuffle(tabs)
            A, Bh = tabs[:len(tabs) // 2], tabs[len(tabs) // 2:]
            tsB = tabsets(Bh, use_base)
            (ht, nt), vt = n1(A, tsB, use_base, 300)
            zt.append(zp(ht, vt)[2])
        line = (f'{label} [{tag}] receipts {len(rd)} docs, {ntyp} site-term types: {h} hit own-site tablets '
                f'vs N1 site-swap {m:.1f}+-{sd:.1f} (z {z:.1f}, P {p:.4f}); vs N2 random receipts {m2:.1f} (z {z2:.1f}, P {p2:.4f}); '
                f'{oline}; tablet-half->tablet-half z mean {sum(zt)/len(zt):.1f}')
        print(line, flush=True)
        out.append(line)
        res[tag] = dict(hit=h, types=ntyp, n1=m, z=z, p=p, n2=m2, z2=z2, p2=p2, zo=zo, zt=sum(zt) / len(zt))
    return res


def per_site(docs, label, out):
    ts = tabsets(docs, False)
    tsites = {d['site'] for d in docs if d['cls'] == 'T'}
    rd = [d for d in docs if d['cls'] == 'R' and d['site'] in tsites and C.doc_terms(d)]
    sites = [d['site'] for d in rd]
    by = collections.defaultdict(set)
    for d in rd:
        for t in C.doc_terms(d):
            by[d['site']].add(t)
    for s, terms in sorted(by.items(), key=lambda x: -len(x[1])):
        if len(terms) < 3:
            continue
        hit = [t for t in terms if ts[s][t]]
        # expected under N1: per-site permutation share
        vals = []
        for _ in range(500):
            s2 = sites[:]
            R.shuffle(s2)
            tt = {t for d, ss in zip(rd, s2) if ss == s for t in C.doc_terms(d)}
            vals.append(sum(1 for t in tt if ts[s][t]) / max(1, len(tt)))
        m = sum(vals) / len(vals)
        p = (1 + sum(1 for v in vals if v >= len(hit) / len(terms))) / 501
        line = f'  {label} {s}: {len(hit)}/{len(terms)} receipt types on own tablets ({len(hit)/len(terms):.2f}) vs N1 share {m:.2f}, P {p:.3f}; hits {sorted(hit)[:14]}'
        print(line, flush=True)
        out.append(line)


def planted(docs, out, fracs=(0.05, 0.1, 0.2, 0.3), k=1, reps=30):
    tsites = {d['site'] for d in docs if d['cls'] == 'T'}
    for f in fracs:
        zs = []
        for rep in range(reps):
            dd = json.loads(json.dumps(docs))
            rd = [d for d in dd if d['cls'] == 'R' and d['site'] in tsites and C.doc_terms(d)]
            types = sorted({(d['site'], t) for d in rd for t in C.doc_terms(d)})
            tabs = collections.defaultdict(list)
            for d in dd:
                if d['cls'] == 'T':
                    tabs[d['site']].append(d)
            for s, t in R.sample(types, int(round(f * len(types)))):
                for tab in R.sample(tabs[s], min(k, len(tabs[s]))):
                    tab['items'].append(dict(terms=[t], nums=[R.randint(1, 20)]))
            ts = tabsets(dd, False)
            (h, n), vals = n1(rd, ts, False, 300)
            zs.append(zp(h, vals)[2])
        det = sum(1 for z in zs if z > 2.33) / reps
        line = f'  planted f={f:.2f} k={k}: mean z {sum(zs)/reps:.1f}, detected (z>2.33) {det:.2f}'
        print(line, flush=True)
        out.append(line)


def deposit(docs, out):
    ht = [d for d in docs if d['site'] == 'Haghia Triada']
    rterms = {t for d in ht if d['cls'] == 'R' for t in C.doc_terms(d)}
    tabs = [d for d in ht if d['cls'] == 'T' and C.doc_terms(d)]
    room = [d for d in tabs if d['findspot'].startswith('Portico 11')]
    def sc(ts):
        u = set().union(*[C.doc_terms(d) for d in ts]) if ts else set()
        return len(u & rterms)
    obs = sc(room)
    size = sum(len(C.doc_terms(d)) for d in room)
    vals = []
    while len(vals) < 3000:
        s = R.sample(tabs, len(room))
        sz = sum(len(C.doc_terms(d)) for d in s)
        if abs(sz - size) <= 0.25 * size:
            vals.append(sc(s))
    m, sd, z, p = zp(obs, vals)
    line = (f'  HT deposit: {len(room)} Portico 11/Room 13 tablets ({[d["id"] for d in room]}, {size} term types) '
            f'carry {obs} HT receipt terms ({sorted(set().union(*[C.doc_terms(d) for d in room]) & rterms)}) '
            f'vs random size-matched HT tablet sets {m:.2f} (z {z:.1f}, P {p:.3f})')
    print(line, flush=True)
    out.append(line)
    # by findspot: share of each room's tablet terms that are HT receipt terms
    by = collections.defaultdict(list)
    for d in tabs:
        by[d['findspot'] or '?'].append(d)
    for f, ds in by.items():
        u = set().union(*[C.doc_terms(d) for d in ds])
        line = f'    findspot {f!r}: {len(ds)} tablets, {len(u & rterms)}/{len(u)} term types are HT receipt terms'
        print(line); out.append(line)


if __name__ == '__main__':
    out = []
    la = C.load_la()
    lb = C.load_lb()
    out.append('== Linear A ==')
    rla = analyse(la, 'LA', out)
    per_site(la, 'LA', out)
    deposit(la, out)
    out.append('== Linear A, roundels only ==')
    analyse([d for d in la if not (d['cls'] == 'R' and d['support'] != 'Roundel')], 'LA-roundels', out, 1000)
    out.append('== Linear A, nodules only ==')
    analyse([d for d in la if not (d['cls'] == 'R' and d['support'] != 'Nodule')], 'LA-nodules', out, 1000)
    out.append('== Linear B control ==')
    rlb = analyse(lb, 'LB', out)
    per_site(lb, 'LB', out)
    out.append('== planted, Linear A size ==')
    planted(la, out)
    json.dump(dict(la=rla, lb=rlb), open(os.path.join(C.CK, 'c1.json'), 'w'))
    open(os.path.join(C.CK, 'c1.out'), 'w').write('\n'.join(out) + '\n')
