#!/usr/bin/env python3
"""LA-12 cycle 3: what do the joins share, and do joins give words a role?

The arithmetic counts in cycles 1-2 ask 'are there more closures than chance?'. This cycle asks
an independent question that the number shuffle cannot fake: are the tablets that the search
joins (a) written by the same scribe, (b) from the same findspot, (c) two sides of one tablet,
(d) close in GORILA number (find groups), (e) sharing a word, more often than the tablets joined
by the SAME search on number-shuffled corpora? Shuffling keeps every tablet's scribe, findspot
and words, so any excess comes from the real numbers linking related tablets.
Join kinds: J1 = a written total equals a block of another tablet (cycle-1 k1);
            JC = own section + one block of another tablet = the total (cycle-1 c1);
            JS = any quantity equals a multi-item sum of another tablet (cycle-2 k1).
Word roles: for JS joins, the labels of the 'summary lines' and the first words of the donor
tablets, against the same lists in the null joins.
Other sites: Khania and Zakros tablets, cycle-1 and cycle-2 statistics with N1 nulls.
Usage: la12_c3.py [mode] [nnull]
"""
import sys, re, random, time
from multiprocessing import Pool
from la12_common import *
import la12_c1 as C1
import la12_c2 as C2

STOP = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO'}


def joins(recs, mode, cval):
    B, T = blocks_targets(recs, mode, cval)
    T = [t for t in T if t['key'] > 0]
    idx = make_indexes(B)
    J = []
    for t in T:
        I = idx.get(t['tag'], idx['*']) if t['tag'] != '*' else idx['*']
        for k, tag, rid, desc in I.blocks:
            if rid == t['rid']:
                continue
            if k == t['key']:
                J.append(('J1', t['rid'], rid, None))
            if 0 < t['own'] < t['key'] and k == t['key'] - t['own']:
                J.append(('JC', t['rid'], rid, None))
    H, D, TT = C2.hits(recs, mode, cval, kmax=1)
    for ti, rs, w, k, ds in H:
        J.append(('JS', TT[ti]['rid'], rs[0], TT[ti]['label']))
    return J


def num(i):
    m = re.match(r'^[A-Z]+(\d+)', i)
    return int(m.group(1)) if m else None


def base(i):
    m = re.match(r'^([A-Z]+\d+)', i)
    return m.group(1) if m else i


def attrs(a, b):
    sa, sb = a['scribe'], b['scribe']
    fa, fb = a['findspot'], b['findspot']
    wa = {w for w in a['words'] if '-' in w and w not in STOP}
    wb = {w for w in b['words'] if '-' in w and w not in STOP}
    na, nb = num(a['id']), num(b['id'])
    return {'scribe_known': bool(sa and sb), 'same_scribe': bool(sa and sa == sb),
            'fs_known': bool(fa and fb), 'same_fs': bool(fa and fa == fb),
            'same_tablet': base(a['id']) == base(b['id']),
            'near_id': na is not None and nb is not None and abs(na - nb) <= 5,
            'shared_word': bool(wa & wb)}


def rates(J, recs):
    out = {}
    for kind in ('J1', 'JC', 'JS', 'ALL'):
        jj = [j for j in J if kind == 'ALL' or j[0] == kind]
        pairs = {(min(x[1], x[2]), max(x[1], x[2])) for x in jj}
        n = len(pairs)
        A = [attrs(recs[a], recs[b]) for a, b in pairs]
        sk = sum(x['scribe_known'] for x in A); fk = sum(x['fs_known'] for x in A)
        out[kind] = {'n': n,
                     'same_scribe': sum(x['same_scribe'] for x in A) / sk if sk else float('nan'),
                     'same_fs': sum(x['same_fs'] for x in A) / fk if fk else float('nan'),
                     'same_tablet': sum(x['same_tablet'] for x in A) / n if n else float('nan'),
                     'near_id': sum(x['near_id'] for x in A) / n if n else float('nan'),
                     'shared_word': sum(x['shared_word'] for x in A) / n if n else float('nan')}
    return out


def roles(J, recs):
    tl = Counter(); dw = Counter()
    for kind, tr, dr, lab in J:
        if kind != 'JS':
            continue
        if lab and '-' in str(lab):
            tl[lab] += 1
        fw = next((w for w in recs[dr]['words'] if '-' in w and w not in STOP), None)
        if fw:
            dw[fw] += 1
    return tl, dw


def _job(a):
    seed, recs, mode, cval = a
    rng = random.Random(seed)
    R2 = shuffle_numbers(recs, rng, totals_too=False)
    J = joins(R2, mode, cval)
    return rates(J, recs), roles(J, recs)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'V'
    nnull = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    out = open(os.path.join(OUT, 'c3_%s.txt' % mode), 'w')
    t0 = time.time()
    R = la_records('HT')
    J = joins(R, mode, LA_CVAL)
    real = rates(J, R); rtl, rdw = roles(J, R)
    with Pool(2) as P:
        res = P.map(_job, [(7000 + i, R, mode, LA_CVAL) for i in range(nnull)], chunksize=4)
        for kind in ('J1', 'JC', 'JS', 'ALL'):
            s = ['n %d (null %.1f)' % (real[kind]['n'], sum(r[0][kind]['n'] for r in res) / len(res))]
            for a in ('same_scribe', 'same_fs', 'same_tablet', 'near_id', 'shared_word'):
                xs = [r[0][kind][a] for r in res if r[0][kind][a] == r[0][kind][a]]
                v = real[kind][a]
                if not xs or v != v:
                    continue
                p = (1 + sum(x >= v for x in xs)) / (1 + len(xs))
                s.append('%s %.3f (null %.3f, P=%.3f)' % (a, v, sum(xs) / len(xs), p))
            out.write('HT %s joins: %s\n' % (kind, '; '.join(s)))
        # join list with attributes (real)
        out.write('Real joined pairs (kind: target tablet <- donor tablet; attributes):\n')
        seen = set()
        for kind, tr, dr, lab in J:
            if (kind, tr, dr) in seen:
                continue
            seen.add((kind, tr, dr))
            at = attrs(R[tr], R[dr])
            flags = [k for k in ('same_scribe', 'same_fs', 'same_tablet', 'near_id', 'shared_word') if at[k]]
            if flags:
                out.write('  %s %s <- %s %s label=%s\n' % (kind, R[tr]['id'], R[dr]['id'], flags, lab))
        # word roles
        out.write('Summary-line labels in real JS joins vs mean in null joins (words with real >= 2):\n')
        for w, c in rtl.most_common():
            if c < 2:
                break
            xs = [r[1][0].get(w, 0) for r in res]
            out.write('  line-label %s real %d null mean %.2f P=%.3f\n' % (w, c, sum(xs) / len(xs), (1 + sum(x >= c for x in xs)) / (1 + len(xs))))
        for w, c in rdw.most_common():
            if c < 2:
                break
            xs = [r[1][1].get(w, 0) for r in res]
            out.write('  donor-first-word %s real %d null mean %.2f P=%.3f\n' % (w, c, sum(xs) / len(xs), (1 + sum(x >= c for x in xs)) / (1 + len(xs))))
        out.flush()
        # other sites
        for site in ('KH', 'ZA'):
            Rs = la_records(site)
            c, B, T, idx = C1.counts(Rs, mode, LA_CVAL)
            if not c:
                out.write('%s: no targets\n' % site)
            else:
                realc = {k: v['real'] for k, v in c.items()}
                n1, n2 = C1.run_null(P, Rs, mode, LA_CVAL, 60, 300)
                out.write('%s cycle-1 stats (records %d, targets %d) vs N1: %s\n' % (site, len(Rs), len(realc), C1.compare(realc, n1)[0]))
            C2.evaluate(P, Rs, mode, LA_CVAL, 30, 400, '%s cycle-2' % site, out)
    out.write('time %.0fs\n' % (time.time() - t0))


if __name__ == '__main__':
    main()
