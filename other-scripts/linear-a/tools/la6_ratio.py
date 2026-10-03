#!/usr/bin/env python3
"""LA-6 cycle 1: exchange rates from numbers and commodity ideograms only.

For every pair of commodities written against the same entry (same label), collect
r = log(q_b / q_a). Statistics per pair:
  SD   = standard deviation of r (small = fixed ratio)
  HIT  = largest number of entries whose ratio lies within +-10% of one common ratio
Nulls (2,000 runs each):
  N1 'shuffle within commodity': every commodity's quantities are permuted across all
     entries that carry that commodity anywhere in the corpus (one- or many-commodity).
  N2 'within pair': q_b permuted among the entries of that pair only (keeps both marginals
     of the pair, breaks the pairing).
Positive control: the same code on Linear B (DAMOS) line-level entries, where fixed rations
  are known to exist (KN Fs, PY Ab etc.); quantities converted to major units with the
  conventional dry/liquid ratios.
Run: python3 la6_ratio.py [--nofrac] [--doc]
"""
import sys, math, random
from collections import defaultdict, Counter
sys.path.insert(0, __import__('os').path.dirname(__file__))
from la6_common import la_entries, lb_docs

random.seed(6)
NOFRAC = '--nofrac' in sys.argv
DOC = '--doc' in sys.argv
R = 2000
TOL = math.log(1.1)


def pairs_of(entries, minn=4):
    P = defaultdict(list)
    for i, e in enumerate(entries):
        cs = sorted(c for c, v in e.items() if v > 0)
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                P[(cs[a], cs[b])].append(i)
    return {k: v for k, v in P.items() if len(v) >= minn}


def stats(rs, docs=None):
    """docs: optional doc id per ratio; then HIT counts distinct documents in the window."""
    n = len(rs)
    m = sum(rs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (n - 1))
    o = sorted(range(n), key=lambda i: rs[i])
    s = [rs[i] for i in o]
    best, j = 0, 0
    for i in range(n):
        while s[i] - s[j] > 2 * TOL: j += 1
        best = max(best, i - j + 1 if docs is None else len({docs[o[k]] for k in range(j, i + 1)}))
    return sd, best


def run(entries, label, out, minn=4, extra_pool=None, docs=None):
    """entries: list of {com: float}. extra_pool: {com: [values]} from single-commodity entries."""
    P = pairs_of(entries, minn)
    pool = defaultdict(list)
    for e in entries:
        for c, v in e.items():
            if v > 0: pool[c].append(v)
    if extra_pool:
        for c, vs in extra_pool.items(): pool[c].extend(vs)
    obs = {}
    clusters = {}
    for k, idx in P.items():
        a, b = k
        rs = [math.log(entries[i][b] / entries[i][a]) for i in idx]
        dd = [docs[i] for i in idx] if docs else None
        obs[k] = stats(rs, dd) + (len(idx), math.exp(sorted(rs)[len(rs) // 2]))
        # members of the best cluster
        best = None
        for c in rs:
            mem = [i for i, r in zip(idx, rs) if abs(r - c) <= TOL]
            sc = len({docs[i] for i in mem}) if docs else len(mem)
            if best is None or sc > best[0]: best = (sc, c, mem)
        clusters[k] = (math.exp(best[1]), [(docs[i] if docs else i, round(entries[i][a], 2), round(entries[i][b], 2)) for i in best[2]])
    # nulls
    cnt1 = {k: [0, 0, 0.0, 0.0] for k in P}
    cnt2 = {k: [0, 0, 0.0, 0.0] for k in P}
    gl_obs = sum(o[1] for o in obs.values())
    gl1 = gl2 = 0
    for _ in range(R):
        g1 = g2 = 0
        for k, idx in P.items():
            a, b = k
            ra = [random.choice(pool[a]) for _ in idx]
            rb = [random.choice(pool[b]) for _ in idx]
            dd = [docs[i] for i in idx] if docs else None
            sd1, h1 = stats([math.log(y / x) for x, y in zip(ra, rb)], dd)
            qb = [entries[i][b] for i in idx]
            random.shuffle(qb)
            sd2, h2 = stats([math.log(y / entries[i][a]) for i, y in zip(idx, qb)], dd)
            o = obs[k]
            cnt1[k][0] += sd1 <= o[0]; cnt1[k][1] += h1 >= o[1]; cnt1[k][2] += sd1; cnt1[k][3] += h1
            cnt2[k][0] += sd2 <= o[0]; cnt2[k][1] += h2 >= o[1]; cnt2[k][2] += sd2; cnt2[k][3] += h2
            g1 += h1; g2 += h2
        gl1 += g1 >= gl_obs; gl2 += g2 >= gl_obs
    out.append(f'## {label}: {len(P)} pairs with n>={minn}; global HIT sum {gl_obs}: P(N1)={(gl1+1)/(R+1):.4f} P(N2)={(gl2+1)/(R+1):.4f}')
    out.append('pair | n | median ratio b/a | SD log | N1 SD mean, P | N2 SD mean, P | HIT | N1 HIT mean, P | N2 HIT mean, P')
    for k in sorted(P, key=lambda k: -obs[k][2]):
        o = obs[k]
        c1, c2 = cnt1[k], cnt2[k]
        out.append(f'{k[0]}:{k[1]} | {o[2]} | {o[3]:.3g} | {o[0]:.2f} | {c1[2]/R:.2f} {(c1[0]+1)/(R+1):.3f} | {c2[2]/R:.2f} {(c2[0]+1)/(R+1):.3f} | '
                   f'{o[1]} | {c1[3]/R:.2f} {(c1[1]+1)/(R+1):.3f} | {c2[3]/R:.2f} {(c2[1]+1)/(R+1):.3f}')
    out.append('best common-ratio cluster per pair (ratio b/a; members doc, q_a, q_b):')
    for k in sorted(P, key=lambda k: -obs[k][2]):
        out.append(f'  {k[0]}:{k[1]} ~{clusters[k][0]:.3g}: {clusters[k][1]}')
    return obs


def la_sets():
    E = la_entries(use_frac=not NOFRAC)
    if DOC:
        D = defaultdict(lambda: defaultdict(float))
        for e in E:
            if e['role'] in ('total', 'grand', 'deficit'): continue
            for c, v in e['com'].items(): D[e['doc']][c] += float(v)
        ents = [dict(v) for v in D.values()]
        return ents, list(D.keys())
    keep = [e for e in E if e['role'] in ('entry', 'head', 'post')]
    ents = [{c: float(v) for c, v in e['com'].items()} for e in keep]
    return ents, [e['doc'] for e in keep]


def main():
    out = [f'# LA-6 cycle 1 ratio test (frac={"off" if NOFRAC else "conventional"}, level={"document" if DOC else "entry"})']
    ents, dl = la_sets()
    mi = [i for i, e in enumerate(ents) if sum(1 for v in e.values() if v > 0) >= 2]
    multi = [ents[i] for i in mi]
    mdocs = [dl[i] for i in mi]
    single = [e for e in ents if sum(1 for v in e.values() if v > 0) == 1]
    extra = defaultdict(list)
    for e in single:
        for c, v in e.items():
            if v > 0: extra[c].append(v)
    out.append(f'LA entries {len(ents)}, multi-commodity {len(multi)}, single-commodity {len(single)}')
    run(multi, 'Linear A (HIT = entries)', out, 4, extra)
    run(multi, 'Linear A (HIT = distinct tablets)', out, 4, extra, mdocs)
    # HT vs rest for the pairs present
    # Linear B positive control
    L = lb_docs()
    lb_lines = [{c: float(v) for c, v in l.items()} for d in L for l in d['lines']]
    lb_ids = [d['id'] for d in L for l in d['lines']]
    lmi = [i for i, e in enumerate(lb_lines) if sum(1 for v in e.values() if v > 0) >= 2]
    lmulti = [lb_lines[i] for i in lmi]
    lsingle = defaultdict(list)
    for e in lb_lines:
        if sum(1 for v in e.values() if v > 0) == 1:
            for c, v in e.items():
                if v > 0: lsingle[c].append(v)
    out.append(f'\nLB line entries {len(lb_lines)}, multi-commodity {len(lmulti)}')
    run(lmulti, 'Linear B (positive control, HIT = distinct tablets)', out, 4, lsingle, [lb_ids[i] for i in lmi])
    ldoc = [({c: float(v) for c, v in d['com'].items()}, d['id']) for d in L if sum(1 for v in d['com'].values() if v > 0) >= 2]
    lsd = defaultdict(list)
    for d in L:
        if sum(1 for v in d['com'].values() if v > 0) == 1:
            for c, v in d['com'].items():
                if v > 0: lsd[c].append(float(v))
    out.append(f'\nLB document-level entries, multi-commodity {len(ldoc)}')
    run([x[0] for x in ldoc], 'Linear B document level (positive control, HIT = distinct tablets)', out, 4, lsd, [x[1] for x in ldoc])
    txt = '\n'.join(out)
    print(txt)
    tag = ('_nofrac' if NOFRAC else '') + ('_doc' if DOC else '')
    open(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'data', f'la6_ratio{tag}.out'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
