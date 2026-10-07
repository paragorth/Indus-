"""v86 cycle 2: real Voynich. Rule search + rule-free subsequence recoverability per slot class:
E (line end, not paragraph end), Epe (line end of the paragraph's last line: no pressure), Ib (before a drawing
intrusion), Ia (after an intrusion), B (line start: an edge without pressure). Kill controls: words permuted inside
lines (10 seeds) and words re-dealt over the page (5 seeds)."""
import os, random
from multiprocessing import Pool
import v86_lib as L

NR = int(os.environ.get('NR', 4000))
CACHE = {}


def get(name):
    if name not in CACHE:
        CACHE[name] = L.voynich_pages(name)[0]
    return CACHE[name]


def relabel_pe(pages):
    """Class 'Epe' for the line-final word of a paragraph's last line."""
    out = {}
    for f, ls in pages.items():
        out[f] = [dict(ln, cls=[('Epe' if (c == 'E' and ln['para_end']) else c) for c in ln['cls']]) for ln in ls]
    return out


def job(args):
    tr, ctl, seed, target = args
    pages = relabel_pe(get(tr))
    A = L.alphabet_of(pages)
    VF = {g for g in A if sum(w.count(g) for ls in pages.values() for ln in ls for w in ln['units'] if w) >= 200}
    R = [L.random_rule(random.Random(86), sorted(VF)) for _ in range(1)]  # placeholder to keep rng api
    rng = random.Random(86)
    R = [L.random_rule(rng, VF) for _ in range(NR)]
    if ctl == 'pos':
        pages = L.shuffle_positions(pages, random.Random(1000 + seed))
    elif ctl == 'page':
        pages = L.shuffle_lines_within_page(pages, random.Random(2000 + seed))
    res = {}
    for sp in (0, 1):
        s = L.search(pages, R, target=target, seed=sp)
        res[sp] = dict(best_disc=s['best_disc'], held_top=s['held_top'], held_top5=s['held_top5'],
                       n=s['n_target_oov'], surv=s['surv'][:8])
    ss = None
    if ctl == 'real' or seed < 3:
        r = L.subseq_recovery1(pages, list(pages), target)
        ss = dict(R_t=r['R_t'], R_c=r['R_c'], delta=r['delta'], ci=r['ci'], n_t=r['n_t'],
                  marks=r['mark_t'].most_common(8), marks_c=r['mark_c'].most_common(8),
                  protect=L.protect_profile(r))
    out = dict(tr=tr, ctl=ctl, seed=seed, target=target, search=res, subseq=ss)
    L.save('c2_%s_%s_%d_%s.json' % (tr, ctl, seed, '+'.join(target)), out)
    return out


if __name__ == '__main__':
    jobs = []
    for tr in ('ZL3b', 'IT2a'):
        for tg in (('E',), ('Epe',), ('Ib',), ('Ia',), ('B',)):
            jobs.append((tr, 'real', 0, tg))
    for tg in (('E',), ('Ib',), ('B',)):
        for s in range(8):
            jobs.append(('ZL3b', 'pos', s, tg))
        for s in range(4):
            jobs.append(('ZL3b', 'page', s, tg))
    with Pool(2) as p:
        for o in p.imap_unordered(job, jobs):
            s0, s1 = o['search'][0], o['search'][1]
            line = '%s %s s%d %s n%d held_top %.4f/%.4f top5 %.4f/%.4f' % (o['tr'], o['ctl'], o['seed'], '+'.join(o['target']),
                    s0['n'], s0['held_top'], s1['held_top'], s0['held_top5'], s1['held_top5'])
            if o['subseq']:
                q = o['subseq']; line += ' | subseq dT-C %.4f [%.4f,%.4f] Rt %.3f' % (q['delta'], q['ci'][0], q['ci'][1], q['R_t'])
            print(line, flush=True)
            if o['ctl'] == 'real':
                print('    rules:', [(x['rule'], round(x['held_delta'], 4)) for x in s0['surv'][:4]], flush=True)
                if o['subseq']:
                    pr = sorted(o['subseq']['protect'].items(), key=lambda kv: kv[1][0] - kv[1][1])
                    print('    dropped:', [(g, round(a - b, 2), int(n)) for g, (a, b, n) in pr[:6]], 'kept:',
                          [(g, round(a - b, 2), int(n)) for g, (a, b, n) in pr[-5:]], 'marks', o['subseq']['marks'][:5], flush=True)
