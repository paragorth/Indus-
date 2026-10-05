"""v61 cycle 1: exhaustive + random-class scan of edge-rewrite rules on controls, Voynich and nulls."""
import sys, json, math
from multiprocessing import Pool
import numpy as np
import v61_lib as L


def corpora():
    C = {}
    C['VMS-ZL'] = L.load_vms('ZL3b')
    C['VMS-IT'] = L.load_vms('IT2a') if False else None
    return C


def build(name):
    if name == 'VMS-ZL':
        return L.load_vms('ZL3b')
    if name == 'VMS-IT':
        return L.load_vms('IT2a')
    if name in ('VMS-ZL-A', 'VMS-ZL-B', 'VMS-IT-A', 'VMS-IT-B'):
        return [l for l in L.load_vms('ZL3b' if '-ZL-' in name else 'IT2a') if l['lang'] == name[-1]]
    if name == 'Sanskrit11k':
        return L.opaque(L.load_sanskrit(max_tokens=11000))[0]
    if name == 'Sanskrit':
        return L.opaque(L.load_sanskrit())[0]
    if name == 'Welsh':
        return L.opaque(L.load_welsh())[0]
    if name == 'Italian':
        return L.opaque(L.load_italian())[0]
    if name == 'VMS-planted':
        # invented regressive sandhi on the Voynich: base -dy -> surface -ty before q/o; -in -> -ir before C/S
        return L.plant_sandhi(L.load_vms('ZL3b'), [('dy', 'ty', set('qo')), ('in', 'ir', set('CS'))], prob=0.9)
    if name.startswith('null-shuf:'):
        return L.within_line_shuffle(build(name.split(':', 1)[1]), seed=3)
    if name.startswith('null-pair:'):
        return L.pair_resample(build(name.split(':', 1)[1]), seed=3)
    if name.startswith('null-pairblind:'):
        return L.pair_resample(build(name.split(':', 1)[1]), seed=3, line_blind_start=True)
    if name.startswith('null-selfcit:'):
        return L.self_citation(build(name.split(':', 1)[1]), seed=3)
    raise KeyError(name)


def summarize(name, lines, recs, nh, info, direction):
    good = [r for r in recs if r['z_test'] > 3]
    top = recs[:50]
    out = {'name': name, 'dir': direction, 'nh': nh, 'info': info, 'n_rules': len(recs),
           'n_test3': len(good), 'frac_test3_top50': sum(r['z_test'] > 3 for r in top) / max(1, len(top))}
    le = []
    for r in recs[:200]:
        x = L.line_end_llr(r, 'both'); x.update({'S': r['S'], 'B': r['B'], 'cls': r['cls'], 'z_train': r['z_train'],
                                                 'z_test': r['z_test'], 'xz': L.xline_z(r)})
        le.append(x)
    sel = [x for x in le if x['z_test'] > 3 and x['n'] >= 10]
    out['le_top'] = le[:40]
    out['n_sel'] = len(sel)
    out['n_sandhi_consistent'] = sum(1 for x in sel if x['llN'] > max(x['llC'], 0) + 2)
    out['n_base_flipped'] = sum(1 for x in sel if x['llC'] > max(x['llN'], 0) + 2)
    out['mean_llN_minus_llC'] = float(np.mean([x['llN'] - x['llC'] for x in sel])) if sel else None
    out['median_xz'] = float(np.median([x['xz'] for x in sel])) if sel else None
    # recovery precision for controls: apply sandhi-consistent rules top 15
    rules = [(x['S'], x['B'], set(x['cls'])) for x in sel if x['llN'] > max(x['llC'], 0) + 2][:15]
    if rules:
        new, nrw = L.apply_rules(lines, rules)
        if lines[0].get('base'):
            tp = ex = nmk = 0
            for A, Bl in zip(lines, new):
                for i, (w0, w1) in enumerate(zip(A['words'], Bl['words'])):
                    if A['marked'][i]:
                        nmk += 1
                    if w0 != w1:
                        tp += A['marked'][i]; ex += (w1 == A['base'][i])
            out['recovery'] = {'rewrites': nrw, 'precision_marked': tp / max(1, nrw), 'exact_base': ex / max(1, nrw),
                               'recall_marked': tp / max(1, nmk)}
        out['mi_before'] = L.coupling_mi(lines); out['mi_after'] = L.coupling_mi(new)
        out['types_before'] = len({w for l in lines for w in l['words']}); out['types_after'] = len({w for l in new for w in l['words']})
    return out


def job(arg):
    name, direction = arg
    lines = build(name)
    if direction == 'P':
        lines = L.reverse_text(lines)
    recs, nh, info = L.scan(lines, n_random_classes=4000, seed=1)
    s = summarize(name, lines, recs, nh, info, direction)
    L.jsave('c1_%s_%s.json' % (name.replace(':', '_'), direction), {'summary': s, 'recs': recs[:400]})
    return s


if __name__ == '__main__':
    names = ['Sanskrit', 'Italian', 'Welsh', 'VMS-planted', 'VMS-ZL', 'VMS-IT',
             'null-shuf:VMS-ZL', 'null-pair:VMS-ZL', 'null-selfcit:VMS-ZL', 'null-pair:Sanskrit', 'null-shuf:Sanskrit']
    jobs = [(n, 'R') for n in names] + [(n, 'P') for n in ['Welsh', 'Italian', 'VMS-ZL', 'VMS-IT', 'null-pair:VMS-ZL', 'Sanskrit']]
    jobs += [(n, d) for n in ['null-pairblind:VMS-ZL', 'null-pairblind:Welsh', 'null-pair:Welsh'] for d in 'RP']
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    with Pool(2) as p:
        for s in p.imap_unordered(job, jobs):
            print(json.dumps({k: v for k, v in s.items() if k != 'le_top'}), flush=True)
