"""v86 cycle 1: calibration of the recoverability search. Languages laid out in the Voynich template,
with and without planted medieval abbreviation at squeezed slots; Voynich-based meaning and habit plants."""
import sys, os, random, json
from collections import Counter
from multiprocessing import Pool
import v86_lib as L

NR = int(os.environ.get('NR', 4000))
VP, META = L.voynich_pages('ZL3b')
NTOK = sum(len(ln['units']) for ls in VP.values() for ln in ls)
VA = L.alphabet_of(VP)
VFREQ = {g for g in VA if sum(w.count(g) for ls in VP.values() for ln in ls for w in ln['units'] if w) >= 200}


def corpus(name):
    rng = random.Random(7)
    if name.startswith('lang:'):
        _, key, pl = name.split(':')
        pages = L.layout_like(VP, L.ref_words(key, NTOK))
        if pl == 'abbr':
            pages = L.plant(pages, L.lat_abbrev, p=0.5, seed=3)
        A = L.alphabet_of(pages) | {'~', '9'}
        A = {g for g in A if g.isalpha() or g in '~9'}
        return pages, A
    base = L.shuffle_positions(VP, rng)
    if name == 'vshuf:none':
        return base, VFREQ
    if name == 'vshuf:det':
        fn, D, k = L.make_det_compress(VFREQ, seed=11)
        print('det plant drops', D, 'cap', k, flush=True)
        return L.plant(base, fn, p=0.3, seed=5), VFREQ
    if name == 'vshuf:habit':
        return L.plant(base, L.make_habit(VFREQ), p=0.3, seed=5), VFREQ
    raise ValueError(name)


def run(name):
    pages, A = corpus(name)
    rng = random.Random(hash(name) % 1000)
    R = [L.random_rule(rng, A) for _ in range(NR)]
    out = {}
    for seed in (0, 1):
        s = L.search(pages, R, seed=seed)
        out['search%d' % seed] = s
    folios = list(pages)
    ss = L.subseq_recovery1(pages, folios, ('E', 'Ib'), exclude_para_end=True)
    prof = L.protect_profile(ss)
    out['subseq'] = {k: v for k, v in ss.items() if k in ('R_t', 'R_c', 'delta', 'ci', 'n_t')}
    out['marks_t'] = ss['mark_t'].most_common(6)
    out['marks_c'] = ss['mark_c'].most_common(6)
    out['protect'] = prof
    L.save('c1_%s.json' % name.replace(':', '_'), out)
    return name, out


if __name__ == '__main__':
    names = ['lang:Latin-Caesar:none', 'lang:Latin-Caesar:abbr', 'lang:Italian-Dante:abbr', 'lang:German-Kafka:abbr',
             'lang:Italian-Dante:none', 'vshuf:none', 'vshuf:det', 'vshuf:habit']
    with Pool(2) as p:
        for name, o in p.imap_unordered(run, names):
            s0, s1 = o['search0'], o['search1']
            print('==', name, 'nOOV', s0['n_target_oov'], 'best_disc %.4f/%.4f' % (s0['best_disc'], s1['best_disc']),
                  'held_top %.4f/%.4f' % (s0['held_top'], s1['held_top']), 'held_top5 %.4f/%.4f' % (s0['held_top5'], s1['held_top5']), flush=True)
            print('   top rules:', [(x['rule'], round(x['held_delta'], 4)) for x in s0['surv'][:3]], flush=True)
            print('   subseq', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o['subseq'].items()}, 'marks', o['marks_t'][:4], flush=True)
            pr = sorted(o['protect'].items(), key=lambda kv: kv[1][0] - kv[1][1])
            print('   most dropped:', [(g, round(a - b, 2), int(n)) for g, (a, b, n) in pr[:6]], flush=True)
            print('   most kept   :', [(g, round(a - b, 2), int(n)) for g, (a, b, n) in pr[-6:]], flush=True)
