"""v61 cycle 3: rewrite the book in the recovered 'base forms' and measure what that does.
For each corpus/direction, take the greedy system from cycle 2b (selected on train pages for edge agreement),
apply it to the whole text and report vocabulary (types, hapax, Zipf, adjacent repeats), junction coupling,
and held-out page prediction (NB accuracy for section, and Currier language for the Voynich), against
(a) the original text, (b) the same rules with context classes replaced by random classes (shuffled neighbour
pairs), (c) for controls, the true base text. Also: the base-form lexicon (most frequent merged types)."""
import sys, json, random
from collections import Counter
from multiprocessing import Pool
import numpy as np
import v61_lib as L
from v61_c1 import build


def fwd(lines, d):
    return L.reverse_text(lines) if d == 'P' else lines


def job(arg):
    name, d = arg
    lines = build(name)
    sysd = L.jload('c2b_%s_%s.json' % (name.replace(':', '_'), d))
    rules = [(S, B, set(C)) for S, B, C in sysd['greedy_rules']]
    X = fwd(lines, d)
    new, nrw = L.apply_rules(X, rules) if rules else (X, 0)
    rng = random.Random(5)
    glyphs = sorted({w[0] for l in X for w in l['words']})
    shuf_rules = [(S, B, set(rng.sample(glyphs, min(len(glyphs), len(C))))) for S, B, C in rules]
    sh, nsh = L.apply_rules(X, shuf_rules) if rules else (X, 0)
    new, sh = fwd(new, d), fwd(sh, d)
    out = {'name': name, 'dir': d, 'n_rules': len(rules), 'rewrites': nrw, 'rewrites_shuf': nsh}
    vars_ = ['sec'] + (['lang'] if name.startswith('VMS') else [])
    for tag, T in (('orig', lines), ('base', new), ('ctxshuf', sh)) + ((('gold', [dict(l, words=l['base']) for l in lines]),) if lines[0].get('base') else ()):
        st = L.vocab_stats(T); st['mi'] = L.coupling_mi(T)
        for v in vars_:
            st['acc_' + v] = L.nb_page_score(T, v, reps=10)[1]
        out[tag] = st
    # merged lexicon: which surface types were folded into which base types (by token count)
    mg = Counter()
    for a, b in zip(lines, new):
        for w0, w1 in zip(a['words'], b['words']):
            if w0 != w1:
                mg[(w0, w1)] += 1
    if name.startswith('VMS'):
        out['merges'] = [(L.unglyph(a), L.unglyph(b), n) for (a, b), n in mg.most_common(25)]
        out['rules'] = [(L.unglyph(S[::-1] if d == 'P' else S), L.unglyph(B[::-1] if d == 'P' else B), L.unglyph(''.join(sorted(C)))) for S, B, C in rules]
    L.jsave('c3_%s_%s.json' % (name.replace(':', '_'), d), out)
    return out


if __name__ == '__main__':
    jobs = [('Sanskrit', 'R'), ('Italian', 'R'), ('Welsh', 'P'), ('VMS-ZL', 'R'), ('VMS-ZL', 'P'), ('VMS-IT', 'R'), ('VMS-IT', 'P'),
            ('null-pairblind:VMS-ZL', 'R'), ('null-pairblind:VMS-ZL', 'P'), ('VMS-planted', 'R')]
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            print(json.dumps(r), flush=True)
