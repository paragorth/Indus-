#!/usr/bin/env python3
"""la65 cycle 1: one random-template search run.
usage: la65_c1.py CORPUS SEED NTEMPL
  CORPUS = LA | LAN (LA without metadata features) | LBd (Linear B draw d at LA size, content only) |
           PLs (LA + planted dossier, seed s)
  SEED   = 0 for the real corpus, k >= 1 for entry-shuffled null k
"""
import sys, os, json, random, time
import la65_common as K


def corpus(name):
    T = K.la_tabs()
    meta = True
    if name == 'LAN':
        meta = False
    elif name.startswith('LB'):
        T = K.lb_draw(len(T), int(name[2:]))
        meta = False
    elif name.startswith('PL'):
        s = int(name[2:])
        rng = random.Random(K.seed('la65-plant-%d' % s))
        src, P = K.plant_dossier(T, rng, tag='PL%d' % s)
        T = T + P
    return T, meta


def main():
    name, s, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fn = os.path.join(K.CK, 'c1_%s_%d.json' % (name, s))
    if os.path.exists(fn):
        return
    T, meta = corpus(name)
    if s > 0:
        T = K.shuffle_ents(T, random.Random(K.seed('la65-null-%s-%d' % (name, s))))
    t0 = time.time()
    ix = K.Index(T, meta)
    best = K.search(ix, n, random.Random(K.seed('la65-search-%s-%d' % (name, s))))
    out = [[[T[i]['id'] for i in G], round(h, 4), list(tm)] for G, (h, tm) in best.items()]
    json.dump({'name': name, 'seed': s, 'n': n, 'groups': out, 'sec': time.time() - t0}, open(fn + '.tmp', 'w'))
    os.replace(fn + '.tmp', fn)
    print(name, s, len(out), round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
