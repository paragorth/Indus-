"""v36 cycle 2: PLANTED controls run through the v30 search.
Two halves of one corpus; the second half gets a planted change; the v30 guided search (30 rules,
n_cand 200) learns X -> Y on fit pages; the learned rules are then scored by v36 COH/CTX/SUB.
  H_*  Hangul (NSMC jamo; v25 jamo stroke features)
       FEAT = y-vowels lose one tick (ya->a, yeo->eo, yo->o, yu->u) + aspirates lose their extra stroke (k->g, t->d, ch->j)
       ARB  = 7 frequency-matched jamo outside those classes, each mapped to a random other jamo
  P_*  Voynich Currier A (v25 hand strokes)
       FEAT = benched gallows lose the bench (cth->t, ckh->k, cph->p, cfh->f) + plume lost (sh->ch, s->e)
       ARB  = 6 frequency-matched glyphs mapped to random other glyphs
  G_*  Bavarian German (phonetic letter features)
       FEAT = final devoicing (b d g -> p t k word-finally) + intervocalic lenition (p t k -> b d g between vowels)
       ARB  = 6 frequency-matched letters mapped to random letters, same positions mix (half final-only)
Each planted change fires on a token with probability 0.7.  NULL = the same halves without a plant.
Usage: python3 v36_plant.py [names...]
"""
import os, sys, json, random, time, collections
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as L
import v25_lib as V25
from v18_lib import glyphs as vglyphs
from v30_lib import run_search, split_pages, distance

RATE = 0.7
VOW_DE = set('aeiouy')


def pages_of(words, size=200):
    return [words[i:i + size] for i in range(0, len(words) - size + 1, size)]


def base(script):
    if script == 'H':
        ws = [''.join(w) for w in V25.hangul(tok=260000)]
    elif script == 'P':
        ws = [w for p in L.corpora30()['V_A'] for w in p]
    else:
        ws = [w for p in L.corpora30()['G_Bav1'] for w in p]
    P = pages_of(ws)
    return P[0::2], P[1::2]


def feat_map(script):
    if script == 'H':
        return {'ㅑ': 'ㅏ', 'ㅕ': 'ㅓ', 'ㅛ': 'ㅗ', 'ㅠ': 'ㅜ', 'ㅋ': 'ㄱ', 'ㅌ': 'ㄷ', 'ㅊ': 'ㅈ'}
    if script == 'P':
        return {'cth': 't', 'ckh': 'k', 'cph': 'p', 'cfh': 'f', 'sh': 'ch', 's': 'e'}
    return None


def tok(script):
    return vglyphs if script == 'P' else list


def arb_map(script, seed, X):
    rng = random.Random(seed)
    T = tok(script)
    fr = collections.Counter(g for p in X for w in p for g in T(w))
    if script == 'G':
        fm = {'b': 1, 'd': 1, 'g': 1, 'p': 1, 't': 1, 'k': 1}
    else:
        fm = feat_map(script)
    alph = [g for g, c in fr.most_common() if c >= 20 and (script != 'P' or g in L.S.VOYNICH)]
    pool = [g for g in alph if g not in fm]
    src = []
    for g in fm:  # nearest-frequency glyph not yet used, chosen at random among the 3 nearest
        cand = sorted((g2 for g2 in pool if g2 not in src), key=lambda x: abs(fr[x] - fr[g]))[:3]
        src.append(rng.choice(cand))
    return {g: rng.choice([h for h in alph if h != g]) for g in src}


def plant_word(w, script, kind, mp, rng):
    T = tok(script)
    gs = T(w)
    out = []
    for i, g in enumerate(gs):
        if script == 'G' and kind == 'FEAT':
            fin = i == len(gs) - 1
            prev_v = i > 0 and gs[i - 1] in VOW_DE; next_v = i + 1 < len(gs) and gs[i + 1] in VOW_DE
            if fin and g in 'bdg' and rng.random() < RATE:
                out.append({'b': 'p', 'd': 't', 'g': 'k'}[g]); continue
            if prev_v and next_v and g in 'ptk' and rng.random() < RATE:
                out.append({'p': 'b', 't': 'd', 'k': 'g'}[g]); continue
            out.append(g); continue
        if script == 'G' and kind == 'ARB':
            keys = list(mp)
            ok = (keys.index(g) % 2 == 0 and i == len(gs) - 1) or (keys.index(g) % 2 == 1) if g in mp else False
            if ok and rng.random() < RATE:
                out.append(mp[g]); continue
            out.append(g); continue
        if g in mp and rng.random() < RATE:
            out.append(mp[g])
        else:
            out.append(g)
    return ''.join(out)


def make(name):
    script, kind = name.split('_')[0], name.split('_')[1]
    X, Y = base(script)
    mp = {}
    if kind == 'FEAT':
        mp = feat_map(script) or {}
    elif kind.startswith('ARB'):
        mp = arb_map(script, 100 + int(kind[3:] or 1), X)
    rng = random.Random(7)
    if kind != 'NULL':
        Y = [[plant_word(w, script, 'ARB' if kind.startswith('ARB') else kind, mp, rng) for w in p] for p in Y]
    return X, Y, mp


def job(name):
    out = os.path.join(L.CK, f'plant_{name}.json')
    if os.path.exists(out):
        return
    t = time.time()
    X, Y, mp = make(name)
    xf, xh, _ = split_pages(X, 3000, 1500, 1)
    yf, yh, _ = split_pages(Y, 3000, 1500, 2)
    floor = distance(yf[:len(yh)], yh)
    path = run_search(xf, yf, xh, yh, kmax=30, n_cand=200, seed=0, guided=True)
    json.dump(dict(name=name, map=mp, floor=floor, path=path, secs=time.time() - t), open(out, 'w'), default=float, ensure_ascii=False)
    print(name, 'done', round(time.time() - t), 'D0', round(path[0]['held'], 3), 'Dk', round(path[-1]['held'], 3), 'F', round(floor[0], 3), mp, flush=True)


NAMES = ['P_FEAT', 'P_ARB1', 'H_FEAT', 'H_ARB1', 'G_FEAT', 'G_ARB1', 'P_NULL', 'H_NULL', 'P_ARB2', 'H_ARB2', 'G_ARB2']

if __name__ == '__main__':
    names = sys.argv[1:] or NAMES
    with Pool(2) as p:
        list(p.imap_unordered(job, names))
