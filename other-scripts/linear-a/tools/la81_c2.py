"""la81 cycle 2: THE ARROW OF WRITING TIME.
A list is written in time. If entries were written as events happened (deliveries, arrivals) or by a
running procedure, some entry properties drift or cycle in writing order; a list written from a sorted
source or by rank has a different arrow; a list whose order carries nothing has none.
Random codings of entries (random partitions of one or two entry properties into 2-3 states) are scored
for a signed arrow on lists published by CUT (trend of a state's position; cyclic flux for 3 states),
with an exact list-reversal null (each list's direction flipped).  Survivors are frozen and scored on
lists published later.  Controls: whole lottery rerun on direction-flipped training lists; planted drift;
Linear B KN -> PY run through the same code (structural only); commodity coding as a known positive.
First and last entries of every list are dropped (headings and totals are positional by construction)."""
import json, os, re, sys, math, random, collections, unicodedata
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from la81_common import CK, DATA, sha
from la78_common import pub_year, _rights
from la15_common import _pub_source

CUT = int(os.environ.get('CUT', 1950))
NH = int(os.environ.get('NH', 20000))
NSH = int(os.environ.get('NSH', 20))
ZT = float(os.environ.get('ZT', 3.0))
FE = ['first', 'last', 'len', 'mag', 'frac', 'logo', 'wfreq']


def mag(v):
    if v is None: return None
    v = float(v)
    return 0 if v < 1 else 1 if v < 2 else 2 if v < 5 else 3 if v < 10 else 4 if v < 50 else 5


def entries_from_tokens(toks):
    """toks: list of (type, value, extra); entry closes at a number"""
    E = []; cur = dict(word=None, logo=None)
    for t, v, x in toks:
        if t == 'word': cur['word'] = v
        elif t == 'logo': cur['logo'] = v
        elif t == 'num':
            E.append(dict(word=cur['word'], logo=cur['logo'], num=v, frac=x))
            cur = dict(word=None, logo=None)
    return E


def load_la():
    src = _pub_source(); url = _rights()
    C = json.load(open(os.path.join(DATA, 'corpus_ra.json')))
    L = []
    for d in C:
        if d['support'] != 'Tablet': continue
        y = pub_year(d['id'], src.get(d['id'], 'blank'), url.get(d['id'], ''))[0]
        toks = []
        for t in d['tokens']:
            if t['t'] == 'word' and t.get('st') in ('read', 'damaged'): toks.append(('word', tuple(t['s']), None))
            elif t['t'] == 'word': toks.append(('word', None, None))
            elif t['t'] == 'logo': toks.append(('logo', t['v'].split('+')[0], None))
            elif t['t'] == 'num' and t.get('st') in ('read', 'damaged'):
                toks.append(('num', t['v'], bool(t.get('frac'))))
        E = entries_from_tokens(toks)
        L.append(dict(id=d['id'], site=d['site'], year=y, E=E))
    return L


def _clean_lb(t):
    t = unicodedata.normalize('NFD', t)
    t = ''.join(ch for ch in t if unicodedata.category(ch) != 'Mn')
    t = re.sub(r"[\[\]⟦⟧⌞⌟⸢⸣'\"?!<>{}]", '', t)
    return t.strip('-')


def load_lb():
    L = []
    for line in open(os.path.join(DATA, 'damos_items.jsonl')):
        x = json.loads(line)
        if not x.get('content') or not x.get('heading'): continue
        h = x['heading']; site = h[:2]
        if site not in ('KN', 'PY'): continue
        toks = []
        for raw in x['content'].split():
            if raw.startswith('.') or raw in (',', '/', '//', ':'): continue
            t = _clean_lb(raw)
            if not t or '.' in t: continue
            if t.lower() in ('vacat', 'vac', 'lat', 'sup', 'inf', 'mut', 'vest', 'deest', 'fr', 'v'): continue
            if re.fullmatch(r'\d+', t): toks.append(('num', int(t), False)); continue
            if any(ch.islower() for ch in t):
                if not re.fullmatch(r'[a-z0-9*\-]+', t): continue
                s = [p.upper() for p in t.split('-') if p]
                if s: toks.append(('word', tuple(s), None))
                continue
            if re.fullmatch(r"[A-Z*0-9+]+", t) and len(t) > 1:
                if len(t) == 1 or t in ('M', 'N', 'P', 'Q', 'S', 'T', 'V', 'Z'):  # unit letters: fraction-like
                    if toks and toks[-1][0] == 'num': toks[-1] = ('num', toks[-1][1], True)
                    continue
                toks.append(('logo', t.split('+')[0], None))
            elif t in ('M', 'N', 'P', 'Q', 'S', 'T', 'V', 'Z'):
                # LB unit letter followed by a number: the number belongs to a sub-unit -> mark fraction on next num
                toks.append(('unit', t, None))
        # unit letters: a 'unit' token makes the following number a sub-unit (fraction analogue)
        t2 = []; pend = False
        for t in toks:
            if t[0] == 'unit': pend = True; continue
            if t[0] == 'num' and pend:
                if t2 and t2[-1][0] == 'num': t2[-1] = ('num', t2[-1][1], True); pend = False; continue
                pend = False
            t2.append(t)
        L.append(dict(id=h, site=site, year=1 if site == 'KN' else 2, E=entries_from_tokens(t2)))
    return L


def featurize(L, lex):
    for d in L:
        for e in d['E']:
            w = e['word']
            e['f'] = dict(first=w[0] if w else None, last=w[-1] if w else None,
                          len=(min(len(w), 4) if w else None), mag=mag(e['num']), frac=int(bool(e['frac'])),
                          logo=int(e['logo'] is not None),
                          wfreq=(None if not w else (0 if lex.get(w, 0) <= 1 else 1 if lex[w] <= 4 else 2)),
                          com=e['logo'])
    return L


def interior(L, minlen=4):
    out = []
    for d in L:
        if len(d['E']) >= minlen + 2: out.append(dict(d, I=d['E'][1:-1]))
    return out


def random_codings(nh, vals, seed):
    rng = random.Random(seed); H = []
    for i in range(nh):
        nf = 1 if rng.random() < 0.6 else 2
        fs = rng.sample(FE, nf); K = rng.choice([2, 3])
        maps = {}
        for f in fs:
            vs = list(vals[f]); rng.shuffle(vs)
            kf = min(len(vs), rng.choice([2, 3]))
            maps[f] = {v: (j % kf if j >= kf else j) for j, v in enumerate(vs)}
            # random partition: assign remaining values at random
            for v in vs[kf:]: maps[f][v] = rng.randrange(kf)
        H.append(dict(fs=fs, maps=maps, K=K, salt=rng.randrange(10 ** 9)))
    return H


def code_entry(h, e):
    s = 0
    for f in h['fs']:
        v = e['f'][f]
        if v is None or v not in h['maps'][f]: return None
        s = s * 7 + h['maps'][f][v]
    # fold the combined value into K states with a fixed random fold
    return (s * 2654435761 + h['salt']) % 1000003 % h['K'] if len(h['fs']) > 1 else s % h['K']


def list_stats(seq, K):
    """per list: trend of state 0 (sum of centred ranks), cyclic flux (K=3)"""
    n = len(seq)
    if n < 3 or len(set(seq)) < 2: return 0.0, 0.0
    r = [(i / (n - 1)) - 0.5 for i in range(n)]
    tr = sum(ri for ri, s in zip(r, seq) if s == 0)
    fl = 0.0
    if K == 3:
        for a, b in zip(seq, seq[1:]):
            if (b - a) % 3 == 1: fl += 1
            elif (b - a) % 3 == 2: fl -= 1
    tr = round(tr, 9)
    return (0.0 if abs(tr) < 1e-9 else tr), fl


def score(H, lists):
    """z of trend and flux under the exact list-flip null; returns arrays (nh, 2)"""
    Z = np.zeros((len(H), 2))
    for i, h in enumerate(H):
        tr = []; fl = []
        for d in lists:
            seq = [c for c in (code_entry(h, e) for e in d['I']) if c is not None]
            a, b = list_stats(seq, h['K']); tr.append(a); fl.append(b)
        tr = np.array(tr); fl = np.array(fl)
        Z[i, 0] = tr.sum() / math.sqrt((tr ** 2).sum()) if (tr ** 2).sum() > 0 else 0
        Z[i, 1] = fl.sum() / math.sqrt((fl ** 2).sum()) if (fl ** 2).sum() > 0 else 0
    return Z


def flip(lists, seed):
    rng = random.Random(seed); out = []
    for d in lists:
        out.append(dict(d, I=d['I'][::-1] if rng.random() < 0.5 else d['I']))
    return out


def run(train, test, tag, H, nsh=NSH):
    Ztr = score(H, train)
    surv = [(i, j, float(np.sign(Ztr[i, j]))) for i in range(len(H)) for j in (0, 1) if abs(Ztr[i, j]) >= ZT]
    Zte = score(H, test)
    def agree(sv):
        if not sv: return float('nan'), float('nan')
        v = np.array([s * Zte[i, j] for i, j, s in sv])
        return float((v > 0).mean()), float(v.mean())
    real = agree(surv)
    sh = []
    for s in range(nsh):
        Zs = score(H, flip(train, 500 + s))
        sv = [(i, j, float(np.sign(Zs[i, j]))) for i in range(len(H)) for j in (0, 1) if abs(Zs[i, j]) >= ZT]
        sh.append((len(sv),) + agree(sv))
    return dict(tag=tag, n_train=len(train), n_test=len(test), n_surv=len(surv), agree=real[0], mean_signed_z=real[1],
                shuffle=[list(x) for x in sh],
                shuffle_nsurv_mean=float(np.mean([x[0] for x in sh])) if sh else None,
                shuffle_agree_mean=float(np.nanmean([x[1] for x in sh])) if sh else None,
                p_agree=float(np.mean([(x[2] if x[2] == x[2] else -9) >= real[1] for x in sh])) if sh else None,
                p_nsurv=float(np.mean([x[0] >= len(surv) for x in sh])) if sh else None), surv, Ztr, Zte


def vals_of(lists):
    V = {f: set() for f in FE}
    for d in lists:
        for e in d['I']:
            for f in FE:
                if e['f'][f] is not None: V[f].add(e['f'][f])
    return {f: sorted(v, key=str) for f, v in V.items()}


def commodity_control(train, test):
    """known positive: commodity order. Codings = random 3-partitions of commodity bases."""
    rng = random.Random(7); cs = sorted({e['f']['com'] for d in train + test for e in d['I'] if e['f']['com']})
    H = []
    for i in range(300):
        m = {c: rng.randrange(3) for c in cs}
        H.append(dict(fs=['com'], maps={'com': m}, K=3, salt=0))
    for d in train + test:
        for e in d['I']: e['f'].setdefault('com', e['logo'])
    return H


def main():
    mode = os.environ.get('MODE', 'la')
    if mode == 'lb':
        L = load_lb(); lex = collections.Counter(e['word'] for d in L if d['site'] == 'KN' for e in d['E'] if e['word'])
        L = interior(featurize(L, lex)); train = [d for d in L if d['site'] == 'KN']; test = [d for d in L if d['site'] == 'PY']
    elif mode == 'lbpy':  # within Pylos, leave-series-out: series A-E vs others
        L = [d for d in load_lb() if d['site'] == 'PY']
        ser = lambda d: (d['id'].split()[1] if len(d['id'].split()) > 1 else 'Z')[:1]
        lex = collections.Counter(e['word'] for d in L if ser(d) <= 'E' for e in d['E'] if e['word'])
        L = interior(featurize(L, lex)); train = [d for d in L if ser(d) <= 'E']; test = [d for d in L if ser(d) > 'E']
    else:
        L = load_la()
        lex = collections.Counter(e['word'] for d in L if d['year'] <= CUT for e in d['E'] if e['word'])
        L = interior(featurize(L, lex))
        train = [d for d in L if d['year'] <= CUT]; test = [d for d in L if d['year'] > CUT]
    PL = os.environ.get('PLANT')
    if PL:  # planted drift: in a share of lists, entries with word length >= 3 are moved later (stable sort on a noisy key)
        share = float(PL); rng = random.Random(9)
        for d in train + test:
            if rng.random() < share:
                key = [(i / len(d['I']) + (0.6 if (e['word'] and len(e['word']) >= 3) else 0) + rng.random() * 0.5, i)
                       for i, e in enumerate(d['I'])]
                d['I'] = [d['I'][i] for _, i in sorted(key)]
    print(mode, 'train lists', len(train), 'test lists', len(test),
          'interior entries', sum(len(d['I']) for d in train), sum(len(d['I']) for d in test))
    H = random_codings(NH, vals_of(train + test), 81)
    res, surv, Ztr, Zte = run(train, test, mode, H)
    tag = mode + (f'_plant{PL}' if PL else '') + f'_{CUT}'
    if mode == 'la' and not PL:
        frozen = dict(cut=CUT, seed=81, nh=NH, zt=ZT, survivors=[dict(i=i, stat=j, sign=s, fs=H[i]['fs'], K=H[i]['K'],
                      maps={f: {str(k): v for k, v in H[i]['maps'][f].items()} for f in H[i]['fs']}, z_train=float(Ztr[i, j]))
                      for i, j, s in surv], train_ids=[d['id'] for d in train])
        h = sha(frozen)
        json.dump(frozen, open(os.path.join(DATA, f'la81_frozen_c2_{CUT}.json'), 'w'))
        open(os.path.join(DATA, f'la81_frozen_c2_{CUT}.sha256'), 'w').write(h + '\n')
        res['sha256'] = h
    # describe top survivors
    top = sorted(surv, key=lambda x: -abs(Ztr[x[0], x[1]]))[:15]
    res['top'] = [dict(fs=H[i]['fs'], K=H[i]['K'], stat=['trend', 'flux'][j], ztr=round(float(Ztr[i, j]), 2),
                       zte=round(float(Zte[i, j]), 2),
                       state0={f: [str(k) for k, v in H[i]['maps'][f].items() if v == 0][:12] for f in H[i]['fs']})
                  for i, j, s in top]
    # feature-level summary: survivors by feature
    res['surv_by_feature'] = dict(collections.Counter(f for i, j, s in surv for f in H[i]['fs']))
    # commodity positive control
    Hc = commodity_control(train, test)
    Zc_tr = score(Hc, train); Zc_te = score(Hc, test)
    res['commodity'] = dict(max_abs_z_train_flux=float(np.abs(Zc_tr[:, 1]).max()),
                            n_surv_flux=int((np.abs(Zc_tr[:, 1]) >= ZT).sum()),
                            agree_flux=float(np.mean([np.sign(Zc_tr[k, 1]) * Zc_te[k, 1] > 0 for k in range(len(Hc)) if abs(Zc_tr[k, 1]) >= ZT])) if (np.abs(Zc_tr[:, 1]) >= ZT).any() else None,
                            n_surv_trend=int((np.abs(Zc_tr[:, 0]) >= ZT).sum()))
    print(json.dumps({k: v for k, v in res.items() if k != 'shuffle'}, indent=1, default=str))
    json.dump(res, open(os.path.join(CK, f'c2_{tag}.json'), 'w'), default=str)


if __name__ == '__main__':
    main()
