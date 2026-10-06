"""pe68 cycle 2: thousands of random completion models (trained without the joined tablets) predict what each
fragment lost; predictions are frozen by hash, then compared with the joined text. Nulls: corpus frequencies,
random same-size tablets, the reading with shuffled sign roles, a cache-only model. Same procedure on
geometry-matched pseudo-fragments of held-out PE, proto-cuneiform and Ur III tablets.

usage: python3 pe68_c2.py predict   (writes and hashes the frozen predictions; never reads hidden lines)
       python3 pe68_c2.py score     (scores everything)
"""
import os, sys, json, math, random, hashlib, time
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe68_lib import *  # noqa

NMOD = 3000
NSHUF = 49
MAXC = 700


def split(T, tag='pe68'):
    tr, ho = [], []
    for t in T:
        (tr if int(hashlib.sha256((tag + t['id']).encode()).hexdigest(), 16) % 2 == 0 else ho).append(t)
    return tr, ho


def real_cases():
    T = {t['id']: t for t in corpus('PE')}
    J = json.load(open(os.path.join(CK, 'c1_joins.json')))
    out = []
    for pid, F in sorted(J['fragments'].items()):
        t = T[pid]
        n = len(t['lines'])
        for lab, idx in F.items():
            hid = [i for i in range(n) if i not in set(idx)]
            c = case_from(t, idx, hid, 'join')
            c['frag'] = lab
            c['conf'] = J['fragmap'][pid]['conf']
            out.append(c)
    return out


def fracs_real():
    return [len(c['V']) / (len(c['V']) + len(c['H'])) for c in real_cases()]


def setup(name, rng):
    T = corpus(name)
    if name == 'PE':
        T = [t for t in T if t['id'] not in JOINED_IDS]
    tr, ho = split(T)
    rs = role_sets(name)
    st = Stats(tr, rs)
    srng = random.Random(seed('pe68-shuf-' + name))
    rs_list = [st.role]
    for k in range(NSHUF):
        st2 = Stats(tr, shuffled_role_sets(rs, tr, srng))
        rs_list.append(st2.role)
    fr = fracs_real()
    ctr = pseudo_cuts(tr, fr, random.Random(seed('pe68-ctr-' + name)))
    cte = pseudo_cuts(ho, fr, random.Random(seed('pe68-cte-' + name)))
    rng.shuffle(ctr)
    rng.shuffle(cte)
    return T, tr, ho, st, rs_list, ctr[:MAXC], cte[:MAXC]


def predictive(case_V, st, rates, m, nH):
    """the frozen prediction for one fragment from its visible lines only."""
    vk = tablet_kind(case_V)
    vcls = [l['cls'] for l in case_V if l['cls']]
    if m['SYS']:
        q = rates.get(vk, st.base_cap)
        q = m['sysmix'] * q + (1 - m['sysmix']) * (vcls.count('CAP') + m['a'] * st.base_cap) / (len(vcls) + m['a'])
    else:
        q = (vcls.count('CAP') + m['a'] * st.base_cap) / (len(vcls) + m['a'])
    vf = Counter(l['fin'] for l in case_V if l['fin'])
    nv = sum(vf.values())
    lc, lr, lu = m['lam']
    if not nv:
        lu += lc; lc = 0
    if not m['ROLE']:
        lu += lr; lr = 0
    cand = set(vf) | set(st.role[vk]) | set(s for s, _ in st.fins.most_common(40))
    P = {s: lc * (vf.get(s, 0) / nv if nv else 0) + lr * st.role[vk].get(s, 0) + lu * st.uni(s) for s in cand}
    top = sorted(P.items(), key=lambda kv: -kv[1])[:8]
    vx = [math.log10(l['v']) for l in case_V if l['v'] and l['v'] > 0]
    if m['SYS']:
        qq = rates.get(vk, st.base_cap)
        mu0 = qq * st.mu['CAP'] + (1 - qq) * st.mu['CNT']
    else:
        mu0 = st.base_cap * st.mu['CAP'] + (1 - st.base_cap) * st.mu['CNT']
    mu = mu0 if not vx else m['w'] * float(np.mean(vx)) + (1 - m['w']) * mu0
    sd = max(st.sd['CAP'], st.sd['CNT']) * m['sds']
    hd = Counter()
    for h in st.hdr:
        hd[h] = st.hdr_p(h, vk if m['HDR'] else None)
    # total gap: the reading's rule (a single reverse numeric line is the sum of the entries)
    tot = [l for l in case_V if l['role'] == 'T' and l['v']]
    ents = [l for l in case_V if l['role'] == 'E']
    gap = None
    if tot and all(l['v'] for l in ents):
        gap = tot[0]['v'] - sum(l['v'] for l in ents)
    return {'visible_kind': vk, 'p_cap_per_hidden_line': round(q, 4),
            'p_any_cap_hidden': round(1 - (1 - q) ** max(nH, 0), 4) if nH else None,
            'top_class_signs': [(s, round(p, 4)) for s, p in top],
            'log10_value_mean': round(mu, 3), 'log10_value_sd': round(sd, 3),
            'header_top3': [(h, round(p, 4)) for h, p in hd.most_common(3)],
            'total_gap_value': gap}


def main_predict():
    rng = np.random.default_rng(seed('pe68-models'))
    pyr = random.Random(seed('pe68-setup'))
    T, tr, ho, st, rs_list, ctr, cte = setup('PE', pyr)
    pc_tr = [precompute(c, st, rs_list[:1]) for c in ctr]
    rates = sys_rates(pc_tr)
    models = [random_model(rng) for _ in range(NMOD)]
    # select READ (all reading flags on) and CACHE (all off) on training pseudo-cuts only
    sc = np.array([np.mean([score_model(p, m, st, rates)['total'] for p in pc_tr]) for m in models])
    full = [i for i, m in enumerate(models) if m['SYS'] and m['ROLE'] and m['HDR']]
    none = [i for i, m in enumerate(models) if not (m['SYS'] or m['ROLE'] or m['HDR'])]
    iR = min(full, key=lambda i: sc[i])
    iC = min(none, key=lambda i: sc[i])
    cases = real_cases()
    preds = []
    for c in cases:
        nH = sum(1 for l in c['H'] if l['cls'])   # size of the hidden part (geometry), not its content
        preds.append({'tablet': c['id'], 'fragment': c['frag'], 'conf': c['conf'], 'n_visible_lines': len(c['V']),
                      'n_hidden_numeric_lines': nH,
                      'READ': predictive(c['V'], st, rates, models[iR], nH),
                      'CACHE': predictive(c['V'], st, rates, models[iC], nH)})
    frozen = {'models': {'READ': models[iR], 'CACHE': models[iC]}, 'rates': rates,
              'train_bits': {'READ': float(sc[iR]), 'CACHE': float(sc[iC])}, 'predictions': preds,
              'fragmap_sha': json.load(open(os.path.join(CK, 'c1_joins.json')))['sha_fragmap'],
              'made': time.strftime('%Y-%m-%d %H:%M')}
    h = sha(frozen)
    frozen['sha256'] = h
    json.dump(frozen, open(os.path.join(DATA, 'pe68_frozen_join_predictions.json'), 'w'), indent=1)
    json.dump({'models': models, 'train_scores': sc.tolist()}, open(os.path.join(CK, 'c2_models_PE.json'), 'w'))
    print('frozen', h[:16], 'READ train bits %.2f CACHE %.2f' % (sc[iR], sc[iC]))


def eval_models(P, models, st, rates, ridx=0):
    return np.array([[score_model(p, m, st, rates, ridx)['total'] for p in P] for m in models])


def feat_means(P, m, st, rates, ridx=0):
    acc = defaultdict(list)
    for p in P:
        s = score_model(p, m, st, rates, ridx)
        for k, v in s.items():
            acc[k].append(v)
    return {k: (float(np.sum(v)), len(v)) for k, v in acc.items()}


def randtab_null(cases, pool, st, rates, m, rs_list, rng, draws=20):
    """replace the visible fragment by a random training tablet's lines of similar size."""
    bys = defaultdict(list)
    for t in pool:
        bys[len(t['lines'])].append(t)
    sizes = sorted(bys)
    out = []
    for c in cases:
        n = len(c['V'])
        near = [t for s in sizes if abs(s - n) <= max(2, n // 5) for t in bys[s]] or pool
        tot = []
        for _ in range(draws):
            t = rng.choice(near)
            k = min(len(t['lines']), n)
            s0 = rng.randrange(0, len(t['lines']) - k + 1)
            fake = {'V': t['lines'][s0:s0 + k], 'H': c['H']}
            tot.append(score_model(precompute(fake, st, rs_list[:1]), m, st, rates)['total'])
        out.append(float(np.mean(tot)))
    return np.array(out)


def signflip(d, rng, n=20000):
    d = np.asarray(d)
    obs = d.mean()
    s = rng.choice([-1, 1], size=(n, len(d)))
    return float(obs), float((np.abs((s * d).mean(1)) >= abs(obs)).mean())


def main_score():
    t0 = time.time()
    F = json.load(open(os.path.join(DATA, 'pe68_frozen_join_predictions.json')))
    h = F.pop('sha256')
    assert sha(F) == h, 'frozen file changed'
    F['sha256'] = h
    res = {'sha': h}
    rng = np.random.default_rng(seed('pe68-score'))
    for name in ('PE', 'PC', 'U3'):
        pyr = random.Random(seed('pe68-setup') if name == 'PE' else seed('pe68-setup-' + name))
        T, tr, ho, st, rs_list, ctr, cte = setup(name, pyr)
        pc_tr = [precompute(c, st, rs_list) for c in ctr]
        pc_te = [precompute(c, st, rs_list) for c in cte]
        rates = sys_rates(pc_tr)
        if name == 'PE':
            assert all(abs(rates[k] - F['rates'][k]) < 1e-9 for k in F['rates']), 'rates drifted'
            M = json.load(open(os.path.join(CK, 'c2_models_PE.json')))['models']
            mR, mC = F['models']['READ'], F['models']['CACHE']
        else:
            mrng = np.random.default_rng(seed('pe68-models'))
            M = [random_model(mrng) for _ in range(NMOD)]
            sc = np.array([np.mean([score_model(p, m, st, rates)['total'] for p in pc_tr]) for m in M])
            full = [i for i, m in enumerate(M) if m['SYS'] and m['ROLE'] and m['HDR']]
            none = [i for i, m in enumerate(M) if not (m['SYS'] or m['ROLE'] or m['HDR'])]
            mR = M[min(full, key=lambda i: sc[i])]
            mC = M[min(none, key=lambda i: sc[i])]
        sets = {'test_pseudo': (cte, pc_te)}
        if name == 'PE':
            rc = real_cases()
            sets['real_joins'] = (rc, [precompute(c, st, rs_list) for c in rc])
            sets['real_joins_M'] = ([c for c in rc if c['conf'] == 'M'],
                                    [precompute(c, st, rs_list) for c in rc if c['conf'] == 'M'])
        R = {'n_train_cases': len(pc_tr), 'rates': rates}
        for sname, (cases, P) in sets.items():
            if not P:
                continue
            r = {'n': len(P)}
            read = np.array([score_model(p, mR, st, rates)['total'] for p in P])
            cache = np.array([score_model(p, mC, st, rates)['total'] for p in P])
            corp = np.array([score_model(p, CORPUS_NULL, st, rates)['total'] for p in P])
            shuf = np.array([[score_model(p, mR, st, rates, k)['total'] for p in P] for k in range(1, NSHUF + 1)])
            rt = randtab_null(cases, tr, st, rates, mR, rs_list, random.Random(seed('rt' + name + sname)),
                              draws=10 if len(P) > 100 else 30)
            r['bits_per_case'] = {'READ': float(read.mean()), 'CACHE': float(cache.mean()),
                                  'CORPUS': float(corp.mean()), 'RANDTAB': float(rt.mean()),
                                  'SHUF_mean': float(shuf.mean()), 'SHUF_min': float(shuf.mean(1).min())}
            r['p_shuf'] = float(((shuf.mean(1) <= read.mean()).sum() + 1) / (NSHUF + 1))
            for nm, other in (('CACHE', cache), ('CORPUS', corp), ('RANDTAB', rt)):
                d, p = signflip(other - read, rng)
                r['gain_vs_' + nm] = {'bits': d, 'p': p, 'wins': int((other > read).sum())}
            r['features'] = {nm: feat_means(P, m, st, rates) for nm, m in (('READ', mR), ('CACHE', mC),
                                                                          ('CORPUS', CORPUS_NULL))}
            r['features']['SHUF0'] = feat_means(P, mR, st, rates, 1)
            # random model cloud: flag effects
            S = eval_models(P, M, st, rates).mean(1)
            for flag in ('SYS', 'ROLE', 'HDR'):
                on = [S[i] for i, m in enumerate(M) if m[flag]]
                off = [S[i] for i, m in enumerate(M) if not m[flag]]
                r['flag_' + flag] = float(np.mean(off) - np.mean(on))
            r['cloud_best'] = float(S.min())
            r['cloud_read_rank'] = int((S < read.mean()).sum())
            if sname.startswith('real'):
                r['per_case'] = [{'tablet': c['id'], 'frag': c['frag'], 'READ': float(a), 'CACHE': float(b),
                                  'CORPUS': float(cc), 'RANDTAB': float(d)}
                                 for c, a, b, cc, d in zip(cases, read, cache, corp, rt)]
            R[sname] = r
            print(name, sname, json.dumps(r['bits_per_case']), 'p_shuf', r['p_shuf'],
                  {k: (round(v['bits'], 2), round(v['p'], 4)) for k, v in r.items() if k.startswith('gain')},
                  'flags', {k: round(v, 2) for k, v in r.items() if k.startswith('flag')}, '%.0fs' % (time.time() - t0),
                  flush=True)
        res[name] = R
    json.dump(res, open(os.path.join(CK, 'c2_scores.json'), 'w'), indent=1)


if __name__ == '__main__':
    {'predict': main_predict, 'score': main_score}[sys.argv[1]]()
