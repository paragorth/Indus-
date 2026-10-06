#!/usr/bin/env python3
"""PE-65 cycle 3: (a) model adequacy (nearest-neighbour distance of each corpus to the bank vs held-out sims);
(b) SIGN KNOCKOUT read-off of institutional markers: delete every line containing one sign (or word), recompute
the panel, and read how far the forest's P(shared institutions) = 1 - P(N) and P(Susa-centred state) fall.
Signs whose removal pushes the corpus toward 'no institutions' carry the shared-institution signal.
Null: frequency-matched knockouts of signs found at one unit only. Controls: Ur III titles and Linear B office
words at PE shape (AUC of truth titles among knockout effects); planted capital+offices world (title class AUC)."""
import sys, os, json, pickle, collections, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe65_common import *
from pe65_run import load_bank
from pe65_fit import clean, read, auc

UR_TIT = {'ugula', 'nu-banda3', 'szabra', 'sanga', 'ensi2', 'lugal', 'dub-sar', 'sukkal', 'kuruszda', 'kiszib3',
          'giri3', 'maszkim', 'sipa', 'engar', 'muhaldim', 'lu2-kin-gi4-a', 'gurusz', 'geme2', 'e2', 'ki'}
LB_TIT = {'wa-na-ka', 'ra-wa-ke-ta', 'e-qe-ta', 'te-re-ta', 'ko-re-te', 'po-ro-ko-re-te', 'da-mo', 'ka-ke-u',
          'i-je-re-ja', 'do-e-ro', 'do-e-ra', 'qa-si-re-u', 'ke-ro-si-ja', 'e-re-ta', 'mo-ro-pa2', 'ko-re-te-re',
          'o-pa', 'ka-ra-wi-po-ro', 'te-o-jo', 'to-so', 'to-sa'}


def tokens_of(word, corpus):
    return word.split('.') if corpus == 'PE' else word.split(' ')


def knock(docs, tok, corpus):
    out = []
    for d in docs:
        out.append(dict(d, words=[w for w in d['words'] if tok not in tokens_of(w, corpus)]))
    return out


def summary(rf, s):
    r = read(rf, s)
    return 1 - r['type'].get('N', 0), r['susa_state'], r['type']


def run_knock(rf, docs, corpus, minn=5, maxn=400):
    cnt = collections.Counter(); units = collections.defaultdict(set)
    for d in docs:
        for w in d['words']:
            for t in set(tokens_of(w, corpus)):
                cnt[t] += 1; units[t].add(d['site'])
    base_inst, base_susa, base_type = summary(rf, stats(docs))
    toks = [t for t, n in cnt.most_common(maxn) if n >= minn]
    rows = []
    for t in toks:
        i, s, _ = summary(rf, stats(knock(docs, t, corpus)))
        outp = len([u for u in units[t] if u >= NB]); hub = len([u for u in units[t] if u < NB])
        rows.append(dict(tok=t, n=cnt[t], hub=hub, outposts=outp, d_inst=base_inst - i, d_susa=base_susa - s))
    return dict(base=dict(inst=base_inst, susa=base_susa, type=base_type), rows=rows)


def adequacy(S, med, mad, corp, nhold=500):
    Z = ((S - med) / mad).astype(np.float32)
    rng = np.random.RandomState(3); hold = rng.choice(len(S), nhold, replace=False)
    mask = np.ones(len(S), bool); mask[hold] = False
    Zb = Z[mask][:150000]
    def nn(z):
        return float(np.sqrt(((Zb - z) ** 2).sum(1)).min())
    ref = np.array([nn(Z[h]) for h in hold])
    out = {}
    for name, s in corp.items():
        z = ((clean(s[None, :])[0] - med) / mad).astype(np.float32)
        d = nn(z)
        out[name] = dict(nn=round(d, 2), ref_median=round(float(np.median(ref)), 2), pct=round(float((ref < d).mean()), 3))
        print('adequacy', name, out[name], flush=True)
    return out


def main():
    rf = pickle.load(open(os.path.join('/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad', 'pe65_rf_c1.pkl'), 'rb'))
    S, T, meta = load_bank('PE'); S = clean(S)
    med = np.median(S, 0); mad = np.median(np.abs(S - med), 0) * 1.4826
    mad[mad < 1e-3] = np.maximum(S.std(0)[mad < 1e-3], 1e-3)
    pe = pe_docs_all('pub')
    corp = {'PE_pub': stats(pe), 'PE_rand': stats(pe_docs_all('rand')), 'PE_shuf0': stats(shuffle_units(pe, 'c3')),
            'UR0': stats(control_docs('UR', 0)), 'LB0': stats(control_docs('LB', 0))}
    res = dict(adequacy=adequacy(S, med, mad, corp))
    del S
    # knockouts
    res['PE'] = run_knock(rf, pe, 'PE')
    print('PE knockouts', len(res['PE']['rows']), res['PE']['base'], flush=True)
    for nm, tit in (('UR', UR_TIT), ('LB', LB_TIT)):
        docs = control_docs(nm, 0)
        r = run_knock(rf, docs, nm)
        y = [row['tok'] in tit for row in r['rows']]
        r['title_auc_inst'] = auc(np.array([row['d_inst'] for row in r['rows']]), y)
        r['title_auc_susa'] = auc(np.array([row['d_susa'] for row in r['rows']]), y)
        r['n_titles'] = int(sum(y))
        res[nm] = r
        print(nm, 'titles', r['n_titles'], 'AUC inst', r['title_auc_inst'], 'AUC centre', r['title_auc_susa'], flush=True)
    # planted capital + satellite offices at Susa: word classes known
    from pe65_c1 import plant
    pa = []
    for rep in range(3):
        _, th = plant('susa_capsat', 200 + rep)
        docs, _ = sim_docs(seed('pe65-c3pl-%d' % rep), force_theta=th)
        cls = {}
        for d in docs:
            for w, c in zip(d['words'], d['wclass']):
                cls[w] = c
        # words are atomic in sims: knock whole words
        docs2 = [dict(d, words=list(d['words'])) for d in docs]
        r = run_knock(rf, docs2, 'SIM', maxn=250)
        y = [cls.get(row['tok']) in (2, 4) for row in r['rows']]
        pa.append(dict(auc_inst=auc(np.array([row['d_inst'] for row in r['rows']]), y), n_inst=int(sum(y)),
                       base=r['base']))
        print('planted', pa[-1], flush=True)
    res['planted'] = pa
    jdump(res, 'c3_results.json')


if __name__ == '__main__':
    main()
