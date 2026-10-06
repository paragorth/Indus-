#!/usr/bin/env python3
"""pe74 cycle 2, item I: the accounting skeleton (pe59 frozen reading as a whole-tablet generator) on a filtered
corpus, with pe59's own code: fit on the training half, held-out numeral and total bits (G vs N2), closing totals
vs re-dealt totals, the M288 60k rule vs other lines in the same position, and shuffled-role / random-unit twins.
usage: python3 pe74_c2_pe59.py MODE [n_shuffle] [n_metro]  -> data/pe74_ckpt/<MODE>/c2_pe59.json
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe74_harness as H
MODE = sys.argv[1]
NSH = int(sys.argv[2]) if len(sys.argv) > 2 else 8
NME = int(sys.argv[3]) if len(sys.argv) > 3 else 6
ROOT = H.activate(MODE)
import pe59_lib as P
H.redirect_ck(P, 'pe59', copy=['pc_tabs.json', 'pc_admin_ids.json'])
sys.argv = [sys.argv[0], 'PE']   # pe59_cycle1 reads the corpus name from argv
import pe59_cycle1 as M
M.CK = P.CK
M.CORPUS = 'PE'
from multiprocessing import Pool
import numpy as np


def after_count(tabs, want_allot, cap, cntm, mults=(60,)):   # verbatim from pe59_cycle3
    k = n = 0
    for t in tabs:
        prev = None
        for l in t['lines']:
            if l['role'] != 'E':
                continue
            if prev is not None and prev['numclean'] and P.ncls(prev['nums']) == 'AMB' and l['numclean'] and l['signs'] \
                    and prev['signs'] and prev['signs'][-1] != l['signs'][-1]:
                is_allot = l['signs'][-1] == 'M288'
                ku = P.value(prev['nums'], cntm)
                if is_allot == want_allot and ku and ku <= 60:
                    v = P.value(l['nums'], cap)
                    if v is not None and P.ncls(l['nums']) in ('CAP', 'AMB'):
                        n += 1
                        k += v in [m * ku for m in mults]
            prev = l
    return k, n


def tot(s, key):
    return sum(s[key]['G']) if key == 'num' else sum(s['tot']['G'])


if __name__ == '__main__':
    T, roles, maps = M.corpus()
    tr, ho = P.split(T)
    cap, cnt = P.pe_maps()
    rate60 = {'M288': after_count(ho, True, cap, cnt['sex2']), 'other_lines': after_count(ho, False, cap, cnt['sex2'])}
    g = M.fit(tr, roles, maps)
    s, raw = M.heldout(g, ho)
    out = {'mode': MODE, 'n_train': len(tr), 'n_heldout': len(ho), 'n_num': s['n_num'], 'n_tot': s['n_tot']}
    out['num_bits'] = {k: sum(v) for k, v in s['num'].items()}
    out['tot_bits'] = {k: sum(v) for k, v in s['tot'].items()}
    out['closure'] = M.closure(g, ho, raw[7], random.Random(P.seed('pe59clos')))
    out['rate60'] = rate60
    print(MODE, json.dumps({k: out[k] for k in ('n_num', 'n_tot', 'num_bits', 'tot_bits', 'rate60')}), flush=True)
    print(MODE, 'closure', out['closure'], flush=True)
    with Pool(2) as pool:
        sh = pool.map(M.shuffled_job, [(k,) for k in range(NSH)])
        me = pool.map(M.metro_job, [(k,) for k in range(NME)])
    out['shuffled_num_G'] = [sum(x['num']['G']) for x in sh]
    out['shuffled_tot_G'] = [sum(x['tot']['G']) for x in sh]
    out['metro_num_G'] = [sum(x['num']['G']) for x in me]
    out['metro_tot_G'] = [sum(x['tot']['G']) for x in me]
    out['shuffled_closure'] = [x['closure']['read'] for x in sh]
    real = out['num_bits']['G']
    out['twins_beaten'] = {'shuffled_num': sum(x > real for x in out['shuffled_num_G']), 'n_sh': NSH,
                           'metro_num': sum(x > real for x in out['metro_num_G']), 'n_me': NME,
                           'metro_tot': sum(x > out['tot_bits']['G'] for x in out['metro_tot_G'])}
    print(MODE, 'twins', out['twins_beaten'], 'sh mean %.4f metro mean %.4f' % (np.mean(out['shuffled_num_G']), np.mean(out['metro_num_G'])), flush=True)
    json.dump(out, open(os.path.join(ROOT, 'c2_pe59.json'), 'w'), indent=1, default=str)
