"""Re-evaluate planted partition runs (families a, g) with the consensus-AUC recovery criterion.
The hidden partition is regenerated from the job seed exactly as in r1_run.fam_a."""
import sys, glob, os, json
import numpy as np
import r1_lib as L, r1_seq as S
from r1_run import consensus_auc

for f in sorted(glob.glob(os.path.join(L.RES, f'c{sys.argv[1]}_[ag]_*_planted*.json'))):
    d = json.load(open(f))
    if 'consensus_auc' in d:
        continue
    j = d['job']
    rng = np.random.default_rng(1000 + j['seed'])
    c = L.load(j['script'])
    _, hidden = S.planted_partition_corpus(c, rng)
    d['recovered_top_nmi'] = d.get('recovered')
    d['consensus_auc'] = consensus_auc(d, hidden, rng)
    d['recovered'] = bool(d['consensus_auc'] and d['consensus_auc']['auc'] > 0.6 and d['consensus_auc']['p'] < 0.01)
    L.save(os.path.basename(f), d)
    print(os.path.basename(f), d['consensus_auc'], d['recovered'])
