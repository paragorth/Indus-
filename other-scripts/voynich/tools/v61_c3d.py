"""v61 cycle 3d: Currier A vs B, and size-matched Sanskrit: scan (cycle-1 code) then rule systems (cycle-2b code)."""
import json, sys
from multiprocessing import Pool
import v61_c1, v61_c2b

def job(arg):
    s1 = v61_c1.job(arg)
    s2 = v61_c2b.run(arg, n_random=2000)
    return {'name': arg[0], 'dir': arg[1], 'greedy_test': s2['greedy_test'], 'gold': s2.get('gold_test'),
            'random_median_train_dA_rel': s2['random_median_train_dA_rel'], 'A0': s2['base_test']['A'], 'n_greedy': len(s2['greedy_rules'])}

if __name__ == '__main__':
    jobs = [('VMS-ZL-A', 'R'), ('VMS-IT-A', 'R'), ('Sanskrit11k', 'R'), ('VMS-ZL-B', 'R'), ('VMS-ZL-A', 'P'), ('null-pairblind:VMS-ZL-A', 'R')]
    with Pool(2) as p:
        for o in p.imap_unordered(job, jobs):
            print(json.dumps(o), flush=True)
