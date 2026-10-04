"""R1 job runner: python3 r1_run.py <cycle> [families comma list] [N_real] [N_ctrl] [n_ctrl_seeds]
Runs every (family, script, mode, seed) job with at most 2 worker processes; each job writes its own
JSON under voynich/data/results/r1/ and is skipped if that file exists (checkpointing)."""
import sys, os, time, json, traceback
import numpy as np
from multiprocessing import Pool
import r1_lib as L

SCRIPTS = ['voynich', 'linear_a', 'proto_elamite']
LOG = os.path.join(L.RES, 'r1_run.log')


def log(msg):
    with open(LOG, 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')


def job_name(j):
    return f"c{j['cycle']}_{j['fam']}_{j['script']}_{j['mode']}{j['seed']}.json"


def run_job(j):
    name = job_name(j)
    if L.done(name):
        return name, 'skip'
    t0 = time.time()
    try:
        out = FAMS[j['fam']](j)
        out.update({'job': j, 'secs': time.time() - t0})
        L.save(name, out)
        log(f'done {name} {time.time()-t0:.0f}s stage1={out.get("n_stage1")} rep={out.get("n_replicated")} '
            f'recovered={out.get("recovered")}')
        return name, 'ok'
    except Exception:
        log(f'FAIL {name}\n' + traceback.format_exc())
        return name, 'fail'


# ---------------- family wrappers ----------------
def fam_a(j):
    import r1_seq as S
    from sklearn.metrics import normalized_mutual_info_score as nmi
    rng = np.random.default_rng(1000 + j['seed'])
    c = L.load(j['script'])
    hidden = None
    if j['mode'] == 'shuffle':
        c = L.global_shuffle(c, __import__('random').Random(j['seed']))
    elif j['mode'] == 'planted':
        c, hidden = S.planted_partition_corpus(c, rng)
    if j['fam'] == 'g':
        out = S.run_partitions_climb(c, j['N'], seed=j['seed'] + 19, log=log)
    else:
        out = S.run_partitions(c, j['N'], seed=j['seed'] + 17, log=log)
    if hidden is not None:
        signs = out['signs']
        h = [hidden.get(s, 0) for s in signs]
        rec = []
        for r in out['stage2']:
            if not r.get('replicated'):
                continue
            v = nmi(h, r['assign'])
            base = [nmi(h, rng.integers(0, r['k'], len(signs))) for _ in range(200)]
            rec.append({'k': r['k'], 'z2': r['z2'], 'nmi': v, 'nmi_null99': float(np.quantile(base, .99))})
        best = max(rec, key=lambda x: x['z2']) if rec else None
        out['planted_eval'] = rec[:50]
        out['recovered'] = bool(best and best['nmi'] > best['nmi_null99'])
    if j['fam'] == 'g':
        out['stage2'] = out['stage2'][:40]
    for r in out['stage2']:
        if hidden is None and not r.get('replicated'):
            r.pop('assign', None)
    if hidden is None:
        out['consensus'] = consensus(out)
    return out


def consensus(out):
    """Which sign pairs do replicated partitions put together more than chance (1/k)?"""
    reps = [r for r in out['stage2'] if r.get('replicated')]
    if not reps:
        return []
    signs = out['signs'][:40]
    n = len(signs)
    M = np.zeros((n, n))
    for r in reps:
        a = np.array(r['assign'][:n])
        M += (a[:, None] == a[None, :]) - 1.0 / r['k']
    M /= len(reps)
    pairs = [(float(M[i, j]), signs[i], signs[j]) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(reverse=True)
    return pairs[:15] + pairs[-5:]


def fam_c(j):
    import r1_seq as S
    rng = np.random.default_rng(2000 + j['seed'])
    c = L.load(j['script'])
    op = None
    if j['mode'] == 'shuffle':
        c = L.global_shuffle(c, __import__('random').Random(j['seed']))
    elif j['mode'] == 'planted':
        c, op = S.planted_transform_corpus(c, rng)
    out = S.run_transforms_fast(c, j["N"], seed=j["seed"] + 31, log=log)
    if op is not None:
        out['planted_op'] = list(op)
        for r in out['stage2'][:30]:
            r['restoration'] = S.restoration(c, r['ops'])
        reps = [r for r in out['stage2'][:30] if r['replicated']]
        best = max(reps, key=lambda r: r['z2']) if reps else None
        out['identity_restoration'] = S.restoration(c, [])
        out['recovered'] = bool(best and best['restoration'] >= 0.9)
    out['stage2'] = out['stage2'][:60]
    return out


def fam_other(j):
    import r1_tab as T
    return T.run(j, log)


FAMS = {'a': fam_a, 'g': fam_a, 'c': fam_c, 'b': fam_other, 'd': fam_other, 'e': fam_other, 'm': fam_other}


def main():
    cycle = int(sys.argv[1])
    fams = sys.argv[2].split(',')
    n_real = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    n_ctrl = int(sys.argv[4]) if len(sys.argv) > 4 else n_real
    k = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    scripts = sys.argv[6].split(',') if len(sys.argv) > 6 else SCRIPTS
    jobs = []
    for f in fams:
        for s in scripts:
            jobs.append(dict(cycle=cycle, fam=f, script=s, mode='real', seed=0, N=n_real))
            for sd in range(1, k + 1):
                jobs.append(dict(cycle=cycle, fam=f, script=s, mode='shuffle', seed=sd, N=n_ctrl))
                jobs.append(dict(cycle=cycle, fam=f, script=s, mode='planted', seed=sd, N=n_ctrl))
    log(f'cycle {cycle} start: {len(jobs)} jobs')
    with Pool(2) as p:
        for name, st in p.imap_unordered(run_job, jobs):
            print(name, st, flush=True)
    log(f'cycle {cycle} end')


if __name__ == '__main__':
    main()
