"""pe52 job runner: python3 pe52_run.py <tag> <corpus> <mode> <null> <S> <M> [seed]
corpus: PE | PEmulti | HERD | PERS | PLANT0 | PLANT1 ...   null: none | qsys | qtab | swithin | stab
Writes data/pe52_ckpt/<tag>_<corpus>_<mode>_<null>.json (per-sign table, scores, explained shares)."""
import sys, os, json, time
import pe52_lib as L


def corpus(name):
    if name == 'PE':
        return L.pe_corpus(), None
    if name == 'PEmulti':
        return L.pe_corpus(minlen=2), None
    if name == 'HERD':
        return L.herd_corpus(), None
    if name == 'HERD2':
        return L.herd2_corpus(), None
    if name == 'PERS':
        return L.pers_corpus(), None
    if name.startswith('PLANT'):
        C, eff = L.planted(L.pe_corpus(), seed=int(name[5:]))
        return C, eff
    raise ValueError(name)


NULLS = dict(none=lambda C, s: C, qsys=L.null_qshuf_sys, qtab=L.null_qshuf_tab, swithin=L.null_sign_within,
             stab=L.null_sign_tab)


def job(tag, cname, mode, null, S, M, seed=0):
    out = os.path.join(L.CK, f'{tag}_{cname}_{mode}_{null}.json')
    if os.path.exists(out):
        return out
    t0 = time.time()
    C, eff = corpus(cname)
    C = NULLS[null](C, 1000 + seed)
    D, R = L.run_corpus(C, mode, S, M, seed0=seed * 100)
    tab = L.summarise(D, R)
    lab = {}
    if cname in ('PERS', 'HERD2'):
        from collections import defaultdict
        acc = defaultdict(list)
        for r in C:
            for w, l in zip(r['w'], r['lab']):
                acc[w].append(l)
        lab = {w: sum(v) / len(v) for w, v in acc.items()}
    res = dict(tag=tag, corpus=cname, mode=mode, null=null, S=S, M=M, n=D.n, scores=L.scores(R),
               explained=L.explained(D, tab), table=tab, planted=eff, namefrac=lab,
               gainsB_pos=[sum(g > 0 for g in r['gainsB']) / len(r['gainsB']) for r in R],
               surv_gC=[[sv['gC'] for sv in r['surv']] for r in R], secs=time.time() - t0)
    json.dump(res, open(out, 'w'))
    return out


if __name__ == '__main__':
    a = sys.argv[1:]
    print(job(a[0], a[1], a[2], a[3], int(a[4]), int(a[5]), int(a[6]) if len(a) > 6 else 0))
