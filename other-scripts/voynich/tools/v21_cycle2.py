"""v21 cycle 2: the neural forger (F8) and learned discriminators.

1. F8: unit-level GRU LM conditioned on section, line type and the previous line, sampled to the real
   skeleton. Scored with the cycle-1 feature battery (ridge, boosting, 300 random 6-feature subsets).
2. Learned discriminators: 3-width CNN over unit sequences of single LINES and of whole PARAGRAPHS,
   5-fold CV by page, 2 seeds, against F3, F7 (best hand-built forger of cycle 1) and F8.
   Controls: null (F7 vs F7), negative (F7 output re-forged by refitted F7), positive (Latin, Italian
   herbal vs F7), planted (long-range repeat planted in F7 output vs re-forged).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v21_cycle1 import corpora, make_forger, LAB, disc_battery, plant_longrange
from v21_neural import *
from multiprocessing import Pool

EPOCHS = int(os.environ.get('EPOCHS', 18))
OUT = os.path.join(LOOPS, 'v21_cycle2.txt')


def get_neural(cname, C):
    path = os.path.join(CK, f'c2_lm_{cname}.pt')
    if os.path.exists(path):
        nf = NeuralForger.__new__(NeuralForger); nf.name = 'F8'
        nf.voc = Vocab(C); nf.m = LM(len(nf.voc.itos)); nf.m.load_state_dict(torch.load(path)); nf.m.eval()
        return nf
    nf = NeuralForger(C, epochs=EPOCHS, log=lambda s: print(cname, s, flush=True))
    torch.save(nf.m.state_dict(), path)
    return nf


def forged_corpus(tag, cname, fname, seed, par):
    """Returns (real, forged) per control tag; cached as json."""
    ck = f'c2_corp_{tag}_{cname}_{fname}_{seed}.json'
    got = load(ck)
    if got: return got['real'], got['forged']
    Cs = corpora(); C = Cs[cname]; rng = random.Random(seed)
    if fname == 'F8':
        nf = get_neural(cname, C)
        real, forged = C, nf.forge(C, rng)
    elif tag == 'main':
        real, forged = C, make_forger(fname, C, par).forge(C, rng)
    elif tag == 'null':
        fg = make_forger(fname, C, par); real, forged = fg.forge(C, random.Random(10_000 + seed)), fg.forge(C, rng)
    elif tag == 'neg':
        S = make_forger(fname, C, par).forge(C, random.Random(20_000 + seed))
        real, forged = S, make_forger(fname, S, par).forge(S, rng)
    elif tag == 'plant':
        S = make_forger(fname, C, par).forge(C, random.Random(30_000 + seed))
        P = plant_longrange(S, random.Random(40_000 + seed))
        real, forged = P, make_forger(fname, P, par).forge(P, rng)
    save(ck, {'real': real, 'forged': forged})
    return real, forged


def job_cnn(args):
    tag, cname, fname, seed, par = args
    ck = f'c2_cnn_{tag}_{cname}_{fname}_{seed}.json'
    got = load(ck)
    if got: return got
    real, forged = forged_corpus(tag, cname, fname, seed, par)
    lu, lp = cnn_auc(real, forged, 'line', seed=seed, nfold=3, epochs=3)
    pu, pp = cnn_auc(real, forged, 'para', seed=seed, nfold=3, epochs=6)
    res = dict(tag=tag, corpus=cname, forger=fname, seed=seed, line_auc=lu, line_page_auc=lp, para_auc=pu, para_page_auc=pp)
    save(ck, res)
    return res


def job_feat_f8(seed):
    ck = f'c2_feat_main_V_F8_{seed}.json'
    got = load(ck)
    if got: return got
    real, forged = forged_corpus('main', 'V', 'F8', seed, None)
    FZ = Featurizer(real, LAB)
    Xr, keys = FZ.matrix(real); Xf, _ = FZ.matrix(forged, keys)
    res = disc_battery(Xr, Xf, seed, 300, keys)
    res.update(tag='main', corpus='V', forger='F8', seed=seed, keys=keys)
    save(ck, res)
    return res


def main():
    t0 = time.time()
    Cs = corpora()
    pars = {cn: load(f'c1_cal_{cn}.json') for cn in Cs}
    get_neural('V', Cs['V'])
    print('lm ready', time.time() - t0, flush=True)
    with Pool(2) as pool:
        for r in pool.imap_unordered(job_feat_f8, range(3)):
            print(f"F8 features s{r['seed']} ridge {r['lr']:.3f} gbm {r['gbm']:.3f} subMax {max(a for a, _ in r['sub']):.3f}", flush=True)
        jobs = [('main', 'V', 'F7', 0, pars['V']), ('null', 'V', 'F7', 0, pars['V']), ('main', 'V', 'F3', 0, pars['V']),
                ('main', 'V', 'F8', 0, pars['V']), ('neg', 'V', 'F7', 0, pars['V']), ('plant', 'V', 'F7', 0, pars['V']),
                ('main', 'LA', 'F7', 0, pars['LA']), ('main', 'IT', 'F7', 0, pars['IT']),
                ('main', 'V', 'F7', 1, pars['V']), ('null', 'V', 'F7', 1, pars['V'])]
        for r in pool.imap_unordered(job_cnn, jobs):
            print(f"CNN {r['tag']:5s} {r['corpus']:2s} {r['forger']} s{r['seed']} line {r['line_auc']:.3f} (page {r['line_page_auc']:.3f}) "
                  f"para {r['para_auc']:.3f} (page {r['para_page_auc']:.3f}) [{time.time() - t0:.0f}s]", flush=True)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
