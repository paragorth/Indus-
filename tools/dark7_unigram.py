"""S-DARK-7 cycle 2: unigram-resampled null corpora (tokens drawn iid from each corpus's unigram distribution, same
text lengths) for the main corpora; same statistics/bootstraps as dark7_fingerprint.py. Output appended to loop7_boot_uni.npz."""
import sys, os, json, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark7_fingerprint as F

def unigram(texts, seed):
    r = random.Random(seed)
    cnt = collections.Counter(t for toks, _ in texts for t in toks)
    ks, ws = zip(*cnt.items())
    return [([r.choices(ks, ws)[0] for _ in toks], m) for toks, m in texts]

def main():
    B, N = int(sys.argv[1]), int(sys.argv[2])
    ind = F.load_indus('seq_raw'); W = F.target_weights(ind); stats = F.make_stats(F.SEED)
    corp = {'IND_raw': ind, 'PE': F.load_pe(), 'LA': F.load_la(), 'LB_words': F.load_lb(False), 'LB_syll': F.load_lb(True),
            'UR3_words': F.load_ur3(False), 'UR3_syll': F.load_ur3(True), 'VOY_words': F.load_voy(False)}
    res = {}
    for k, t in corp.items():
        R, info = F.run_corpus(k, unigram(t, 11), stats, W, B, N, 101 + hash(k) % 1000)
        res[k + '_UNI'] = R
        print(k, 'done', flush=True)
    np.savez_compressed(os.path.join(F.ROOT, 'data/derived/dark/loop7_boot_uni.npz'), **res)
    print('saved')

if __name__ == '__main__':
    main()
