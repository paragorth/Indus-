"""S-DARK-7 cycle 3: extra corpora -- Linear Elamite segments (language, royal/dedicatory, syllables), Ur III seal-owner
names as syllable strings (names only), Indus seals only vs Indus non-seal objects (tablets etc.). Same statistics."""
import sys, os, re, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark7_fingerprint as F
SP = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/seal_line1.txt'

def load_le():
    le = json.load(open(os.path.join(F.ROOT, 'data/derived/linear-elamite-corpus-hled.json')))
    out = []
    for x in le:
        for seg in (x.get('translit') or '').split('||'):
            toks = [p for p in re.split(r'[-\s]+', seg.strip()) if p and p != 'X' and '…' not in p and '[' not in p]
            if toks:
                out.append(([(t, False) for t in toks], None))
    return out

def load_names():
    out = []
    for l in open(SP, errors='ignore'):
        l = re.sub(r'[#\[\]!?<>]', '', l.strip().lower())
        if not l or ' ' in l or 'x' in l.split('-'):
            continue
        t = [p for p in l.split('-') if p]
        if t:
            out.append(([(p, False) for p in t], None))
    return out

def main():
    B, N = int(sys.argv[1]), int(sys.argv[2])
    ind = F.load_indus('seq_raw'); W = F.target_weights(ind); stats = F.make_stats(F.SEED)
    corp = {'LE_syll': load_le(), 'UR3_names_syll': load_names(),
            'IND_seals': [(t, m) for t, m in ind if str(m.get('type', '')).startswith('SEAL')],
            'IND_nonseal': [(t, m) for t, m in ind if not str(m.get('type', '')).startswith('SEAL')]}
    res = {}; infos = {}
    for k, t in corp.items():
        R, info = F.run_corpus(k, t, stats, W, B, N, 202 + hash(k) % 1000)
        res[k] = R; info['n_used'] = sum(1 for x, _ in t if F.LMIN <= len(x) <= F.LMAX); infos[k] = info
        print(k, info, flush=True)
    np.savez_compressed(os.path.join(F.ROOT, 'data/derived/dark/loop7_boot_c3.npz'), **res)
    json.dump(infos, open(os.path.join(F.ROOT, 'data/derived/dark/loop7_c3_infos.json'), 'w'))
    print('saved')

if __name__ == '__main__':
    main()
