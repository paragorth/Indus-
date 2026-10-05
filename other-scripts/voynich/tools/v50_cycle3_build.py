"""v50 cycle 3 positive controls: hide a message along arbitrary RANDOM multi-move paths of the cycle-3 random
family (600k paths, seed 50), inside the Markov filler (MK) and inside the real pages (REAL).
  PR1  Latin letters in the first glyph of each visited word (path picked at random among W-unit, sel 0, m>=3
       paths with 1,500-6,000 visited cells over the pages)
  PR2  German code words (German word types -> Voynich word types by rank) as whole words along a sel-5 path
Writes sets <PRk>_<base> and 4 line-shuffled replicates each, plus data/v50_ckpt/c3_plants.json."""
import os, sys, json, gzip, random, subprocess, zlib
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v50_build as B
import v50_lib as L


def rand_specs():
    return [l.split() for l in gzip.open(os.path.join(L.CK, 'rand_specs.txt.gz'), 'rt')]


def positions(setname, idx):
    fn = L.write_idx([idx], os.path.join(L.CK, 'pos.idx'))
    p = subprocess.run([L.BIN, os.path.join(L.SETS, setname + '.txt'), 'POS', 'subrand:600000:50', fn, '0'],
                       capture_output=True, text=True, check=True)
    return [tuple(map(int, l.split())) for l in p.stdout.splitlines() if not l.startswith('#')]


def main():
    rng = random.Random(5050); R = rand_specs()
    def pick(sel):
        c = [int(r[0]) for r in R if r[1] == 'unit=0' and r[2] == f'sel={sel}']
        while True:
            i = rng.choice(c); pos = positions('MK', i)
            cells = len(set(pos))
            if 1500 <= len(pos) <= 6000 and cells > 0.9 * len(pos): return i, pos
    out = {}
    base = {'MK': json.load(open(os.path.join(L.SETS, 'MK.json'))), 'REAL': json.load(open(os.path.join(L.SETS, 'REAL.json')))}
    txt = B.letters('la', 60000, skip=20000)
    M = B.letter_map(txt, 'first'); V = B.vocab_by_slot('first')
    gw = B.words_of('de')[8000:]; gr = [w for w, _ in Counter(gw).most_common()]; vr = [w for w, _ in B.WC.most_common()]
    code = {w: vr[i % len(vr)] for i, w in enumerate(gr)}
    for kind, sel in (('PR1', 0), ('PR2', 5)):
        i, _ = pick(sel)
        out[kind] = {'path': i, 'spec': ' '.join(R[i][1:])}
        for bname, pages in base.items():
            pos = positions(bname, i)
            pg = [(f, [(ps, list(ws)) for ps, ws in p]) for f, p in pages]
            k = 0
            for (p, li, col) in pos:
                ws = pg[p][1][li][1]
                if kind == 'PR1':
                    g = M[txt[k % len(txt)]]; ks, wt = V[g]; ws[col] = rng.choices(ks, weights=wt)[0]
                else:
                    ws[col] = code[gw[k % len(gw)]]
                k += 1
            name = f'{kind}_{bname}'
            B.write_set(name, pg)
            for r in (1, 2, 3, 4): B.write_set(f'{name}_LS{r}', B.line_shuffle(pg, 77 * r + zlib.crc32(name.encode()) % 991))
            out[kind][bname + '_cells'] = len(pos)
            print(name, i, len(pos), flush=True)
    json.dump(out, open(os.path.join(L.CK, 'c3_plants.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
