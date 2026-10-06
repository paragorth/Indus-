"""v77 report helper: load checkpoint JSONs and print compact tables (used to write the cycle rows)."""
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v77_lib as L


def load(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(L.CK, pattern))):
        out[os.path.basename(f)] = json.load(open(f))
    return out


def c1():
    D = load('c1_*.json')
    for k, d in D.items():
        if k == 'c1_all.json': continue
        a = d['k12']; k4 = d['k4']; k6 = d['k6']
        s = '%-22s id_clean raw %+.4f E1c %+.4f | page E1c %.3f | dir %.3f body %.3f junc %.3f (beyond junc %.3f) | pad_in %.2f free %.2f frame_in %.2f free %.2f n %d | l>y %d/%d z %.1f h0 %.1f h1 %.1f' % (
            d['label'], a['raw']['id_clean'], a['E1c']['id_clean'], a['E1c']['page'], a['raw']['dir'], a['raw']['dir_body'],
            a['raw']['dir_junc'], a['raw']['dir'] - a['raw']['dir_junc'], k4['pad_in'], k4['pad_free'], k4['frame_in'], k4['frame_free'], k4['n_in'],
            k6['hNone']['ly'], k6['hNone']['yl'], k6['hNone']['z'], k6['h0']['z'], k6['h1']['z'])
        print(s)
        if 'k3' in d:
            k3 = d['k3']
            print('   K3 xfer word/f3/l3 %.2f %.2f %.2f | withinA %.3f %.3f %.3f | map id %.3f learned %.3f rand %.3f q95 %.3f pct %.3f' % (
                k3['xfer_word'], k3['xfer_first3'], k3['xfer_last3'], k3['withinA_word'], k3['withinA_first3'], k3['withinA_last3'],
                k3['map_identity_te'], k3['map_learned_te'], k3['map_rand_te_mean'], k3['map_rand_te_q95'], k3['map_pct']))


if __name__ == '__main__':
    globals()[sys.argv[1]]()
