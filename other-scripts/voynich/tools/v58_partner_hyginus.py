"""v58: partner test for the bio x Hyginus lead. A true source must also be found through a translation:
Hyginus De astronomia book 2 in Latin (Latin Library) and in English (theoi.com, Grant tr.), each tested
alone against the bio paragraphs (200 + 200 nulls, ZL3b and IT2a)."""
import os, re, html, json, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v58_lib as L, v58_texts as TX

def eng_b2():
    out = []
    for f in ['theoi1.html', 'theoi2.html']:
        s = open(os.path.join(TX.SRC, 'hyginus', f), encoding='utf-8', errors='replace').read()
        s = re.sub(r'<script.*?</script>', '', s, flags=re.S)
        t = re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s)))
        parts = re.split(r'\bII\.(\d+) ', t)
        for k in range(1, len(parts), 2):
            body = parts[k + 1]
            cut = re.search(r'(Theoi Project|Copyright|BIBLIOGRAPHY|Previous Part|Next Part)', body)
            if cut: body = body[:cut.start()]
            out.append(TX.unit('2.' + parts[k], [body]))
    seen, res = set(), []
    for u in out:
        if u['t'] in seen: continue
        seen.add(u['t']); res.append(u)
    res.sort(key=lambda u: int(u['t'].split('.')[1]))
    return res

def main():
    T = json.load(open(os.path.join(L.CK, 'texts.json')))
    lat = T['hyginus_astr']['units']
    i0 = next(i for i, u in enumerate(lat) if u['t'].startswith('I. Igitur, ut supra'))
    i1 = next(i for i, u in enumerate(lat) if i > i0 and u['t'].startswith('I. ') )
    latb2 = lat[i0:i1]
    eng = eng_b2()
    print('latin book 2 chapters', len(latb2), 'english', len(eng), [u['w'] for u in eng[:10]])
    rng = random.Random(5859)
    res = {}
    PAR = dict(lam=1.0, s=0.30, pm=0.30, g=0.5)
    # sanity: Latin vs English book 2 (should align)
    x = [u['w'] for u in latb2]; y = [u['w'] for u in eng]
    obs, c = L.align(x, y, PAR); nb = [L.align(L.block_shuffle(x, rng), y, PAR)[0] for _ in range(200)]
    res['lat_vs_eng_b2'] = dict(obs=round(obs, 2), zb=round(float(L.zp(obs, nb)[0]), 2))
    print(res)
    for tr in ['ZL3b', 'IT2a']:
        S = L.voynich_sequences(tr); xv = S['bio_paras'][0]
        for nm, ys in [('latin_b2', latb2), ('english_b2', eng), ('latin_all', lat)]:
            y = [u['w'] for u in ys]
            obs, c = L.align(xv, y, PAR)
            nx = [L.align(rng.sample(xv, len(xv)), y, PAR)[0] for _ in range(200)]
            nb = [L.align(L.block_shuffle(xv, rng), y, PAR)[0] for _ in range(200)]
            res[tr + '|' + nm] = dict(obs=round(obs, 2), zx=round(float(L.zp(obs, nx)[0]), 2), zb=round(float(L.zp(obs, nb)[0]), 2))
            print(tr, nm, res[tr + '|' + nm], flush=True)
    json.dump(res, open(os.path.join(L.CK, 'partner_hyginus.json'), 'w'))

if __name__ == '__main__':
    main()
