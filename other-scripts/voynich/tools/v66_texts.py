"""v66: tokenised units (entry/page = unit) for the concept-network alignment.

Reuses the v58 parsers but keeps the words. Output data/v66_ckpt/texts.json:
  {name: {lang, genre, units: [[tokens...], ...]}}
Herbal/medical texts keep their natural entries (long entries are cut into
~200-word pieces); prose controls (chronicle, psalter, novel) are cut into
fixed 170-word chunks (about one Voynich page).
"""
import os, re, sys, json, unicodedata
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v58_texts as V

VD = os.path.dirname(HERE)
CK = os.path.join(VD, 'data', 'v66_ckpt')
os.makedirs(CK, exist_ok=True)


def norm(w):
    w = unicodedata.normalize('NFD', w.lower())
    w = ''.join(c for c in w if unicodedata.category(c) != 'Mn')
    return w.replace('ſ', 's').replace('j', 'i').replace('v', 'u') if False else w


def toks(text):
    text = ''.join(ch for ch in unicodedata.normalize('NFD', text) if unicodedata.category(ch) != 'Mn')
    return [w.lower() for w in V.WRE.findall(text)]


def unit_tok(title, paras):
    t = []
    for p in paras:
        if p.strip(): t += toks(p)
    return {'t': (title or '')[:60], 'w': len(t), 'tok': t}


V.unit = unit_tok   # parsers call the module-level name


def cut(units, maxw=400, piece=200, minw=30):
    out = []
    for u in units:
        t = u['tok'] if isinstance(u, dict) else u
        if len(t) < minw: continue
        if len(t) > maxw:
            for i in range(0, len(t), piece):
                if len(t[i:i + piece]) >= minw: out.append(t[i:i + piece])
        else:
            out.append(t)
    return out


def chunks(text, size=170):
    t = toks(text)
    return [t[i:i + size] for i in range(0, len(t) - size + 1, size)]


def gut(path):
    return V.gut_body(path)


def main():
    T = {}
    D = os.path.join(VD, 'data')
    SRC = V.SRC
    jobs = [
        ('culpeper', 'en', 'herbal', V.culpeper),
        ('konrad_plants', 'de', 'herbal', lambda: [u for u in V.konrad() if u['t'][:5] in ('kon_4', 'kon_5')]),
        ('konrad_body', 'de', 'encycl', lambda: [u for u in V.konrad() if u['t'][:5] not in ('kon_4', 'kon_5')]),
        ('hildegard', 'la', 'herbal', V.hildegard),
        ('macer', 'la', 'herbal', V.macer),
        ('circa_fr', 'fr', 'herbal', lambda: V.ocr_caps_entries(os.path.join(SRC, 'ia', 'BIUSante_pharma_032591.txt'))),
        ('celsus_lat', 'la', 'medical', lambda: V.tei_units(os.path.join(SRC, 'celsus_lat.xml'), V.is_chap)),
        ('celsus_eng', 'en', 'medical', lambda: V.tei_units(os.path.join(SRC, 'celsus_eng.xml'), V.is_chap)),
        ('regimen', 'la', 'regimen', V.regimen),
        ('hyginus', 'la', 'astro', V.hyginus_astr),
    ]
    for k, lang, genre, fn in jobs:
        try:
            us = cut(fn())
            T[k] = {'lang': lang, 'genre': genre, 'units': us}
            print('%-14s %s %-8s units=%4d words=%7d' % (k, lang, genre, len(us), sum(map(len, us))), flush=True)
        except Exception as e:
            print('FAIL', k, repr(e)[:200], flush=True)
    for k, us in V.v21_herbals().items():
        us = cut(us)
        T['v21_' + k] = {'lang': k.lower(), 'genre': 'herbal', 'units': us}
        print('%-14s units=%d words=%d' % ('v21_' + k, len(us), sum(map(len, us))))
    prose = [('caesar', 'la', 'chronicle', gut(os.path.join(D, 'pg218.txt'))),
             ('dalimil', 'cs', 'chronicle', open(os.path.join(D, 'plain', 'cs.txt'), encoding='utf-8').read()),
             ('manzoni', 'it', 'novel', gut(os.path.join(D, 'pg45334.txt'))),
             ('kafka', 'de', 'novel', gut(os.path.join(D, 'pg22367.txt'))),
             ('descartes', 'la', 'philosophy', gut(os.path.join(D, 'pg23306.txt')))]
    for k, lang, genre, txt in prose:
        us = chunks(txt)[:300]
        T[k] = {'lang': lang, 'genre': genre, 'units': us}
        print('%-14s %s %-8s units=%4d' % (k, lang, genre, len(us)))
    try:
        us = []
        for u in V.sefaria('psalms_en.json'): us += [u['tok']]
        flat = [w for u in us for w in u]
        T['psalms_en'] = {'lang': 'en', 'genre': 'psalter', 'units': [flat[i:i + 170] for i in range(0, len(flat) - 169, 170)][:300]}
        print('psalms_en units=%d' % len(T['psalms_en']['units']))
    except Exception as e:
        print('FAIL psalms', e)
    json.dump(T, open(os.path.join(CK, 'texts.json'), 'w'))


if __name__ == '__main__':
    main()
