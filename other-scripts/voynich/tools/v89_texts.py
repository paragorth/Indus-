"""v89: candidate texts with word TOKENS per natural entry (re-uses the v58 parsers; v58 kept only lengths).
Output data/v89_ckpt/texts.json : {id: {lang, genre, src, units: [{t, w, c, p, tok:[words]}]}}"""
import os, re, json, unicodedata
import v58_texts as T

HERE = os.path.dirname(os.path.abspath(__file__))
VD = os.path.dirname(HERE)
T.CK = os.path.join(VD, 'data', 'v89_ckpt')

def toks(text):
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return [w.lower() for w in T.WRE.findall(text)]

_old_unit = T.unit
def unit(title, paras):
    u = _old_unit(title, paras)
    u['tok'] = [w for p in paras if p.strip() for w in toks(p)]
    return u
T.unit = unit

def gerard():
    d = json.load(open(os.path.join(VD, 'data', 'derived', 'v13_gerard.json')))['pages']
    out = []
    for k in sorted(d, key=lambda z: int(re.sub(r'\D', '', z) or 0)):
        ws = [w.lower() for w in d[k]['ocr'] if sum(ch.isalpha() for ch in w) >= 2]
        ws = [re.sub(r'[^\w]', '', w) for w in ws]
        if len(ws) >= 5: out.append({'t': 'p' + k, 'w': len(ws), 'c': sum(len(w) for w in ws), 'p': [len(ws)], 'tok': ws})
    return out
T.gerard = gerard

if __name__ == '__main__':
    T.main()
