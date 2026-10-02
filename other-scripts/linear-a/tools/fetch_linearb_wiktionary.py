#!/usr/bin/env python3
"""Fetch Linear B (Mycenaean Greek) word spellings from Wiktionary categories.

Only the spellings are used (titles written in Linear B Unicode, converted to
conventional sign values from the Unicode character names). Wiktionary glosses
are not used. Output: data/linearb_words.json  {category: [spelling, ...]}
"""
import json, os, unicodedata, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data', 'linearb_words.json')
API = 'https://en.wiktionary.org/w/api.php'
CATS = ['Category:Mycenaean_Greek_lemmas', 'Category:Mycenaean_Greek_proper_nouns',
        'Category:Mycenaean_Greek_non-lemma_forms']

def members(cat):
    out, cont = [], {}
    while True:
        q = {'action': 'query', 'list': 'categorymembers', 'cmtitle': cat, 'cmlimit': '500',
             'format': 'json', 'cmnamespace': '0'}
        q.update(cont)
        req = urllib.request.Request(API + '?' + urllib.parse.urlencode(q),
                                     headers={'User-Agent': 'script-research/0.1 (data fetch)'})
        d = json.load(urllib.request.urlopen(req, timeout=60))
        out += [m['title'] for m in d['query']['categorymembers']]
        if 'continue' not in d: break
        cont = {'cmcontinue': d['continue']['cmcontinue']}
    return out

def translit(title):
    signs = []
    for ch in title:
        c = ord(ch)
        if not (0x10000 <= c <= 0x100FF): return None      # Linear B blocks only
        n = unicodedata.name(ch, '')
        if 'SYLLABLE' in n:
            signs.append(n.split()[-1])                    # e.g. LINEAR B SYLLABLE B008 A -> A
        elif 'SYMBOL' in n:
            signs.append('*' + n.split()[-1].lstrip('B').lstrip('0'))
        else:
            return None
    return '-'.join(signs) if signs else None

def main():
    res = {}
    for cat in CATS:
        try:
            titles = members(cat)
        except Exception as e:
            print('FAILED', cat, e); continue
        sp = sorted({t for t in (translit(x) for x in titles) if t})
        res[cat] = sp
        print(cat, 'titles', len(titles), 'Linear B spellings', len(sp))
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=0)

if __name__ == '__main__':
    main()
