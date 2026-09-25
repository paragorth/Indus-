"""Strategy 131: direct test of the one published outside-anchor proposal (Fuls 2024: 20 Indus signs given
Linear Elamite values). Count Indus windows made only of anchored signs whose reading is an Elamite word form
(Old Elamite from HLED Linear Elamite + later Elamite from CDLI), vs the same count when the 20 values are
shuffled among the 20 signs (10,000 permutations) and when values are drawn from the LE syllable inventory."""
import json, re, random, collections, unicodedata
import indus_core as C
def red(s):
    s=unicodedata.normalize('NFKD',s.lower()); s=''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z]','',re.sub(r'\d','',s))
key={k:red(v) for k,v in json.load(open('data/derived/fuls2024-le-indus-key.json'))['key'].items()}
le=json.load(open('data/derived/linear-elamite-corpus-hled.json'))
lexO=set(); syl=collections.Counter()
for x in le:
    for w in re.split(r'\s+|\|\|',x.get('translit','')):
        for s in w.split('-'):
            r=red(s)
            if r: syl[r]+=1
        r=red(w.replace('-',''))
        if 2<=len(r)<=15: lexO.add(r)
lexC={w.strip() for w in open('data/derived/lex_elamite.txt') if w.strip()}
texts=C.unique_texts(C.load_corpus())
segs=[s for t in texts for s in t]
def count(k,lex):
    n=0; hits=collections.Counter()
    for s in segs:
        for L in (2,3,4):
            for i in range(len(s)-L+1):
                w=s[i:i+L]
                if all(x in k for x in w):
                    r=''.join(k[x] for x in w)
                    if len(r)>=3 and r in lex: n+=1; hits[r]+=1
    return n,hits
if __name__=='__main__':
    rng=random.Random(0); signs=list(key); vals=[key[s] for s in signs]
    sylpool=[s for s,_ in syl.most_common(60)]
    for name,lex in (('Old Elamite (LE, 304 forms)',lexO),('Elamite CDLI (3,004 forms)',lexC),('both',lexO|lexC)):
        obs,h=count(key,lex)
        perm=[]; rand=[]
        for _ in range(2000):
            v=vals[:]; rng.shuffle(v); perm.append(count(dict(zip(signs,v)),lex)[0])
            rand.append(count(dict(zip(signs,rng.choices(sylpool,k=len(signs)))),lex)[0])
        print(f"{name}: observed {obs} {dict(h.most_common(8))} | shuffled-values mean {sum(perm)/len(perm):.1f}, P={sum(p>=obs for p in perm)/len(perm):.3f} | random LE values mean {sum(rand)/len(rand):.1f}, P={sum(p>=obs for p in rand)/len(rand):.3f}")
