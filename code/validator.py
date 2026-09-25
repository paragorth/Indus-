"""Blind validator for any proposed Indus sign-to-sound key.
A key passes only if, on HELD-OUT texts, its lexicon-hit rate exceeds (a) the same key on within-text scrambled
sequences and (b) the null from fake lexicons (letters shuffled within each real word) by more than 3 SD.
Usage: python validator.py key.json lexicon.txt texts.txt
  key.json    : {"sign_number": "string", ...}
  lexicon.txt : one word per line, skeletonised (lowercase a-z)
  texts.txt   : one text per line, sign numbers separated by spaces, reading order, 0 = damage
"""
import sys, json, random, numpy as np
def load_texts(p):
    T=[]
    for line in open(p):
        t=[int(x) for x in line.split() if x.isdigit() and x!='0']
        if len(t)>=2: T.append(t)
    return T
def rate(key,texts,lex):
    h=n=0
    for t in texts:
        v=[key.get(str(s)) for s in t]
        for i in range(len(t)):
            for j in range(i+2,min(len(t),i+5)+1):
                w=v[i:j]
                if all(w): n+=1; h+=(''.join(w) in lex)
    return h/max(1,n)
def scrambled(texts,rng):
    out=[]
    for t in texts: l=t[:]; rng.shuffle(l); out.append(l)
    return out
def fake(lex,rng):
    out=set()
    for w in lex: l=list(w); rng.shuffle(l); out.add(''.join(l))
    return out
if __name__=='__main__':
    key=json.load(open(sys.argv[1])); lex=set(w.strip() for w in open(sys.argv[2]) if w.strip()); T=load_texts(sys.argv[3])
    rng=random.Random(0); rng.shuffle(T); held=T[len(T)//2:]
    real=rate(key,held,lex); scr=np.mean([rate(key,scrambled(held,rng),lex) for _ in range(10)])
    null=[rate(key,held,fake(lex,rng)) for _ in range(30)]
    z=(real-np.mean(null))/(np.std(null)+1e-9)
    print(f"held-out real {100*real:.2f}%  scrambled {100*scr:.2f}%  fake-lexicon null {100*np.mean(null):.2f}% ± {100*np.std(null):.2f}  z={z:.1f}")
    print("PASS" if (real>scr and z>3) else "FAIL")
