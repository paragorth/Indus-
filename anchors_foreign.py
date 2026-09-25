"""Do Indus texts found abroad follow the home grammar?

Foreign = West Asian / Gulf / Central Asian find-spots.  For each text: share of
its adjacent sign pairs attested in home-site texts, jar-final, opener-initial.
Compared with length-matched random home texts (10,000 draws).  If the foreign
texts are local names or words spelt in Indus signs, they should use attested
pairs less and the home formulae (opener, jar) less.
"""
import json, random, statistics
from collections import Counter

FOREIGN = {"Ur", "Kish", "Susa", "Nippur", "Tell Umma", "Luristan", "Failaka", "Qala'at al-Bahrain",
           "Hajar", "Karzakan", "Ra's al-Junayz", "Salut", "Altyn Depe"}
d = json.load(open("data/derived/merged-corpus-reading-order.json"))
seqs = [(x["site"], x["type"], [s for s in x["seq"] if s != 0]) for x in d]
home = [q for s, t, q in seqs if s not in FOREIGN and len(q) >= 2]
home_u = [list(t) for t in {tuple(q) for q in home}]
foreign = [(s, t, q) for s, t, q in seqs if s in FOREIGN and len(q) >= 2]
pairs = Counter((q[i], q[i + 1]) for q in home_u for i in range(len(q) - 1))
freq = Counter(x for q in home_u for x in q)


def stats(q, excl=None):
    ps = [(q[i], q[i + 1]) for i in range(len(q) - 1)]
    att = sum(1 for p in ps if pairs[p] - (1 if excl else 0) > 0) / len(ps)
    return att, q[-1] == 740, q[0] in (817, 861, 820), sum(1 for x in q if freq[x] == 0) / len(q)


fs = [stats(q) for _, _, q in foreign]
print(f"foreign texts (>=2 signs): {len(foreign)}")
for (s, t, q), st in zip(foreign, fs):
    print(f"  {s:20s} {t:8s} {q}  attested-pairs {st[0]:.2f}  jar-final {st[1]}  unseen-signs {st[3]:.2f}")
obs = [statistics.mean(x[i] for x in fs) for i in range(4)]
bylen = {}
for q in home_u:
    bylen.setdefault(len(q), []).append(q)
rng = random.Random(0)
null = []
for _ in range(10000):
    samp = []
    for _, _, q in foreign:
        L = len(q)
        while L not in bylen:
            L -= 1
        samp.append(stats(rng.choice(bylen[L]), excl=True))
    null.append([statistics.mean(x[i] for x in samp) for i in range(4)])
names = ["attested pairs", "jar-final", "opener-initial", "signs unseen at home"]
for i, n in enumerate(names):
    col = [x[i] for x in null]
    lo = sum(c <= obs[i] for c in col) / len(col)
    print(f"{n:22s} foreign {obs[i]:.3f}  home (length-matched) {statistics.mean(col):.3f}  P(home <= foreign) = {lo:.4f}")
