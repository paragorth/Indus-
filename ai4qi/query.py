"""Answer questions from the Ai4Qi corpus (no network, no dependencies).

    python3 query.py "hip fracture delay anticoagulation"      # ranked audits + matching topic cards
    python3 query.py "clozapine" --specialty psychiatry --closed
    python3 query.py --topic "Operation note quality"          # the card and every audit on it
    python3 query.py --id 1136                                  # one entry in full
    python3 query.py --topics [--specialty surgery]             # topic list with counts
    add --json for machine-readable output (for the site's backend)

Ranking is BM25 over title, topic, standard, finding, change, cycle results,
intervention and abstract, with a boost for detailed, closed-loop and UK/Ireland audits.
"""
import argparse
import collections
import json
import math
import os
import re
import signal
import sys

signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # quiet when piped into head

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "ai4qi-library.json")
TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("the a an of in on for and or to with by at from is are was were be this that audit audits "
           "re quality improvement project closed loop cycle".split())


def load():
    with open(LIB, encoding="utf-8") as f:
        return json.load(f)


def text_of(e):
    d, p = e.get("detail") or {}, e.get("paper") or {}
    parts = [e.get("title"), e.get("topic"), e.get("specialty"), e.get("standard"), e.get("finding"),
             e.get("change"), e.get("next_audit"), d.get("intervention"), d.get("other_findings"),
             (d.get("cycle1") or {}).get("result"), ((d.get("cycle2") or {}) or {}).get("result"),
             p.get("abstract", "")[:1500]]
    return " ".join(x for x in parts if isinstance(x, str))


def toks(s):
    return [t for t in TOKEN.findall((s or "").lower()) if t not in STOP]


class Index:
    def __init__(self, audits):
        self.docs = [toks(text_of(e)) for e in audits]
        self.tf = [collections.Counter(d) for d in self.docs]
        self.df = collections.Counter(t for d in self.tf for t in d)
        self.avg = sum(len(d) for d in self.docs) / max(len(self.docs), 1)
        self.n = len(self.docs)

    def score(self, i, q):
        s, tf, L = 0.0, self.tf[i], len(self.docs[i])
        for t in q:
            if t in tf:
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * L / self.avg))
        return s


def quality_boost(e):
    d, p = e.get("detail") or {}, e.get("paper") or {}
    b = 1.0
    if e.get("status") in ("published, detailed",) or (d and e["id"] <= 1145):
        b += 0.3
    if d.get("loop_closed") in ("yes", True):
        b += 0.3
    if (d.get("country") in ("UK", "Ireland")) or p.get("uk_ireland"):
        b += 0.15
    if e.get("status") == "recurring topic, not individually linked":
        b -= 0.4
    return b


def brief(e):
    d = e.get("detail") or {}
    c1, c2 = d.get("cycle1") or {}, d.get("cycle2") or {}
    return {"id": e["id"], "title": e.get("title"), "specialty": e.get("specialty"), "topic": e.get("topic"),
            "status": e.get("status"), "standard": e.get("standard"), "cycle1": c1.get("result") or e.get("finding"),
            "intervention": d.get("intervention") or e.get("change"), "cycle2": c2.get("result"),
            "loop_closed": d.get("loop_closed"), "fix_type": d.get("fix_type"),
            "country": d.get("country") or (e.get("paper") or {}).get("country_from_affiliation"),
            "year": (e.get("paper") or {}).get("year"), "source": e.get("source"),
            "citation": d.get("citation") or (e.get("paper") or {}).get("citation")}


def show(b):
    print(f"[{b['id']}] {b['title']}")
    print(f"    {b['specialty']} | topic: {b['topic'] or '-'} | {b['status']} | {b['country'] or ''} {b['year'] or ''}")
    for k in ("standard", "cycle1", "intervention", "cycle2", "fix_type"):
        if b.get(k) and b[k] != "not reported":
            print(f"    {k}: {b[k]}")
    print(f"    source: {b['source'] or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q", nargs="*")
    ap.add_argument("--specialty")
    ap.add_argument("--topic")
    ap.add_argument("--id", type=int)
    ap.add_argument("--topics", action="store_true")
    ap.add_argument("--closed", action="store_true", help="closed-loop audits only")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--proposed", action="store_true", help="search the proposed (not yet run) audits instead")
    a = ap.parse_args()
    lib = load()
    audits, cards = lib["audits"], lib.get("topic_knowledge", [])
    out = {}

    def keep(e):
        if a.specialty and a.specialty.lower() not in (e.get("specialty") or "").lower():
            return False
        if a.closed and (e.get("detail") or {}).get("loop_closed") not in ("yes", True):
            return False
        return True

    if a.proposed:
        pa = []
        for fn in ("ortho_new_audits.json", "nonortho_new_audits.json"):
            p = os.path.join(HERE, "new_audits", fn)
            if os.path.exists(p):
                pa += json.load(open(p, encoding="utf-8"))
        if a.specialty:
            pa = [x for x in pa if a.specialty.lower() in (x.get("area") or "").lower()]
        q = toks(" ".join(a.q))
        def fbw(x):   # down-voted proposals sink, up-voted rise
            f = x.get("feedback") or {}
            return 1 + 0.2 * f.get("up", 0) - 0.5 * f.get("down", 0)
        sc = sorted(((sum(t in toks(json.dumps(x)) for t in q) * fbw(x), x) for x in pa), key=lambda z: -z[0])
        res = [x for s, x in sc if s > 0][:a.n]
        if a.json:
            print(json.dumps(res, indent=1, ensure_ascii=False))
        else:
            for x in res:
                fb = x.get("feedback")
                note = f"  [feedback: {fb['up']} up / {fb['down']} down {', '.join(fb['reasons'])}]" if fb else ""
                print(f"{x['id']}  {x['question']}{note}\n    standard: {x['standard'].get('source')} | change: {x['change']} | template: {x['template_file']}")
        return
    if a.id:
        e = next((x for x in audits if x["id"] == a.id), None)
        out = e or {"error": "no such id"}
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return
    if a.topics:
        c = collections.Counter(e["topic"] for e in audits if e.get("topic") and keep(e))
        out = dict(c.most_common())
        if a.json:
            print(json.dumps(out, ensure_ascii=False))
        else:
            for t, n in c.most_common():
                print(f"{n:4d}  {t}")
        return
    if a.topic:
        tl = a.topic.lower()
        cs = [c for c in cards if c["topic"].lower() == tl]
        es = [brief(e) for e in audits if (e.get("topic") or "").lower() == tl and keep(e)]
        es.sort(key=lambda b: b["country"] not in ("UK", "Ireland"))
        out = {"cards": cs, "audits": es}
    else:
        q = toks(" ".join(a.q))
        idx = Index(audits)
        scored = sorted(((idx.score(i, q) * quality_boost(e), e) for i, e in enumerate(audits) if keep(e)),
                        key=lambda x: -x[0])
        pool = [e for s, e in scored[:max(a.n * 4, 40)] if s > 0]
        def uk(e):   # UK and Ireland audits always listed first, then the rest, each by relevance
            d, p = e.get("detail") or {}, e.get("paper") or {}
            c = d.get("country")
            if c and c != "not reported":
                return c in ("UK", "Ireland")
            return bool(p.get("uk_ireland"))
        hits = [brief(e) for e in ([e for e in pool if uk(e)] + [e for e in pool if not uk(e)])[:a.n]]
        topics = collections.Counter(h["topic"] for h in hits if h["topic"])
        cs = [c for c in cards if c["topic"] in topics]
        out = {"query": " ".join(a.q), "cards": cs, "audits": hits}
    if a.json:
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return
    for c in out["cards"]:
        tag = c.get("status", "seed card (from the original library)")
        print(f"== TOPIC CARD: {c['topic']}  [{tag}] ({c.get('audits_in_library')} audits)")
        for k in ("usual_baseline", "fix_that_works", "fix_that_fails", "consultant_advice"):
            print(f"   {k.replace('_', ' ')}: {c.get(k)}")
        print()
    for b in out["audits"]:
        show(b)


if __name__ == "__main__":
    sys.exit(main())
