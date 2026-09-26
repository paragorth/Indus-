"""In-session alternative to topics.py (no API key needed).

    python3 topics_local.py export      # writes work/topics_in/topic_names.json
    (Claude writes work/topics_out/mapping.json: {extracted topic: canonical topic})
    python3 topics_local.py apply       # applies mapping, writes work/topics_in/groups/*.json
                                        # for every topic with >= 5 detailed audits
    (Claude writes work/topics_out/cards/<slug>.json with the four card fields)
    python3 topics_local.py cards       # adds the cards to the library as drafts

Same rules as topics.py: seed topics and seed cards are never changed; new cards
are marked "Draft, needs consultant sign-off" and list their evidence ids.
"""
import collections
import glob
import json
import os
import re
import sys
import time

import pipeline
from topics import DRAFT, MIN_DETAILED, SEED_MAX_ID, is_detailed

IN = os.path.join(pipeline.WORK, "topics_in")
OUT = os.path.join(pipeline.WORK, "topics_out")
FIELDS = ["usual_baseline", "fix_that_works", "fix_that_fails", "consultant_advice"]


def slug(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:80]


def export():
    lib = pipeline.base_library()
    os.makedirs(IN, exist_ok=True)
    c = collections.Counter()
    spec = collections.defaultdict(collections.Counter)
    for e in lib["audits"]:
        if e["id"] > SEED_MAX_ID and e.get("topic") and e["topic"] != "not reported":
            c[e["topic"]] += 1
            spec[e["topic"]][e["specialty"]] += 1
    seed = sorted({e["topic"] for e in lib["audits"] if e["id"] <= SEED_MAX_ID and e.get("topic")})
    data = {"seed_topics": seed,
            "topics": [{"topic": t, "n": n, "specialties": dict(spec[t])} for t, n in c.most_common()]}
    json.dump(data, open(os.path.join(IN, "topic_names.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{len(c)} distinct extracted topics exported")


def apply():
    lib = pipeline.base_library()
    mapping = json.load(open(os.path.join(OUT, "mapping.json"), encoding="utf-8"))
    for e in lib["audits"]:
        if e["id"] > SEED_MAX_ID and e.get("topic") in mapping:
            e.setdefault("topic_extracted", e["topic"])
            e["topic"] = mapping[e["topic"]]
    pipeline._write_lib(lib)
    groups = collections.defaultdict(list)
    for e in lib["audits"]:
        if e.get("topic") and is_detailed(e):
            groups[e["topic"]].append(e)
    gdir = os.path.join(IN, "groups")
    os.makedirs(gdir, exist_ok=True)
    for f in glob.glob(os.path.join(gdir, "*.json")):
        os.remove(f)
    seed_names = {t["topic"] for t in lib.get("topic_knowledge", []) if t.get("generated_by") != "pipeline"}
    n = 0
    for topic, es in groups.items():
        if len(es) < MIN_DETAILED or (topic in seed_names and all(e["id"] <= SEED_MAX_ID for e in es)):
            continue
        ev = [{"id": e["id"], "title": e["title"], "standard": e.get("standard"), **(e.get("detail") or {})}
              for e in es]
        json.dump({"topic": topic, "is_seed_topic": topic in seed_names, "audits": ev},
                  open(os.path.join(gdir, slug(topic) + ".json"), "w"), indent=1, ensure_ascii=False)
        n += 1
    print(f"mapping applied; {n} topics with >= {MIN_DETAILED} detailed audits exported for cards")


def cards():
    lib = pipeline.base_library()
    by_topic = collections.defaultdict(list)
    for e in lib["audits"]:
        if e.get("topic") and is_detailed(e):
            by_topic[e["topic"]].append(e)
    seed_cards = [t for t in lib.get("topic_knowledge", []) if t.get("generated_by") != "pipeline"]
    seed_names = {t["topic"] for t in seed_cards}
    new = []
    for f in sorted(glob.glob(os.path.join(OUT, "cards", "*.json"))):
        body = json.load(open(f, encoding="utf-8"))
        topic = body["topic"]
        es = by_topic.get(topic, [])
        missing = [k for k in FIELDS if not body.get(k)]
        if missing or len(es) < MIN_DETAILED:
            print(f"skip {topic}: missing {missing} or only {len(es)} detailed audits")
            continue
        years = sorted(int(y) for e in es for y in [(e.get("paper") or {}).get("year", "")] if str(y).isdigit())
        card = {"topic": topic, "audits_in_library": len(es),
                "countries": sorted({(e.get("detail") or {}).get("country") or "not reported" for e in es}),
                "years_seen": f"{years[0]}–{years[-1]}" if years else "not reported",
                **{k: body[k] for k in FIELDS},
                "evidence": "Detailed records in this library; judgement written by Claude from them",
                "evidence_ids": [e["id"] for e in es], "status": DRAFT, "generated_by": "pipeline",
                "model": "Claude (in-session, Claude Code)", "generated_at": time.strftime("%Y-%m-%d")}
        if topic in seed_names:
            card["revises_seed_card"] = topic
        new.append(card)
    lib["topic_knowledge"] = seed_cards + new
    pipeline._write_lib(lib)
    print(f"{len(new)} draft cards added (+{len(seed_cards)} seed cards kept)")


if __name__ == "__main__":
    {"export": export, "apply": apply, "cards": cards}[sys.argv[1]]()
