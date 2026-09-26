"""Group audits by topic and draft topic cards (Anthropic API).

1. Canonicalise the free-text topics that extraction produced, so audits of the
   same thing share one name.  Seed topic names are kept as they are, and seed
   entries' topics are never changed.
2. For every topic with at least MIN_DETAILED detailed audits, draft the usual
   baseline, the fix that works, the fix that fails and consultant advice, from
   those audits only.  Every card is marked "Draft, needs consultant sign-off".
   The three seed cards are kept verbatim; if new audits join one of those
   topics, a separate draft revision card is added next to it.
"""
import collections
import json
import re
import time

import pipeline
from extract import MODEL, client

MIN_DETAILED = 5
DRAFT = "Draft, needs consultant sign-off"
SEED_MAX_ID = 1145

MAP_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["mapping"],
              "properties": {"mapping": {"type": "array", "items": {
                  "type": "object", "additionalProperties": False, "required": ["topic", "canonical"],
                  "properties": {"topic": {"type": "string"}, "canonical": {"type": "string"}}}}}}
CARD_SCHEMA = {"type": "object", "additionalProperties": False,
               "required": ["usual_baseline", "fix_that_works", "fix_that_fails", "consultant_advice"],
               "properties": {k: {"type": "string"} for k in
                              ["usual_baseline", "fix_that_works", "fix_that_fails", "consultant_advice"]}}

CARD_SYSTEM = """You write short topic summaries for a clinical audit library used by UK trainees and reviewed by consultants.
You are given every detailed audit in the library on one topic. Using ONLY those audits:
- usual_baseline: what cycle 1 usually finds, with typical numbers.
- fix_that_works: interventions that were followed by real improvement at re-audit, with the numbers and which audits.
- fix_that_fails: interventions followed by little or no improvement, or items that stayed poor. If the audits contain no failed fix, say "No failed fix reported in these audits" rather than inventing one.
- consultant_advice: 2-3 sentences a consultant would tell a trainee choosing this audit: whether it is overdone, and what would be more useful to measure.
Cite audits by their library id in brackets, e.g. [1136]. Plain British English, no hedging filler. Do not use knowledge outside the audits given."""


def _call(c, system, user, schema, max_tokens=8000):
    msg = c.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                            output_config={"format": {"type": "json_schema", "schema": schema}},
                            messages=[{"role": "user", "content": user}])
    if msg.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason {msg.stop_reason}")
    return json.loads(next(b.text for b in msg.content if b.type == "text"))


def is_detailed(e):
    d = e.get("detail") or {}
    c1 = (d.get("cycle1") or {}).get("result") or ""
    return bool(d) and c1 and not re.match(r"not reported|see (paper|abstract|poster)", c1, re.I)


def canonicalise(c, lib):
    seed_topics = sorted({e["topic"] for e in lib["audits"] if e["id"] <= SEED_MAX_ID and e.get("topic")})
    new = collections.Counter(e["topic"] for e in lib["audits"]
                              if e["id"] > SEED_MAX_ID and e.get("topic") and e["topic"] != "not reported")
    if not new:
        return {}
    by_spec = collections.defaultdict(list)
    for e in lib["audits"]:
        if e["id"] > SEED_MAX_ID and e.get("topic") in new:
            by_spec[pipeline.library_of(e)].append(e["topic"])
    mapping = {}
    for libname, tops in by_spec.items():
        tops = sorted(set(tops))
        for i in range(0, len(tops), 300):
            chunk = tops[i:i + 300]
            user = ("Merge these audit topic names so that audits of the same thing share one short canonical name. "
                    "Reuse a seed name exactly when it fits. Do not merge different clinical questions.\n\n"
                    "Seed names:\n" + "\n".join(seed_topics) + "\n\nTopics (" + libname + "):\n" +
                    "\n".join(f"{t} ({new[t]})" for t in chunk))
            out = _call(c, "You normalise clinical audit topic names.", user, MAP_SCHEMA, 32000)
            mapping.update({m["topic"]: m["canonical"] for m in out["mapping"] if m["topic"] in chunk})
    return mapping


def run():
    lib = pipeline.base_library()
    c = client()
    mapping = canonicalise(c, lib)
    for e in lib["audits"]:
        if e["id"] > SEED_MAX_ID and e.get("topic") in mapping:
            e["topic_extracted"] = e["topic"]
            e["topic"] = mapping[e["topic"]]
    groups = collections.defaultdict(list)
    for e in lib["audits"]:
        if e.get("topic") and is_detailed(e):
            groups[e["topic"]].append(e)
    seed_cards = [t for t in lib.get("topic_knowledge", []) if t.get("generated_by") != "pipeline"]
    seed_names = {t["topic"] for t in seed_cards}
    cards = []
    for topic, es in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(es) < MIN_DETAILED:
            continue
        if topic in seed_names and all(e["id"] <= SEED_MAX_ID for e in es):
            continue            # seed card already covers exactly this evidence
        evidence = [{"id": e["id"], "title": e["title"], "standard": e.get("standard"),
                     "country": (e.get("detail") or {}).get("country"),
                     "cycle1": (e.get("detail") or {}).get("cycle1"),
                     "intervention": (e.get("detail") or {}).get("intervention"),
                     "cycle2": (e.get("detail") or {}).get("cycle2"),
                     "loop_closed": (e.get("detail") or {}).get("loop_closed"),
                     "fix_type": (e.get("detail") or {}).get("fix_type"),
                     "other_findings": (e.get("detail") or {}).get("other_findings")} for e in es]
        body = _call(c, CARD_SYSTEM, f"Topic: {topic}\n\nAudits:\n" + json.dumps(evidence, ensure_ascii=False),
                     CARD_SCHEMA)
        years = sorted(int(y) for e in es for y in [(e.get("paper") or {}).get("year", "")] if str(y).isdigit())
        card = {"topic": topic, "audits_in_library": len(es),
                "countries": sorted({(e.get("detail") or {}).get("country") or "not reported" for e in es}),
                "years_seen": f"{years[0]}–{years[-1]}" if years else "not reported",
                **body, "evidence": "Detailed records in this library; judgement written by Claude from them",
                "evidence_ids": [e["id"] for e in es], "status": DRAFT, "generated_by": "pipeline",
                "model": MODEL, "generated_at": time.strftime("%Y-%m-%d")}
        if topic in seed_names:
            card["revises_seed_card"] = topic
        cards.append(card)
        print(f"  card: {topic} ({len(es)} detailed audits)")
    lib["topic_knowledge"] = seed_cards + cards
    pipeline._write_lib(lib)
    print(f"topics: {len(mapping)} topic names canonicalised; {len(cards)} draft cards "
          f"(+{len(seed_cards)} seed cards kept)")
