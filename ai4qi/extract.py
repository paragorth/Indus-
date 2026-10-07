"""Structured extraction of each kept paper with the Anthropic API.

Credentials: the SDK's zero-argument client (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN
or an `ant auth login` profile).  Model: AI4QI_MODEL (default claude-opus-5).

Input is the full-text JATS body when the paper is open access, otherwise the
title and abstract.  The prompt forbids guessing: anything the text does not
state comes back as "not reported".
"""
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import pipeline

MODEL = os.environ.get("AI4QI_MODEL", "claude-opus-5")
EFFORT = os.environ.get("AI4QI_EFFORT", "medium")
NR = "not reported"

CYCLE = {"type": "object", "additionalProperties": False,
         "required": ["dates", "sample_size", "result"],
         "properties": {
             "dates": {"type": "string", "description": "Data collection period as stated, or 'not reported'"},
             "sample_size": {"type": "string", "description": "Number of patients/cases/notes as digits, or 'not reported'"},
             "result": {"type": "string", "description": "Main result against the standard WITH the numbers (percentages, counts, times), or 'not reported'"}}}
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["is_audit", "specialty", "topic", "setting", "country", "standard", "cycle1",
                 "intervention", "cycle2", "loop_closed", "other_findings", "fix_type"],
    "properties": {
        "is_audit": {"type": "string", "enum": ["yes", "no", "unclear"],
                     "description": "Is this a clinical audit or audit-based quality improvement project measured against a standard?"},
        "specialty": {"type": "string"},
        "topic": {"type": "string", "description": "Short reusable topic name shared by audits of the same thing, e.g. 'Operation note quality'"},
        "setting": {"type": "string", "description": "Hospital/unit and type, as stated"},
        "country": {"type": "string"},
        "standard": {"type": "string", "description": "The guideline or standard audited against, with the target if stated"},
        "cycle1": CYCLE,
        "intervention": {"type": "string", "description": "What was changed between cycles"},
        "cycle2": CYCLE,
        "loop_closed": {"type": "string", "enum": ["yes", "no", NR],
                        "description": "yes only if a second measurement after the change is reported"},
        "other_findings": {"type": "string"},
        "fix_type": {"type": "string", "enum": ["form or template", "system change", "education only", "mixed", NR],
                     "description": "form or template = proforma, sticker, checklist, electronic template; system change = new pathway, list, rota, IT workflow, staffing; education only = teaching, posters, emails, reminders; mixed = more than one of these"},
    },
}

SYSTEM = """You extract structured data from published clinical audits and quality improvement projects for a library that trainees use to plan audits.

Rules:
- Use only what the supplied text states. Never infer, estimate or fill from general knowledge. If a field is not stated, write exactly "not reported".
- Results must keep the numbers exactly as the paper gives them (percentages, n/N, medians, p values). Report cycle 1 (baseline) and cycle 2 (re-audit after the change) separately. If there were more than two cycles, put the final re-audit in cycle 2 and mention the others in other_findings.
- specialty: the clinical specialty of the audited service, in plain British English (e.g. "Orthopaedics – Hip fracture", "General surgery", "Emergency medicine").
- topic: a short, generic, reusable name so that audits of the same thing group together. Prefer one of the existing topic names listed below when it fits exactly.
- fix_type: classify the intervention described between cycles; "not reported" if none is described.
- British spelling, plain sentences, no marketing language."""


def jats_text(path):
    """Readable text of a JATS article: title, abstract, body, figure/table captions and tables.
    References are dropped (they are not evidence about the audit)."""
    root = ET.parse(path).getroot()
    for ref in list(root.iter("ref-list")):
        ref.clear()
    parts = []
    for tag in ("article-title", "abstract", "body"):
        for el in root.iter(tag):
            parts.append(re.sub(r"\s+", " ", " ".join(el.itertext())).strip())
            if tag == "article-title":
                break
    return "\n\n".join(p for p in parts if p)


def build_input(rec):
    ft = os.path.join(pipeline.FT_DIR, f"{rec.get('pmcid')}.xml")
    if rec.get("pmcid") and os.path.exists(ft):
        try:
            return "full text", jats_text(ft)
        except ET.ParseError:
            pass
    return "title and abstract", f"{rec['title']}\n\n{rec.get('abstract') or '(no abstract)'}"


def request_params(rec, topics):
    kind, text = build_input(rec)
    head = (f"Journal: {rec.get('journal', '')} ({rec.get('year', '')})\n"
            f"Affiliations: {'; '.join(rec.get('affiliations', [])[:5]) or 'not given'}\n"
            f"Text supplied: {kind}\n\n")
    system = SYSTEM + "\n\nExisting topic names:\n" + "\n".join(f"- {t}" for t in topics)
    return kind, dict(
        model=MODEL, max_tokens=16000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": head + text}],
    )


def _parse(msg):
    if msg.stop_reason == "refusal":
        return None, "refused"
    if msg.stop_reason == "max_tokens":
        return None, "hit max_tokens"
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"


def _todo(pas, limit):
    recs, scr = pipeline.load("records.json", {}), pipeline.load("screen.json", {})
    ext = pipeline.load("extractions.json", {})
    keys = [k for k, s in scr.items() if s["keep"] and s["pass"] == pas and k in recs
            and not (ext.get(k) or {}).get("result")]
    # UK and Ireland first, then full-text papers, so a partial run covers the most useful ones.
    keys.sort(key=lambda k: (not __import__("classify").uk_ireland(recs[k].get("affiliations")),
                             not recs[k].get("open_access"), k))
    return (keys[:limit] if limit else keys), recs, ext


def existing_topics():
    lib = pipeline.base_library()
    t = {e.get("topic") for e in lib["audits"] if e.get("topic") and e.get("topic") != NR}
    t |= {c["topic"] for c in lib.get("topic_knowledge", [])}
    return sorted(t)


def client():
    import anthropic
    try:
        return anthropic.Anthropic()
    except Exception as e:  # no credentials at all
        raise SystemExit(f"Anthropic client could not start ({e}). Set ANTHROPIC_API_KEY or run "
                         f"`ant auth login`, then re-run: python3 pipeline.py extract --pass <pass>")


def run(pas, limit=0, batch=False):
    keys, recs, ext = _todo(pas, limit)
    if not keys:
        print("nothing to extract")
        return
    topics = existing_topics()
    c = client()
    print(f"extracting {len(keys)} papers with {MODEL} ({'batch' if batch else 'direct'})")
    if batch:
        return _run_batch(c, pas, keys, recs, ext, topics)

    def one(k):
        kind, params = request_params(recs[k], topics)
        # Server-side refusal fallback: if the model declines, the API reruns the request
        # on a fallback model inside the same call.
        msg = c.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
        res, err = _parse(msg)
        return k, {"result": res, "error": err, "model": msg.model, "input": kind,
                   "at": time.strftime("%Y-%m-%d %H:%M")}

    done = 0
    with ThreadPoolExecutor(max_workers=int(os.environ.get("AI4QI_WORKERS", "4"))) as pool:
        futs = [pool.submit(one, k) for k in keys]
        for f in as_completed(futs):
            try:
                k, v = f.result()
            except Exception as e:  # typed API errors are retried by the SDK already
                import sources
                sources.log_failure("extract", "?", f"{e.__class__.__name__}: {e}")
                continue
            ext[k] = v
            done += 1
            if done % 20 == 0:
                pipeline.save("extractions.json", ext)
                print(f"  {done}/{len(keys)}")
    pipeline.save("extractions.json", ext)
    print(f"extracted {done}; errors {sum(1 for k in keys if (ext.get(k) or {}).get('error'))}")


def _run_batch(c, pas, keys, recs, ext, topics):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    state = pipeline.load(f"batch_{pas}.json", {})
    if not state.get("id"):
        reqs, kinds = [], {}
        for i, k in enumerate(keys):
            kind, params = request_params(recs[k], topics)
            kinds[f"r{i}"] = [k, kind]
            reqs.append(Request(custom_id=f"r{i}", params=MessageCreateParamsNonStreaming(**params)))
        b = c.messages.batches.create(requests=reqs)
        state = {"id": b.id, "map": kinds}
        pipeline.save(f"batch_{pas}.json", state)
        print(f"batch {b.id} submitted ({len(reqs)} requests)")
    while True:
        b = c.messages.batches.retrieve(state["id"])
        if b.processing_status == "ended":
            break
        print(f"  batch {b.processing_status}: {b.request_counts.processing} processing")
        time.sleep(60)
    for res in c.messages.batches.results(state["id"]):
        k, kind = state["map"][res.custom_id]
        if res.result.type == "succeeded":
            parsed, err = _parse(res.result.message)
            ext[k] = {"result": parsed, "error": err, "model": res.result.message.model, "input": kind,
                      "at": time.strftime("%Y-%m-%d %H:%M")}
        else:
            ext[k] = {"result": None, "error": res.result.type, "model": MODEL, "input": kind}
    pipeline.save("extractions.json", ext)
    pipeline.save(f"batch_{pas}.json", {})
    print(f"batch collected: {len(state['map'])} results")
