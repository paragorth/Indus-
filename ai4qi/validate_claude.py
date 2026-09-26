"""Check and ingest in-session extractions (work/claude_out/batch_NN.json) into
work/extractions.json, validating every record against extract.SCHEMA."""
import glob, json, sys, time
import pipeline
from extract import SCHEMA, build_input


def check(obj, schema, path="$"):
    errs = []
    if schema["type"] == "object":
        if not isinstance(obj, dict):
            return [f"{path}: not an object"]
        for k in schema["required"]:
            if k not in obj:
                errs.append(f"{path}.{k}: missing")
        for k in obj:
            if k not in schema["properties"]:
                errs.append(f"{path}.{k}: unexpected")
        for k, sub in schema["properties"].items():
            if k in obj:
                errs += check(obj[k], sub, f"{path}.{k}")
    elif schema["type"] == "string":
        if not isinstance(obj, str) or not obj.strip():
            errs.append(f"{path}: not a non-empty string")
        elif "enum" in schema and obj not in schema["enum"]:
            errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    return errs


def main(ingest):
    ext = pipeline.load("extractions.json", {})
    recs = pipeline.load("records.json", {})
    bad, good = {}, 0
    for fn in sorted(glob.glob(f"{pipeline.WORK}/claude_out/batch_*.json")):
        data = json.load(open(fn, encoding="utf-8"))
        for k, v in data.items():
            e = check(v, SCHEMA)
            if e:
                bad[k] = e
                continue
            good += 1
            if ingest:
                ext[k] = {"result": v, "error": None, "model": "Claude (in-session, Claude Code)",
                          "input": build_input(recs[k])[0], "at": time.strftime("%Y-%m-%d %H:%M")}
    if ingest:
        pipeline.save("extractions.json", ext)
    print(f"valid {good}, invalid {len(bad)}")
    for k, e in list(bad.items())[:20]:
        print(" ", k, e[:3])


if __name__ == "__main__":
    main("--ingest" in sys.argv)
