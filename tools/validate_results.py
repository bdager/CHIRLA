#!/usr/bin/env python3
"""Check docs/leaderboard/results.json before it is published.

Usage: python tools/validate_results.py [path/to/results.json]
Exits with status 1 and a list of problems if anything is wrong.
"""
import json
import re
import sys
from pathlib import Path

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/leaderboard/results.json")
REID_SCENARIOS = ["long_term", "multi_camera_long_term", "multi_camera", "reappearance"]
REID_METRICS = ["r1", "r5", "r10", "map"]
TRACK_GROUPS = {
    "all": ["mota", "idf1", "mt", "ml", "fn", "fp", "idsw", "fm"],
    "brief_occlusions": ["mota", "idf1", "idsw"],
    "multiple_people_occlusions": ["mota", "idf1", "idsw"],
}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
errors = []


def err(where, msg):
    errors.append(f"{where}: {msg}")


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_common(e, where, papers, seen):
    for f in ["id", "method", "cite", "reported_in", "kind", "added", "results"]:
        if f not in e:
            err(where, f"missing '{f}'")
    if e.get("id") in seen:
        err(where, f"duplicate id '{e.get('id')}'")
    seen.add(e.get("id"))
    if e.get("kind") not in ("baseline", "community"):
        err(where, "kind must be 'baseline' or 'community'")
    if "added" in e and not DATE.match(str(e["added"])):
        err(where, "added must be YYYY-MM-DD")
    for k in list(e.get("cite", [])) + [e.get("reported_in")]:
        if k and k not in papers:
            err(where, f"unknown paper key '{k}' (add it under 'papers')")


def main():
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"Cannot read {PATH}: {exc}")
        return 1
    papers = data.get("papers", {})
    for key, p in papers.items():
        for f in ["type", "authors", "title", "venue", "year"]:
            if f not in p:
                err(f"papers.{key}", f"missing '{f}'")
        if p.get("type") not in ("article", "inproceedings", "misc"):
            err(f"papers.{key}", "type must be article, inproceedings or misc")
    if not DATE.match(str(data.get("updated", ""))):
        err("updated", "must be YYYY-MM-DD")

    seen = set()
    for track in ["reid_closed", "reid_open"]:
        for i, e in enumerate(data.get(track, [])):
            where = f"{track}[{i}] {e.get('id', '?')}"
            check_common(e, where, papers, seen)
            for f in ["input", "pretrain"]:
                if f not in e:
                    err(where, f"missing '{f}' (use null if not applicable)")
            for s in REID_SCENARIOS:
                res = e.get("results", {}).get(s)
                if res is None:
                    err(where, f"missing scenario '{s}'")
                    continue
                for m in REID_METRICS:
                    v = res.get(m)
                    if not is_num(v) or not 0 <= v <= 100:
                        err(where, f"{s}.{m} must be a number between 0 and 100")
                if all(is_num(res.get(m)) for m in ["r1", "r5", "r10"]) and not res["r1"] <= res["r5"] <= res["r10"]:
                    err(where, f"{s}: expected r1 <= r5 <= r10")

    for i, e in enumerate(data.get("tracking", [])):
        where = f"tracking[{i}] {e.get('id', '?')}"
        check_common(e, where, papers, seen)
        for g, ms in TRACK_GROUPS.items():
            res = e.get("results", {}).get(g)
            if res is None:
                err(where, f"missing group '{g}'")
                continue
            for m in ms:
                v = res.get(m)
                if not is_num(v):
                    err(where, f"{g}.{m} must be a number")
                elif m in ("idf1",) and not 0 <= v <= 100:
                    err(where, f"{g}.{m} must be between 0 and 100")
                elif m not in ("mota", "idf1") and v < 0:
                    err(where, f"{g}.{m} cannot be negative")

    if errors:
        print(f"{len(errors)} problem(s) in {PATH}:")
        for e in errors:
            print("  -", e)
        return 1
    n = sum(len(data.get(t, [])) for t in ["reid_closed", "reid_open", "tracking"])
    print(f"OK: {n} results, {len(papers)} papers in {PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
