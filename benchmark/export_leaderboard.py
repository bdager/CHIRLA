#!/usr/bin/env python3
"""
Turn CHIRLA evaluation output into a leaderboard entry (the format of
docs/leaderboard/results.json), ready to paste into the submission form or a pull request.

Re-ID: reads the CSV written by benchmark/reid/run_all_evaluations.py
  python benchmark/export_leaderboard.py reid --csv evaluation_results.csv \
      --method "MyMethod (ResNet50)" --input 256x128 --pretrain MSMT17

  The CSV can hold several models; use --model to pick one (default: all, one entry each).

Tracking: reads TrackEval's pedestrian_summary.txt for the three benchmarks
  python benchmark/export_leaderboard.py tracking \
      --trackers-folder benchmark/tracking/data/trackers --tracker BoT-SORT-reid \
      --method "BoT-SORT" --appearance "OSNet x0.25 (MSMT17)" --detector YOLO11n

No third-party packages needed.
"""
import argparse
import csv
import datetime
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REID_SCENARIOS = ["long_term", "multi_camera_long_term", "multi_camera", "reappearance"]
TRACK_BENCHMARKS = {"all": "CHIRLA_all", "brief_occlusions": "CHIRLA_brief", "multiple_people_occlusions": "CHIRLA_multi"}
TRACK_FIELDS = {"mota": "MOTA", "idf1": "IDF1", "mt": "MT", "ml": "ML", "fn": "CLR_FN", "fp": "CLR_FP", "idsw": "IDSW", "fm": "Frag"}
TRACK_KEEP = {"all": list(TRACK_FIELDS), "brief_occlusions": ["mota", "idf1", "idsw"],
              "multiple_people_occlusions": ["mota", "idf1", "idsw"]}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def base_entry(args):
    return {
        "id": args.id or slug(args.method),
        "method": args.method,
        "cite": [args.paper_key] if args.paper_key else [],
        "reported_in": args.paper_key or "",
        "kind": "community",
        "verified": False,
        "added": datetime.date.today().isoformat(),
        **({"code": args.code} if args.code else {}),
    }


def normalize_scenario(name):
    name = name.lower()
    for prefix in ("reid_",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def reid(args):
    rows = list(csv.DictReader(open(args.csv, newline="")))
    by_model = defaultdict(dict)
    for r in rows:
        scenario = normalize_scenario(r["scenario"])
        if scenario not in REID_SCENARIOS:
            print(f"skipping unknown scenario '{r['scenario']}'", file=sys.stderr)
            continue
        by_model[r["model"]][scenario] = {
            "r1": float(r["CMC@1"]), "r5": float(r["CMC@5"]), "r10": float(r["CMC@10"]), "map": float(r["mAP"])}
    models = [args.model] if args.model else sorted(by_model)
    entries = []
    for m in models:
        res = by_model.get(m)
        if not res:
            sys.exit(f"model '{m}' not found in {args.csv}; available: {sorted(by_model)}")
        missing = [s for s in REID_SCENARIOS if s not in res]
        if missing:
            print(f"warning: {m} has no results for {missing}", file=sys.stderr)
        e = base_entry(args)
        if len(models) > 1:
            e["id"] = slug(f"{e['id']}-{m}")
            e["method"] = f"{args.method} ({m})"
        e.update({"recipe": args.recipe, "pretrain": args.pretrain, "input": args.input.replace("x", "×"),
                  "weights": args.weights,
                  "results": {s: res[s] for s in REID_SCENARIOS if s in res}})
        entries.append(e)
    return entries


def read_trackeval_summary(path):
    lines = [l.split() for l in Path(path).read_text().splitlines() if l.strip()]
    if len(lines) < 2:
        sys.exit(f"cannot parse {path}")
    return dict(zip(lines[0], (float(v) for v in lines[1])))


def tracking(args):
    root = Path(args.trackers_folder)
    results = {}
    for group, bench in TRACK_BENCHMARKS.items():
        path = root / f"{bench}-{args.split}" / args.tracker / "pedestrian_summary.txt"
        if not path.exists():
            print(f"warning: {path} not found, skipping {group}", file=sys.stderr)
            continue
        s = read_trackeval_summary(path)
        out = {}
        for key in TRACK_KEEP[group]:
            v = s.get(TRACK_FIELDS[key])
            if v is None:
                sys.exit(f"{path} has no '{TRACK_FIELDS[key]}' column")
            if key in ("mota", "idf1"):
                v = v * 100 if abs(v) <= 1.5 else v      # TrackEval may write fractions or percentages
                out[key] = round(v, 2)
            else:
                out[key] = int(round(v))
        results[group] = out
    e = base_entry(args)
    e.update({"appearance": args.appearance, "detector": args.detector, "results": results})
    return [e]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)
    for name in ("reid", "tracking"):
        p = sub.add_parser(name)
        p.add_argument("--method", required=True, help="name shown on the leaderboard")
        p.add_argument("--id", help="unique id (default: from --method)")
        p.add_argument("--paper-key", help="BibTeX key of your paper (add the paper under 'papers' too)")
        p.add_argument("--code", help="link to your code")
        p.add_argument("--out", help="write JSON here instead of printing it")
        if name == "reid":
            p.add_argument("--csv", required=True, help="CSV from run_all_evaluations.py")
            p.add_argument("--model", help="model name in the CSV (default: every model)")
            p.add_argument("--input", default="256x128")
            p.add_argument("--pretrain", default=None)
            p.add_argument("--recipe", default=None)
            p.add_argument("--weights", default=None)
        else:
            p.add_argument("--trackers-folder", required=True, help="TrackEval TRACKERS_FOLDER")
            p.add_argument("--tracker", required=True, help="tracker folder name, e.g. BoT-SORT-reid")
            p.add_argument("--split", default="test")
            p.add_argument("--appearance", default=None)
            p.add_argument("--detector", default=None)
    args = ap.parse_args()
    entries = reid(args) if args.task == "reid" else tracking(args)
    text = json.dumps(entries[0] if len(entries) == 1 else entries, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
