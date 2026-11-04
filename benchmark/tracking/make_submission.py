#!/usr/bin/env python3
"""
From tracker results to a CHIRLA leaderboard entry in one command.

    python benchmark/tracking/make_submission.py --tracker MyTracker \\
        --method "MyTracker" --detector YOLO11n --appearance "OSNet x0.25 (MSMT17)"

Needs the ground truth from step 1 of the tracking README and your tracker's results on
CHIRLA_all in MOT format:

    <trackers-folder>/CHIRLA_all-test/<tracker>/data/<sequence>.txt

1. Checks that every test sequence has a valid results file
2. Builds the CHIRLA_brief and CHIRLA_multi results from them (best IoU match per GT box)
3. Runs TrackEval on CHIRLA_all, CHIRLA_brief and CHIRLA_multi
4. Writes the leaderboard entry (JSON) and prints the results for the submission form
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

import export_leaderboard  # noqa: E402
from generate_subset_trackers import generate_subset  # noqa: E402

SOURCE = "CHIRLA_all"
SUBSETS = ["CHIRLA_brief", "CHIRLA_multi"]
BENCHMARKS = [SOURCE] + SUBSETS
MAX_EXAMPLES = 3


def read_seqmap(path):
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    return lines[1:] if lines and lines[0] == "name" else lines


def check_results_file(path):
    """Problems TrackEval would stop on: bad rows, w/h <= 0 and an ID used twice in a frame"""
    problems = []
    seen = Counter()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.replace(" ", "").split(",")
        try:
            if len(parts) < 7:
                raise ValueError(f"{len(parts)} columns, expected 10 (frame,id,x,y,w,h,conf,-1,-1,-1)")
            frame, tid = int(float(parts[0])), int(float(parts[1]))
            w, h = float(parts[4]), float(parts[5])
            float(parts[2]), float(parts[3]), float(parts[6])
        except ValueError as e:
            problems.append(f"line {n}: {e}")
            continue
        if frame < 1:
            problems.append(f"line {n}: frame {frame}, frames start at 1")
        if w <= 0 or h <= 0:
            problems.append(f"line {n}: width and height must be positive")
        seen[(frame, tid)] += 1
    dupes = [k for k, c in seen.items() if c > 1]
    if dupes:
        problems.append(f"{len(dupes)} (frame, id) pair(s) appear more than once, e.g. frame {dupes[0][0]} id {dupes[0][1]}")
    return problems


def check_inputs(args):
    """Return a list of problems, empty when everything is in place"""
    problems = []
    script = args.trackeval_dir / "scripts" / "run_chirla_challenge.py"
    if not script.exists():
        problems.append(f"TrackEval not found at {args.trackeval_dir}, see step 0 of the tracking README "
                        "(git clone https://github.com/bdager/TrackEval.git benchmark/tracking/TrackEval)")
    for b in BENCHMARKS:
        seqmap = args.gt_root / "seqmaps" / f"{b}-{args.split}.txt"
        if not seqmap.exists() or not (args.gt_root / f"{b}-{args.split}").is_dir():
            problems.append(f"ground truth for {b}-{args.split} not found in {args.gt_root}, see step 1 of the tracking README")
    if problems:
        return problems

    data_dir = args.trackers_folder / f"{SOURCE}-{args.split}" / args.tracker / "data"
    if not data_dir.is_dir():
        return [f"tracker results not found: {data_dir}"]
    needed = set(read_seqmap(args.gt_root / "seqmaps" / f"{SOURCE}-{args.split}.txt"))
    for b in SUBSETS:
        needed |= set(read_seqmap(args.gt_root / "seqmaps" / f"{b}-{args.split}.txt"))
    missing = sorted(s for s in needed if not (data_dir / f"{s}.txt").exists())
    if missing:
        more = f" (+{len(missing) - MAX_EXAMPLES} more)" if len(missing) > MAX_EXAMPLES else ""
        problems.append(f"{len(missing)} of {len(needed)} sequence file(s) missing in {data_dir}: "
                        + ", ".join(f"{s}.txt" for s in missing[:MAX_EXAMPLES]) + more)
    for s in sorted(needed - set(missing)):
        for p in check_results_file(data_dir / f"{s}.txt")[:MAX_EXAMPLES]:
            problems.append(f"{s}.txt {p}")
    return problems


def run_trackeval(args, benchmark):
    summary = args.trackers_folder / f"{benchmark}-{args.split}" / args.tracker / "pedestrian_summary.txt"
    # A summary from an earlier run must not end up in the entry
    summary.unlink(missing_ok=True)
    cmd = [sys.executable, str(args.trackeval_dir / "scripts" / "run_chirla_challenge.py"),
           "--GT_FOLDER", str(args.gt_root), "--TRACKERS_FOLDER", str(args.trackers_folder),
           "--BENCHMARK", benchmark, "--SPLIT_TO_EVAL", args.split,
           "--TRACKERS_TO_EVAL", args.tracker, "--METRICS", "CLEAR", "Identity",
           "--USE_PARALLEL", "False", "--PRINT_CONFIG", "False", "--PLOT_CURVES", "False"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not summary.exists():
        sys.exit(f"TrackEval failed on {benchmark}:\n{(result.stdout + result.stderr)[-3000:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracker", required=True, help="Tracker folder name, e.g. BoT-SORT-reid")
    ap.add_argument("--method", required=True, help="Name shown on the leaderboard")
    ap.add_argument("--detector", required=True, help="Detector, e.g. YOLO11n")
    ap.add_argument("--appearance", default=None, help="Appearance model, if any")
    ap.add_argument("--paper-key", default=None, help="BibTeX key of your paper")
    ap.add_argument("--code", default=None, help="Link to your code")
    ap.add_argument("--id", default=None, help="Unique id on the leaderboard (default: from --method)")
    ap.add_argument("--gt-root", type=Path, default=SCRIPT_DIR / "data" / "gt", help="Ground truth from step 1")
    ap.add_argument("--trackers-folder", type=Path, default=SCRIPT_DIR / "data" / "trackers",
                    help="Folder with <benchmark>-<split>/<tracker>/data/")
    ap.add_argument("--trackeval-dir", type=Path, default=SCRIPT_DIR / "TrackEval", help="Clone of the TrackEval fork")
    ap.add_argument("--split", default="test", help="Split to evaluate (leaderboard results use test)")
    ap.add_argument("--min-iou", type=float, default=0.5, help="IoU for matching GT boxes in brief/multi")
    ap.add_argument("--out", default="submission.json", help="Leaderboard entry JSON (default: submission.json)")
    args = ap.parse_args()
    for name in ("gt_root", "trackers_folder", "trackeval_dir"):
        setattr(args, name, getattr(args, name).resolve())

    print("Checking inputs...")
    problems = check_inputs(args)
    if problems:
        for p in problems:
            print(f"❌ {p}")
        sys.exit(f"\n{len(problems)} problem(s) found, fix them before evaluating")
    print("✅ Ground truth, TrackEval and tracker results found")

    print("\nBuilding CHIRLA_brief and CHIRLA_multi results...")
    for b in SUBSETS:
        if not generate_subset(args.gt_root, args.trackers_folder, SOURCE, b, args.split, args.tracker, args.min_iou):
            sys.exit(f"Could not build {b} results")

    for b in BENCHMARKS:
        print(f"Running TrackEval on {b}-{args.split}...")
        run_trackeval(args, b)

    export_args = argparse.Namespace(
        trackers_folder=str(args.trackers_folder), tracker=args.tracker, split=args.split,
        method=args.method, appearance=args.appearance, detector=args.detector,
        id=args.id, paper_key=args.paper_key, code=args.code)
    entry = export_leaderboard.tracking(export_args)[0]
    with open(args.out, "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
        f.write("\n")

    r = entry["results"]
    print(f"\n{'Benchmark':28} {'MOTA':>7} {'IDF1':>7} {'IDSW':>6}")
    for g in ("all", "brief_occlusions", "multiple_people_occlusions"):
        print(f"{g:28} {r[g]['mota']:7.2f} {r[g]['idf1']:7.2f} {r[g]['idsw']:6d}")
    print(f"\nLeaderboard entry: {args.out}")
    # Same layout as the results box of the submission form
    print("\nResults for the submission form:")
    lines = [f'  {json.dumps(g) + ":":29} {json.dumps(r[g])}' for g in r]
    print("{\n" + ",\n".join(lines) + "\n}")


if __name__ == "__main__":
    main()
