#!/usr/bin/env python3
"""
From embeddings to a CHIRLA leaderboard entry in one command.

    python make_submission.py embeddings/mymethod \\
        --method "MyMethod (ResNet50)" --input 256x128 --pretrain MSMT17

1. Checks the gallery and query files of the four scenarios (see check_embeddings.py)
2. Evaluates each scenario following the benchmark protocol (averaged over subsets)
3. Writes the metrics CSV and the leaderboard entry (JSON)

Paste the printed results into the submission form, or add the JSON entry to
docs/leaderboard/results.json in a pull request.
"""
import argparse
import csv
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from check_embeddings import DEFAULT_METADATA_DIR, SCENARIOS, check_dir, h5_filename  # noqa: E402
import export_leaderboard  # noqa: E402

TOPK = [1, 5, 10]


def evaluate_all(embeddings_dir, model_name):
    from evaluate_reid import run_evaluation
    rows = []
    for scenario in SCENARIOS:
        gallery = os.path.join(embeddings_dir, h5_filename(scenario, "gallery"))
        query = os.path.join(embeddings_dir, h5_filename(scenario, "query"))
        print(f"Evaluating {scenario}...")
        metrics = run_evaluation(gallery, query, topk=TOPK, per_subset=True, show_logs=False)
        row = {"model": model_name, "scenario": scenario,
               "gallery_file": os.path.basename(gallery), "query_file": os.path.basename(query)}
        for k, score in zip(TOPK, metrics["cmc_scores"]):
            row[f"CMC@{k}"] = round(float(score) * 100, 2)
        row["mAP"] = round(float(metrics["mAP"]) * 100, 2)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("embeddings_dir", help="Folder with the {scenario}_{split}_embeddings.h5 files")
    ap.add_argument("--method", required=True, help="Name shown on the leaderboard, e.g. 'MyMethod (ResNet50)'")
    ap.add_argument("--input", required=True, help="Input size as HxW, e.g. 256x128")
    ap.add_argument("--pretrain", default=None, help="Pretraining data, e.g. MSMT17")
    ap.add_argument("--recipe", default=None, help="Training recipe, e.g. BoT or AGW")
    ap.add_argument("--weights", default=None, help="Link to the model weights")
    ap.add_argument("--paper-key", default=None, help="BibTeX key of your paper")
    ap.add_argument("--code", default=None, help="Link to your code")
    ap.add_argument("--id", default=None, help="Unique id on the leaderboard (default: from --method)")
    ap.add_argument("--metadata-dir", default=DEFAULT_METADATA_DIR, help="Folder with the CHIRLA metadata CSVs")
    ap.add_argument("--out", default="submission.json", help="Leaderboard entry JSON (default: submission.json)")
    ap.add_argument("--csv-out", default="evaluation_results.csv",
                    help="Metrics per scenario (default: evaluation_results.csv)")
    args = ap.parse_args()

    if not os.path.isdir(args.embeddings_dir):
        sys.exit(f"Not a folder: {args.embeddings_dir}")

    print("Checking embedding files...")
    problems = check_dir(args.embeddings_dir, args.metadata_dir, require_all=True)
    if problems:
        sys.exit(f"\n{len(problems)} problem(s) found, fix them before evaluating")

    model_name = os.path.basename(os.path.normpath(args.embeddings_dir))
    rows = evaluate_all(args.embeddings_dir, model_name)
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    export_args = argparse.Namespace(
        csv=args.csv_out, model=model_name, method=args.method, id=args.id,
        paper_key=args.paper_key, code=args.code, input=args.input,
        pretrain=args.pretrain, recipe=args.recipe, weights=args.weights)
    entry = export_leaderboard.reid(export_args)[0]
    with open(args.out, "w") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n{'Scenario':24} {'R1':>6} {'R5':>6} {'R10':>6} {'mAP':>6}")
    for s in SCENARIOS:
        r = entry["results"][s]
        print(f"{s:24} {r['r1']:6.2f} {r['r5']:6.2f} {r['r10']:6.2f} {r['map']:6.2f}")
    print(f"\nMetrics: {args.csv_out}\nLeaderboard entry: {args.out}")
    # Same layout as the results box of the submission form
    print("\nResults for the submission form:")
    lines = [f'  {json.dumps(s) + ":":25} {json.dumps(entry["results"][s])}' for s in SCENARIOS]
    print("{\n" + ",\n".join(lines) + "\n}")


if __name__ == "__main__":
    main()
