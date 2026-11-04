#!/usr/bin/env python3
"""
Save and check CHIRLA Re-ID embedding files before evaluating or submitting.

Using your own model? Write each file with save_embeddings():

    from check_embeddings import save_embeddings
    save_embeddings("embeddings/mymethod/long_term_query_embeddings.h5",
                    embeddings,   # [N, D] floats
                    ids,          # [N] ints, the 'id' column of the CSV
                    paths)        # [N] the 'image_path' column of the CSV

Files are named {scenario}_{split}_embeddings.h5 with one file per metadata CSV,
e.g. benchmark/metadata/reid_long_term_query.csv -> long_term_query_embeddings.h5

Then check a folder:

    python check_embeddings.py embeddings/mymethod

Each file is compared with its metadata CSV: every image there exactly once, the
same IDs, and no empty, NaN or all-zero rows. Exits with status 1 on any problem.
"""
import argparse
import csv
import os
import sys

import h5py
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METADATA_DIR = os.path.join(SCRIPT_DIR, "..", "metadata")
SCENARIOS = ["long_term", "multi_camera_long_term", "multi_camera", "reappearance"]
SPLITS = ["gallery", "query", "train", "val"]
REQUIRED_SPLITS = ["gallery", "query"]
MAX_EXAMPLES = 5


def h5_filename(scenario, split):
    return f"{scenario}_{split}_embeddings.h5"


def save_embeddings(h5_path, embeddings, ids, paths):
    """Write embeddings in the layout the CHIRLA evaluation reads"""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    ids = np.asarray(ids)
    paths = [str(p) for p in paths]
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape [N, D], got {embeddings.shape}")
    if not (len(embeddings) == len(ids) == len(paths)):
        raise ValueError(f"embeddings, ids and paths must have the same length, "
                         f"got {len(embeddings)}, {len(ids)} and {len(paths)}")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"ids must be integers, got {ids.dtype}")
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings)
        f.create_dataset("ids", data=ids.astype(np.int32))
        f.create_dataset("paths", data=paths, dtype=h5py.string_dtype())


def benchmark_key(path, scenario):
    """Path from 'reid/<scenario>/' on, so relative and absolute paths compare equal"""
    path = path.replace("\\", "/")
    marker = f"reid/{scenario}/"
    i = path.find(marker)
    return path[i:] if i >= 0 else None


def read_metadata(csv_path):
    with open(csv_path, newline="") as f:
        return {row["image_path"]: int(row["id"]) for row in csv.DictReader(f)}


def _examples(items):
    items = list(items)
    more = f" (+{len(items) - MAX_EXAMPLES} more)" if len(items) > MAX_EXAMPLES else ""
    return ", ".join(repr(x) for x in items[:MAX_EXAMPLES]) + more


def check_file(h5_path, csv_path, scenario):
    """Return (errors, embedding dim or None) for one HDF5 file"""
    errors = []
    try:
        f = h5py.File(h5_path, "r")
    except OSError as e:
        return [f"cannot open as HDF5: {e}"], None
    with f:
        missing = [k for k in ("embeddings", "ids", "paths") if k not in f]
        if missing:
            return [f"missing dataset(s) {missing}, expected 'embeddings', 'ids' and 'paths'"], None
        emb = f["embeddings"][:]
        ids = f["ids"][:]
        paths = f["paths"][:].astype(str)

    if emb.ndim != 2:
        return [f"'embeddings' must have shape [N, D], got {emb.shape}"], None
    if not (len(emb) == len(ids) == len(paths)):
        return [f"'embeddings', 'ids' and 'paths' have different lengths "
                f"({len(emb)}, {len(ids)}, {len(paths)})"], None
    if not np.issubdtype(emb.dtype, np.floating):
        errors.append(f"'embeddings' must be floats, got {emb.dtype}")
    if not np.issubdtype(ids.dtype, np.integer):
        errors.append(f"'ids' must be integers, got {ids.dtype}")
        return errors, emb.shape[1]

    emb = emb.astype(np.float64)
    bad = ~np.isfinite(emb).all(axis=1)
    if bad.any():
        errors.append(f"{bad.sum()} row(s) with NaN or inf values, rows {_examples(np.flatnonzero(bad).tolist())}")
    zero = np.isfinite(emb).all(axis=1) & (np.linalg.norm(emb, axis=1) == 0)
    if zero.any():
        errors.append(f"{zero.sum()} all-zero row(s), usually images that failed to load, "
                      f"rows {_examples(np.flatnonzero(zero).tolist())}")

    expected = read_metadata(csv_path)
    keys = [benchmark_key(p, scenario) for p in paths]
    unknown = [p for p, k in zip(paths, keys) if k is None]
    if unknown:
        errors.append(f"{len(unknown)} path(s) without 'reid/{scenario}/' in them: {_examples(unknown)}")

    seen = {}
    dupes = set()
    for k, pid in zip(keys, ids.tolist()):
        if k is None:
            continue
        if k in seen:
            dupes.add(k)
        seen[k] = pid
    if dupes:
        errors.append(f"{len(dupes)} image(s) appear more than once: {_examples(sorted(dupes))}")

    absent = sorted(set(expected) - set(seen))
    if absent:
        errors.append(f"{len(absent)} of {len(expected)} image(s) from {os.path.basename(csv_path)} "
                      f"are missing: {_examples(absent)}")
    extra = sorted(set(seen) - set(expected))
    if extra:
        errors.append(f"{len(extra)} image(s) not in {os.path.basename(csv_path)}: {_examples(extra)}")
    wrong = sorted(k for k in set(seen) & set(expected) if seen[k] != expected[k])
    if wrong:
        errors.append(f"{len(wrong)} image(s) with a different ID than the CSV: "
                      + _examples(f"{k} has {seen[k]}, CSV says {expected[k]}" for k in wrong))
    return errors, emb.shape[1]


def check_dir(embeddings_dir, metadata_dir=DEFAULT_METADATA_DIR, scenarios=SCENARIOS,
              require_all=False, verbose=True):
    """Check every {scenario}_{split}_embeddings.h5 in a folder, return the list of problems"""
    say = print if verbose else (lambda *a, **k: None)
    problems = []
    checked = 0
    for scenario in scenarios:
        dims = {}
        for split in SPLITS:
            h5_path = os.path.join(embeddings_dir, h5_filename(scenario, split))
            csv_path = os.path.join(metadata_dir, f"reid_{scenario}_{split}.csv")
            name = os.path.basename(h5_path)
            if not os.path.exists(h5_path):
                if split in REQUIRED_SPLITS:
                    msg = f"{name}: not found"
                    if require_all:
                        problems.append(msg)
                        say(f"❌ {msg}")
                    else:
                        say(f"⚠️  {msg}, {scenario} can't be evaluated")
                continue
            if not os.path.exists(csv_path):
                problems.append(f"{name}: metadata CSV {csv_path} not found, pass --metadata-dir")
                say(f"❌ {problems[-1]}")
                continue
            checked += 1
            errors, dims[split] = check_file(h5_path, csv_path, scenario)
            if errors:
                say(f"❌ {name}")
                for e in errors:
                    say(f"     - {e}")
                problems += [f"{name}: {e}" for e in errors]
            else:
                say(f"✅ {name}")
        if dims.get("gallery") and dims.get("query") and dims["gallery"] != dims["query"]:
            msg = (f"{scenario}: gallery and query embeddings have different sizes "
                   f"({dims['gallery']} vs {dims['query']})")
            problems.append(msg)
            say(f"❌ {msg}")
    if checked == 0 and not problems:
        problems.append(f"no *_embeddings.h5 files found in {embeddings_dir}")
        say(f"❌ {problems[-1]}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("embeddings_dir", help="Folder with the {scenario}_{split}_embeddings.h5 files")
    ap.add_argument("--metadata-dir", default=DEFAULT_METADATA_DIR, help="Folder with the CHIRLA metadata CSVs")
    ap.add_argument("--require-all", action="store_true",
                    help="Fail if any scenario is missing its gallery or query file (needed for a submission)")
    args = ap.parse_args()
    if not os.path.isdir(args.embeddings_dir):
        sys.exit(f"Not a folder: {args.embeddings_dir}")

    problems = check_dir(args.embeddings_dir, args.metadata_dir, require_all=args.require_all)
    if problems:
        print(f"\n{len(problems)} problem(s) found")
        sys.exit(1)
    print("\nAll files are valid")


if __name__ == "__main__":
    main()
