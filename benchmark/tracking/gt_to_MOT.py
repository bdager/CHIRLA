import os
import argparse
import json
from typing import Dict, List, Any

"""Convert CHIRLA tracking JSON ground-truth to MOT16 txt format.

Input JSON structure (one file per camera sequence):
{
  "<track_id>": [
      {
        "frame_init": int,  # inclusive (1-based frame index assumed)
        "frame_end": int,    # inclusive
        "bbox_init": [x1,y1,x2,y2],  # top-left & bottom-right at frame_init
        "bbox_end" : [x1,y1,x2,y2]   # top-left & bottom-right at frame_end
      }, ... (multiple disjoint segments for same id)
  ],
  ... other ids ...
}

Output lines (MOT16 style):
<frame>,
<id>,
<bb_left>,
<bb_top>,
<bb_width>,
<bb_height>,
<conf>,
<class>,
<visibility>

Confidence, class and visibility are equal to 1.00. (class = '1' correspondences to "Pedestrian" class in MOT dataset format).
Here is an example:

1, 3, 794.27, 247.59, 71.245, 174.88, 1.00, 1, 1
"""


def iter_json_files(root_dir: str):
    for r, _d, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.json'):
                yield os.path.join(r, f)


def derive_output_path(json_path: str, input_root: str, output_root: str, mot_suffix: str = '.txt') -> str:
    rel = os.path.relpath(json_path, input_root)
    base = os.path.splitext(rel)[0] + mot_suffix
    return os.path.join(output_root, base)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def box_to_ltwh(box):
    x1, y1, x2, y2 = box
    # Ensure ordering just in case
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


def convert_file(json_path: str, out_path: str, *, conf: float, verbose: bool):
    data = load_json(json_path)
    lines: List[str] = []
    for id_str, segments in data.items():
        try:
            track_id = int(id_str)
        except ValueError:
            # Skip non-integer ids
            if verbose:
                print(f"[warn] Non-integer id '{id_str}' in {json_path}, skipping")
            continue
        if not isinstance(segments, list):
            continue
        for seg in segments:
            try:
                f0 = int(seg['frame_init']) 
                f1 = int(seg['frame_end'])
                b0 = list(seg['bbox_init'])
                b1 = list(seg['bbox_end'])
            except Exception as e:
                if verbose:
                    print(f"[warn] Bad segment for id {track_id} in {json_path}: {e}")
                continue
            if f1 < f0:
                f0, f1 = f1, f0
                b0, b1 = b1, b0
            for frame in range(f0, f1 + 1):
                box = b0 if frame == f0 else b1
                x, y, w, h = box_to_ltwh(box)
                if w <= 0 or h <= 0:
                    continue
                line = f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.2f},1,1"
                lines.append(line)

    # Sort: frame then id
    lines.sort(key=lambda s: (int(s.split(',')[0]), int(s.split(',')[1])))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def main(args):
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    json_paths = sorted(iter_json_files(input_dir))
    if args.limit is not None:
        json_paths = json_paths[:args.limit]
    if not json_paths:
        print(f"No JSON files found under {input_dir}")
        return

    converted = skipped = errors = 0
    print(f"Found {len(json_paths)} JSON files. Converting...")
    for jp in json_paths:
        out_path = derive_output_path(jp, input_dir, output_dir)
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            if args.verbose:
                print(f"[skip] {out_path} exists")
            continue
        try:
            convert_file(jp, out_path, conf=args.conf, verbose=args.verbose)
            converted += 1
            if args.verbose:
                print(f"[ok] {jp} -> {out_path}")
        except Exception as e:
            errors += 1
            print(f"[err] {jp}: {e}")

    print("\n=== Summary ===")
    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")
    print(f"Output root: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CHIRLA tracking JSON to MOT16 txt format")
    parser.add_argument("--input_dir", type=str, 
        default="data/CHIRLA/benchmark/tracking", help="Root directory containing JSON annotation files")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory to write MOT txt files")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N JSON files (debug)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--conf", type=float, default=1.0, help="Confidence score to assign to each GT box")
    args = parser.parse_args()
    main(args)
