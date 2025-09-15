"""Generate tracker data for CHIRLA_brief and CHIRLA_multi benchmarks

This script builds synthetic tracker result files for sparse benchmarks (brief, multi)
by sampling existing tracker outputs produced for the full CHIRLA_all benchmark.

For every sequence present in the target benchmark GT, and for every GT row (frame,id,bbox),
the script looks up the corresponding CHIRLA_all tracker file for each tracker method and:
  * Collects all tracker detections at that frame.
  * Computes IoU with the GT bbox for each detection.
  * Selects the detection with maximum IoU (if IoU > 0, configurable min_iou).
  * Writes a MOTChallenge-format line with: frame, tracker_id, x, y, w, h, iou_as_conf, -1,-1,-1

If no detection exists at that frame (or max IoU below threshold), nothing is written
for that GT object (i.e., it will count as a miss in evaluation consistent with absence).

Directory layout produced mirrors TrackEval expected structure:
  data/trackers/CHIRLA_<benchmark>-<split>/<TrackerName>/data/<sequence>.txt

Usage:
  python generate_subset_trackers.py \\
      --source_benchmark CHIRLA_all \\
      --target_benchmarks CHIRLA_brief CHIRLA_multi \\
      --splits test train \\
      --trackers BoT-SORT ByteTrack \\
      --min_iou 0.5 \\
      --gt_root benchmark/tracking/data/gt \\
      --tracker_root benchmark/tracking/data/trackers

Assumptions:
  * Source tracker data exists only (or at least) for the test split of CHIRLA_all; if not,
    script will attempt both train/test when specified.
  * GT files use MOT format: frame,id,x,y,w,h,conf,*,*
  * Tracker files use standard MOT format: frame,id,x,y,w,h,conf,*,*,*

"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set


@dataclass
class Box:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float

    def to_line(self, iou_as_conf: float) -> str:
        # return f"{self.frame},{self.track_id},{self.x:.2f},{self.y:.2f},{self.w:.2f},{self.h:.2f},{iou_as_conf:.6f},-1,-1,-1"
        return f"{self.frame},{self.track_id},{self.x:.2f},{self.y:.2f},{self.w:.2f},{self.h:.2f},-1,-1,-1,-1"


def iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = min(ax2, bx2) - max(ax, bx)
    if inter_w <= 0:
        return 0.0
    inter_h = min(ay2, by2) - max(ay, by)
    if inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_tracker_file(path: Path) -> Dict[int, List[Box]]:
    frames: Dict[int, List[Box]] = {}
    if not path.exists():
        return frames
    with path.open('r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            try:
                frame = int(row[0])
                track_id = int(float(row[1]))  # sometimes ids are float formatted
                x, y, w, h = map(float, row[2:6])
                conf = float(row[6]) if len(row) > 6 else -1.0
            except ValueError:
                continue
            frames.setdefault(frame, []).append(Box(frame, track_id, x, y, w, h, conf))
    return frames


def load_gt_file(path: Path) -> List[Tuple[int, int, Tuple[float, float, float, float]]]:
    rows: List[Tuple[int, int, Tuple[float, float, float, float]]] = []
    with path.open('r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            try:
                frame = int(row[0])
                obj_id = int(row[1])
                x, y, w, h = map(float, row[2:6])
            except ValueError:
                continue
            rows.append((frame, obj_id, (x, y, w, h)))
    return rows


def synthesize_sequence(gt_seq_dir: Path, source_tracker_file: Path, out_file: Path, min_iou: float) -> int:
    gt_file = gt_seq_dir / 'gt' / 'gt.txt'
    if not gt_file.exists():
        return 0
    gt_rows = load_gt_file(gt_file)
    tracker_frames = load_tracker_file(source_tracker_file)
    written = 0
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open('w') as f:
        for frame, obj_id, gt_box in gt_rows:
            candidates = tracker_frames.get(frame, [])
            if not candidates:
                continue
            best_iou = 0.0
            best_box: Box | None = None
            for box in candidates:
                current_iou = iou((box.x, box.y, box.w, box.h), gt_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_box = box
            if best_box is None or best_iou < min_iou:
                continue
            f.write(best_box.to_line(best_iou) + "\n")
            written += 1
    return written


def dedupe_frame_ids(file_path: Path) -> int:
    """Remove duplicate (frame,id) rows keeping the first occurrence.

    Returns number of removed duplicates.
    """
    if not file_path.exists():
        return 0
    seen: Set[Tuple[int, int]] = set()
    kept_lines: List[str] = []
    removed = 0
    with file_path.open('r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                frame = int(parts[0])
                tid = int(float(parts[1]))
            except ValueError:
                continue
            key = (frame, tid)
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            kept_lines.append(line.rstrip('\n'))
    if removed:
        with file_path.open('w') as f:
            f.write('\n'.join(kept_lines) + ('\n' if kept_lines else ''))
    return removed


def main():
    parser = argparse.ArgumentParser(description="Generate subset tracker data for sparse CHIRLA benchmarks")
    parser.add_argument('--source_benchmark', default='CHIRLA_all', help='Benchmark providing full tracker outputs')
    parser.add_argument('--target_benchmarks', nargs='+', default=['CHIRLA_brief', 'CHIRLA_multi'])
    parser.add_argument('--splits', nargs='+', default=['test'], help='Splits to process (e.g., test train)')
    parser.add_argument('--trackers', nargs='+', required=True, help='Tracker method names (directory names)')
    parser.add_argument('--gt_root', default='benchmark/tracking/data/gt')
    parser.add_argument('--tracker_root', default='benchmark/tracking/data/trackers')
    parser.add_argument('--min_iou', type=float, default=0.5)
    parser.add_argument('--dry_run', action='store_true', help='Only report planned creations')
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    tracker_root = Path(args.tracker_root)
    source_root = tracker_root

    for benchmark in args.target_benchmarks:
        for split in args.splits:
            gt_split_dir = gt_root / f"{benchmark}-{split}"
            if not gt_split_dir.exists():
                print(f"[WARN] Missing GT split dir: {gt_split_dir}")
                continue
            # sequences are directories under gt_split_dir
            seq_dirs = [p for p in gt_split_dir.iterdir() if p.is_dir()]
            for tracker_name in args.trackers:
                # Source tracker location (we try same split under source benchmark)
                source_tracker_dir = source_root / f"{args.source_benchmark}-{split}" / tracker_name / 'data'
                if not source_tracker_dir.exists():
                    print(f"[WARN] Source tracker dir missing: {source_tracker_dir}; skipping {tracker_name} {split}")
                    continue
                out_tracker_dir = tracker_root / f"{benchmark}-{split}" / tracker_name / 'data'
                created_total = 0
                written_total = 0
                for seq_dir in seq_dirs:
                    seq_name = seq_dir.name  # e.g., seq_000_camera_1_...
                    source_file = source_tracker_dir / f"{seq_name}.txt"
                    out_file = out_tracker_dir / f"{seq_name}.txt"
                    created_total += 1
                    if args.dry_run:
                        continue
                    written = synthesize_sequence(seq_dir, source_file, out_file, args.min_iou)
                    # Deduplicate frame-id pairs post generation (simple safeguard)
                    dups = dedupe_frame_ids(out_file)
                    if dups:
                        print(f"[DEDUP] Removed {dups} duplicate frame-id rows in {out_file.name}")
                    written_total += (written - dups)
                if args.dry_run:
                    print(f"[DRY] {benchmark}-{split} {tracker_name}: would process {created_total} sequences -> {out_tracker_dir}")
                else:
                    print(f"[OK] {benchmark}-{split} {tracker_name}: processed {created_total} sequences, wrote {written_total} rows -> {out_tracker_dir}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted', file=sys.stderr)
        sys.exit(130)
