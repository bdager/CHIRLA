import os
import argparse
import json
from typing import Dict, List, Any, Tuple, Optional
import re

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


def derive_sequence_name(json_path: str, input_root: str) -> str:
    """Derive flattened sequence name seq_xxx_camera_x from path.

    Expected relative path: .../seq_xxx/camera_x.json (possibly nested deeper).
    """
    rel = os.path.relpath(json_path, input_root)
    parts = rel.split(os.sep)
    # find seq_* and camera_* tokens
    seq_token = None
    cam_token = os.path.splitext(parts[-1])[0]  # filename without .json
    for p in parts:
        if p.startswith('seq_'):
            seq_token = p
            break
    if seq_token is None:
        # fallback: use first directory and camera token
        seq_token = parts[-2] if len(parts) >= 2 else 'seq_unknown'
    return f"{seq_token}_{cam_token}"


def build_sequence_folder(output_root: str, benchmark_split_dir: str, sequence_name: str) -> str:
    seq_dir = os.path.join(output_root, benchmark_split_dir, sequence_name)
    gt_dir = os.path.join(seq_dir, 'gt')
    os.makedirs(gt_dir, exist_ok=True)
    return os.path.join(gt_dir, 'gt.txt')


def write_seqinfo(seq_dir: str, seq_name: str, seq_length: int, width: int = 1920, height: int = 1080,
                  frame_rate: int = 30):
    ini_path = os.path.join(seq_dir, 'seqinfo.ini')
    if os.path.exists(ini_path):
        # Overwrite with updated length if different
        pass
    # img_dir = os.path.join(seq_dir, 'img1')
    # os.makedirs(img_dir, exist_ok=True)
    content = (
        "[Sequence]\n"
        f"name={seq_name}\n"
        "imDir=imgs\n"
        f"frameRate={frame_rate}\n"
        f"seqLength={seq_length}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.png\n"
    )
    with open(ini_path, 'w', encoding='utf-8') as f:
        f.write(content)


def scenario_to_benchmark(scenario: str) -> str:
    sc = scenario.lower()
    if sc.startswith('brief'):
        return 'CHIRLA_brief'
    if sc.startswith('multi') or 'multiple' in sc:
        return 'CHIRLA_multi'
    raise ValueError(f"Unknown scenario '{scenario}'. Use 'brief' or 'multi'.")


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


def convert_file(json_path: str, out_path: str, *, conf: float, verbose: bool) -> Optional[int]:
    data = load_json(json_path)
    lines: List[str] = []
    max_frame = 0
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
            # Endpoint-only representation: only start and end frames.
            for frame, box in [(f0, b0), (f1, b1)]:
                x, y, w, h = box_to_ltwh(box)
                if w <= 0 or h <= 0:
                    continue
                line = f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.2f},1,1"
                lines.append(line)
                if frame > max_frame:
                    max_frame = frame

    # Sort: frame then id
    lines.sort(key=lambda s: (int(s.split(',')[0]), int(s.split(',')[1])))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return max_frame if lines else None


def update_seqmap(seqmaps_dir: str, benchmark: str, split: str, sequences: List[str]):
    os.makedirs(seqmaps_dir, exist_ok=True)
    seqmap_path = os.path.join(seqmaps_dir, f"{benchmark}-{split}.txt")
    sequences = sorted(sequences)
    with open(seqmap_path, 'w', encoding='utf-8') as f:
        f.write('name\n')
        for s in sequences:
            f.write(f"{s}\n")
    print(f"[seqmap] wrote {seqmap_path} ({len(sequences)} sequences)")


def scan_sequences_for_benchmark(output_dir: str, benchmark: str) -> List[str]:
    seqs = []
    for split in ['train', 'test']:
        root = os.path.join(output_dir, f"{benchmark}-{split}")
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            seq_path = os.path.join(root, name, 'gt', 'gt.txt')
            if os.path.isfile(seq_path):
                seqs.append((split, name))
    return seqs


def main(args):
    input_dir = os.path.join(args.input_dir, args.scenario, args.split)
    output_dir = args.output_dir
    scenario = args.scenario
    split = args.split

    benchmark = scenario_to_benchmark(scenario)
    benchmark_split_dir = f"{benchmark}-{split}"
    print(f"[config] scenario={scenario} -> benchmark={benchmark} split={split}")

    json_paths = sorted(iter_json_files(input_dir))
    if args.limit is not None:
        json_paths = json_paths[:args.limit]
    if not json_paths:
        print(f"No JSON files found under {input_dir}")
        return

    converted = skipped = errors = 0
    seq_max_frames: Dict[str, int] = {}
    sequence_names: List[str] = []
    print(f"Found {len(json_paths)} JSON files. Converting to {benchmark_split_dir} ...")
    for jp in json_paths:
        seq_name = derive_sequence_name(jp, input_dir)
        out_path = build_sequence_folder(output_dir, benchmark_split_dir, seq_name)
        sequence_names.append(seq_name)
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            if args.verbose:
                print(f"[skip] {out_path} exists")
            continue
        try:
            max_frame = convert_file(jp, out_path, conf=args.conf, verbose=args.verbose)
            if max_frame:
                seq_max_frames[seq_name] = max(seq_max_frames.get(seq_name, 0), max_frame)
            converted += 1
            if args.verbose:
                print(f"[ok] {jp} -> {out_path} (max_frame={seq_max_frames.get(seq_name)})")
        except Exception as e:
            errors += 1
            print(f"[err] {jp}: {e}")

    # Write seqinfo.ini for each sequence
    unique_sequences = sorted(set(sequence_names))
    for seq in unique_sequences:
        seq_dir = os.path.join(output_dir, benchmark_split_dir, seq)
        length = seq_max_frames.get(seq, 0)
        if length <= 0:  # fallback: parse frames from file
            gt_file = os.path.join(seq_dir, 'gt', 'gt.txt')
            if os.path.isfile(gt_file):
                frames = []
                with open(gt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            frame = int(line.split(',')[0])
                            frames.append(frame)
                        except Exception:
                            pass
                if frames:
                    length = max(frames)
        length = max(length, 1)
        write_seqinfo(seq_dir, seq, length)
    print(f"[seqinfo] wrote seqinfo.ini for {len(unique_sequences)} sequences")

    # Update seqmap for this benchmark-split
    update_seqmap(args.seqmaps_dir, benchmark, split, unique_sequences)

    print("\n=== Summary ===")
    print(f"Converted: {converted}")
    print(f"Skipped:   {skipped}")
    print(f"Errors:    {errors}")
    print(f"Benchmark split dir: {os.path.join(output_dir, benchmark_split_dir)}")
    print(f"Seqmaps dir: {args.seqmaps_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CHIRLA tracking JSON to MOT16 txt format with scenario-aware output and seqmaps generation")
    parser.add_argument("--input_dir", type=str,
                        default="data/CHIRLA/benchmark/tracking", help="Root directory containing JSON annotation files")
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory to write MOT structures (GT root)")
    parser.add_argument("--scenario", type=str, default='all', 
        help="Scenario: brief or multi (accepts 'brief_occlusions', 'multiple_people_occlusions')")
    parser.add_argument("--split", type=str, default='all', 
        choices=['train', 'test', 'all'], help="Dataset split")
    parser.add_argument("--seqmaps_dir", type=str, default=None, help="Directory to store seqmaps (default: <output_dir>/seqmaps)")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N JSON files (debug)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--conf", type=float, default=1.0, help="Confidence score to assign to each GT box")
    args = parser.parse_args()
    if args.seqmaps_dir is None:
        args.seqmaps_dir = os.path.join(args.output_dir, 'seqmaps')

    if args.scenario == 'all':
        for sc in ['brief_occlusions', 'multiple_people_occlusions']:
            args.scenario = sc
            if args.split == 'all':
                for sp in ['train', 'test']:
                    args.split = sp
                    main(args)
                args.split = 'all'
            else:
                main(args)
    else:
        if args.split == 'all':
            for sp in ['train', 'test']:
                args.split = sp
                main(args)
        else:   
            main(args)

