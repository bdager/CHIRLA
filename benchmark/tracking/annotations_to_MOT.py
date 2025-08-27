import os
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set

# Ensure project root (parent of 'benchmark') is on sys.path when running as a script
_CURR_DIR = Path(__file__).resolve()
_PROJECT_ROOT = _CURR_DIR.parents[2]  # .../CHIRLA
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    # Absolute import (works when run as module: python -m benchmark.tracking_scripts.all_json_to_MOT)
    from benchmark.tracking.utils.convert_json_to_MOT import convert_to_txt, read_json_file
except ModuleNotFoundError:
    # Fallback relative import if executed differently
    from benchmark.tracking.utils.convert_json_to_MOT import convert_to_txt, read_json_file  # type: ignore

# Read all files in the seq directory
def iter_json_files(root_dir):
    """Yield absolute paths to all .json annotation files under root_dir."""
    for r, _d, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.json'):
                yield os.path.join(r, f)

def derive_flat_sequence_name(json_path: str, input_root: str) -> str:
    """Flatten seq/camera into seq_xxx_camera_y name for MOT sequence."""
    rel = os.path.relpath(json_path, input_root)
    parts = rel.split(os.sep)
    seq_tok = None
    cam_tok = os.path.splitext(parts[-1])[0]
    for p in parts:
        if p.startswith('seq_'):
            seq_tok = p
            break
    if not seq_tok:
        seq_tok = 'seq_unknown'
    return f"{seq_tok}_{cam_tok}"


def mot_gt_path(root: str, benchmark_split: str, seq_name: str) -> str:
    seq_dir = os.path.join(root, benchmark_split, seq_name, 'gt')
    os.makedirs(seq_dir, exist_ok=True)
    return os.path.join(seq_dir, 'gt.txt')


def write_seqinfo(seq_root: str, seq_name: str, seq_length: int, width: int = 1920, height: int = 1080, frame_rate: int = 30):
    # img_dir = os.path.join(seq_root, 'img1')
    # os.makedirs(img_dir, exist_ok=True)
    ini_path = os.path.join(seq_root, 'seqinfo.ini')
    content = (
        '[Sequence]\n'
        f'name={seq_name}\n'
        'imDir=imgs\n'
        f'frameRate={frame_rate}\n'
        f'seqLength={seq_length}\n'
        f'imWidth={width}\n'
        f'imHeight={height}\n'
        'imExt=.jpg\n'
    )
    with open(ini_path, 'w', encoding='utf-8') as f:
        f.write(content)


def write_seqmap(seqmaps_dir: str, benchmark_split: str, sequences: List[str]):
    os.makedirs(seqmaps_dir, exist_ok=True)
    path = os.path.join(seqmaps_dir, f'{benchmark_split}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('name\n')
        for s in sorted(sequences):
            f.write(f'{s}\n')
    print(f'[seqmap] wrote {path} ({len(sequences)} sequences)')


def main(args):
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    benchmark = 'CHIRLA_all'
    train_seq_ids: Set[str] = set([s.strip() for s in args.train_seqs.split(',') if s.strip()])
    seqmaps_dir = os.path.join(output_dir, 'seqmaps')

    json_paths = sorted(iter_json_files(input_dir))
    if args.limit is not None:
        json_paths = json_paths[:args.limit]
    if not json_paths:
        print(f'No JSON files found under {input_dir}')
        return

    converted = skipped = errors = 0
    split_seq_frames: Dict[str, Dict[str, int]] = {'train': {}, 'test': {}}
    print(f'[config] benchmark={benchmark} train_ids={sorted(train_seq_ids)} total_json={len(json_paths)}')
    for jp in json_paths:
        seq_name = derive_flat_sequence_name(jp, input_dir)
        base_seq = seq_name.split('_')[0] + '_' + seq_name.split('_')[1]
        split = 'train' if base_seq in train_seq_ids else 'test'
        benchmark_split = f'{benchmark}-{split}'
        gt_path = mot_gt_path(output_dir, benchmark_split, seq_name)
        if not args.overwrite and os.path.exists(gt_path):
            skipped += 1
            if args.verbose:
                print(f'[skip] {gt_path} exists')
            continue
        try:
            data = read_json_file(jp)
            # Original JSON may have trailing empty frames (no detections). We want seqLength to reflect
            # the true last frame index present in the annotation JSON keys, not just the last frame
            # that contains a detection written to gt.txt.
            try:
                orig_max_frame = max(int(k) for k in data.keys()) if data else 0
            except Exception:
                orig_max_frame = 0
            convert_to_txt(data, gt_path)
            # We still compute max frame with detections (for diagnostics only)
            det_max_frame = 0
            with open(gt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        fr = int(line.split(',')[0])
                        if fr > det_max_frame:
                            det_max_frame = fr
                    except Exception:
                        continue
            # Store the original maximum frame index (may be > det_max_frame if trailing empty frames exist)
            seq_len = orig_max_frame
            split_seq_frames[split][seq_name] = max(split_seq_frames[split].get(seq_name, 0), seq_len)
            converted += 1
            if args.verbose:
                print(f'[ok] ({split}) {jp} -> {gt_path} (seqLength(orig_json_max)={seq_len} det_max={det_max_frame})')
        except Exception as e:
            errors += 1
            print(f'[err] Failed {jp}: {e}')

    # Write seqinfo.ini and seqmaps per split
    for split, seq_frames in split_seq_frames.items():
        if not seq_frames:
            continue
        benchmark_split = f'{benchmark}-{split}'
        for seq_name, length in seq_frames.items():
            seq_root = os.path.join(output_dir, benchmark_split, seq_name)
            write_seqinfo(seq_root, seq_name, max(length, 1))
        write_seqmap(seqmaps_dir, f'{benchmark}-{split}', list(seq_frames.keys()))
        print(f'[seqinfo] {split}: wrote {len(seq_frames)} seqinfo.ini files')

    print('\n=== Summary ===')
    print(f'Converted: {converted}')
    print(f'Skipped:   {skipped}')
    print(f'Errors:    {errors}')
    for split, seq_frames in split_seq_frames.items():
        print(f'  {split}: {len(seq_frames)} sequences')
    print(f'Seqmaps dir: {seqmaps_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CHIRLA annotation JSON to MOT format with fixed train/test split for CHIRLA_all benchmark.")
    parser.add_argument("--input_dir", type=str, default="data/CHIRLA/annotations", help="Root annotations directory (contains seq_* folders)")
    parser.add_argument("--output_dir", type=str, default="benchmark/tracking_MOT", help="Root output directory (will create CHIRLA_all-all)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N JSON files (debug)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing txt outputs")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-file logging")
    parser.add_argument("--train_seqs", type=str, default="seq_004,seq_026", help="Comma-separated base sequence ids to assign to train split (others go to test)")
    args = parser.parse_args()
    main(args)
