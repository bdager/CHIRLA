import os
import argparse
import shutil
from typing import List, Dict, Set, Tuple

SCENARIO_MAP = {
    'brief': 'brief_occlusions',
    'brief_occlusions': 'brief_occlusions',
    'multi': 'multiple_people_occlusions',
    'multiple': 'multiple_people_occlusions',
    'multiple_people_occlusions': 'multiple_people_occlusions'
}

BENCHMARK_MAP = {
    'brief_occlusions': 'CHIRLA_brief',
    'multiple_people_occlusions': 'CHIRLA_multi'
}

def iter_camera_tracker_files(scenario_root: str) -> List[Tuple[str,str,str]]:
    """Yield (seq_name, camera_file_base, full_path) for each camera txt file within scenario_root.
    Expected structure: scenario_root/test/<TrackerName>/seq_xxx/camera_*.txt
    """
    results = []
    if not os.path.isdir(scenario_root):
        return results
    # Only 'test' currently; could generalize
    test_root = os.path.join(scenario_root, 'test')
    if not os.path.isdir(test_root):
        return results
    for tracker in os.listdir(test_root):
        tracker_dir = os.path.join(test_root, tracker)
        if not os.path.isdir(tracker_dir):
            continue
        for seq in os.listdir(tracker_dir):
            seq_dir = os.path.join(tracker_dir, seq)
            if not os.path.isdir(seq_dir):
                continue
            if not seq.startswith('seq_'):
                continue
            for f in os.listdir(seq_dir):
                if not f.endswith('.txt'):
                    continue
                full = os.path.join(seq_dir, f)
                cam_base = os.path.splitext(f)[0]  # camera_1_...
                results.append((tracker, seq, cam_base, full))
    return results


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_or_link(src: str, dst: str, overwrite: bool, symlink: bool):
    if os.path.exists(dst):
        if not overwrite:
            return False, 'exists'
        os.remove(dst)
    if symlink:
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)
    return True, 'ok'


def build_destination(base_out: str, benchmark: str, split: str, tracker: str, seq_flat: str) -> str:
    # TRACKERS_FOLDER / benchmark-split / TrackerName / data / seq_flat.txt
    dest_dir = os.path.join(base_out, f'{benchmark}-{split}', tracker, 'data')
    ensure_dir(dest_dir)
    return os.path.join(dest_dir, seq_flat + '.txt')


def process_scenario(scenario_root: str, output_root: str, split: str, overwrite: bool, symlink: bool, train_seq_ids: Set[str], include_train_test_split: bool, scenario_label: str):
    scenario_name = os.path.basename(scenario_root.rstrip('/'))
    benchmark = BENCHMARK_MAP.get(scenario_name)
    if benchmark is None:
        raise ValueError(f'Unknown scenario root name: {scenario_name}')

    records = iter_camera_tracker_files(scenario_root)
    created = skipped = 0
    per_tracker_counts: Dict[str,int] = {}
    for tracker, seq, cam_base, src_path in records:
        base_seq = seq  # seq_xxx
        seq_flat = f'{seq}_{cam_base}'  # seq_xxx_camera_...
        target_split = 'test'
        if include_train_test_split and base_seq in train_seq_ids:
            target_split = 'train'
        dst_path = build_destination(output_root, benchmark, target_split, tracker, seq_flat)
        ok, status = copy_or_link(src_path, dst_path, overwrite=overwrite, symlink=symlink)
        if ok:
            created += 1
            per_tracker_counts[tracker] = per_tracker_counts.get(tracker, 0) + 1
        else:
            skipped += 1
    print(f'[scenario {scenario_label}] tracker files processed: {created} (skipped {skipped})')
    for trk, cnt in per_tracker_counts.items():
        print(f'  - {trk}: {cnt} sequence files')


def main(args):
    input_root = os.path.abspath(args.input_dir)
    output_root = os.path.abspath(args.output_dir)
    ensure_dir(output_root)

    # Determine scenarios to process
    if args.scenario == 'all':
        scenario_keys = ['brief_occlusions', 'multiple_people_occlusions']
    else:
        key = SCENARIO_MAP.get(args.scenario.lower())
        if key is None:
            raise ValueError(f'Invalid scenario {args.scenario}')
        scenario_keys = [key]

    train_seq_ids = set([s.strip() for s in args.train_seqs.split(',') if s.strip()])
    include_train_test_split = args.include_train_test_split

    for scen in scenario_keys:
        scen_root = os.path.join(input_root, scen)
        if not os.path.isdir(scen_root):
            print(f'[warn] scenario directory missing: {scen_root}')
            continue
        process_scenario(
            scen_root,
            output_root,
            split='test',  # source is test partition currently
            overwrite=args.overwrite,
            symlink=args.symlink,
            train_seq_ids=train_seq_ids,
            include_train_test_split=include_train_test_split,
            scenario_label=scen
        )

    print('\nDone.')
    print(f'Tracker MOT folders at: {output_root}')
    print('Expected structure example:')
    print('  <output_root>/CHIRLA_brief-test/<Tracker>/data/seq_000_camera_1_xxx.txt')
    if include_train_test_split:
        print('  <output_root>/CHIRLA_brief-train/<Tracker>/data/seq_004_camera_1_xxx.txt (train sequences)')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Restructure tracker outputs into MOTChallenge format for CHIRLA scenarios.')
    ap.add_argument('--input_dir', type=str, default='benchmark/tracking/data/trackers', help='Root of existing tracker scenario directories')
    ap.add_argument('--output_dir', type=str, default='benchmark/tracking/data/trackers_mot', help='Destination root for MOT-formatted tracker results')
    ap.add_argument('--scenario', type=str, default='all', help='Scenario: brief, multi, all')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing destination files')
    ap.add_argument('--symlink', action='store_true', help='Use symlinks instead of copying (saves space)')
    ap.add_argument('--train_seqs', type=str, default='seq_004,seq_026', help='Comma-separated base seq ids that belong to train split')
    ap.add_argument('--include_train_test_split', action='store_true', help='Split into train/test outputs based on train_seqs list')
    args = ap.parse_args()
    main(args)
