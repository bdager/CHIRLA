#!/usr/bin/env python

"""Batch apply the tracker in tracker.py to every video in the dataset.

Scans a videos root directory (default: data/CHIRLA/videos) for video files and
produces MOT-format tracking result txt files mirroring the directory structure
under an output root (default: benchmark/tracking/results).

Output line format (same as tracker.py):
frame,id,x,y,w,h,-1,-1,-1,-1

Example:
  python tracker_all_videos.py \
	  --videos-root data/CHIRLA/videos \
	  --output-root benchmark/tracking/results \
	  --model benchmark/tracking/models/yolo11n.pt
"""

import os
import argparse
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import List, Set, Optional
from ultralytics import YOLO
from boxmot import BoostTrack
from boxmot import StrongSort
from boxmot import OcSort
from boxmot import DeepOcSort
from boxmot import ByteTrack
from boxmot import BotSort

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def list_videos(root: str) -> List[str]:
	vids = []
	for r, _d, files in os.walk(root):
		for f in files:
			ext = os.path.splitext(f)[1].lower()
			if ext in VIDEO_EXTS:
				vids.append(os.path.join(r, f))
	vids.sort()
	return vids


def rel_output_path(video_path: str, videos_root: str, output_root: str) -> str:
	rel = os.path.relpath(video_path, videos_root)
	stem, _ext = os.path.splitext(rel)
	return os.path.join(output_root, stem + ".txt")


def mot_output_path(video_path: str, videos_root: str, mot_root: str, benchmark: str, split: str, tracker_name: str) -> str:
	"""Return MOT-style tracker file path.

	Pattern: <mot_root>/<benchmark>-<split>/<tracker_name>/data/<seq>_<video_stem>.txt
	Where <seq> is the directory component starting with 'seq_'.
	If the video stem already starts with seq_, it won't be duplicated.
	"""
	rel = os.path.relpath(video_path, videos_root)
	parts = rel.split(os.sep)
	seq_name = next((p for p in parts if p.startswith('seq_')), None)
	video_stem, _ext = os.path.splitext(parts[-1])
	if seq_name and not video_stem.startswith(seq_name + '_'):
		file_stem = f"{seq_name}_{video_stem}"
	else:
		file_stem = video_stem
	return os.path.join(mot_root, f"{benchmark}-{split}", tracker_name, 'data', file_stem + '.txt')


def ensure_dir(path: str):
	os.makedirs(os.path.dirname(path), exist_ok=True)


def track_single_video(model: YOLO, video_path: str, out_txt: str, *, 
	tracker, device, conf: float, verbose: bool, classes: Optional[List[int]] = None) -> int:
	
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Could not open video: {video_path}")
	frame_idx = 0
	lines_written = 0
	with open(out_txt, "w") as f_out:
		while True:
			ok, frame = cap.read()
			if not ok:
				break
			frame_idx += 1
			results = model(
				source=frame,
				classes=classes if classes is not None else [0],  # default person only
				verbose=verbose,
				device=device,
				conf=conf,  
			)
			result = results[0]

			# Build detections array: (x1,y1,x2,y2,conf,cls)
			if result.boxes is not None and len(result.boxes) > 0:
				boxes = result.boxes.xyxy.cpu().numpy()  # (N,4)
				scores = result.boxes.conf.cpu().numpy()
				labels = result.boxes.cls.cpu().numpy()
				detections = np.concatenate([boxes, scores[:, None], labels[:, None]], axis=1)
			else:
				detections = np.empty((0, 6), dtype=float)

			try:
				# Update tracker and draw results
				#   INPUT:  M X (x, y, x, y, conf, cls)
				#   OUTPUT: M X (x, y, x, y, id, conf, cls, ind)
				tracks = tracker.update(detections, frame)
			except Exception as e:
				print(f"[warn] tracker.update failed (frame {frame_idx}): {e}")
				tracks = []

			for track in tracks:
				if len(track) < 5:
					continue
				x1, y1, x2, y2, track_id = track[:5]
				try:
					track_id = int(track_id)
				except ValueError:
					continue
				if track_id < 0:
					continue
				w = x2 - x1
				h = y2 - y1
				line = f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1\n"
				f_out.write(line)
				lines_written += 1

	cap.release()
	return lines_written


def parse_args():
	ap = argparse.ArgumentParser(description="Batch run ByteTrack on all videos and save MOT txt outputs")
	ap.add_argument("--videos-root", default="data/CHIRLA/videos", help="Root directory containing sequence subfolders with videos")
	ap.add_argument("--output-root", default="benchmark/tracking/results", help="Root directory to mirror for MOT txt outputs")
	ap.add_argument("--mot-root", default=None, help="If set, also (or only if --only-mot) write MOT tracker structure root (e.g. benchmark/tracking/data/trackers)")
	ap.add_argument("--benchmark", default="CHIRLA_all", help="Benchmark name for MOT directory (e.g. CHIRLA_all)")
	ap.add_argument("--split", default="test", help="Split name for MOT directory (e.g. test or train)")
	ap.add_argument("--only-mot", action="store_true", help="Do not write mirrored output-root, only MOT structure")
	ap.add_argument("--classes", nargs='+', type=int, default=[0], help="YOLO class indices to keep (default person=0)")
	ap.add_argument("--metadata-csv", default=None, help="Optional metadata CSV containing a 'sequence' column; restrict processing to listed sequences (seq_XXX)")
	ap.add_argument("--model", default="benchmark/tracking/models/yolo11n.pt", help="YOLO model path")
	ap.add_argument("--tracker", default="boosttrack", help="Tracker to use", choices=["boosttrack", "boosttrack+", "boosttrack++", "strongsort", "ocsort", "deepocsort", "bytetrack", "botsort"])
	ap.add_argument("--reid-weights", default="benchmark/tracking/models/osnet_x0_25_msmt17.pt", help="Path to ReID weights for appearance-based trackers")
	ap.add_argument("--with-reid", action="store_true", help="Use ReID weights for appearance-based trackers")
	ap.add_argument("--device", default=0, help="Device (e.g. 0 or cpu)")
	ap.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
	ap.add_argument("--overwrite", action="store_true", help="Overwrite existing txt files")
	ap.add_argument("--limit", type=int, default=None, help="Process only first N videos (debug)")
	ap.add_argument("--verbose", action="store_true", help="Verbose per-frame model output")
	return ap.parse_args()


def load_sequences_from_metadata(csv_path: str) -> Set[str]:
	seqs: Set[str] = set()
	with open(csv_path, newline="") as f:
		reader = csv.DictReader(f)
		if 'sequence' not in reader.fieldnames:
			raise ValueError(f"metadata CSV {csv_path} lacks required 'sequence' column")
		for row in reader:
			seq = row.get('sequence')
			if seq and seq.startswith('seq_'):
				seqs.add(seq)
	return seqs


def main(args):
	videos_root = os.path.abspath(args.videos_root)
	output_root = os.path.abspath(args.output_root)
	mot_root = os.path.abspath(args.mot_root) if args.mot_root else None
	videos = list_videos(videos_root)

	# Filter by metadata sequences if provided
	if args.metadata_csv:
		meta_csv = os.path.abspath(args.metadata_csv)
		if not os.path.isfile(meta_csv):
			raise FileNotFoundError(f"Metadata CSV not found: {meta_csv}")
		allowed_seqs = load_sequences_from_metadata(meta_csv)
		if not allowed_seqs:
			print(f"Warning: no sequences extracted from {meta_csv}; nothing will be processed.")
			videos = []
		else:
			filtered = []
			for v in videos:
				# Expect path like .../seq_000/.../camera_*.avi
				parts = v.split(os.sep)
				seq_name = next((p for p in parts if p.startswith('seq_')), None)
				if seq_name and seq_name in allowed_seqs:
					filtered.append(v)
			videos = filtered
			print(f"Metadata filter: {len(videos)} videos in {len(allowed_seqs)} allowed sequences")
	if args.limit is not None:
		videos = videos[: args.limit]
	if not videos:
		print(f"No videos found under {videos_root}")
		return
	print(f"Found {len(videos)} videos. Loading model once ...")
	model = YOLO(args.model)
	converted = skipped = errors = 0
	total_lines = 0
	device = args.device

	reid_path = Path(args.reid_weights) if Path(args.reid_weights) is not None else Path('osnet_x0_25_msmt17.pt')
	tracker_cfg = args.tracker

	if args.tracker in {"boosttrack", "strongsort", "deepocsort", "botsort"} and not reid_path.is_file():
		print(f"[warn] ReID weights not found: {reid_path}")

	# Initialize tracker once
	if tracker_cfg == "boosttrack": # no ReID
		tracker = BoostTrack(reid_weights=reid_path, device=device, half=False,
			with_reid=False,    # <-- no appearance embeddings 
			use_sb=False,       # <-- no soft-BIoU
			use_ecc=True        # camera motion compensation (ECC) on is default
		)
		tracker_name = "BoostTrack"
	elif tracker_cfg == "boosttrack+": # use ReID
		tracker = BoostTrack(reid_weights=reid_path, device=device, half=False,
			with_reid=True
		)
		tracker_name = "BoostTrack+"
	elif tracker_cfg == "boosttrack++": # soft-BIoU, soft boosting, ...
		tracker = BoostTrack(reid_weights=reid_path, device=device, half=False,
			with_reid=True, use_sb=True, use_rich_s=True, use_vt=True
		)
		tracker_name = "BoostTrack++"
	elif tracker_cfg == "strongsort": # Use ReID by default (not optional)
		tracker = StrongSort(reid_weights=reid_path, device=device, half=False)
		tracker_name = "StrongSORT"
	elif tracker_cfg == "ocsort": # Not Reid-based
		tracker = OcSort()
		tracker_name = "OC-SORT"
	elif tracker_cfg == "deepocsort": # Use ReID by default (not optional)
		tracker = DeepOcSort(reid_weights=reid_path, device=device, half=False)
		tracker_name = "Deep-OC-SORT"
	elif tracker_cfg == "bytetrack": # Not Reid-based
		tracker = ByteTrack()
		tracker_name = "ByteTrack"
	elif tracker_cfg == "botsort": # ReID optionally
		tracker = BotSort(reid_weights=reid_path, device=device, half=False,
					with_reid=args.with_reid)
		tracker_name = "BoT-SORT-reid" if args.with_reid else "BoT-SORT"
	else:
		raise ValueError(f"Unknown tracker: {tracker_cfg}")
	
	for vid in videos:
		paths_to_write = []
		if not args.only_mot:
			out_path = rel_output_path(vid, videos_root, output_root)
			paths_to_write.append(out_path)
		if mot_root:
			mot_path = mot_output_path(vid, videos_root, mot_root, args.benchmark, args.split, tracker_name)
			paths_to_write.append(mot_path)
		if not paths_to_write:
			continue
		# If any existing and not overwrite, skip entirely
		if any(os.path.exists(p) for p in paths_to_write) and not args.overwrite:
			skipped += 1
			continue
		for p in paths_to_write:
			ensure_dir(p)
		try:
			# Use first path for actual tracking writing, then copy lines to others if multiple
			primary = paths_to_write[0]
			lines = track_single_video(model, vid, primary, tracker=tracker, device=device, conf=args.conf, verbose=args.verbose, classes=args.classes)
			# Copy to other destinations if more than one
			if len(paths_to_write) > 1:
				with open(primary, 'r') as fr:
					data = fr.read()
				for extra in paths_to_write[1:]:
					with open(extra, 'w') as fw:
						fw.write(data)
			converted += 1
			total_lines += lines
			print(f"[ok] {vid} -> {', '.join(paths_to_write)} ({lines} boxes)")
		except Exception as e:
			errors += 1
			print(f"[err] {vid}: {e}")
			
	print("\n=== Summary ===")
	print(f"Processed videos: {converted}")
	print(f"Skipped (exists): {skipped}")
	print(f"Errors:          {errors}")
	print(f"Total boxes:     {total_lines}")
	print(f"Output root:     {output_root if not args.only_mot else '(disabled)'}")
	if mot_root:
		print(f"MOT root:        {mot_root}/{args.benchmark}-{args.split}/{tracker_name}/data")


if __name__ == "__main__":
	main(parse_args())

