#!/usr/bin/env python3
"""
Build the assets for the CHIRLA project page explorer.

Reads the raw CHIRLA release (videos/ and annotations/) and writes:
  <out>/ids/id_XX/<seq>_c<cam>.jpg        person crop (best few per identity and session)
  <out>/ids/id_XX/<seq>_c<cam>_full.jpg   the full camera frame that crop came from
  <out>/sync/<seq>/c<cam>.jpg             one synchronized moment per session, all cameras
  <out>/explorer_data.js                  manifest the page loads (window.CHIRLA_EXPLORER)

Frames are saved clean; the page draws the boxes itself from the manifest.

Curation: put an explorer_exclude.txt next to this script (or pass --exclude).
One entry per line, '#' starts a comment:
  id_06/seq_024_c3     drop this crop; the next-best camera of that session replaces it
  id_17                drop an identity entirely
  sync/seq_024         drop a session from the seven-camera view
  sync/seq_024@1530    use frame 1530 for that session's seven-camera view
Open the page with #curate at the end of the URL to see each item's key.

Usage:
  python tools/build_explorer.py --data /path/to/CHIRLA --out docs/assets/explorer

Identities with fewer than --min-per-id crops get extra ones from other moments
(file names ending in _f<frame>), so people seen in only one session still get a few.

Requires: opencv-python
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
CAM_RE = re.compile(r"camera_(\d+)")


def parse_meta(path):
    cam = CAM_RE.search(path.stem)
    date = DATE_RE.search(path.stem)
    return (int(cam.group(1)) if cam else None, date.group(1) if date else None)


def find_video(videos_dir, seq, ann_path, cam):
    exact = videos_dir / seq / (ann_path.stem + ".avi")
    if exact.exists():
        return exact
    matches = sorted((videos_dir / seq).glob(f"camera_{cam}*"))
    return matches[0] if matches else None


def load_exclude(path):
    crops, ids, sync_skip, sync_frame = set(), set(), set(), {}
    if not path or not path.exists():
        return crops, ids, sync_skip, sync_frame
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("sync/"):
            rest = line[5:]
            if "@" in rest:
                seq, frame = rest.split("@", 1)
                sync_frame[seq.strip()] = int(frame)
            else:
                sync_skip.add(rest)
        elif "/" in line:
            crops.add(line)
        else:
            ids.add(int(line.replace("id_", "")))
    print(f"exclusions from {path}: {len(crops)} crops, {len(ids)} identities, "
          f"{len(sync_skip)} sync sessions skipped, {len(sync_frame)} sync frames forced")
    return crops, ids, sync_skip, sync_frame


def read_frames(video_path, frames):
    """Yield (frame_number_1based, image) for the requested frames, in order."""
    cap = cv2.VideoCapture(str(video_path))
    for f in sorted(set(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, img = cap.read()
        if ok:
            yield f, img
    cap.release()


def save_resized_width(img, path, width, quality):
    scale = width / img.shape[1]
    out = cv2.resize(img, (width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, quality])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="CHIRLA root with videos/ and annotations/")
    ap.add_argument("--out", default=Path("docs/assets/explorer"), type=Path)
    ap.add_argument("--url-prefix", default="assets/explorer", help="path of --out as seen from index.html")
    ap.add_argument("--exclude", type=Path, default=Path(__file__).with_name("explorer_exclude.txt"))
    ap.add_argument("--per-session", type=int, default=2, help="crops per identity per session (different cameras)")
    ap.add_argument("--min-per-id", type=int, default=6,
                    help="top up identities with fewer crops than this using extra moments (0 disables)")
    ap.add_argument("--min-gap", type=float, default=5.0,
                    help="seconds between extra crops taken from the same camera and session")
    ap.add_argument("--min-height", type=int, default=140, help="prefer boxes at least this tall (pixels)")
    ap.add_argument("--crop-height", type=int, default=256)
    ap.add_argument("--full-width", type=int, default=800, help="width of the enlarged frame behind each crop")
    ap.add_argument("--no-full", action="store_true", help="skip full frames (smaller site, no enlarged view)")
    ap.add_argument("--sync-width", type=int, default=640)
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()

    ex_crops, ex_ids, ex_sync, forced_sync = load_exclude(args.exclude)
    ann_dir, vid_dir, out = args.data / "annotations", args.data / "videos", args.out
    out.mkdir(parents=True, exist_ok=True)

    best = {}                                            # (pid, seq, cam) -> (score, frame, bbox, video, date)
    per_frame = defaultdict(lambda: defaultdict(list))   # seq -> frame -> [(cam, pid, bbox)]
    videos = {}                                          # (seq, cam) -> (video, date)
    ann_files = {}                                       # (seq, cam) -> annotation path

    for seq_dir in sorted(p for p in ann_dir.iterdir() if p.is_dir()):
        seq = seq_dir.name
        for ann_path in sorted(seq_dir.glob("*.json")):
            cam, date = parse_meta(ann_path)
            video = find_video(vid_dir, seq, ann_path, cam)
            if cam is None or video is None:
                print(f"skip {ann_path} (no camera number or video)")
                continue
            videos[(seq, cam)] = (video, date)
            ann_files[(seq, cam)] = ann_path
            for frame_key, dets in json.loads(ann_path.read_text()).items():
                frame = int(frame_key)
                for det in dets:
                    pid = int(det["id"])
                    if pid <= 0:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in det["BboxP"])
                    per_frame[seq][frame].append((cam, pid, [x1, y1, x2, y2]))
                    if pid in ex_ids:
                        continue
                    key = (pid, seq, cam)
                    score = (y2 - y1 >= args.min_height, (x2 - x1) * (y2 - y1))
                    if key not in best or score > best[key][0]:
                        best[key] = (score, frame, [x1, y1, x2, y2], video, date)
        print(f"indexed {seq}")

    # Identities with at least one box above --min-height only use those;
    # identities that are only ever seen small keep their largest boxes anyway.
    has_tall = {pid for (pid, _, _), v in best.items() if v[0][0]}
    small_only = sorted({pid for (pid, _, _) in best} - has_tall)
    if small_only:
        print(f"identities with only small boxes (kept anyway): {small_only}")

    by_session = defaultdict(list)
    for (pid, seq, cam), v in best.items():
        if pid in has_tall and not v[0][0]:
            continue
        by_session[(pid, seq)].append((v[0], cam, v))

    jobs = defaultdict(lambda: defaultdict(list))        # video -> frame -> [(pid, bbox, key, entry)]
    taken = defaultdict(list)                            # (pid, seq, cam) -> frames already used
    count = Counter()

    def add_job(pid, seq, cam, frame, bbox, video, date, key):
        entry = {"key": key, "src": f"{args.url_prefix}/ids/{key}.jpg", "seq": seq, "cam": cam,
                 "date": date, "frame": frame, "box": [round(v, 1) for v in bbox]}
        if not args.no_full:
            entry["full"] = f"{args.url_prefix}/ids/{key}_full.jpg"
        jobs[video][frame].append((pid, bbox, key, entry))
        taken[(pid, seq, cam)].append(frame)
        count[pid] += 1

    for (pid, seq), cands in by_session.items():
        n_taken = 0
        for _, cam, (_, frame, bbox, video, date) in sorted(cands, key=lambda c: c[0], reverse=True):
            if n_taken >= args.per_session:
                break
            key = f"id_{pid:02d}/{seq}_c{cam}"
            if key in ex_crops:
                continue
            add_job(pid, seq, cam, frame, bbox, video, date, key)
            n_taken += 1

    # Identities seen in few sessions or cameras end up with few crops. Top them up
    # with extra moments, spread over cameras and sessions and at least --min-gap
    # seconds apart, so within-session changes (jacket on/off) can show up too.
    all_pids = sorted({pid for (pid, _, _) in best})
    few = {pid for pid in all_pids if count[pid] < args.min_per_id}
    if args.min_per_id > 0 and few:
        gap = int(args.min_gap * 30)
        extra = defaultdict(list)                        # pid -> [(score, seq, cam, frame, bbox)]
        for (seq, cam), ann_path in ann_files.items():
            for frame_key, dets in json.loads(ann_path.read_text()).items():
                for det in dets:
                    pid = int(det["id"])
                    if pid in few:
                        x1, y1, x2, y2 = (float(v) for v in det["BboxP"])
                        score = (y2 - y1 >= args.min_height, (x2 - x1) * (y2 - y1))
                        extra[pid].append((score, seq, cam, int(frame_key), [x1, y1, x2, y2]))
        for pid in sorted(few):
            cands = sorted(extra[pid], key=lambda c: c[0], reverse=True)
            if pid in has_tall:
                cands = [c for c in cands if c[0][0]]
            while count[pid] < args.min_per_id:
                pick = None
                for c in cands:                          # least-used camera/session first, then largest box
                    _, seq, cam, frame, bbox = c
                    used = taken[(pid, seq, cam)]
                    if any(abs(frame - f) < gap for f in used):
                        continue
                    key = f"id_{pid:02d}/{seq}_c{cam}_f{frame}"
                    if key in ex_crops:
                        continue
                    if pick is None or len(used) < len(taken[(pid, pick[1], pick[2])]):
                        pick = c
                        if not used:
                            break
                if pick is None:
                    break
                _, seq, cam, frame, bbox = pick
                video, date = videos[(seq, cam)]
                add_job(pid, seq, cam, frame, bbox, video, date, f"id_{pid:02d}/{seq}_c{cam}_f{frame}")

    manifest_ids = defaultdict(list)
    for video, frames in jobs.items():
        for frame, img in read_frames(video, frames):
            H, W = img.shape[:2]
            full_saved = set()
            for pid, (x1, y1, x2, y2), key, entry in frames[frame]:
                mx, my = 0.06 * (x2 - x1), 0.04 * (y2 - y1)
                xa, ya = max(0, int(x1 - mx)), max(0, int(y1 - my))
                xb, yb = min(W, int(x2 + mx)), min(H, int(y2 + my))
                crop = img[ya:yb, xa:xb]
                if crop.size == 0:
                    continue
                scale = args.crop_height / crop.shape[0]
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), args.crop_height),
                                  interpolation=cv2.INTER_AREA)
                path = out / "ids" / f"{key}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                if not args.no_full and key not in full_saved:
                    save_resized_width(img, out / "ids" / f"{key}_full.jpg", args.full_width, args.quality)
                    full_saved.add(key)
                entry["w"], entry["h"] = W, H
                manifest_ids[str(pid)].append(entry)
        print(f"crops from {video.name}")

    for pid in manifest_ids:
        manifest_ids[pid].sort(key=lambda e: (e["date"] or "9999", e["seq"], e["cam"], e["frame"]))

    # One synchronized moment per session: the frame seen by the most cameras at once,
    # unless a frame is forced in the exclude file.
    sync = []
    for seq, frames in sorted(per_frame.items()):
        if seq in ex_sync:
            continue
        if seq in forced_sync:
            frame = forced_sync[seq]
        else:
            frame = max(frames, key=lambda f: (len({c for c, _, _ in frames[f]}), len(frames[f])))
        views, date = [], None
        for (s, cam), (video, vdate) in sorted(videos.items()):
            if s != seq:
                continue
            date = date or vdate
            for _, img in read_frames(video, [frame]):
                rel = f"sync/{seq}/c{cam}.jpg"
                save_resized_width(img, out / rel, args.sync_width, args.quality)
                people = [{"id": pid, "box": [round(v, 1) for v in box]}
                          for c, pid, box in frames.get(frame, []) if c == cam]
                views.append({"cam": cam, "src": f"{args.url_prefix}/{rel}",
                              "w": img.shape[1], "h": img.shape[0], "people": people})
        sync.append({"key": f"sync/{seq}", "seq": seq, "date": date, "frame": frame, "views": views})
        print(f"sync frame {frame} for {seq}")

    all_ids = {pid for frames in per_frame.values() for dets in frames.values() for _, pid, _ in dets}
    manifest = {"total_ids": len(all_ids),
                "ids": dict(sorted(manifest_ids.items(), key=lambda kv: int(kv[0]))), "sync": sync}
    (out / "explorer_data.js").write_text(
        "// Generated by tools/build_explorer.py; edit explorer_exclude.txt instead of this file.\n"
        "window.CHIRLA_EXPLORER = " + json.dumps(manifest, separators=(",", ":")) + ";\n"
    )
    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print("\nper identity: sessions and cameras in the annotations -> crops written")
    for pid in all_pids:
        seqs = {s for (p, s, _) in best if p == pid}
        cams = {c for (p, _, c) in best if p == pid}
        print(f"  ID {pid:>2}: {len(seqs):>2} sessions, {len(cams)} cameras -> {len(manifest_ids.get(str(pid), []))} crops")
    print(f"done: {sum(len(v) for v in manifest_ids.values())} crops for {len(manifest_ids)} identities, "
          f"{len(sync)} sync sessions, {size_mb:.1f} MB -> {out}")


if __name__ == "__main__":
    main()