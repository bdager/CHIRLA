# 📈 Tracking Benchmark Guide

[![Leaderboard](https://img.shields.io/badge/%F0%9F%8F%86-Leaderboard-orange)](https://bdager.github.io/CHIRLA/leaderboard.html#tracking)
[![arXiv](https://img.shields.io/badge/%F0%9F%93%84%20ArXiv-2502.06681-b31b1b.svg)](https://arxiv.org/abs/2502.06681)

🏆 **[Tracking leaderboard](https://bdager.github.io/CHIRLA/leaderboard.html#tracking)**: compare results of all evaluated trackers and submit your own.

This repository contains scripts to:
- Prepare CHIRLA ground truth (GT) in MOT format
- Run trackers over videos
- Save results in TrackEval layout
- Evaluate with TrackEval

## 📂 Directory Layout
- `data/` – MOT-style GT and tracker outputs created by these scripts
- `TrackEval/` – Evaluation toolkit ([TrackEval](https://github.com/bdager/TrackEval))
- Scripts:
  - `benchmark_to_MOT.py` – Convert CHIRLA JSON annotations to MOT GT with seqmaps
  - `annotations_to_MOT.py` – Convert raw annotations to CHIRLA_all MOT GT (seqinfo + seqmaps)
  - `tracker_botmox_all_videos.py` – Run YOLO + boxmot trackers over all videos to generate results
  - `generate_subset_trackers.py` – Build brief/multi trackers from CHIRLA_all results


## 📊 Evaluation Metrics
We use **CLEAR** and **Identity** metrics (MOTA, IDF1, etc.), evaluated with [TrackEval](https://github.com/bdager/TrackEval) and using MOT-style result format (10 columns):
```
frame,id,x,y,w,h,conf,-1,-1,-1
```


## 0) Setup
Commands below are run from the repository root, with the dataset in `data/CHIRLA`. Install the dependencies and clone our [TrackEval](https://github.com/bdager/TrackEval) fork, which adds the CHIRLA evaluation script:

```bash
pip install -r benchmark/tracking/requirements.txt
git clone https://github.com/bdager/TrackEval.git benchmark/tracking/TrackEval
```

## 1) Prepare Ground Truth in MOT format
Create MOT-style GT for **CHIRLA_all** (all annotations from CHIRLA dataset) and seqmaps:

```bash
python benchmark/tracking/annotations_to_MOT.py \
	--input_dir data/CHIRLA/annotations \
	--output_dir benchmark/tracking/data/gt \
	--verbose
```

Create scenario-specific GT (brief and multiple-people-occlusion) with seqmaps:

```bash
python benchmark/tracking/benchmark_to_MOT.py \
	--input_dir data/CHIRLA/benchmark/tracking \
	--output_dir benchmark/tracking/data/gt \
	--scenario all --split all --verbose
```

Outputs:
- `benchmark/tracking/data/gt/CHIRLA_all-{train,test}/<seq_cam_time>/gt/gt.txt`
- `benchmark/tracking/data/gt/CHIRLA_{brief,multi}-{train,test}/<seq_cam_time>/gt/gt.txt`
- `benchmark/tracking/data/gt/seqmaps/CHIRLA_*-{train,test}.txt`

## 2) Run tracking over videos (write MOT results)
We provide scripts to run trackers via [boxmot](https://github.com/mikel-brostrom/boxmot) with YOLO as detector:  

Supported trackers: [BoostTrack](https://github.com/vukasin-stanojevic/BoostTrack), [BoostTrack+](https://github.com/vukasin-stanojevic/BoostTrack), [BoostTrack++](https://github.com/vukasin-stanojevic/BoostTrack), [BoT-SORT](https://github.com/NirAharon/BoT-SORT), [ByteTrack](https://github.com/FoundationVision/ByteTrack), [OC-SORT](https://github.com/noahcao/OC_SORT), [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT), [StrongSORT](https://github.com/dyhBUPT/StrongSORT).

<!-- ### Supported Trackers
- [BoostTrack](https://github.com/vukasin-stanojevic/BoostTrack) –  *Boosting the similarity measure and detection confidence for improved multiple object tracking*   [Paper (Springer, 2024)](https://link.springer.com/article/10.1007/s00138-024-01531-5)

- [BoostTrack+ / BoostTrack++](https://github.com/vukasin-stanojevic/BoostTrack) –  *Using tracklet information to detect more objects with multiple object tracking*    [ArXiv (2024)](https://arxiv.org/abs/2408.13003)

- [BoT-SORT](https://github.com/NirAharon/BoT-SORT) –    *Robust Associations Multi-Pedestrian Tracking*    [ArXiv (2022)](https://arxiv.org/abs/2206.14651)

- [ByteTrack](https://github.com/FoundationVision/ByteTrack) –    *Multi-Object Tracking by Associating Every Detection Box*    [ECCV 2022](https://arxiv.org/abs/2110.06864)

- [OC-SORT](https://github.com/noahcao/OC_SORT) –    *Observation-Centric SORT for Robust Multi-Object Tracking*    [ArXiv (2022)](https://arxiv.org/abs/2203.14360)

- [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT) –    *Multi-Pedestrian Tracking by Adaptive Re-Identification*    [ArXiv (2023)](https://arxiv.org/abs/2302.11813)

- [StrongSORT](https://github.com/dyhBUPT/StrongSORT) –    *Make DeepSORT Great Again*    [GitHub reference](https://github.com/dyhBUPT/StrongSORT) -->

We run these methods with **Python 3.11** using the module versions specified in [requirements.txt](requirements.txt).

**Tracker folder names:** the CLI name (`--tracker`) and the folder the results are written to are different. Steps 3 to 5 use the folder name:

| `--tracker` | Folder |
|---|---|
| `boosttrack` / `boosttrack+` / `boosttrack++` | `BoostTrack` / `BoostTrack+` / `BoostTrack++` |
| `bytetrack` | `ByteTrack` |
| `ocsort` / `deepocsort` | `OC-SORT` / `Deep-OC-SORT` |
| `strongsort` | `StrongSORT` |
| `botsort` | `BoT-SORT`, or `BoT-SORT-reid` with `--with-reid` |

### Batch Process All Videos
```
python benchmark/tracking/tracker_botmox_all_videos.py \
	--videos-root data/CHIRLA/videos \
	--output-root benchmark/tracking/results \
	--mot-root benchmark/tracking/data/trackers \
	--benchmark CHIRLA_all --split test \
	--model benchmark/tracking/models/yolo11n.pt \
	--tracker bytetrack --conf 0.25 --device 0
```
Tracker name format (CLI): boosttrack, boosttrack+, boosttrack++, strongsort, ocsort, deepocsort, bytetrack, botsort.

### Appearance-based Trackers
For trackers with ReID support ([BoostTrack+](https://github.com/vukasin-stanojevic/BoostTrack), [BoostTrack++](https://github.com/vukasin-stanojevic/BoostTrack), [BoT-SORT](https://github.com/NirAharon/BoT-SORT),  [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT), [StrongSORT](https://github.com/dyhBUPT/StrongSORT)), enable `--with-reid` and specify weights:

```bash
python benchmark/tracking/tracker_botmox_all_videos.py \
	--videos-root data/CHIRLA/videos \
	--mot-root benchmark/tracking/data/trackers \
	--benchmark CHIRLA_all --split test \
	--tracker botsort --with-reid \
	--reid-weights benchmark/tracking/models/osnet_x0_25_msmt17.pt
```

**Creates:** `<mot_root>/CHIRLA_all-{train,test}/<Tracker>/data/<seq_camera_time>.txt`

> If you run your own tracker, ensure outputs follow the same MOT layout and directory structure.

<!-- ### 3) Restructure tracker outputs (optional)
If you have per-camera files under scenario folders, convert them to TrackEval’s expected layout:
```
python benchmark/tracking/restructure_trackers_to_MOT.py \
	--input_dir benchmark/tracking/data/trackers \
	--output_dir benchmark/tracking/data/trackers_mot \
	--scenario all --include_train_test_split --train_seqs seq_004,seq_026
```
Creates: `<output>/<CHIRLA_brief|CHIRLA_multi>-{train,test}/<Tracker>/data/<seq_camera>.txt`.

<output_root>/CHIRLA_brief-train/<Tracker>/data/seq_004_camera_1_xxx.txt -->

## 3) Generate Subset Tracker Results
After producing CHIRLA_all tracker results, you can build **brief/multi** tracker files by sampling CHIRLA_all outputs using per-frame IoU best-match:

```bash
python benchmark/tracking/generate_subset_trackers.py \
	--source_benchmark CHIRLA_all \
	--target_benchmarks CHIRLA_brief CHIRLA_multi \
	--splits test \
	--trackers ByteTrack BoT-SORT-reid \
	--gt_root benchmark/tracking/data/gt \
	--tracker_root benchmark/tracking/data/trackers
```

## 4) Evaluate with TrackEval
Run [TrackEval](https://github.com/bdager/TrackEval) once per benchmark (`CHIRLA_all`, `CHIRLA_brief` and `CHIRLA_multi`):
```bash
for b in CHIRLA_all CHIRLA_brief CHIRLA_multi; do
python benchmark/tracking/TrackEval/scripts/run_chirla_challenge.py \
	--GT_FOLDER benchmark/tracking/data/gt \
	--TRACKERS_FOLDER benchmark/tracking/data/trackers \
	--BENCHMARK $b \
	--SPLIT_TO_EVAL test \
	--TRACKERS_TO_EVAL ByteTrack BoT-SORT-reid \
	--METRICS CLEAR Identity
done
```

Results are written to `<TRACKERS_FOLDER>/<benchmark>-test/<TrackerName>/pedestrian_summary.txt`.

Notes:
- Seqmaps are read from `<GT_FOLDER>/seqmaps/<benchmark>-<split>.txt` (created by step 1).
- Tracker outputs must be placed under:  
  `<TRACKERS_FOLDER>/CHIRLA_{all,brief,multi}-{split}/<TrackerName>/data/<sequence>.txt`  
  with 10 MOT columns.

## 5) Create the Leaderboard Entry
Turn the TrackEval results of one tracker into a [leaderboard](https://bdager.github.io/CHIRLA/leaderboard.html#submit) entry:

```bash
python benchmark/export_leaderboard.py tracking \
	--trackers-folder benchmark/tracking/data/trackers \
	--tracker BoT-SORT-reid \
	--method "BoT-SORT" --appearance "OSNet x0.25 (MSMT17)" --detector YOLO11n \
	--out submission.json
```

It reads the three `pedestrian_summary.txt` files from step 4 and writes the entry in the leaderboard format.


### 📌 Baselines
Results of all evaluated methods, including the baselines from our paper, are on the
**[CHIRLA leaderboard](https://bdager.github.io/CHIRLA/leaderboard.html)**, which also explains how to submit yours.
For the baseline configurations, see our [paper](https://arxiv.org/pdf/2502.06681).

## 🙏 Acknowledgements
We thank the open-source community and tools that made this benchmark possible:   [boxmot](https://github.com/mikel-brostrom/boxmot), [BoostTrack](https://github.com/vukasin-stanojevic/BoostTrack),  [BoT-SORT](https://github.com/NirAharon/BoT-SORT), [ByteTrack](https://github.com/FoundationVision/ByteTrack), [OC-SORT](https://github.com/noahcao/OC_SORT), [Deep-OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT), [StrongSORT](https://github.com/dyhBUPT/StrongSORT), [TrackEval](https://github.com/JonathonLuiten/TrackEval).