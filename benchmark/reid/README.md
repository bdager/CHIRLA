# 📈  Re-ID Benchmark Guide

[![Leaderboard](https://img.shields.io/badge/%F0%9F%8F%86-Leaderboard-orange)](https://bdager.github.io/CHIRLA/leaderboard.html#reid)
[![arXiv](https://img.shields.io/badge/%F0%9F%93%84%20ArXiv-2502.06681-b31b1b.svg)](https://arxiv.org/abs/2502.06681)

🏆 **[Re-ID leaderboard](https://bdager.github.io/CHIRLA/leaderboard.html#reid)**: compare results of all evaluated methods and submit your own.

Workflow summary:

1. **Create embeddings** for gallery and query sets, and store them in [HDF5 format](#hdf5-format) with the following folder convention:

   ```
   embeddings/{method_name}/{model_name}/{scenario}_{split}_embeddings.h5
   ```

2. **Run evaluation** with the provided scripts.  
   See [Run Evaluation](#run-evaluation) for examples.

## 🚀 Quick Start: From Your Model to a Leaderboard Entry

**1. Save one HDF5 file per metadata CSV** (gallery and query of the four scenarios) with `save_embeddings`:

```python
import pandas as pd
from check_embeddings import save_embeddings

rows = pd.read_csv("../metadata/reid_long_term_query.csv")
# embeddings = your model's features for rows["image_path"], in the same order
save_embeddings("embeddings/mymethod/long_term_query_embeddings.h5",
                embeddings,          # [N, D] features from your model
                rows["id"],          # person IDs from the CSV
                rows["image_path"])  # image paths from the CSV
```

**2. Check the files.** This compares each file with its CSV (all images present once, same IDs, no empty or NaN rows):

```bash
python check_embeddings.py embeddings/mymethod
```

**3. Evaluate and create the leaderboard entry** in one command:

```bash
python make_submission.py embeddings/mymethod \
  --method "MyMethod (ResNet50)" --input 256x128 --pretrain MSMT17
```

It prints the results for the [submission form](https://github.com/bdager/CHIRLA/issues/new?template=submit-result.yml) and writes `evaluation_results.csv` and `submission.json`, the entry in the leaderboard format.



## 📊 Evaluation Metrics

Evaluation is performed per subset pair, where each query split `test_k` is matched against the corresponding gallery split `train_k`.  
Final results are reported as the average across all such pairs.

The benchmark uses standard person Re-ID metrics:

| Metric  | Description |
|---------|-------------|
| **CMC@k** | Rank-k retrieval accuracy (probability that the correct ID is within the top-k results) |
| **mAP**   | Mean Average Precision across all queries, measuring overall retrieval quality |


## 🧩 HDF5 Format

Embeddings must be stored as HDF5 files with the following structure:

- `embeddings`: float32 array of shape `[N, D]` (features)  
- `ids`: int32 array of shape `[N]` (person IDs; negatives allowed for distractors)  
- `paths`: UTF-8 string array of length `N` (original image paths)

**Notes:**
- Paths are hierarchical, e.g.:  
  ```
  .../reid/<scenario>/<split>/<subset>/seq_xxx/imgs/<camera>/<id>/frame_xxx.png
  ```
- Subset parsing uses `/train/train_k/` and `/test/test_k/` in paths.
- Negative IDs (e.g., `-1`, `-4`) in queries represent **unknown identities**.


## Data Split Roles

Each ReID scenario provides multiple CSV files corresponding to different roles:

| CSV suffix     | Subset       | Purpose                          | Use during development? | Use in final report? |
|----------------|-------------|----------------------------------|--------------------------|----------------------|
| `_train`       | `train_0`   | Small training subset             | ✅ Fine-tuning     | ❌ |
| `_val`         | `test_0`    | Validation subset                 | ✅ Model selection, hyperparameter tuning | ❌ |
| `_gallery`     | train – `train_0` removed | Main gallery for evaluation | ⚠️ Only for feature extraction | ✅  |
| `_query`       | test – `test_0` removed   | Main queries for evaluation | ❌ | ✅ |

**Protocol:**  
- Use `_train` + `_val` **only for development** (fine-tuning, ablations, hyperparameter search).  
- Report final results using `_gallery` (as gallery) vs `_query` (as queries).  
- ⚠️ Do **not** tune on `_query`, as it is reserved for final evaluation only.


## Create Embeddings

We provide scripts to generate embeddings from the gallery and query CSV files using existing ReID frameworks such as [CION_ReIDZoo](https://github.com/Zplusdragon/CION_ReIDZoo), [FastReID](https://github.com/JDAI-CV/fast-reid), and [Centroids-ReID](https://github.com/bdager/centroids-reid).  

To run the embedding extraction, make sure to:
- Clone the corresponding repositories.  
- Download the required pre-trained model weights and configuration files from each repository.  


Use `create_embeddings.py` for one CSV or `get_embeddings_*.py` scripts for batches.

Example (single CSV):
```
python create_embeddings.py \
  --csv metadata/reid_multi_camera_gallery.csv \
  --input ../data/CHIRLA/benchmark \
  --output embeddings/fastreid \
  --config configs/Market1501/bagtricks_R101-ibn.yml \
  --cktp checkpoints/market_bot_R101-ibn.pth
```

**DIFFER / SapiensID** (`--model differ` / `--model sapiensid`): clone [DIFFER](https://github.com/xliangp/DIFFER) (commit `67acb5d`) and [SapiensID](https://github.com/mk-minchul/sapiensid) (commit `a8d8011`) into `models/DIFFER` and `models/sapiensid` (or pass `--repo-dir`), and get their weights from each repo. SapiensID also needs `~/.cache/yolov8n-pose.pt`.

> ⚠️ SapiensID crashes on crops where no person is detected. Apply this one-line fix after cloning:
> ```
> sed -i 's/len(_keypoints\[0\]) > 0/len(_keypoints) > 0/' models/sapiensid/tasks/sapiensID/src/aligners/keypoint_predictor/__init__.py
> ```


## Run Evaluation

We provide flexible evaluation scripts supporting different evaluation modes.

#### Per-Subset Evaluation
Report averaged metrics over all `train_k` / `test_k` subset pairs of specific gallery and query files:

```bash
python evaluate_reid.py     --gallery embeddings/gallery_embeddings.h5     --query embeddings/query_embeddings.h5     --topk 1 5 10     --per-subset
```

Add `--out-csv results.csv` to also save each query's ranked gallery (`results_test_k.csv` per subset plus `results.csv` with all of them; `--save-topk N` keeps only the top N). `--visualize` needs `--data-root <CHIRLA>/benchmark`, since the HDF5 paths are relative.

#### Single Subset Evaluation
To test a specific subset pair (`train_k` vs `test_k`):

```bash
python evaluate_reid.py     --gallery embeddings/gallery_train_k.h5     --query embeddings/query_test_k.h5     --topk 1 5 10
```

#### Batch Evaluation
Run evaluation across scenarios for a specific Re-ID model embedding:

```bash
python run_all_evaluations.py     --base-dir fastreid     --output evaluation_results.csv
```

Arguments:  
- `--base-dir`: Base directory containing HDF5 files (e.g., `fastreid`, `CION/` or any model you use to extract the embeddings)  
- `--output`: Output CSV file for results (default: `evaluation_results.csv`)  

By default, batch evaluation averages metrics across subsets.

<!-- ### 11. Code Reference
| Script | Purpose |
|--------|---------|
| `create_embeddings.py` | Single CSV embedding extraction |
| `get_embeddings_fastreid.py` / `get_embeddings_centroids.py` / `get_embeddings_cion.py` | Batch extraction per method |
| `evaluate_reid.py` | Core metric computation (CMC, mAP, optional per-subset) |
| `run_all_evaluations.py` | Batch evaluate all generated embeddings |
| `inference_*.py` | Convenience inference on single image or set | -->



## 📌 Open / Closed Set Evaluation 

We report **closed-set evaluation** for baseline results, excluding distractors (unknown IDs) from the test/query set.  

Since CHIRLA also provides distractors in the test set, it naturally supports **open-set evaluation**, where models must distinguish between known and unknown IDs by applying a similarity threshold. This setting better reflects real-world deployment conditions, and we encourage the community to explore it.

To approximate **open-set mAP**, add a threshold argument when running the evaluation scripts (`evaluate_reid.py` or `run_all_evaluations.py`):

```
--open-map-threshold <value>
```

Results of all evaluated methods, including the baselines from our paper, are on the
**[CHIRLA leaderboard](https://bdager.github.io/CHIRLA/leaderboard.html)**, which also explains how to submit yours.
For the experimental settings of the baselines, please refer to our [paper](https://arxiv.org/pdf/2502.06681).


## 🙏 Acknowledgements

We thank the great works and open-source repositories: [CION_ReIDZoo](https://github.com/Zplusdragon/CION_ReIDZoo), [FastReID](https://github.com/JDAI-CV/fast-reid), [Centroids-ReID](https://github.com/mikwieczorek/centroids-reid), [DIFFER](https://github.com/xliangp/DIFFER) and [SapiensID](https://github.com/mk-minchul/sapiensid).