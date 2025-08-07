# CHIRLA Benchmark

The **CHIRLA (Comprehensive High-resolution Identification and Re-identification for Large-scale Analysis)** benchmark is designed for evaluating person re-identification and tracking algorithms in real-world scenarios with varying temporal gaps and camera configurations.

## 📁 Benchmark Data Structure

After downloading the benchmark data, you will find the following hierarchical organization:

```
Benchmark/
├── Task (reid, tracking)
│   ├── Scenario (long_term, multi_camera, etc.)
│   │   ├── Split (train/test)
│   │   │   ├── Subset (train_0, test_0, etc.)
│   │   │   │   ├── Sequences (seq_001, seq_002, ...)
│   │   │   │   │   ├── imgs/
│   │   │   │   │   │   ├── Camera (camera_X_timestamp)
│   │   │   │   │   │   │   ├── ID (person identifier)
│   │   │   │   │   │   │   │   └── Images (frame_XXXX.png)
```

### Example Path Structure
```
reid/long_term/seq_024/imgs/camera_6_2023-12-01-11:05:52/<ID>/frame_XXXX.png
```

## 🎯 Tasks and Scenarios

### Person Re-Identification (ReID)

The ReID task focuses on matching person identities across different cameras and time periods.

#### Scenarios:

1. **Long-term ReID** (`long_term`)
   - Matching individuals across extended time periods
   - Challenges: appearance changes, lighting variations

2. **Multi-camera ReID** (`multi_camera`) 
   - Cross-camera person matching at similar timestamps
   - Challenges: viewpoint variations, camera differences

3. **Multi-camera Long-term ReID** (`multi_camera_long_term`)
   - Combined challenges of multiple cameras and time gaps
   - Most challenging scenario

4. **Reappearance ReID** (`reappearance`)
   - Detecting when a person reappears after absence
   - Challenges: clothing changes, temporal gaps

### Person Tracking

The tracking task focuses on maintaining consistent identity assignment over time sequences.

#### Scenarios:

1. **Brief Occlusions** (`brief_occlusions`)
   - Tracking through short-term occlusions
   - Challenges: partial visibility, motion blur

2. **Multiple People Occlusions** (`multiple_people_occlusions`)
   - Tracking in crowded scenes with person-to-person occlusions
   - Challenges: identity switches, group interactions

## 📊 Metadata Files

The benchmark provides structured metadata in CSV format for easy access and evaluation.

### Enhanced Data Organization

The metadata extraction script now generates multiple CSV files per scenario based on different data subsets:

#### Subset-based File Generation

- **Main Files** (excluding `train_0` and `test_0` subsets):
  - **Gallery/Train Files**: For ReID scenarios, training data is saved as `reid_{scenario}_gallery.csv`
  - **Query/Test Files**: For ReID scenarios, test data is saved as `reid_{scenario}_query.csv`
  - **Standard Files**: For tracking scenarios, files follow the pattern `{task}_{scenario}_{split}.csv`

- **Special Subset Files**:
  - **Training Subset (`train_0`)**: Saved as `reid_{scenario}_train.csv`
  - **Validation Subset (`test_0`)**: Saved as `reid_{scenario}_val.csv`

This organization allows for:
- **Flexible evaluation**: Use different subsets as gallery/query combinations
- **Cross-validation**: Leverage different train/test splits
- **Fine-tuning validation**: Use `test_0` subset for hyperparameter optimization

### Data Splits

Each ReID scenario includes multiple data files based on the subset organization:

- **Gallery Set** (`*_gallery.csv`): Main training data (excludes `train_0` subset) used as gallery for evaluation
- **Query Set** (`*_query.csv`): Main test data (excludes `test_0` subset) used as queries for evaluation
- **Training Set** (`*_train.csv`): Specific training subset (`train_0`) for model training or alternative gallery
- **Validation Set** (`*_val.csv`): Derived from `test_0` subset, specifically designed for:
  - **Fine-tuning pre-trained models** on CHIRLA-specific characteristics
  - **Hyperparameter optimization** without touching the main test set
  - **Model selection** and early stopping during adaptation
  - **Domain adaptation** from other ReID datasets to CHIRLA

> ⚠️ **Important**: The validation set (`*_val.csv`) should be used for method development and fine-tuning, while the main query set (`*_query.csv`) should only be used for final evaluation and comparison with other methods.

### ReID Metadata Files

| File | Description | Source Subsets |
|------|-------------|----------------|
| `reid_long_term_gallery.csv` | Gallery data for long-term ReID | All train subsets except `train_0` |
| `reid_long_term_query.csv` | Query data for long-term ReID | All test subsets except `test_0` |
| `reid_long_term_train.csv` | Training subset for long-term ReID | `train_0` subset only |
| `reid_long_term_val.csv` | Validation data for long-term ReID | `test_0` subset only |
| `reid_multi_camera_gallery.csv` | Gallery data for multi-camera ReID | All train subsets except `train_0` |
| `reid_multi_camera_query.csv` | Query data for multi-camera ReID | All test subsets except `test_0` |
| `reid_multi_camera_train.csv` | Training subset for multi-camera ReID | `train_0` subset only |
| `reid_multi_camera_val.csv` | Validation data for multi-camera ReID | `test_0` subset only |
| `reid_multi_camera_long_term_gallery.csv` | Gallery data for multi-camera long-term ReID | All train subsets except `train_0` |
| `reid_multi_camera_long_term_query.csv` | Query data for multi-camera long-term ReID | All test subsets except `test_0` |
| `reid_multi_camera_long_term_train.csv` | Training subset for multi-camera long-term ReID | `train_0` subset only |
| `reid_multi_camera_long_term_val.csv` | Validation data for multi-camera long-term ReID | `test_0` subset only |
| `reid_reappearance_gallery.csv` | Gallery data for reappearance ReID | All train subsets except `train_0` |
| `reid_reappearance_query.csv` | Query data for reappearance ReID | All test subsets except `test_0` |
| `reid_reappearance_train.csv` | Training subset for reappearance ReID | `train_0` subset only |
| `reid_reappearance_val.csv` | Validation data for reappearance ReID | `test_0` subset only |

### Tracking Metadata Files

| File | Description | Source Subsets |
|------|-------------|----------------|
| `tracking_brief_occlusions_train.csv` | Training data for brief occlusion tracking | All train subsets except `train_0` |
| `tracking_brief_occlusions_test.csv` | Test data for brief occlusion tracking | All test subsets except `test_0` |
| `tracking_multiple_people_occlusions_train.csv` | Training data for multi-person occlusion tracking | All train subsets except `train_0` |
| `tracking_multiple_people_occlusions_test.csv` | Test data for multi-person occlusion tracking | All test subsets except `test_0` |

### CSV Structure

Each CSV file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | string | Relative path to the image file |
| `id` | integer | Person identifier (positive or negative) |
| `camera` | string | Camera identifier with timestamp |
| `sequence` | string | Sequence identifier (e.g., seq_024) |
| `subset` | string | Data subset (train_0, test_0, etc.) |

## 🚀 Getting Started

### 1. Generate Metadata

Extract metadata from the raw dataset structure using the enhanced extraction script:

```bash
python extract_metadata.py --root-dir data/CHIRLA/benchmark --output-dir benchmark/metadata
```

This will generate multiple CSV files per scenario:
- **Gallery/Query files**: Main evaluation files (e.g., `reid_long_term_gallery.csv`, `reid_long_term_query.csv`)
- **Training files**: Specific training subsets (e.g., `reid_long_term_train.csv`)  
- **Validation files**: For fine-tuning and development (e.g., `reid_long_term_val.csv`)

### 2. Create Embeddings

Generate feature embeddings using pre-trained models. You can use different CSV files based on your evaluation needs:

```bash
# Example: Using gallery/query files for standard evaluation
cd fast-reid
python create_embeddings.py \
    --csv ../CHIRLA/benchmark/metadata/reid_long_term_gallery.csv \
    --input ../CHIRLA/data/CHIRLA/benchmark \
    --output ../CHIRLA/benchmark/fastreid/embeddings \
    --config configs/Market1501/bagtricks_R101-ibn.yml \
    --cktp checkpoints/market_bot_R101-ibn.pth

# Example: Using training subset for model training
python create_embeddings.py \
    --csv ../CHIRLA/benchmark/metadata/reid_long_term_train.csv \
    --input ../CHIRLA/data/CHIRLA/benchmark \
    --output ../CHIRLA/benchmark/fastreid/train_embeddings

# Multiple models (automated)
python ../get_embeddings.py
```

### 3. Run Evaluation

Evaluate model performance with flexible gallery/query combinations:

Evaluate model performance using standard ReID metrics:

```bash
# Standard evaluation (gallery vs query)
python evaluate_reid.py \
    --gallery embeddings/gallery_embeddings.h5 \
    --query embeddings/query_embeddings.h5 \
    --topk 1 5 10

# Per-subset evaluation with averaging
python evaluate_reid.py \
    --gallery embeddings/gallery_embeddings.h5 \
    --query embeddings/query_embeddings.h5 \
    --topk 1 5 10 \
    --per-subset

# Cross-validation using different subset combinations
python evaluate_reid.py \
    --gallery embeddings/train_embeddings.h5 \
    --query embeddings/val_embeddings.h5 \
    --topk 1 5 10

# Batch evaluation for all models and scenarios
python run_all_evaluations.py \
    --base-dir fastreid \
    --output evaluation_results.csv
```

## 📈 Evaluation Metrics

The benchmark uses standard person re-identification metrics with enhanced per-subset evaluation:

- **CMC (Cumulative Matching Characteristics)**: Rank-k accuracy
- **mAP (mean Average Precision)**: Overall retrieval quality
- **Per-subset Analysis**: Individual evaluation for each data subset
- **Averaged Results**: Weighted and simple averages across subsets
- **Visualization**: Top-k retrieval results for qualitative analysis

### Enhanced Evaluation Features

The evaluation script now provides:

1. **Per-Subset Metrics**: Individual performance for each subset (e.g., test_1, test_2, etc.)
2. **Weighted Average**: Results weighted by the number of queries per subset
3. **Simple Average**: Equal weight for each subset (person-level performance)
4. **Overall Evaluation**: Traditional approach for comparison


## 🔧 Tools and Scripts

| Script | Purpose |
|--------|---------|
| `extract_metadata.py` | Generate CSV metadata from dataset structure |
| `create_embeddings.py` | Extract feature embeddings using ReID models |
| `evaluate_reid.py` | Compute ReID evaluation metrics |
| `run_all_evaluations.py` | Batch evaluation across all model/scenario combinations |
| `get_embeddings.py` | Automated embedding generation for multiple models |


## 🏆 Benchmark Characteristics

- **Scale**: Thousands of identities across multiple scenarios
- **Temporal Range**: From minutes to months between appearances
- **Camera Diversity**: Multiple viewpoints and lighting conditions  
- **Real-world Challenges**: Clothing changes, pose variations, occlusions
- **Evaluation Rigor**: Standardized protocols for fair comparison

## 📝 Best Practices

### Proper Use of Data Splits

1. **Gallery Set** (`*_gallery.csv`): Candidate images pool for evaluation
2. **Query Set** (`*_query.csv`): Images query for evaluation
3. **Training Set** (`*_train.csv`): Small subset of training images for fine-tuning purpuses
4. **Validation Set** (`*_val.csv`): Use for:
   - Fine-tuning pre-trained models on CHIRLA characteristics
   - Hyperparameter optimization 
   - Model architecture selection and ablation studies

### Evaluation Protocol

- **Do not** use main query set (`*_query.csv`) during model development
- **Do not** perform hyperparameter search on main test queries
- **Use** validation set (`*_val.csv`) for all development activities
- **Report** results on both validation and main query sets separately
- **Use** per-subset evaluation for detailed analysis of model performance
- **Use** the same evaluation metrics (CMC, mAP) across all splits

### Fine-tuning Guidelines

When fine-tuning pre-trained ReID models:
- Start with models trained on large-scale datasets (Market1501, MSMT17, etc.)
- Use validation set to monitor adaptation progress
- Apply domain-specific augmentations based on CHIRLA characteristics
- Consider temporal and multi-camera aspects during fine-tuning