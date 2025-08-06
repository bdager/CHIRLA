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

### Data Splits

Each ReID scenario includes three data splits:

- **Training Set**: Used for training models from scratch or as gallery for evaluation
- **Test Set**: Main evaluation set for reporting final performance metrics  
- **Validation Set**: Derived from `test_0` subset, specifically designed for:
  - **Fine-tuning pre-trained models** on CHIRLA-specific characteristics
  - **Hyperparameter optimization** without touching the test set
  - **Model selection** and early stopping during adaptation
  - **Domain adaptation** from other ReID datasets to CHIRLA

> ⚠️ **Important**: The validation set should be used for method development and fine-tuning, while the test set should only be used for final evaluation and comparison with other methods.

### ReID Metadata

| File | Description |
|------|-------------|
| `reid_long_term_train.csv` | Training data for long-term ReID |
| `reid_long_term_test.csv` | Test data for long-term ReID |
| `reid_long_term_test_val.csv` | Validation data for long-term ReID (for fine-tuning) |
| `reid_multi_camera_train.csv` | Training data for multi-camera ReID |
| `reid_multi_camera_test.csv` | Test data for multi-camera ReID |
| `reid_multi_camera_test_val.csv` | Validation data for multi-camera ReID (for fine-tuning) |
| `reid_multi_camera_long_term_train.csv` | Training data for multi-camera long-term ReID |
| `reid_multi_camera_long_term_test.csv` | Test data for multi-camera long-term ReID |
| `reid_multi_camera_long_term_test_val.csv` | Validation data for multi-camera long-term ReID (for fine-tuning) |
| `reid_reappearance_train.csv` | Training data for reappearance ReID |
| `reid_reappearance_test.csv` | Test data for reappearance ReID |
| `reid_reappearance_test_val.csv` | Validation data for reappearance ReID (for fine-tuning) |

### Tracking Metadata 

| File | Description |
|------|-------------|
| `tracking_brief_occlusions_train.csv` | Training data for brief occlusion tracking |
| `tracking_brief_occlusions_test.csv` | Test data for brief occlusion tracking |
| `tracking_multiple_people_occlusions_train.csv` | Training data for multi-person occlusion tracking |
| `tracking_multiple_people_occlusions_test.csv` | Test data for multi-person occlusion tracking |

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

Extract metadata from the raw dataset structure:

```bash
python extract_metadata.py --root-dir data/CHIRLA/benchmark --output-dir benchmark/metadata
```

### 2. Create Embeddings

Generate feature embeddings using pre-trained models:

```bash
# Single model evaluation
cd fast-reid
python create_embeddings.py \
    --csv ../CHIRLA/benchmark/metadata/reid_long_term_train.csv \
    --input ../CHIRLA/data/CHIRLA/benchmark \
    --output ../CHIRLA/benchmark/fastreid/embeddings \
    --config configs/Market1501/bagtricks_R101-ibn.yml \
    --cktp checkpoints/market_bot_R101-ibn.pth

# Multiple models (automated)
python ../get_embeddings.py
```

### 3. Run Evaluation

Evaluate model performance using standard ReID metrics:

```bash
# Single evaluation
python evaluate_reid.py \
    --gallery embeddings/train_embeddings.h5 \
    --query embeddings/test_embeddings.h5 \
    --topk 1 5 10

# Batch evaluation for all models
python run_all_evaluations.py \
    --base-dir fastreid \
    --output evaluation_results.csv
```

## 📈 Evaluation Metrics

The benchmark uses standard person re-identification metrics:

- **CMC (Cumulative Matching Characteristics)**: Rank-k accuracy
- **mAP (mean Average Precision)**: Overall retrieval quality
- **Visualization**: Top-k retrieval results for qualitative analysis

### Example Output

```
=== Evaluation Results ===
CMC Rank-1: 75.23%
CMC Rank-5: 89.45%
CMC Rank-10: 94.12%
mAP: 68.94%
```

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

1. **Training Set**: Use for initial model training or as gallery for zero-shot evaluation
2. **Validation Set**: Use for:
   - Fine-tuning pre-trained models on CHIRLA characteristics
   - Hyperparameter optimization (learning rate, augmentation, etc.)
   - Model architecture selection and ablation studies
   - Early stopping during domain adaptation
3. **Test Set**: Use only for:
   - Final performance reporting
   - Comparison with other published methods
   - Benchmark submission

### Evaluation Protocol

- **Do not** use test set during model development
- **Do not** perform hyperparameter search on test set
- **Report** validation and test results separately
- **Use** the same evaluation metrics (CMC, mAP) across all splits

### Fine-tuning Guidelines

When fine-tuning pre-trained ReID models:
- Start with models trained on large-scale datasets (Market1501, MSMT17, etc.)
- Use validation set to monitor adaptation progress
- Apply domain-specific augmentations based on CHIRLA characteristics
- Consider temporal and multi-camera aspects during fine-tuning