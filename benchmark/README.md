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
|   |   |   |   |   └── camera_X_timestamp.json            
```

>  Tracking scenarios do not contain **Subsets** for train or test splits.

#### Example Path Structure
```
reid/long_term/seq_024/imgs/camera_6_2023-12-01-11:05:52/<ID>/frame_XXXX.png
```

## 🎯 Tasks and Scenarios

### Person Re-Identification (ReID)

The ReID task focuses on matching person identities across different cameras and time periods.

#### Scenarios:

1. **Reappearance ReID** (`reappearance`)
   - Detecting when a person reappears after absence.
   - The person leaves the field of view of a camera and returns to the scene after a time interval. This reappearance can also include a change in the person appearance (e.g., takes off or puts on a jacket, wears a hat or changes an accessory). The objective of this scenario is to assess the ability of the system to re-identify the person correctly after a break in visibility.

2. **Long-term ReID** (`long_term`)
   - Matching individuals across extended time periods.
   - The person is seen in different recordings captured on different days, implying possible changes in appearance and environment. The objective of this scenario is to evaluate the system’s ability to recognize a person when recordings come from different days, with potential variations in lighting, appearance and other contextual factors.

3. **Multi-camera ReID** (`multi_camera`) 
   - Cross-camera person matching at similar timestamps.
   - The person is captured by several cameras at different locations, with little or no overlap between the fields of view of the cameras. The objective of this scenario is to assess the system’s ability to re-identify the person when viewed from different angles and perspectives.

4. **Multi-camera Long-term ReID** (`multi_camera_long_term`)
   - Combined challenges of multiple cameras and long term re-id.
   - The person is captured by several cameras at different locations and days. The objective of this scenario is to evaluate the system’s robustness in re-identifying the person from various angles and perspectives, ensuring reliable performance under diverse temporal and contextual conditions.


### Person Tracking

The tracking task focuses on maintaining consistent identity assignment over time sequences.

#### Scenarios:

1. **Brief Occlusions** (`brief_occlusions`)
   - Tracking through short-term occlusions.
   - The person is totally hidden by an object or another person for a short period of time and becomes visible again without having left the scene. The objective of this scenario is to assess the ability of the system to maintain the ID and tracking of a person after a brief occlusion.

2. **Multiple People Occlusions** (`multiple_people_occlusions`)
   - Tracking in crowded scenes with person-to-person occlusions.
   - The scene includes several people, some of whom cross each other or are hidden, creating frequent occlusions. The objective of this scenario is to evaluate the system’s ability to track multiple people simultaneously, even in situations where occlusions arise from their interactions.


## 📊 Metadata Files

The benchmark provides structured metadata in CSV format for easy access and evaluation.
All scenario metadata CSVs are in the [`metadata/`](metadata) directory, with detailed per-file descriptions in [`metadata/README.md`](metadata/README.md). 

### Re-ID CSV Files

| File       | Subset basis          | Filename pattern           | Typical use                           |
|------------|----------------------|----------------------------|---------------------------------------|
| Gallery    | Train (excl. `train_0`) | `reid_{scenario}_gallery.csv` | Feature extraction + gallery index     |
| Query      | Test (excl. `test_0`)   | `reid_{scenario}_query.csv`   | Final evaluation queries               |
| Training   | `train_0` only          | `reid_{scenario}_train.csv`   | Fine-tuning / model adjustment           |
| Validation | `test_0` only           | `reid_{scenario}_val.csv`     | Hyper-parameter tuning  |

Where `{scenario}` ∈ {`reappearance`, `long_term`, `multi_camera`, `multi_camera_long_term`}.

> ⚠️ **Important**: The validation set (`*_val.csv`) is for method development and fine-tuning.  The query set (`*_query.csv`) should be reserved for final evaluation and fair comparison.


### Tracking CSV Files

Tracking scenarios do not use **subset** splits. Each split has one file:

```
tracking_{scenario}_{split}.csv
```

Examples:  
- `tracking_brief_occlusions_test.csv`  
- `tracking_multiple_people_occlusions_train.csv`


## Evaluation Protocol

Check [`reid/`](reid) and [`tracking/`](tracking) directories for the specific protocols to evaluate each task-oriented benchmark.