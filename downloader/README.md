# CHIRLA Dataset Downloader

This repository provides a Python script to **download the CHIRLA dataset** from cloud storage. It supports filtering by file types, sequences, dataset splits, and subcategories such as ReID and Tracking scenarios.

## 🎯 Features

- **Download from a list of URLs** with preserved directory structure  
- **Flexible filtering system**:
  - General: `--benchmark`, `--videos`, `--annotations`  
  - Dataset splits: `--train`, `--val`  
  - ReID scenarios: `--reid`, `--long_term`, `--multi_camera`, `--multi_camera_long_term`, `--reappearance`  
  - Tracking scenarios: `--tracking`, `--brief_occlusions`, `--multiple_people_occlusions`  
- **Sequence filtering** with `--specific_seq` for targeted downloads
- **Resume capability**: skip existing files or force re-download with `--no-skip`  
- **Automatic directory creation** preserving the original dataset structure


## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bdager/CHIRLA.git
   cd CHIRLA
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Basic Command

```bash
python downloader/chirla_downloader.py --input-file <path_to_urls.txt> --output-dir <output_directory>
```

### Quick Start

Download all files from the URL list:

```bash
python downloader/chirla_downloader.py --input-file data/CHIRLA_urls.txt --output-dir ./CHIRLA_data
```

This downloads **all files** listed in `data/CHIRLA_urls.txt` into the `CHIRLA_data/CHIRLA/` directory structure.


### 🎛️ Filtering Examples

#### 1. Download only benchmark data

```bash
python chirla_downloader.py --input-file CHIRLA_urls.txt --output-dir ./CHIRLA_data --benchmark
```

#### 2. Download videos and annotations for specific sequences

```bash
python chirla_downloader.py --input-file CHIRLA_urls.txt --output-dir ./CHIRLA_data \
    --videos --annotations --specific_seq 000 001 024 025
```

#### 3. Download ReID multi-camera long-term scenario

```bash
python chirla_downloader.py --input-file CHIRLA_urls.txt --output-dir ./CHIRLA_data \
    --reid --multi_camera_long_term
```

#### 4. Download tracking data with brief occlusions (train split)

```bash
python chirla_downloader.py --input-file CHIRLA_urls.txt --output-dir ./CHIRLA_data \
    --tracking --brief_occlusions --train
```

#### 5. Download complete ReID data for all scenarios

```bash
python chirla_downloader.py --input-file CHIRLA_urls.txt --output-dir ./CHIRLA_data \
    --reid 
```


### 📋 Command Line Arguments

| Argument | Type | Category | Description |
|----------|------|----------|-------------|
| `--input-file` | Required | Core | Path to the `.txt` file containing dataset URLs |
| `--output-dir` | Required | Core | Directory to store downloaded files |
| `--no-skip` | Flag | Core | Force re-download files that already exist |
| `--benchmark` | Flag | Content | Download benchmark data (ReID/tracking scenarios) |
| `--videos` | Flag | Content | Download raw video files |
| `--annotations` | Flag | Content | Download annotation files (JSON format) |
| `--train` | Flag | Split | Download training data splits |
| `--val` | Flag | Split | Download validation data splits |
| `--reid` | Flag | ReID | Download all ReID scenarios |
| `--long_term` | Flag | ReID | Download long-term ReID scenario |
| `--multi_camera` | Flag | ReID | Download multi-camera ReID scenario |
| `--multi_camera_long_term` | Flag | ReID | Download multi-camera long-term ReID scenario |
| `--reappearance` | Flag | ReID | Download reappearance ReID scenario |
| `--tracking` | Flag | Tracking | Download all tracking scenarios |
| `--brief_occlusions` | Flag | Tracking | Download brief occlusions tracking scenario |
| `--multiple_people_occlusions` | Flag | Tracking | Download multiple people occlusions tracking scenario |
| `--specific_seq [SEQ ...]` | List | Sequence | Download only specified sequences: `000`, `001`, `002`, `004`, `006`, `007`, `020`, `024`, `025`, `026` |

> **Note**: All filters use **OR logic** - a file is downloaded if it matches **any** active filter. Sequence filters are applied as an additional constraint.

## 📁 Output Structure

The downloaded files will preserve the original CHIRLA dataset structure:

```
<output-dir>/CHIRLA/
├── videos/                          # Raw video files
│   ├── seq_000/
│   │   ├── camera_0.avi
│   │   ├── camera_1.avi
│   │   └── ...
│   ├── seq_001/
│   └── ...
├── annotations/                     # JSON annotation files  
│   ├── seq_000/
│   │   ├── camera_0.json
│   │   └── ...
│   └── ...
└── benchmark/                       # Processed benchmark data
    ├── reid/                        # Person Re-Identification
    │   ├── long_term/
    │   │   ├── train/
    │   │   │   ├── train_0/
    │   │   │   │   └── seq_XXX/
    │   │   │   └── train_1/
    │   │   └── test/
    │   │       ├── test_0/          # Validation subset
    │   │       └── test_1/
    │   ├── multi_camera/
    │   ├── multi_camera_long_term/
    │   └── reappearance/
    └── tracking/                    # Person Tracking
        ├── brief_occlusions/
        │   ├── train/
        │   └── test/
        └── multiple_people_occlusions/
            ├── train/
            └── test/
```

## 🔗 Related

- [CHIRLA Benchmark README](../benchmark/README.md) - Information about using the downloaded data
- [CHIRLA Main README](../README.md) - Overview of the entire project
 