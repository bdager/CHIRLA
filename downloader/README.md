# CHIRLA Dataset Downloader

This repository provides a Python script to **download the CHIRLA dataset**.  
It supports filtering by file types, sequences, dataset splits, and subcategories such as ReID and Tracking scenarios.


## Features

- Download from a list of URLs with preserved directory structure.  
- Flexible filters:
  - General: `--benchmark`, `--videos`, `--annotations`  
  - Dataset splits: `--train`, `--val`  
  - ReID: `--reid`, `--long_term`, `--multi_camera`, `--multi_camera_long_term`, `--reappearance`  
  - Tracking: `--tracking`, `--brief_occlusions`, `--multiple_people_occlusions`  
- Sequence filtering with `--specific_seq` (e.g., `000`, `001`, `002`, `004`, `006`, `007`, `020`, `024`, `025`, `026`).  
- Skip or overwrite existing files with `--no-skip`.  
- Automatically creates subdirectories in correspondence with the input paths.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bdager/CHIRLA.git
   cd CHIRLA
   ```
2. Install requirements:
   ```bash
   pip install requests
   ```


## Usage

The main script is:

```bash
python downloader/chirla_downloader.py --input-file <path_to_urls.txt> --output-dir <output_directory>
```

### **Basic example**

```bash
python downloader/chirla_downloader.py --input-file data/CHIRLA_urls.txt --output-dir ./CHIRLA_data
```

This downloads **all files** listed in `data/CHIRLA_urls.txt` into `CHIRLA_data/CHIRLA/...`.


### **Filtering examples**

#### 1. Download only **benchmark** files

```bash
python chirla_downloader.py --input-file CHIRLA_urls.txt --output-dir ./CHIRLA_data --benchmark
```

#### 2. Download **videos** and **annotations** for sequences 000 and 001

```bash
python chirla_downloader.py --input-file CHIRLA_urls.txt --output-dir ./CHIRLA_data \
    --videos --annotations --specific_seq 000 001
```

#### 3. Download **ReID multi-camera long-term** data

```bash
python chirla_downloader.py --input-file CHIRLA_urls.txt --output-dir ./CHIRLA_data \
    --reid --multi_camera_long_term
```

#### 4. Download **tracking brief occlusions** for validation split

```bash
python chirla_downloader.py --input-file CHIRLA_urls.txt --output-dir ./CHIRLA_data \
    --tracking --brief_occlusions --val
```


### **Arguments**

| Argument                          | Description                                                                 |
|-----------------------------------|-----------------------------------------------------------------------------|
| `--input-file`                     | Path to the `.txt` file containing dataset URLs                             |
| `--output-dir`                     | Directory to store downloaded files                                        |
| `--no-skip`                        | Do not skip files that already exist                                        |
| **General filters**                | `--benchmark`, `--videos`, `--annotations`                                  |
| **Splits**                         | `--train`, `--val`                                                          |
| **ReID filters**                   | `--reid`, `--long_term`, `--multi_camera`, `--multi_camera_long_term`, `--reappearance` |
| **Tracking filters**               | `--tracking`, `--brief_occlusions`, `--multiple_people_occlusions`          |
| `--specific_seq [SEQ ...]`         | Only download sequences from `{000,001,002,004,006,007,020,024,025,026}`    |

All **filters** are combined with **OR logic**, i.e., a file is downloaded if it matches **any** active filter and (optionally) the sequence filter.

---

## Output Structure

The output directory will preserve the CHIRLA dataset structure:

```plaintext
<output-dir>/CHIRLA/
├── videos/
    ├── seq_000/
        ├── camera_0.avi
        ├── camera_1.avi
        ├── ...
    ├── seq_001/
        ├── ...
    ├── ...
├── annotations/
    ├── seq_000/
        ├── camera_0.json
        ├── ...
    ├── ...
└── benchmark/
    ├── reid
        ├── long_term
        ├── multi_camera
        ├── multi_camera_long_term
        └── reappearance
    └── tracking
        ├── brief_occlusions
        └── multiple_people_occlusions

```
 