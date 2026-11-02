# CHIRLA: Comprehensive High-resolution Identification and Re-identification for Large-scale Analysis

<!-- [[`CHIRLA dataset`](https://doi.org/10.57760/sciencedb.20543)] [[`Paper`](https://arxiv.org/pdf/2502.06681)] [[`BibTeX`](#citation)] -->


<a href='https://huggingface.co/papers/2502.06681'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Paper-yellow'></a>
<a href='https://huggingface.co/datasets/bdager/CHIRLA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow'></a>
<a href='https://doi.org/10.57760/sciencedb.20543'><img src='https://img.shields.io/badge/ScienceDB-Dataset-blue?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0zIDVjMC0xLjY2IDQuMDMtMyA5LTNzOSAxLjM0IDkgM3YxNGMwIDEuNjYtNC4wMyAzLTkgM3MtOS0xLjM0LTktM3oiLz48ZyBmaWxsPSJub25lIiBzdHJva2U9IiM1NTUiIHN0cm9rZS13aWR0aD0iMS42Ij48cGF0aCBkPSJNMyA1YzAgMS42NiA0LjAzIDMgOSAzczktMS4zNCA5LTMiLz48cGF0aCBkPSJNMyAxMC43YzAgMS42NiA0LjAzIDMgOSAzczktMS4zNCA5LTMiLz48cGF0aCBkPSJNMyAxNi4yYzAgMS42NiA0LjAzIDMgOSAzczktMS4zNCA5LTMiLz48L2c%2BPC9zdmc%2B'></a>
[![arXiv](https://img.shields.io/badge/%F0%9F%93%84%20ArXiv-2502.06681-b31b1b.svg)](https://arxiv.org/abs/2502.06681)
[![Project Page](https://img.shields.io/badge/%F0%9F%8C%90%20Project-Page-green)](https://bdager.github.io/CHIRLA/)
[![Leaderboard](https://img.shields.io/badge/%F0%9F%8F%86%20Leaderboard-Re--ID%20%26%20Tracking-orange)](https://bdager.github.io/CHIRLA/leaderboard.html)

🏆 **[Leaderboard](https://bdager.github.io/CHIRLA/leaderboard.html)**: Re-ID and tracking results, sortable, with how to submit · 🌐 **[Project page](https://bdager.github.io/CHIRLA/)**


![CHIRLA dataset](assets/dataset_sample.jpg?raw=true)

The **CHIRLA** dataset (Comprehensive High-resolution Identification and Re-identification for Large-scale Analysis) is designed for long-term person re-identification (Re-ID) in real-world scenarios. The dataset consists of multi-camera video recordings captured over **seven months** in an indoor office environment, featuring **22 individuals** with **963,554 bounding box annotations** across **596,345 frames**.

This dataset aims to facilitate the development and evaluation of Re-ID algorithms capable of handling significant variations in individuals' appearances, including changes in clothing and physical characteristics over extended time periods.


## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| **Duration** | 7 months |
| **Individuals** | 22 unique persons |
| **Cameras** | 7 multi-view cameras |
| **Video Files** | 70 sequences |
| **Total Frames** | 596,345 frames |
| **Annotations** | 963,554 bounding boxes |
| **Resolution** | 1080×720 pixels |
| **Frame Rate** | 30 fps |
| **Environment** | Indoor office setting |

## 🎥 Data Generation Procedures

The dataset was recorded at the Robotics, Vision, and Intelligent Systems Research Group headquarters at the University of Alicante, Spain. Seven strategically placed Reolink RLC-410W cameras were used to capture videos in a typical office setting, covering areas such as laboratories, hallways, and shared workspaces. Each camera features a 1/2.7" CMOS image sensor with a 5.0-megapixel resolution and an 80° horizontal field of view. The cameras were connected via Ethernet and WiFi to ensure stable streaming and synchronization.

A ROS-based interconnection framework was used to synchronize and retrieve images from all cameras. The dataset includes video recordings at a resolution of 1080×720 pixels, with a consistent frame rate of 30 fps, stored in AVI format with DivX MPEG-4 encoding.

## Data Processing Methods and Steps

Data processing involved a semi-automatic labeling procedure:

### 1. Automated Detection and Tracking
- **Detection**: YOLOv8x was used to detect individuals in video frames and extract bounding boxes
- **Tracking**: The Deep SORT algorithm was employed to generate tracklets and assign unique IDs to detected individuals

### 2. Manual Verification and Correction
- **Custom GUI**: A specialized graphical user interface was developed for manual verification and correction
- **Identity Consistency**: Bounding boxes and IDs were manually verified for consistency across different cameras and sequences
- **Quality Control**: All annotations underwent thorough manual review to ensure accuracy

> 🔗 **Labeling Tool**: The custom GUI used for annotation is available at: [CHIRLA Labeling Tool](https://github.com/bdager/preid-labeling-gui)

## 📁 Data Structure and Format

The dataset comprises:

- **Video Files**: 70 videos, each corresponding to a specific camera view in a sequence, stored in AVI format
- **Annotation Files**: JSON files containing frame-wise annotations, including bounding box coordinates and identity labels
- **Benchmark Data**: Processed image crops organized for ReID and tracking evaluation

### Directory Structure

```
CHIRLA/
├── videos/                          # Raw video files
│   └── seq_XXX/
│       └── camera_Y.avi             # Video files for each camera view
├── annotations/                     # Frame-level annotations
│   └── seq_XXX/
│       └── camera_Y.json            # Bounding boxes and IDs
└── benchmark/                       # Processed benchmark data
    ├── reid/                        # Person Re-Identification
    │   ├── long_term/               # Long-term ReID scenario
    │   │   ├── train/
    │   │   │   ├── train_0/
    │   │   │   │   └── seq_XXX/
    │   │   │   └── train_1/
    │   │   └── test/
    │   │       ├── test_0/          # Validation subset
    │   │       └── test_1/          # Test subset
    │   ├── multi_camera/            # Multi-camera ReID
    │   ├── multi_camera_long_term/  # Combined scenario
    │   └── reappearance/            # Reappearance detection
    └── tracking/                    # Person Tracking
        ├── brief_occlusions/        # Short-term occlusions
        └── multiple_people_occlusions/  # Multi-person scenarios
```

## 📥 CHIRLA Dataset Downloader

The CHIRLA dataset is hosted on **🤗 Hugging Face Datasets**:  
👉 [bdager/CHIRLA](https://huggingface.co/datasets/bdager/CHIRLA)  

We also provide a **CLI downloader** for direct access.  

For detailed usage instructions (both Hugging Face and CLI options), see [downloader/README.md](downloader/README.md)


## 📈 Benchmark
We propose a benchmark with different scenarios for tracking and reidentification tasks. Please see [benchmark/README.md](benchmark/README.md) to have all the information on how to run your methods on the different challenges of the benchmark.

## Use Cases and Reusability

The CHIRLA dataset is suitable for:

- Long-term person re-identification
- Multi-camera tracking and re-identification
- Single-camera tracking and re-identification

## License

The dataset is publicly available under the **CC-BY** license. The code in this repository is released under the [MIT License](LICENSE).

## Citation

If you use CHIRLA dataset and benchmark, please cite the work as:

```bibtex
@article{bdager2025chirla, 
    title={CHIRLA: Comprehensive High-resolution Identification and Re-identification for Large-scale Analysis}, 
    author={Dominguez-Dager, Bessie and Escalona, Felix and Gomez-Donoso, Fran and Cazorla, Miguel},  
    journal={arXiv preprint arXiv:2502.06681}, 
    year={2025}, 
}
```

## Contact

For any questions or support, please contact bessie.dominguez@ua.es