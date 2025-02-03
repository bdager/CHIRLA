# CHIRLA: Comprehensive High-resolution Identification and Re-identification for Large-scale Analysis

The **CHIRLA** dataset (Comprehensive High-resolution Identification and Re-identification for Large-scale Analysis) is designed for long-term person re-identification (Re-ID) in real-world scenarios. The dataset consists of multi-camera video recordings captured over seven months in an indoor office environment. This dataset aims to facilitate the development and evaluation of Re-ID algorithms capable of handling significant variations in individuals’ appearances, including changes in clothing and physical characteristics. The dataset includes 22 individuals with 963,554 bounding box annotations across 596,345 frames.

![CHIRLA dataset](dataset_sample.jpg?raw=true)

## Data Generation Procedures

The dataset was recorded at the Robotics, Vision, and Intelligent Systems Research Group headquarters at the University of Alicante, Spain. Seven strategically placed Reolink RLC-410W cameras were used to capture videos in a typical office setting, covering areas such as laboratories, hallways, and shared workspaces. Each camera features a 1/2.7" CMOS image sensor with a 5.0-megapixel resolution and an 80° horizontal field of view. The cameras were connected via Ethernet and WiFi to ensure stable streaming and synchronization.

A ROS-based interconnection framework was used to synchronize and retrieve images from all cameras. The dataset includes video recordings at a resolution of 1080×720 pixels, with a consistent frame rate of 30 fps, stored in AVI format with DivX MPEG-4 encoding.

## Data Processing Methods and Steps

Data processing involved a semi-automatic labeling procedure:

- **Detection**: YOLOv8x was used to detect individuals in video frames and extract bounding boxes.
- **Tracking**: The Deep SORT algorithm was employed to generate tracklets and assign unique IDs to detected individuals.
- **Manual Verification**: A custom graphical user interface (GUI) was developed to facilitate manual verification and correction of the automatically generated labels. The GUI is available in the following repository: [CHIRLA Labeling Tool](https://github.com/bdager/preid-labeling-gui).

Bounding boxes and IDs were assigned consistently across different cameras and sequences to maintain identity coherence.

## Data Structure and Format

The dataset comprises:

- **Video Files**: 70 videos, each corresponding to a specific camera view in a sequence, stored in AVI format.
- **Annotation Files**: JSON files containing frame-wise annotations, including bounding box coordinates and identity labels.

The dataset is structured as follows:

```plaintext
videos/
    seq_XXX/
        camera_Y.avi  # Video files for each camera view
annotations/
    seq_XXX/
        camera_Y.json  # Annotation files providing labeled bounding boxes and IDs
benchmark/
```

## Use Cases and Reusability

The CHIRLA dataset is suitable for:

- Long-term person re-identification
- Multi-camera tracking and re-identification
- Single-camera tracking and re-identification

## License

The dataset is publicly available under the **CC-BY** license.
