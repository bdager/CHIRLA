### CSV Structure

Each CSV file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `image_path` | string | Relative path to the image file |
| `id` | integer | Person identifier (positive or negative) |
| `camera` | string | Camera identifier with timestamp |
| `sequence` | string | Sequence identifier (e.g., seq_024) |
| `subset` | string | Data subset (train_0, test_0, etc.) |

> CSV files for tracking scenarios do not contain `subset` column.
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