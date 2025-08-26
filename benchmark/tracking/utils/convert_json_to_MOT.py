"""
This file converts .json format from our dataset to .txt gt file in the MOTChallenge format:
<frame>,
<id>,
<bb_left>,
<bb_top>,
<bb_width>,
<bb_height>,
<conf>,
<class>,
<visibility>

Confidence, class and visibility are equal to 1. (class = '1' correspondences to "Pedestrian" class in MOT dataset format).
Here is an example:

1, 3, 794.27, 247.59, 71.245, 174.88, 1, 1, 1
"""

import json
import argparse
from typing import Dict, Any


def read_json_file(file_path: str) -> Dict[str, Any]:
    """Read JSON annotation file."""
    with open(file_path, "r") as json_file:
        return json.load(json_file)


def convert_to_txt(json_data: Dict[str, Any], output_file_path: str):
    """
    Convert a JSON dict (frame -> list[detections]) to MOT16 txt format.
    Ensures:
      - Numeric frame ordering
      - Deterministic ordering of detections per frame (by id then bbox)
    """
    # Sort frames numerically (keys are strings like "1", "2", ...)
    frame_keys = sorted(json_data.keys(), key=lambda k: int(k))
    with open(output_file_path, "w") as txt_file:
        for frame_key in frame_keys:
            detections = json_data[frame_key]
            # Sort detections for reproducibility (by id)
            try:
                detections = sorted(detections, key=lambda d: (d.get("id", -1), d.get("BboxP", [0,0,0,0])[0]))
            except Exception:
                pass
            for det in detections:
                id_number = det.get("id", -1)
                bbox = det.get("BboxP", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue  # malformed
                # Convert x1,y1,x2,y2 -> x,y,w,h
                x1, y1, x2, y2 = bbox
                w = x2 - x1
                h = y2 - y1
                line = f"{int(frame_key)},{id_number},{x1},{y1},{w},{h},1,1,1\n"
                txt_file.write(line)


def main(args):
    json_path = args.json
    output_path = args.output

    json_path = json_path + ".json"
    json_data = read_json_file(json_path)

    # Create txt output file
    if output_path is None:
        output_path = json_path.replace(".json", ".txt")
    convert_to_txt(json_data, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument(
        "--json", type=str, help="Json file path. e.g. camera_2_2023-12-01-11:05:52"
    )
    parser.add_argument(
        "--output", type=str, default=None, help=".txt output file path"
    )

    args = parser.parse_args()

    main(args)
