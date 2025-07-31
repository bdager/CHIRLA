#!/usr/bin/env python3

import os
import requests
import argparse
import urllib.parse


ALLOWED_SEQUENCES = {"000", "001", "002", "004", "006", "007", "020", "024", "025", "026"}


def download_files(input_file, output_dir, skip_existing=True, filters=None, seqs=None):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    # Apply keyword filters if provided
    if filters:
        filters_lower = [f.lower() for f in filters]
        urls = [url for url in urls if any(f in url.lower() for f in filters_lower)]
        print(f"Filtered URLs with keywords {filters}: {len(urls)} found.")

    # Apply sequence filters if provided
    if seqs:
        urls = [url for url in urls if any(seq in url for seq in seqs)]
        print(f"Filtered URLs with sequences {seqs}: {len(urls)} found.")

    for url in urls:
        # Determine relative path starting from CHIRLA_dataset
        if "CHIRLA_dataset" not in url:
            print(f"Skipping (no CHIRLA_dataset in path): {url}")
            continue

        relative_path = url.split("CHIRLA_dataset", 1)[1].lstrip("/\\")
        # If &fileName= appears at the end, keep only the last real filename
        if "&fileName=" in relative_path:
            relative_path = relative_path.split("&fileName=")[0]
        output_path = os.path.join(output_dir, "CHIRLA", relative_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if skip_existing and os.path.exists(output_path):
            print(f"Skipping (already exists): {output_path}")
            continue

        print(f"Downloading: {url} -> {output_path}")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(output_path, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)
        except requests.RequestException as e:
            print(f"Failed to download {url}: {e}")

    print(f"\nAll downloads completed to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download files from a list of URLs contained in a text file, "
                    "preserving the folder structure starting from CHIRLA_dataset."
    )
    parser.add_argument("--input-file", type=str,
                        default="data/CHIRLA_urls.txt",
                        help="Path to the text file containing URLs")
    parser.add_argument("--output-dir", type=str,
                        default="data",                        
                        help="Directory to save downloaded files")
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Do not skip already existing files"
    )

    # General filters
    parser.add_argument("--benchmark", action="store_true", help="Only download URLs containing 'benchmark'")
    parser.add_argument("--videos", action="store_true", help="Only download URLs containing 'videos'")
    parser.add_argument("--annotations", action="store_true", help="Only download URLs containing 'annotations'")
    parser.add_argument("--reid", action="store_true", help="Only download URLs containing 'reid'")
    parser.add_argument("--tracking", action="store_true", help="Only download URLs containing 'tracking'")
    parser.add_argument("--train", action="store_true", help="Only download URLs containing 'train'")
    parser.add_argument("--val", action="store_true", help="Only download URLs containing 'val'")

    # Subfolder filters (ReID and Tracking)
    parser.add_argument("--long_term", action="store_true", help="Only URLs containing 'long_term'")
    parser.add_argument("--multi_camera", action="store_true", help="Only URLs containing 'multi_camera'")
    parser.add_argument("--multi_camera_long_term", action="store_true", help="Only URLs containing 'multi_camera_long_term'")
    parser.add_argument("--reappearance", action="store_true", help="Only URLs containing 'reappearance'")
    parser.add_argument("--brief_occlusions", action="store_true", help="Only URLs containing 'brief_occlusions'")
    parser.add_argument("--multiple_people_occlusions", action="store_true", help="Only URLs containing 'multiple_people_occlusions'")

    # Specific sequence filter
    parser.add_argument(
        "--specific_seq",
        nargs="+",
        choices=sorted(ALLOWED_SEQUENCES),
        help="Download only URLs that contain one of these sequence codes."
    )

    args = parser.parse_args()

    # Collect filters based on flags
    filters = []
    if args.benchmark:
        filters.append("benchmark")
    if args.videos:
        filters.append("videos")
    if args.annotations:
        filters.append("annotations")
    if args.reid:
        filters.append("reid")
    if args.tracking:
        filters.append("tracking")
    if args.train:
        filters.append("train")
    if args.val:
        filters.append("val")
    if args.long_term:
        filters.append("long_term")
    if args.multi_camera:
        filters.append("multi_camera")
    if args.multi_camera_long_term:
        filters.append("multi_camera_long_term")
    if args.reappearance:
        filters.append("reappearance")
    if args.brief_occlusions:
        filters.append("brief_occlusions")
    if args.multiple_people_occlusions:
        filters.append("multiple_people_occlusions")

    download_files(
        args.input_file,
        args.output_dir,
        skip_existing=not args.no_skip,
        filters=filters if filters else None,
        seqs=args.specific_seq
    )