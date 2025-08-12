#!/usr/bin/env python3
import os
import csv
import argparse

FILE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def generate_metadata_for_scenario(root_dir, task, scenario, output_dir):
    """
    Generates train and test CSVs for a given scenario.
    """
    scenario_path = os.path.join(root_dir, task, scenario)
    if not os.path.exists(scenario_path):
        print(f"Skipping missing scenario: {scenario_path}")
        return

    for split in ["train", "test"]:
        split_path = os.path.join(scenario_path, split)
        if not os.path.exists(split_path):
            continue

        rows = []
        val_rows = []  # Separate list for validation data
        train_rows = []  # Separate list for training data
        for subset in os.listdir(split_path):  # train_0, test_0, etc.
            subset_path = os.path.join(split_path, subset)
            if not os.path.isdir(subset_path):
                continue

            for seq in os.listdir(subset_path):
                if not seq.startswith("seq_"):
                    continue

                seq_path = os.path.join(subset_path, seq, "imgs")
                if not os.path.exists(seq_path):
                    continue

                for cam_name in os.listdir(seq_path):
                    cam_path = os.path.join(seq_path, cam_name)
                    if not os.path.isdir(cam_path):
                        continue

                    for person_id in os.listdir(cam_path):
                        person_path = os.path.join(cam_path, person_id)
                        if not os.path.isdir(person_path):
                            continue  # Skip non-ID files like JSONs

                        for file in os.listdir(person_path):
                            if file.lower().endswith(FILE_EXTS):
                                file_path = os.path.join(person_path, file)
                                rel_path = os.path.relpath(file_path, root_dir)
                                row_data = {
                                    "image_path": rel_path,
                                    "id": person_id,  # Can be negative
                                    "camera": cam_name,
                                    "sequence": seq,
                                    "subset": subset,
                                    # "split": split,
                                    # "scenario": scenario,
                                    # "task": task
                                }
                                
                                # Add to validation list if subset is test_0
                                if subset == "test_0":
                                    val_rows.append(row_data)
                                elif subset == "train_0":
                                    train_rows.append(row_data)
                                else:
                                    rows.append(row_data)

        # Save CSV for this scenario/split
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main CSV (excluding test_0 subset)
        if rows:
            if split == "train" and task == "reid":
                csv_name = f"{task}_{scenario}_gallery.csv"
            elif split == "test" and task == "reid":
                csv_name = f"{task}_{scenario}_query.csv"
            else:
                csv_name = f"{task}_{scenario}_{split}.csv"
            output_csv = os.path.join(output_dir, csv_name)
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "image_path", "id", "camera", "sequence", "subset"
                ])
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved: {output_csv} ({len(rows)} entries)")

        # Save validation CSV (test_0 subset only)
        if val_rows:
            val_csv = os.path.join(output_dir, f"{task}_{scenario}_val.csv")
            with open(val_csv, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "image_path", "id", "camera", "sequence", "subset"
                ])
                writer.writeheader()
                writer.writerows(val_rows)
            print(f"Saved validation: {val_csv} ({len(val_rows)} entries)")

        if train_rows:
            train_csv = os.path.join(output_dir, f"{task}_{scenario}_train.csv")
            with open(train_csv, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    "image_path", "id", "camera", "sequence", "subset"
                ])
                writer.writeheader()
                writer.writerows(train_rows)
            print(f"Saved training: {train_csv} ({len(train_rows)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 12 CSV files for CHIRLA benchmark metadata.")
    parser.add_argument("--root-dir", default="data/CHIRLA/benchmark", 
                        help="Path to CHIRLA benchmark folder")
    parser.add_argument("--output-dir", default="benchmark/metadata", 
                        help="Folder to save metadata CSVs")
    args = parser.parse_args()

    # Define scenarios per task
    reid_scenarios = ["long_term", "multi_camera", "multi_camera_long_term", "reappearance"]
    tracking_scenarios = ["brief_occlusions", "multiple_people_occlusions"]

    for scenario in reid_scenarios:
        generate_metadata_for_scenario(args.root_dir, "reid", scenario, args.output_dir)

    for scenario in tracking_scenarios:
        generate_metadata_for_scenario(args.root_dir, "tracking", scenario, args.output_dir)
