"""
Convert a COCO annotation file into a one-class COCO annotation file.
"""

import argparse
import json
from pathlib import Path


def convert_to_oneclass(src_path, dst_path, class_id=0, class_name="anomaly"):
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = dict(data)
    converted["categories"] = [{"id": class_id, "name": class_name, "supercategory": class_name}]

    annotations = []
    for ann in data.get("annotations", []):
        ann = dict(ann)
        ann["category_id"] = class_id
        annotations.append(ann)

    converted["annotations"] = annotations

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=str)
    parser.add_argument("--dst", required=True, type=str)
    parser.add_argument("--class-id", default=0, type=int)
    parser.add_argument("--class-name", default="anomaly", type=str)
    args = parser.parse_args()

    convert_to_oneclass(args.src, args.dst, class_id=args.class_id, class_name=args.class_name)
    print(f"Saved one-class annotation to: {args.dst}")


if __name__ == "__main__":
    main()
