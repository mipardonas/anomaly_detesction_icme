import argparse
import json
from pathlib import Path
from PIL import Image


DEFAULT_LABEL_DIR = Path("icme/Track1/output/rtdetrv4_anomaly/test_nms_tuned/labels")
DEFAULT_IMAGE_DIR = Path("/data/processed/ICME/Test")
DEFAULT_OUTPUT_JSON = Path("/data/usrs/lnj/icme/Track1/output/rtdetrv4_anomaly/test_nms_tuned/trios_Track1_2.json")

IMAGE_EXTS = [".bmp", ".png", ".jpg", ".jpeg", ".BMP", ".PNG", ".JPG", ".JPEG"]


def find_image_by_stem(stem, image_dir):
    for ext in IMAGE_EXTS:
        image_path = image_dir / f"{stem}{ext}"
        if image_path.exists():
            return image_path
    return None


def yolo_to_xyxy_norm(x_c, y_c, bw, bh):
    x1 = max(0.0, x_c - bw / 2.0)
    y1 = max(0.0, y_c - bh / 2.0)
    x2 = min(1.0, x_c + bw / 2.0)
    y2 = min(1.0, y_c + bh / 2.0)
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def parse_label_file(label_path):
    defect_info = []

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            category_id = int(float(parts[0]))
            x_c = float(parts[1])
            y_c = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            defect_info.append(
                {
                    "category_id": category_id,
                    "bbox": yolo_to_xyxy_norm(x_c, y_c, bw, bh),
                }
            )

    return defect_info


def build_submission_json(label_dir, image_dir):
    results = []
    label_files = sorted(label_dir.glob("*.txt"))

    for label_path in label_files:
        stem = label_path.stem
        image_path = find_image_by_stem(stem, image_dir)

        if image_path is None:
            print(f"Warning: image not found for label file: {label_path.name}")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        result_item = {
            "file_name": image_path.name,
            "width": width,
            "height": height,
            "defect_info": parse_label_file(label_path),
        }
        results.append(result_item)

    return {"results": results}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main():
    args = parse_args()
    submission_data = build_submission_json(args.label_dir, args.image_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(submission_data, f, ensure_ascii=False, indent=4)

    print(f"Saved submission json to: {args.output_json}")
    print(f"Total files written: {len(submission_data['results'])}")


if __name__ == "__main__":
    main()
