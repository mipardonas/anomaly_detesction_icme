#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a filtered YOLO label directory by removing selected class ids."
    )
    parser.add_argument("--src", required=True, help="Source label directory")
    parser.add_argument("--dst", required=True, help="Destination label directory")
    parser.add_argument(
        "--drop-classes",
        nargs="+",
        type=int,
        required=True,
        help="Class ids to remove from the copied labels",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep empty label files after filtering instead of deleting them",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    drop_classes = set(args.drop_classes)

    if not src.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    total_files = 0
    changed_files = 0
    removed_lines = 0
    emptied_files = 0

    for src_file in sorted(src.glob("*.txt")):
        total_files += 1
        kept_lines = []
        original_lines = src_file.read_text(encoding="utf-8").splitlines()

        for line in original_lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            if cls_id in drop_classes:
                removed_lines += 1
                continue
            kept_lines.append(line)

        if len(kept_lines) != len(original_lines):
            changed_files += 1

        dst_file = dst / src_file.name
        if kept_lines:
            dst_file.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
        elif args.keep_empty:
            emptied_files += 1
            dst_file.write_text("", encoding="utf-8")

    print(f"source: {src}")
    print(f"destination: {dst}")
    print(f"total_files: {total_files}")
    print(f"changed_files: {changed_files}")
    print(f"removed_lines: {removed_lines}")
    print(f"emptied_files: {emptied_files}")


if __name__ == "__main__":
    main()
