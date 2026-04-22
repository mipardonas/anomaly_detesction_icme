#!/usr/bin/env python3
"""Reorganize heatmapscore .npy files based on a new image split.

Example:
    python tools/dataset/reorganize_heatmapscore_by_images.py \
        --images-root /public/home/nc260224a/rt-trv4/ICME/track1compose/images \
        --heatmap-root /public/home/nc260224a/rt-trv4/INP-Former/output/res_nocls/heatmapscore \
        --output-root /public/home/nc260224a/rt-trv4/INP-Former/output/res_nocls/heatmapscore_reorg \
        --execute

By default the script performs a dry run and only prints the planned moves.
Pass --execute to apply changes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move or copy heatmapscore .npy files into new split folders based on "
            "image filenames under images_root/train|val|test."
        )
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Root directory containing image split folders such as train/val/test.",
    )
    parser.add_argument(
        "--heatmap-root",
        type=Path,
        required=True,
        help="Root directory containing heatmapscore split folders with .npy files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root directory to save reorganized .npy files into train/val/test style "
            "subfolders. Default: same as --heatmap-root"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split folder names under images-root to process. Default: train val test",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform file operations. Without this flag, only print a dry run.",
    )
    parser.add_argument(
        "--mode",
        choices=["move", "copy"],
        default="move",
        help="How to place matched .npy files into new split folders. Default: move",
    )
    parser.add_argument(
        "--recursive-images",
        action="store_true",
        help="Recursively scan image files inside each split directory.",
    )
    parser.add_argument(
        "--recursive-heatmaps",
        action="store_true",
        help="Recursively scan .npy files inside heatmap-root.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip when the destination .npy already exists instead of overwriting it.",
    )
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=None,
        help=(
            "Only scan these subdirectories directly under heatmap-root for source .npy "
            "files. Example: --source-splits train val test"
        ),
    )
    parser.add_argument(
        "--prefer-source-splits",
        nargs="+",
        default=["train", "val", "test"],
        help=(
            "When multiple .npy files match the same image stem, prefer files from these "
            "heatmap subdirectories in order. Default: train val test"
        ),
    )
    parser.add_argument(
        "--allow-multiple-matches",
        action="store_true",
        help=(
            "Move all matching .npy files for the same image stem. By default only one "
            "best match is selected."
        ),
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    name = name.strip()
    if name.endswith("_anomaly_score"):
        name = name[: -len("_anomaly_score")]
    return name


def discover_split_dir(root: Path, split_name: str) -> Path | None:
    if not root.exists():
        return None

    exact = root / split_name
    if exact.is_dir():
        return exact

    lower_map = {child.name.lower(): child for child in root.iterdir() if child.is_dir()}
    return lower_map.get(split_name.lower())


def collect_image_stems(split_dir: Path, recursive: bool) -> set[str]:
    pattern = "**/*" if recursive else "*"
    stems = set()
    for path in split_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            stems.add(path.stem)
    return stems


def build_child_dir_lookup(root: Path) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = defaultdict(list)
    for child in root.iterdir():
        if child.is_dir():
            lookup[child.name.lower()].append(child)
    return lookup


def resolve_source_dirs(heatmap_root: Path, source_splits: list[str] | None) -> list[Path]:
    child_lookup = build_child_dir_lookup(heatmap_root)
    exact_lookup = {
        child.name: child
        for child in heatmap_root.iterdir()
        if child.is_dir()
    }
    if source_splits is None:
        return []

    resolved_dirs: list[Path] = []
    seen: set[Path] = set()
    for split_name in source_splits:
        exact_path = exact_lookup.get(split_name)
        if exact_path is not None:
            if exact_path not in seen:
                resolved_dirs.append(exact_path)
                seen.add(exact_path)
            continue

        matched_paths = child_lookup.get(split_name.lower(), [])
        if len(matched_paths) == 1:
            path = matched_paths[0]
            if path not in seen:
                resolved_dirs.append(path)
                seen.add(path)
    return resolved_dirs


def build_heatmap_index(
    source_dirs: list[Path],
    recursive: bool,
) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for source_dir in source_dirs:
        pattern = "**/*.npy" if recursive else "*.npy"
        for path in source_dir.glob(pattern):
            if path.is_file():
                index[normalize_name(path.stem)].append(path)
    return index


def build_preference_rank(preferred_split_names: list[str]) -> dict[str, int]:
    return {name.lower(): idx for idx, name in enumerate(preferred_split_names)}


def choose_best_match(
    matched_paths: list[Path],
    preference_rank: dict[str, int],
) -> Path:
    def sort_key(path: Path) -> tuple[int, str]:
        parent_name = path.parent.name.lower()
        return (preference_rank.get(parent_name, len(preference_rank)), str(path))

    return sorted(matched_paths, key=sort_key)[0]


def choose_target_split_dir(
    output_root: Path,
    split_name: str,
    existing_split_dirs: dict[str, Path],
) -> Path:
    split_key = split_name.lower()
    if split_key in existing_split_dirs:
        return existing_split_dirs[split_key]
    return output_root / split_name


def safe_transfer(src: Path, dst: Path, mode: str, skip_existing: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if skip_existing:
            return "skipped_existing"
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))
    return "done"


def main() -> int:
    args = parse_args()

    if not args.images_root.is_dir():
        print(f"[ERROR] images-root does not exist: {args.images_root}", file=sys.stderr)
        return 1

    if not args.heatmap_root.is_dir():
        print(f"[ERROR] heatmap-root does not exist: {args.heatmap_root}", file=sys.stderr)
        return 1

    output_root = args.output_root or args.heatmap_root

    existing_split_dirs = {
        child.name.lower(): child
        for child in output_root.iterdir()
        if child.is_dir()
    } if output_root.exists() else {}
    requested_source_splits = args.source_splits or args.splits
    source_dirs = resolve_source_dirs(args.heatmap_root, requested_source_splits)
    if not source_dirs:
        print(
            "[ERROR] No source heatmap subdirectories matched --source-splits.",
            file=sys.stderr,
        )
        return 1
    print(
        "[INFO] Heatmap source dirs: "
        + ", ".join(str(path.name) for path in source_dirs)
    )
    print(f"[INFO] Output root: {output_root}")

    split_to_image_stems: dict[str, set[str]] = {}
    for split_name in args.splits:
        split_dir = discover_split_dir(args.images_root, split_name)
        if split_dir is None:
            print(
                f"[WARN] Split folder not found under images-root: {split_name}",
                file=sys.stderr,
            )
            continue

        stems = collect_image_stems(split_dir, recursive=args.recursive_images)
        split_to_image_stems[split_name] = stems
        print(f"[INFO] {split_name}: found {len(stems)} image files in {split_dir}")

    if not split_to_image_stems:
        print("[ERROR] No valid split folders found under images-root.", file=sys.stderr)
        return 1

    heatmap_index = build_heatmap_index(source_dirs, recursive=args.recursive_heatmaps)
    print(f"[INFO] Indexed {sum(len(v) for v in heatmap_index.values())} heatmap files")
    preference_rank = build_preference_rank(args.prefer_source_splits)

    planned_moves: list[tuple[Path, Path]] = []
    missing_names: dict[str, list[str]] = defaultdict(list)
    duplicate_names: dict[str, list[str]] = defaultdict(list)

    for split_name, image_stems in split_to_image_stems.items():
        target_split_dir = choose_target_split_dir(
            output_root, split_name, existing_split_dirs
        )
        for image_stem in sorted(image_stems):
            matched_paths = heatmap_index.get(image_stem, [])

            if not matched_paths:
                missing_names[split_name].append(image_stem)
                continue

            if len(matched_paths) > 1:
                duplicate_names[split_name].append(image_stem)

            selected_paths = (
                matched_paths
                if args.allow_multiple_matches
                else [choose_best_match(matched_paths, preference_rank)]
            )

            for src in selected_paths:
                dst = target_split_dir / src.name
                if src.resolve() == dst.resolve():
                    continue
                planned_moves.append((src, dst))

    print(f"[INFO] Planned {len(planned_moves)} file operations")

    if duplicate_names:
        for split_name, names in duplicate_names.items():
            print(
                f"[WARN] {split_name}: {len(names)} image names matched multiple .npy files",
                file=sys.stderr,
            )
            if not args.allow_multiple_matches:
                preview = ", ".join(names[:10])
                if preview:
                    print(
                        f"       Using preferred source order: "
                        f"{', '.join(args.prefer_source_splits)}",
                        file=sys.stderr,
                    )
                    print(f"       Examples: {preview}", file=sys.stderr)

    if missing_names:
        for split_name, names in missing_names.items():
            print(
                f"[WARN] {split_name}: missing {len(names)} matching .npy files",
                file=sys.stderr,
            )
            preview = ", ".join(names[:10])
            if preview:
                print(f"       Examples: {preview}", file=sys.stderr)

    preview_count = min(20, len(planned_moves))
    for src, dst in planned_moves[:preview_count]:
        print(f"[PLAN] {src} -> {dst}")
    if len(planned_moves) > preview_count:
        print(f"[INFO] ... and {len(planned_moves) - preview_count} more")

    if not args.execute:
        print("[INFO] Dry run only. Add --execute to perform the operations.")
        return 0

    done = 0
    skipped_existing = 0
    failed = 0
    for src, dst in planned_moves:
        try:
            result = safe_transfer(src, dst, args.mode, args.skip_existing)
            if result == "skipped_existing":
                skipped_existing += 1
            else:
                done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[ERROR] Failed: {src} -> {dst} | {exc}", file=sys.stderr)

    print(
        f"[DONE] success={done}, skipped_existing={skipped_existing}, failed={failed}, "
        f"missing={sum(len(v) for v in missing_names.values())}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
