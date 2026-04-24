from __future__ import annotations

import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 1337
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
LEGACY_TRAIN_DIR = DATASET_DIR / "Training"
LEGACY_TEST_DIR = DATASET_DIR / "Testing"
OUTPUT_SPLITS = {
    "train": DATASET_DIR / "train",
    "val": DATASET_DIR / "val",
    "test": DATASET_DIR / "test",
}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_augmented_file(path: Path) -> bool:
    return "aug" in path.stem.lower()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_source_files() -> dict[str, list[Path]]:
    source_files: dict[str, list[Path]] = defaultdict(list)

    for source_root in (LEGACY_TRAIN_DIR, LEGACY_TEST_DIR):
        if not source_root.is_dir():
            raise FileNotFoundError(
                f"Missing source dataset folder: '{source_root}'."
            )

        for class_name in CLASS_NAMES:
            class_dir = source_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class folder: '{class_dir}'."
                )

            for path in class_dir.iterdir():
                if not is_image_file(path) or is_augmented_file(path):
                    continue
                source_files[class_name].append(path)

    return source_files


def deduplicate_files(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen_hashes: set[str] = set()

    for path in sorted(paths):
        digest = file_md5(path)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        unique_paths.append(path)

    return unique_paths


def build_split_counts(total: int) -> dict[str, int]:
    if total < 3:
        raise ValueError(f"Need at least 3 original images per class, found {total}.")

    test_count = max(1, round(total * TEST_RATIO))
    val_count = max(1, round(total * VAL_RATIO))

    if test_count + val_count >= total:
        test_count = 1
        val_count = 1

    train_count = total - val_count - test_count
    if train_count < 1:
        raise ValueError(f"Unable to create train split for class with {total} images.")

    return {"train": train_count, "val": val_count, "test": test_count}


def reset_output_dirs() -> None:
    for split_dir in OUTPUT_SPLITS.values():
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for class_name in CLASS_NAMES:
            (split_dir / class_name).mkdir(parents=True, exist_ok=True)


def write_split(class_name: str, split_name: str, files: list[Path]) -> None:
    destination_dir = OUTPUT_SPLITS[split_name] / class_name
    for index, source_path in enumerate(files, start=1):
        destination_name = f"{class_name}_{index:04d}{source_path.suffix.lower()}"
        shutil.copy2(source_path, destination_dir / destination_name)


def prepare_splits() -> None:
    random.seed(SEED)
    source_files = collect_source_files()
    reset_output_dirs()

    print("Preparing leakage-safe dataset splits...")
    print("Rules: remove offline augmented files, deduplicate exact matches, rebuild train/val/test.")

    for class_name in CLASS_NAMES:
        unique_files = deduplicate_files(source_files[class_name])
        random.shuffle(unique_files)

        split_counts = build_split_counts(len(unique_files))
        train_end = split_counts["train"]
        val_end = train_end + split_counts["val"]

        split_map = {
            "train": unique_files[:train_end],
            "val": unique_files[train_end:val_end],
            "test": unique_files[val_end:],
        }

        for split_name, files in split_map.items():
            write_split(class_name, split_name, files)

        print(
            f"{class_name:12s} total={len(unique_files):4d} "
            f"train={len(split_map['train']):4d} "
            f"val={len(split_map['val']):4d} "
            f"test={len(split_map['test']):4d}"
        )

    print("\nCreated dataset splits:")
    for split_name, split_dir in OUTPUT_SPLITS.items():
        print(f"  {split_name}: {split_dir}")
    print("\nNext steps:")
    print("  1. Run: python prepare_dataset_splits.py")
    print("  2. Run: python main.py")


if __name__ == "__main__":
    prepare_splits()
