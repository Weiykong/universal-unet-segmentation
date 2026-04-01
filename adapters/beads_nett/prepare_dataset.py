import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def list_files(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def normalize_stem(path: Path, is_mask: bool) -> str:
    stem = path.stem
    if is_mask and stem.endswith("_mask"):
        stem = stem[:-5]
    return stem


def validate_pairs(image_paths: List[Path], mask_paths: List[Path]) -> None:
    if not image_paths:
        raise ValueError("No source images found.")
    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Image/mask count mismatch: {len(image_paths)} images vs {len(mask_paths)} masks."
        )

    for image_path, mask_path in zip(image_paths, mask_paths):
        image_key = normalize_stem(image_path, is_mask=False)
        mask_key = normalize_stem(mask_path, is_mask=True)
        if image_key != mask_key:
            raise ValueError(
                f"Pair mismatch: {image_path.name} does not match {mask_path.name}."
            )


def backup_existing_files(source_dir: Path, backup_dir: Path) -> int:
    moved = 0
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.iterdir()):
        if path.name == ".gitkeep":
            continue
        target = backup_dir / path.name
        shutil.move(str(path), str(target))
        moved += 1
    return moved


def clear_prepared_stage(stage_dir: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)


def materialize_files(source_paths: List[Path], dest_dir: Path, mode: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_paths:
        dest_path = dest_dir / source_path.name
        if mode == "symlink":
            dest_path.symlink_to(source_path.resolve())
        else:
            shutil.copy2(source_path, dest_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the train_beads_nett dataset for the existing universal_unet layout."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/Users/weiyuankong/train_beads_nett/ml_dataset"),
        help="Dataset root containing images/ and masks/.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing data/images and data/masks.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Whether to symlink or copy source files into the training layout.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    repo_root = args.repo_root.resolve()

    image_source_dir = source_root / "images"
    mask_source_dir = source_root / "masks"
    image_dest_dir = repo_root / "data" / "images"
    mask_dest_dir = repo_root / "data" / "masks"
    backup_root = repo_root / "data" / "backups" / f"pre_beads_nett_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stage_root = repo_root / "data" / "_beads_nett_stage"

    if not image_source_dir.is_dir() or not mask_source_dir.is_dir():
        raise FileNotFoundError(
            f"Expected source directories at {image_source_dir} and {mask_source_dir}."
        )

    image_paths = list_files(image_source_dir)
    mask_paths = list_files(mask_source_dir)
    validate_pairs(image_paths, mask_paths)

    clear_prepared_stage(stage_root)
    materialize_files(image_paths, stage_root / "images", args.mode)
    materialize_files(mask_paths, stage_root / "masks", args.mode)

    image_dest_dir.mkdir(parents=True, exist_ok=True)
    mask_dest_dir.mkdir(parents=True, exist_ok=True)

    moved_images = backup_existing_files(image_dest_dir, backup_root / "images")
    moved_masks = backup_existing_files(mask_dest_dir, backup_root / "masks")

    for staged_file in sorted((stage_root / "images").iterdir()):
        shutil.move(str(staged_file), str(image_dest_dir / staged_file.name))
    for staged_file in sorted((stage_root / "masks").iterdir()):
        shutil.move(str(staged_file), str(mask_dest_dir / staged_file.name))

    shutil.rmtree(stage_root)

    report = {
        "source_root": str(source_root),
        "repo_root": str(repo_root),
        "mode": args.mode,
        "prepared_images": len(image_paths),
        "prepared_masks": len(mask_paths),
        "backup_root": str(backup_root),
        "backed_up_images": moved_images,
        "backed_up_masks": moved_masks,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
