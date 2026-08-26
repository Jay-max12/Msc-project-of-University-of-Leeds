"""Pre-defined stratified CV splits shared by roi_repr and YOLO one-step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from train_resnet50_roi_classifier import load_bbox_annotations, read_split_file

Sample = Tuple[str, int]
FoldSplits = Dict[str, object]
StratifyMode = Literal["binary", "multiclass", "none"]

_SAMPLE = List  # JSON stores [name, class_id]


def list_valid_samples(dataset_root: Path, split: str) -> List[Sample]:
    bbox = load_bbox_annotations(dataset_root / "bbox_annot.json")
    imgs_dir = dataset_root / "imgs"
    rows = read_split_file(dataset_root / f"{split}.txt")
    out: List[Sample] = []
    for name, cls_id in rows:
        if name not in bbox:
            continue
        if not (imgs_dir / name).exists():
            continue
        out.append((name, int(cls_id)))
    return out


def stratify_labels(samples: List[Sample], stratify: StratifyMode) -> np.ndarray:
    if stratify == "binary":
        return np.array([1 if int(cls_id) == 2 else 0 for _, cls_id in samples], dtype=int)
    if stratify == "multiclass":
        return np.array([int(cls_id) for _, cls_id in samples], dtype=int)
    return np.zeros(len(samples), dtype=int)


def cv_split_filename(n_cv_folds: int, seed: int, stratify: StratifyMode) -> str:
    return f"cv{n_cv_folds}_seed{seed}_{stratify}.json"


def default_cv_split_path(
    dataset_root: Path,
    *,
    n_cv_folds: int = 3,
    seed: int = 42,
    stratify: StratifyMode = "binary",
) -> Path:
    return dataset_root / "cv_splits" / cv_split_filename(n_cv_folds, seed, stratify)


def resolve_cv_split_path(
    dataset_root: Path,
    *,
    n_cv_folds: int,
    seed: int,
    stratify: StratifyMode,
    cv_split_file: Path | str | None = None,
) -> Path:
    if cv_split_file is not None:
        return Path(cv_split_file)
    return default_cv_split_path(dataset_root, n_cv_folds=n_cv_folds, seed=seed, stratify=stratify)


def _samples_to_json(samples: List[Sample]) -> List[_SAMPLE]:
    return [[name, int(cls_id)] for name, cls_id in samples]


def _samples_from_json(rows: List[_SAMPLE]) -> List[Sample]:
    return [(str(name), int(cls_id)) for name, cls_id in rows]


def generate_cv_split_manifest(
    dataset_root: Path,
    *,
    n_cv_folds: int = 3,
    seed: int = 42,
    stratify: StratifyMode = "binary",
) -> dict:
    train_rows = list_valid_samples(dataset_root, "train")
    test_rows = list_valid_samples(dataset_root, "test")
    y = stratify_labels(train_rows, stratify)
    idx = np.arange(len(train_rows))
    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
    folds = []
    for fold_i, (tr_i, va_i) in enumerate(skf.split(idx, y)):
        folds.append(
            {
                "fold": int(fold_i),
                "train": _samples_to_json([train_rows[int(i)] for i in tr_i]),
                "val": _samples_to_json([train_rows[int(i)] for i in va_i]),
            }
        )
    return {
        "version": 1,
        "dataset_root": str(dataset_root.resolve()),
        "n_cv_folds": int(n_cv_folds),
        "seed": int(seed),
        "stratify": stratify,
        "train_split": "train.txt",
        "test_split": "test.txt",
        "test": _samples_to_json(test_rows),
        "folds": folds,
    }


def save_cv_split_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[CV splits] saved: {path}")


def load_cv_split_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_cv_split_manifest(
    dataset_root: Path,
    *,
    n_cv_folds: int = 3,
    seed: int = 42,
    stratify: StratifyMode = "binary",
    cv_split_file: Path | str | None = None,
) -> Path:
    path = resolve_cv_split_path(
        dataset_root,
        n_cv_folds=n_cv_folds,
        seed=seed,
        stratify=stratify,
        cv_split_file=cv_split_file,
    )
    if path.exists():
        return path
    print(f"[CV splits] manifest not found, generating: {path}")
    manifest = generate_cv_split_manifest(
        dataset_root,
        n_cv_folds=n_cv_folds,
        seed=seed,
        stratify=stratify,
    )
    save_cv_split_manifest(path, manifest)
    return path


def load_cv_folds(
    dataset_root: Path,
    *,
    n_cv_folds: int = 3,
    seed: int = 42,
    stratify: StratifyMode = "binary",
    cv_split_file: Path | str | None = None,
) -> Tuple[List[FoldSplits], Path]:
    path = ensure_cv_split_manifest(
        dataset_root,
        n_cv_folds=n_cv_folds,
        seed=seed,
        stratify=stratify,
        cv_split_file=cv_split_file,
    )
    manifest = load_cv_split_manifest(path)
    if int(manifest["n_cv_folds"]) != int(n_cv_folds):
        raise ValueError(
            f"CV split file {path} has n_cv_folds={manifest['n_cv_folds']}, expected {n_cv_folds}"
        )
    if int(manifest["seed"]) != int(seed):
        raise ValueError(f"CV split file {path} has seed={manifest['seed']}, expected {seed}")
    if str(manifest["stratify"]) != stratify:
        raise ValueError(
            f"CV split file {path} has stratify={manifest['stratify']!r}, expected {stratify!r}"
        )

    test_rows = _samples_from_json(manifest["test"])
    folds: List[FoldSplits] = []
    for fold in manifest["folds"]:
        folds.append(
            {
                "fold": int(fold["fold"]),
                "train": _samples_from_json(fold["train"]),
                "val": _samples_from_json(fold["val"]),
                "test": test_rows,
            }
        )
    return folds, path


def stratify_for_task_mode(task_mode: str) -> StratifyMode:
    if task_mode in ("binary", "detect_cls"):
        return "binary"
    if task_mode in ("multiclass",):
        return "multiclass"
    if task_mode in ("detect",):
        return "none"
    raise ValueError(f"Unsupported task_mode for CV splits: {task_mode}")


def main() -> None:
    parser = argparse.ArgumentParser("Generate fixed stratified CV split manifest for GBCU")
    parser.add_argument("--dataset_root", type=str, default="GBCU")
    parser.add_argument("--n_cv_folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify",
        type=str,
        default="binary",
        choices=["binary", "multiclass", "none"],
    )
    parser.add_argument("--output", type=str, default=None, help="Override output JSON path")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_path = (
        Path(args.output)
        if args.output
        else default_cv_split_path(
            dataset_root,
            n_cv_folds=args.n_cv_folds,
            seed=args.seed,
            stratify=args.stratify,
        )
    )
    manifest = generate_cv_split_manifest(
        dataset_root,
        n_cv_folds=args.n_cv_folds,
        seed=args.seed,
        stratify=args.stratify,
    )
    save_cv_split_manifest(out_path, manifest)
    folds, _ = load_cv_folds(
        dataset_root,
        n_cv_folds=args.n_cv_folds,
        seed=args.seed,
        stratify=args.stratify,
        cv_split_file=out_path,
    )
    print(f"[CV splits] verified {len(folds)} folds from {out_path}")


if __name__ == "__main__":
    main()
