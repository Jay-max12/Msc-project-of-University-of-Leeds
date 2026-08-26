#!/usr/bin/env python3
"""Summarize AE hyperparameter sweep results into a ranked table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Summarize AE sweep cv_results.json files")
    p.add_argument(
        "--sweep_root",
        type=str,
        default="outputs/roi_repr/binary/ae_sweep_night",
        help="Root directory containing emb*_cls*_recon0p3/ subfolders",
    )
    p.add_argument(
        "--sort_by",
        type=str,
        default="test_mal_sens_mean",
        choices=[
            "test_mal_sens_mean",
            "test_select_mean",
            "test_balanced_accuracy_mean",
            "test_accuracy_mean",
            "test_mal_spec_mean",
        ],
    )
    return p.parse_args()


def load_row(run_dir: Path) -> dict | None:
    cv_path = run_dir / "cv_results.json"
    if not cv_path.exists():
        return {
            "run": run_dir.name,
            "status": "incomplete",
            "path": str(run_dir),
        }
    with cv_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("cv_summary", {})
    name = run_dir.name
    # emb64_cls1p0_recon0p3
    parts = name.split("_")
    emb = next((p.replace("emb", "") for p in parts if p.startswith("emb")), "?")
    cls = next((p.replace("cls", "").replace("p", ".") for p in parts if p.startswith("cls")), "?")
    return {
        "run": name,
        "status": "done",
        "embedding_dim": emb,
        "lambda_cls": cls,
        "lambda_recon": "0.3",
        "test_acc": summary.get("test_accuracy_mean"),
        "test_bal_acc": summary.get("test_balanced_accuracy_mean"),
        "test_mal_sens": summary.get("test_mal_sens_mean"),
        "test_mal_spec": summary.get("test_mal_spec_mean"),
        "test_select": summary.get("test_select_mean"),
        "test_acc_std": summary.get("test_accuracy_std"),
        "test_mal_sens_std": summary.get("test_mal_sens_std"),
        "path": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.sweep_root)
    if not root.exists():
        raise SystemExit(f"Sweep root not found: {root}")

    rows = [load_row(p) for p in sorted(root.iterdir()) if p.is_dir()]
    done = [r for r in rows if r and r.get("status") == "done"]
    pending = [r for r in rows if r and r.get("status") != "done"]

    sort_key = args.sort_by
    field_map = {
        "test_mal_sens_mean": "test_mal_sens",
        "test_select_mean": "test_select",
        "test_balanced_accuracy_mean": "test_bal_acc",
        "test_accuracy_mean": "test_acc",
        "test_mal_spec_mean": "test_mal_spec",
    }
    key = field_map[sort_key]
    done.sort(key=lambda r: float(r.get(key) or -1), reverse=True)

    print(f"AE sweep summary | root={root} | sort_by={sort_key}")
    print(f"completed={len(done)} pending={len(pending)} total_dirs={len(rows)}")
    print()
    header = (
        f"{'rank':>4}  {'emb':>4}  {'cls':>5}  {'acc':>8}  {'bal':>8}  "
        f"{'sens':>8}  {'spec':>8}  {'select':>8}  run"
    )
    print(header)
    print("-" * len(header))
    for i, row in enumerate(done, start=1):
        print(
            f"{i:4d}  {row['embedding_dim']:>4}  {row['lambda_cls']:>5}  "
            f"{row['test_acc']:8.4f}  {row['test_bal_acc']:8.4f}  "
            f"{row['test_mal_sens']:8.4f}  {row['test_mal_spec']:8.4f}  "
            f"{row['test_select']:8.4f}  {row['run']}"
        )

    if pending:
        print()
        print("Incomplete runs:")
        for row in pending:
            print(f"  - {row['run']}")

    if done:
        best = done[0]
        print()
        print(
            "Best by {sort}: emb={emb} lambda_cls={cls} | "
            "acc={acc:.4f} sens={sens:.4f} spec={spec:.4f} select={sel:.4f}".format(
                sort=sort_key,
                emb=best["embedding_dim"],
                cls=best["lambda_cls"],
                acc=best["test_acc"],
                sens=best["test_mal_sens"],
                spec=best["test_mal_spec"],
                sel=best["test_select"],
            )
        )
        print(f"path: {best['path']}")


if __name__ == "__main__":
    main()
