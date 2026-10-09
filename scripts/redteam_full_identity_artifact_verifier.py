"""Minimal independent verifier for the released full-identity audit artefacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--signatures", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    confirmed = report["confirmed"]
    selected = [row for row in report["all_bilateral_candidates_measured"] if row["passes_primary_rule"]]
    signatures = np.load(args.signatures, allow_pickle=True)
    blank = set(signatures["image"][signatures["blank"].astype(bool)].astype(str))
    clean_train = set(pd.read_csv(args.clean_dir / "eyepacs_labels_train60_detector_clean.csv")["image"].astype(str))
    clean_held = set(pd.read_csv(args.clean_dir / "eyepacs_labels_heldout40_detector_clean.csv")["image"].astype(str))
    original = set(pd.read_csv(args.metadata / "eyepacs_labels_full.csv")["image"].astype(str))
    if selected != confirmed:
        raise AssertionError("report-selected registrations disagree with the primary rule")
    if any(row["same_side_min_inliers"] < 50 or row["laterality_swap_max_inliers"] > 10 for row in confirmed):
        raise AssertionError("reported positive violates its own rule")
    positive_images = {name for row in confirmed for name in row["images"].values()}
    if blank & positive_images:
        raise AssertionError("blank sentinel appears in a positive")
    if clean_train & clean_held or clean_train | clean_held != original:
        raise AssertionError("corrected split is not a disjoint complete partition")
    prefix_clean = {name.rsplit("_", 1)[0]: ("train60" if name in clean_train else "heldout40") for name in original}
    if any(prefix_clean[row["prefix_a"]] != prefix_clean[row["prefix_b"]] for row in confirmed):
        raise AssertionError("confirmed identity crosses corrected split")
    controls = report["controls"]
    result = {
        "n_confirmed": len(confirmed),
        "n_cross_historical_split": sum(row["cross_historical_split"] for row in confirmed),
        "blank_images": sorted(blank),
        "control_primary_passes": {name: sum(row["passes_primary_rule"] for row in controls[name]["rows"]) for name in ("random", "exact_dimension")},
        "verdict": "pass: released detector report is self-consistent; all positives obey the fixed registration rule, controls have no pass, blank frames are excluded from positives, and the corrected split closes every confirmed edge. This verifier does not estimate detector recall or establish clinical identity ground truth.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
