"""Recompute the candidate-pool null, gate interval, and re-acquisition checks.

This intentionally consumes only the released SIFT adjudication report.  It does
not import the retrieval or SIFT implementation, so it is suitable as a compact
artifact-level check for W1--W4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pair_id(row: dict) -> str:
    return "~".join(sorted((str(row["prefix_a"]), str(row["prefix_b"]))))


def accepted_at(rows: list[dict], inlier_gate: int, swap_gate: int) -> set[str]:
    return {
        pair_id(row)
        for row in rows
        if row["same_side_min_inliers"] >= inlier_gate
        and row["laterality_swap_max_inliers"] <= swap_gate
    }


def q(values: np.ndarray, probability: float) -> float:
    # Explicitly use the standard linear sample quantile, recorded in the output.
    return float(np.quantile(values, probability, method="linear"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inlier-gate", type=int, default=50)
    parser.add_argument("--swap-gate", type=int, default=10)
    parser.add_argument("--near-miss-min", type=int, default=40)
    parser.add_argument("--exact-duplicate-error-px", type=float, default=0.05)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = report["all_bilateral_candidates_measured"]
    expected_n = int(report["n_bilateral_dct_candidate_groups"])
    if len(rows) != expected_n:
        raise AssertionError(f"candidate count mismatch: {len(rows)} != {expected_n}")

    observed_primary = {pair_id(row) for row in rows if row["passes_primary_rule"]}
    recomputed_primary = accepted_at(rows, args.inlier_gate, args.swap_gate)
    if observed_primary != recomputed_primary:
        raise AssertionError("emitted primary-rule field disagrees with frozen gates")
    if len(observed_primary) != int(report["n_primary_sift_confirmed_groups"]):
        raise AssertionError("primary confirmed count disagrees with report header")

    inliers = np.asarray([row["same_side_min_inliers"] for row in rows], dtype=float)
    rejected = [row for row in rows if not row["passes_primary_rule"]]
    passing = [row for row in rows if row["passes_primary_rule"]]
    rejected_inliers = np.asarray([row["same_side_min_inliers"] for row in rejected], dtype=float)
    passing_inliers = np.asarray([row["same_side_min_inliers"] for row in passing], dtype=float)
    max_rejected = int(np.max(rejected_inliers))
    min_passing = int(np.min(passing_inliers))
    conservative_gap = list(range(max_rejected + 1, min_passing))

    max_gate = int(np.max(inliers)) + 1
    identical_gates = [
        gate
        for gate in range(0, max_gate + 1)
        if accepted_at(rows, gate, args.swap_gate) == recomputed_primary
    ]
    intervals: list[list[int]] = []
    for gate in identical_gates:
        if not intervals or gate != intervals[-1][1] + 1:
            intervals.append([gate, gate])
        else:
            intervals[-1][1] = gate

    reprojection_errors = np.asarray(
        [
            error
            for row in passing
            for error in (
                row["same_left"]["median_reprojection_error"],
                row["same_right"]["median_reprojection_error"],
            )
        ],
        dtype=float,
    )
    exact_duplicate_pairs = [
        pair_id(row)
        for row in passing
        if row["same_left"]["median_reprojection_error"] < args.exact_duplicate_error_px
        and row["same_right"]["median_reprojection_error"] < args.exact_duplicate_error_px
    ]

    anomaly_id = "26509~39599"
    anomaly = next((row for row in rows if pair_id(row) == anomaly_id), None)
    near_misses = [
        {
            "pair": pair_id(row),
            "same_side_min_inliers": row["same_side_min_inliers"],
            "laterality_swap_max_inliers": row["laterality_swap_max_inliers"],
            "same_left_inliers": row["same_left"]["ransac_inliers"],
            "same_right_inliers": row["same_right"]["ransac_inliers"],
            "swap_left_to_right_inliers": row["swap_left_to_right"]["ransac_inliers"],
            "swap_right_to_left_inliers": row["swap_right_to_left"]["ransac_inliers"],
        }
        for row in rejected
        if row["same_side_min_inliers"] >= args.near_miss_min
    ]
    near_misses.sort(key=lambda row: (-row["same_side_min_inliers"], row["pair"]))

    result = {
        "input": str(args.report),
        "frozen_rule": {
            "same_side_min_inliers_gte": args.inlier_gate,
            "laterality_swap_max_inliers_lte": args.swap_gate,
        },
        "integrity_assertions": {
            "report_candidate_count": expected_n,
            "measured_candidate_count": len(rows),
            "emitted_and_recomputed_primary_sets_match": True,
            "confirmed_count": len(observed_primary),
        },
        "w1_hard_candidate_pool_null": {
            "quantile_method": "numpy.quantile(method=linear)",
            "same_side_min_inliers": {"p50": q(inliers, 0.5), "p90": q(inliers, 0.9), "p99": q(inliers, 0.99)},
            "n_rejected": len(rejected),
            "n_passing": len(passing),
            "max_rejected": max_rejected,
            "min_passing": min_passing,
            "strict_empty_gap": max_rejected < min_passing,
        },
        "w2_gate_insensitivity": {
            "all_integer_intervals_with_identical_primary_accepted_set": intervals,
            "conservative_empty_gap_integer_interval": [max_rejected + 1, min_passing - 1],
            "conservative_interval_has_identical_set": all(
                accepted_at(rows, gate, args.swap_gate) == recomputed_primary
                for gate in conservative_gap
            ),
        },
        "w3_exact_duplicate_test": {
            "rule": f"both same-side median reprojection errors < {args.exact_duplicate_error_px} px",
            "n_exact_duplicate_signature": len(exact_duplicate_pairs),
            "exact_duplicate_signature_pairs": exact_duplicate_pairs,
            "same_side_error_distribution_px": {
                "n": int(reprojection_errors.size),
                "min": float(np.min(reprojection_errors)),
                "median": q(reprojection_errors, 0.5),
                "max": float(np.max(reprojection_errors)),
            },
        },
        "w4_laterality_anomaly": {
            "anomaly_pair": anomaly_id,
            "found": anomaly is not None,
            "anomaly_row": None if anomaly is None else {
                "same_side_min_inliers": anomaly["same_side_min_inliers"],
                "laterality_swap_max_inliers": anomaly["laterality_swap_max_inliers"],
                "passes_primary_rule": anomaly["passes_primary_rule"],
                "same_left_inliers": anomaly["same_left"]["ransac_inliers"],
                "same_right_inliers": anomaly["same_right"]["ransac_inliers"],
                "swap_left_to_right_inliers": anomaly["swap_left_to_right"]["ransac_inliers"],
                "swap_right_to_left_inliers": anomaly["swap_right_to_left"]["ransac_inliers"],
            },
            "near_miss_min_same_side_inliers": args.near_miss_min,
            "rejected_near_misses": near_misses,
        },
    }
    if anomaly is None or anomaly["passes_primary_rule"]:
        raise AssertionError("laterality anomaly was absent or unexpectedly passed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
