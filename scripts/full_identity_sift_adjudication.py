"""Bilateral SIFT/laterality adjudication for full-corpus DCT candidates.

The primary detector rule is intentionally simple and fixed before the scan:
both same-side pairs must have at least 50 SIFT/RANSAC inliers and both
laterality-swapped comparisons must have at most 10. DCT only proposes pairs;
this stage makes the reported detector result reproducible and supplies
matched random and exact-dimension negative controls.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


_WORKER_MATCHER: "Matcher | None" = None


class Matcher:
    def __init__(self, image_root: Path):
        self.image_root = image_root
        self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02, edgeThreshold=10, sigma=1.6)
        self.cache: dict[str, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}

    def view(self, name: str) -> np.ndarray:
        image = cv2.imread(str(self.image_root / f"{name}.jpeg"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows, columns = np.where(gray > 8)
        if not len(rows):
            # Blank frames remain in the manifest and may be proposed by the
            # fixed DCT sentinel.  This featureless image deterministically
            # yields zero SIFT/RANSAC support, so it cannot pass identity.
            return np.zeros((640, 640), dtype=np.uint8)
        top, bottom = max(0, int(rows.min()) - 4), min(gray.shape[0], int(rows.max()) + 5)
        left, right = max(0, int(columns.min()) - 4), min(gray.shape[1], int(columns.max()) + 5)
        crop = image[top:bottom, left:right]
        crop = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_AREA)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop[:, :, 1])

    def features(self, name: str) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        if name not in self.cache:
            self.cache[name] = self.sift.detectAndCompute(self.view(name), None)
        return self.cache[name]

    def match(self, first: str, second: str) -> dict:
        key_first, desc_first = self.features(first)
        key_second, desc_second = self.features(second)
        result: dict[str, float | int | None] = {
            "keypoints_first": len(key_first), "keypoints_second": len(key_second), "ratio_matches": 0,
            "ransac_inliers": 0, "median_reprojection_error": None, "source_hull_fraction": None,
            "source_grid_occupancy_8x8": None,
        }
        if desc_first is None or desc_second is None:
            return result
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_first, desc_second, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.70 * pair[1].distance]
        result["ratio_matches"] = len(good)
        if len(good) < 4:
            return result
        source = np.float32([key_first[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        target = np.float32([key_second[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
        if homography is None or mask is None:
            return result
        mask = mask.ravel().astype(bool)
        result["ransac_inliers"] = int(mask.sum())
        if not mask.any():
            return result
        projected = cv2.perspectiveTransform(source, homography)
        residual = np.linalg.norm(projected - target, axis=2).ravel()
        result["median_reprojection_error"] = float(np.median(residual[mask]))
        source_xy = source[mask, 0, :]
        hull = cv2.convexHull(source_xy.astype(np.float32).reshape(-1, 1, 2))
        result["source_hull_fraction"] = float(cv2.contourArea(hull)) / (640.0 * 640.0) if len(hull) >= 3 else 0.0
        bins = np.floor(source_xy / 80.0).astype(int).clip(0, 7)
        result["source_grid_occupancy_8x8"] = int(len(np.unique(bins[:, 0] * 8 + bins[:, 1])))
        return result


def candidate_groups(candidate_csv: Path) -> list[dict]:
    groups: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    with candidate_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            a, b = str(row["prefix_a"]), str(row["prefix_b"])
            key = tuple(sorted((a, b)))
            side = str(row["side"])
            previous = groups[key].get(side)
            if previous is None or int(row["hamming_distance"]) < int(previous["hamming_distance"]):
                groups[key][side] = row
    output = []
    for (a, b), sides in groups.items():
        if set(sides) == {"left", "right"}:
            output.append({"prefix_a": a, "prefix_b": b, "left": sides["left"], "right": sides["right"]})
    return sorted(output, key=lambda value: (value["prefix_a"], value["prefix_b"]))


def image_for_prefix(row: dict, prefix: str, side: str) -> str:
    candidate = row[side]
    return str(candidate["image_a"] if candidate["prefix_a"] == prefix else candidate["image_b"])


def adjudicate(matcher: Matcher, prefix_a: str, prefix_b: str, left_a: str, right_a: str, left_b: str, right_b: str) -> dict:
    same_left = matcher.match(left_a, left_b)
    same_right = matcher.match(right_a, right_b)
    swap_lr = matcher.match(left_a, right_b)
    swap_rl = matcher.match(right_a, left_b)
    same_min = min(int(same_left["ransac_inliers"]), int(same_right["ransac_inliers"]))
    swap_max = max(int(swap_lr["ransac_inliers"]), int(swap_rl["ransac_inliers"]))
    return {
        "prefix_a": prefix_a,
        "prefix_b": prefix_b,
        "images": {"a_left": left_a, "a_right": right_a, "b_left": left_b, "b_right": right_b},
        "same_left": same_left,
        "same_right": same_right,
        "swap_left_to_right": swap_lr,
        "swap_right_to_left": swap_rl,
        "same_side_min_inliers": same_min,
        "laterality_swap_max_inliers": swap_max,
        "passes_primary_rule": bool(same_min >= 50 and swap_max <= 10),
    }


def initialise_worker(image_root: str) -> None:
    global _WORKER_MATCHER
    _WORKER_MATCHER = Matcher(Path(image_root))


def adjudicate_group_worker(group: dict) -> dict:
    """One independent four-image adjudication for a process-pool worker.

    Clearing the per-process descriptor cache after each group fixes memory
    use and cannot affect the deterministic SIFT/RANSAC output for that group.
    """
    if _WORKER_MATCHER is None:
        raise RuntimeError("worker matcher was not initialised")
    a, b = group["prefix_a"], group["prefix_b"]
    result = adjudicate(
        _WORKER_MATCHER, a, b,
        image_for_prefix(group, a, "left"), image_for_prefix(group, a, "right"),
        image_for_prefix(group, b, "left"), image_for_prefix(group, b, "right"),
    )
    _WORKER_MATCHER.cache.clear()
    return result


def add_labels(frame: pd.DataFrame, row: dict) -> dict:
    grade = {str(item.image): int(item.level) for item in frame.itertuples()}
    split = {str(item.image): str(item.split) for item in frame.itertuples()}
    images = row["images"]
    row["grades"] = {key: grade[value] for key, value in images.items()}
    row["splits"] = {key: split[value] for key, value in images.items()}
    row["cross_historical_split"] = len(set(row["splits"].values())) > 1
    row["any_same_side_label_contradiction"] = bool(
        row["grades"]["a_left"] != row["grades"]["b_left"] or row["grades"]["a_right"] != row["grades"]["b_right"]
    )
    row["patient_grade_vector_contradiction"] = bool(
        (row["grades"]["a_left"], row["grades"]["a_right"])
        != (row["grades"]["b_left"], row["grades"]["b_right"])
    )
    return row


def controls(frame: pd.DataFrame, matcher: Matcher, n: int, seed: int, excluded_edges: set[tuple[str, str]]) -> dict:
    by_prefix = {str(prefix): group.set_index("side") for prefix, group in frame.groupby("prefix")}
    prefixes = sorted(by_prefix)
    dimension_pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for prefix, group in by_prefix.items():
        # Dimensions are read lazily once per image because the preflight JSON
        # intentionally stores only aggregate strata.
        left = matcher.image_root / f"{prefix}_left.jpeg"
        right = matcher.image_root / f"{prefix}_right.jpeg"
        for path in (left, right):
            if not path.is_file():
                raise FileNotFoundError(path)
        image_left = cv2.imread(str(left), cv2.IMREAD_UNCHANGED)
        image_right = cv2.imread(str(right), cv2.IMREAD_UNCHANGED)
        dimension_pairs[(f"{image_left.shape[1]}x{image_left.shape[0]}", f"{image_right.shape[1]}x{image_right.shape[0]}")].append(prefix)
    rng = np.random.default_rng(seed)
    random_pairs, exact_pairs = [], []
    def admissible_pair(values: list[str] | np.ndarray) -> tuple[str, str]:
        # Candidate edges are excluded from both control arms. This does not
        # prove the remaining pairs are different people; it prevents the
        # control from knowingly sampling a detector-suspected positive.
        for _ in range(10000):
            a, b = (str(value) for value in rng.choice(values, size=2, replace=False))
            if tuple(sorted((a, b))) not in excluded_edges:
                return a, b
        raise RuntimeError("could not draw a non-candidate control pair")

    for _ in range(n):
        random_pairs.append(admissible_pair(prefixes))
    strata = [values for values in dimension_pairs.values() if len(values) >= 2]
    weights = np.asarray([len(values) * (len(values) - 1) / 2 for values in strata], dtype=np.float64)
    weights /= weights.sum()
    for _ in range(n):
        values = strata[int(rng.choice(len(strata), p=weights))]
        exact_pairs.append(admissible_pair(values))

    def run(pairs: list[tuple[str, str]]) -> list[dict]:
        output = []
        for a, b in pairs:
            row = adjudicate(matcher, a, b, f"{a}_left", f"{a}_right", f"{b}_left", f"{b}_right")
            output.append({
                "same_side_min_inliers": row["same_side_min_inliers"],
                "laterality_swap_max_inliers": row["laterality_swap_max_inliers"],
                "passes_primary_rule": row["passes_primary_rule"],
            })
        return output

    random_result = run(random_pairs)
    exact_result = run(exact_pairs)
    return {
        "n": n,
        "seed": seed,
        "n_excluded_dct_candidate_edges": len(excluded_edges),
        "random": {
            "max_same_side_min_inliers": int(max(row["same_side_min_inliers"] for row in random_result)),
            "n_primary_pass": int(sum(row["passes_primary_rule"] for row in random_result)),
            "rows": random_result,
        },
        "exact_dimension": {
            "max_same_side_min_inliers": int(max(row["same_side_min_inliers"] for row in exact_result)),
            "n_primary_pass": int(sum(row["passes_primary_rule"] for row in exact_result)),
            "rows": exact_result,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--controls", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--clear-cache-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    frame = pd.read_csv(args.metadata / "eyepacs_labels_full.csv")
    parts = frame["image"].astype(str).str.rsplit("_", n=1)
    frame["prefix"] = parts.str[0]
    frame["side"] = parts.str[1]
    train_names = set(pd.read_csv(args.metadata / "eyepacs_labels_train60.csv")["image"].astype(str))
    frame["split"] = frame["image"].astype(str).map(lambda value: "train60" if value in train_names else "heldout40")
    groups = candidate_groups(args.candidate_csv)
    matcher = Matcher(args.image_root)
    measured = []
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.workers == 1:
        results = (adjudicate_group_worker(group) for group in groups)
        initialise_worker(str(args.image_root))
    else:
        pool = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialise_worker,
            initargs=(str(args.image_root),),
        )
        results = pool.map(adjudicate_group_worker, groups, chunksize=8)
    try:
        for index, row in enumerate(results, 1):
            measured.append(add_labels(frame, row))
            if index % args.clear_cache_every == 0:
                matcher.cache.clear()
                print(f"SIFT_ADJUDICATION completed={index}/{len(groups)}", flush=True)
    finally:
        if args.workers > 1:
            pool.shutdown(wait=True, cancel_futures=True)
    confirmed = [row for row in measured if row["passes_primary_rule"]]
    excluded_candidate_edges = {tuple(sorted((group["prefix_a"], group["prefix_b"]))) for group in groups}
    control_report = controls(frame, matcher, args.controls, args.seed, excluded_candidate_edges)
    confirmed_prefixes = sorted({row["prefix_a"] for row in confirmed} | {row["prefix_b"] for row in confirmed})
    report = {
        "protocol": {
            "candidate_scope": "all full-corpus same-side DCT pairs passing the configured Hamming gate, then requiring both eyes for SIFT adjudication",
            "primary_identity_rule": "both same-side SIFT/RANSAC inliers >=50 and both laterality-swapped SIFT/RANSAC inliers <=10",
            "controls": "full-corpus random prefix pairs and full-corpus exact left/right-dimension matched prefix pairs, same four SIFT comparisons; all DCT-candidate edges excluded from both control draws",
            "guardrail": "the fraction is detector prevalence under this finite retrieval-and-registration rule, not a recall-corrected prevalence of all repeated people",
        },
        "n_bilateral_dct_candidate_groups": int(len(groups)),
        "n_primary_sift_confirmed_groups": int(len(confirmed)),
        "n_unique_prefixes_in_confirmed_groups": int(len(confirmed_prefixes)),
        "detector_prefix_prevalence": float(len(confirmed_prefixes) / frame["prefix"].nunique()),
        "n_cross_historical_split_confirmed_groups": int(sum(row["cross_historical_split"] for row in confirmed)),
        "n_same_side_label_contradictions": int(sum(row["any_same_side_label_contradiction"] for row in confirmed)),
        "n_patient_grade_vector_contradictions": int(sum(row["patient_grade_vector_contradiction"] for row in confirmed)),
        "confirmed": confirmed,
        "all_bilateral_candidates_measured": measured,
        "controls": control_report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key not in {"confirmed", "all_bilateral_candidates_measured", "controls"}}, indent=2))
    print(json.dumps({"controls": {key: value for key, value in control_report.items() if key != "random" and key != "exact_dimension"}, "random": {key: value for key, value in control_report["random"].items() if key != "rows"}, "exact_dimension": {key: value for key, value in control_report["exact_dimension"].items() if key != "rows"}}, indent=2))


if __name__ == "__main__":
    main()
