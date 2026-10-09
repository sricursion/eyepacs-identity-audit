"""All-prefix EyePACS DCT retrieval for the identity-integrity audit.

Every same-side pair in the 35,126-image manifest is compared once. A
256-bit retinal-crop DCT signature is used only as an exhaustive candidate
generator. Identity requires the separate bilateral SIFT/laterality check in
``full_identity_sift_adjudication.py``.

The finite detector is explicit: reported detector prevalence is prevalence of
the DCT-plus-SIFT rule, not an estimate of all repeated people in EyePACS.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.fft import dctn


def signature(path: Path) -> tuple[np.ndarray, bool]:
    with Image.open(path) as image:
        gray = image.convert("L")
        gray.thumbnail((512, 512), Image.Resampling.LANCZOS)
        array = np.asarray(gray)
        rows, columns = np.where(array > 8)
        if not len(rows):
            # Keep blank files in the exhaustive manifest.  A fixed signature
            # can propose blank-to-blank pairs, but they cannot pass the later
            # SIFT registration rule; omitting them would instead change scope.
            return np.zeros(32, dtype=np.uint8), True
        top, bottom = max(0, int(rows.min()) - 2), min(gray.height, int(rows.max()) + 3)
        left, right = max(0, int(columns.min()) - 2), min(gray.width, int(columns.max()) + 3)
        retina = gray.crop((left, top, right, bottom)).resize((64, 64), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(retina, dtype=np.float64), type=2, norm="ortho")[:17, :17]
    return np.packbits((coefficients[1:, 1:] > 0).ravel(), bitorder="big"), False


def dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def image_record(args: tuple[str, Path]) -> tuple[np.ndarray, tuple[int, int], bool]:
    name, path = args
    value, blank = signature(path)
    return value, dimensions(path), blank


def load_frame(metadata: Path) -> pd.DataFrame:
    full = pd.read_csv(metadata / "eyepacs_labels_full.csv")
    train = set(pd.read_csv(metadata / "eyepacs_labels_train60.csv")["image"].astype(str))
    held = set(pd.read_csv(metadata / "eyepacs_labels_heldout40.csv")["image"].astype(str))
    if len(full) != 35126 or set(full["image"].astype(str)) != train | held or train & held:
        raise AssertionError("unexpected EyePACS manifests")
    parts = full["image"].astype(str).str.rsplit("_", n=1)
    full["prefix"] = parts.str[0]
    full["side"] = parts.str[1]
    full["historical_split"] = full["image"].astype(str).map(lambda value: "train60" if value in train else "heldout40")
    if not (full.groupby("prefix")["side"].nunique() == 2).all():
        raise AssertionError("incomplete bilateral prefix")
    if not (full.groupby("prefix")["historical_split"].nunique() == 1).all():
        raise AssertionError("prefix crosses historical split")
    return full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--block", type=int, default=512)
    # 48 is locked by `identity_detector_recall_preflight.py`: it recovers all
    # retained positives satisfying the primary >=50/<=10 SIFT rule.  The
    # widening from the older 32-bit candidate screen is a post-hoc sensitivity
    # expansion and is documented as such; it is not presented as an unbiased
    # estimate of detector recall.
    parser.add_argument("--hamming-max", type=int, default=48)
    parser.add_argument("--reuse-signatures", action="store_true")
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="write the manifest-ordered signature cache, then stop before all-pair retrieval",
    )
    args = parser.parse_args()
    if not 0 <= args.hamming_max <= 256:
        raise ValueError("hamming max must be between 0 and 256")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame(args.metadata)
    signature_path = args.out_dir / "full_dct_signatures.npz"
    if args.reuse_signatures and signature_path.exists():
        packed = np.load(signature_path, allow_pickle=True)
        if not np.array_equal(packed["image"].astype(str), frame["image"].astype(str).to_numpy()):
            raise AssertionError("cached signatures do not match manifest ordering")
        if "blank" not in packed:
            raise AssertionError("signature cache lacks blank-frame inventory")
        signatures = packed["signature"].astype(np.uint8)
        dimensions_array = packed["dimension"].astype(str)
        blank_mask = packed["blank"].astype(bool)
    else:
        paths = [args.image_root / f"{name}.jpeg" for name in frame["image"].astype(str)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise AssertionError(f"missing images: {missing[:3]}")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            records = list(pool.map(image_record, zip(frame["image"].astype(str), paths)))
        blank_mask = np.asarray([blank for _, _, blank in records], dtype=bool)
        signatures = np.vstack([value for value, _, _ in records]).astype(np.uint8)
        dimensions_array = np.asarray([f"{width}x{height}" for _, (width, height), _ in records], dtype=object)
        np.savez_compressed(signature_path, image=frame["image"].astype(str).to_numpy(dtype=object), signature=signatures, dimension=dimensions_array, blank=blank_mask)

    if args.extract_only:
        report = {
            "scope": "signature extraction only; no candidate pairs were compared",
            "n_images": int(len(frame)),
            "n_blank_frames": int(blank_mask.sum()),
            "blank_frame_policy": "retained with an explicit fixed DCT signature; no blank image may pass the subsequent SIFT identity gate",
            "signature_cache": signature_path.name,
        }
        (args.out_dir / "full_dct_signature_extraction_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device requested but unavailable")
    bits = np.unpackbits(signatures, axis=1).astype(np.float16)
    bits = bits * 2.0 - 1.0
    features = torch.from_numpy(bits).to(device)
    side = frame["side"].to_numpy()
    output_csv = args.out_dir / "full_same_side_dct_candidates.csv"
    fieldnames = [
        "image_a", "image_b", "prefix_a", "prefix_b", "side", "hamming_distance",
        "grade_a", "grade_b", "split_a", "split_b", "dimension_a", "dimension_b",
    ]
    relation_counts: Counter[str] = Counter()
    n_rows = 0
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for laterality in ("left", "right"):
            indices = np.flatnonzero(side == laterality)
            side_features = features[torch.from_numpy(indices).to(device)]
            for start in range(0, len(indices), args.block):
                source_positions = np.arange(start, min(start + args.block, len(indices)))
                block = side_features[source_positions]
                # Dot product of +/-1 bits is 256 - 2*Hamming distance.
                distances = ((256.0 - block @ side_features.T) / 2.0).round().to(torch.int16)
                for local_row, position in enumerate(source_positions):
                    matches = torch.nonzero(distances[local_row] <= args.hamming_max, as_tuple=False).flatten().cpu().numpy()
                    for partner_position in matches:
                        if partner_position <= position:
                            continue
                        first = int(indices[position])
                        second = int(indices[partner_position])
                        prefix_a, prefix_b = str(frame.iloc[first]["prefix"]), str(frame.iloc[second]["prefix"])
                        if prefix_a == prefix_b:
                            continue
                        split_a, split_b = str(frame.iloc[first]["historical_split"]), str(frame.iloc[second]["historical_split"])
                        relation_counts["-".join(sorted((split_a, split_b)))] += 1
                        writer.writerow({
                            "image_a": str(frame.iloc[first]["image"]),
                            "image_b": str(frame.iloc[second]["image"]),
                            "prefix_a": prefix_a,
                            "prefix_b": prefix_b,
                            "side": laterality,
                            "hamming_distance": int(distances[local_row, partner_position].item()),
                            "grade_a": int(frame.iloc[first]["level"]),
                            "grade_b": int(frame.iloc[second]["level"]),
                            "split_a": split_a,
                            "split_b": split_b,
                            "dimension_a": str(dimensions_array[first]),
                            "dimension_b": str(dimensions_array[second]),
                        })
                        n_rows += 1
                print(f"DCT_RETRIEVAL side={laterality} completed={min(start + args.block, len(indices))}/{len(indices)} candidates={n_rows}", flush=True)
            del side_features
            if device.type == "cuda":
                torch.cuda.empty_cache()

    protocol = {
        "scope": "all 35,126 EyePACS manifest images; every same-side pair compared once",
        "candidate_generator": "256-bit signed DCT of a cropped retinal canvas; Hamming distance <= configured maximum",
        "identity_gate": "separate bilateral SIFT rule: >=50 same-side inliers and <=10 laterality-swapped inliers",
        "guardrail": "candidate count is detector scope, not all-person identity prevalence",
    }
    report = {
        "protocol": protocol,
        "n_images": int(len(frame)),
        "n_prefixes": int(frame["prefix"].nunique()),
        "n_blank_frames": int(blank_mask.sum()),
        "blank_frame_policy": "retained with an explicit fixed DCT signature; no blank image may pass the subsequent SIFT identity gate",
        "hamming_max": int(args.hamming_max),
        "n_same_side_candidate_pairs": int(n_rows),
        "candidate_pairs_by_historical_split_relation": dict(relation_counts),
        "candidate_csv": output_csv.name,
        "signature_cache": signature_path.name,
    }
    (args.out_dir / "full_dct_retrieval_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
