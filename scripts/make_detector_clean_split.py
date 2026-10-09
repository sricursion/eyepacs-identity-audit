"""Write a detector-clean EyePACS split from confirmed prefix-identity edges.

This does not claim to solve all latent identity overlap. It guarantees only
that no edge confirmed by the declared full-corpus DCT+SIFT detector crosses
the emitted train/held-out boundary.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: str, second: str) -> None:
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--confirmed-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    full = pd.read_csv(args.metadata / "eyepacs_labels_full.csv")
    parts = full["image"].astype(str).str.rsplit("_", n=1)
    full["prefix"] = parts.str[0]
    original_train = set(pd.read_csv(args.metadata / "eyepacs_labels_train60.csv")["image"].astype(str))
    prefix_split = {
        prefix: "train60" if f"{prefix}_left" in original_train else "heldout40"
        for prefix in full["prefix"].astype(str).unique()
    }
    report = json.loads(args.confirmed_report.read_text(encoding="utf-8"))
    confirmed = report.get("confirmed", [])
    union = UnionFind(sorted(prefix_split))
    for row in confirmed:
        union.union(str(row["prefix_a"]), str(row["prefix_b"]))
    components: dict[str, list[str]] = defaultdict(list)
    for prefix in prefix_split:
        components[union.find(prefix)].append(prefix)
    # Deterministic and conservative: a mixed component is assigned to train,
    # so no held-out score is evaluated on a detector-linked train identity.
    corrected = {}
    moved = []
    for root, members in components.items():
        original = {prefix_split[value] for value in members}
        assigned = "train60" if "train60" in original else "heldout40"
        for value in members:
            corrected[value] = assigned
            if corrected[value] != prefix_split[value]:
                moved.append(value)
    for row in confirmed:
        if corrected[str(row["prefix_a"])] != corrected[str(row["prefix_b"])]:
            raise AssertionError("confirmed edge crosses corrected split")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    full["detector_clean_split"] = full["prefix"].map(corrected)
    train = full.loc[full["detector_clean_split"] == "train60", ["image", "level"]].sort_values("image")
    held = full.loc[full["detector_clean_split"] == "heldout40", ["image", "level"]].sort_values("image")
    train.to_csv(args.out_dir / "eyepacs_labels_train60_detector_clean.csv", index=False)
    held.to_csv(args.out_dir / "eyepacs_labels_heldout40_detector_clean.csv", index=False)
    component_rows = []
    for root, members in sorted(components.items()):
        component_rows.append({
            "component": root,
            "prefixes": ";".join(sorted(members)),
            "n_prefixes": len(members),
            "original_splits": ";".join(sorted({prefix_split[value] for value in members})),
            "detector_clean_split": corrected[members[0]],
        })
    pd.DataFrame(component_rows).to_csv(args.out_dir / "detector_identity_components.csv", index=False)
    summary = {
        "protocol": "detector-clean split: union confirmed prefix edges, assign every mixed original component to train60",
        "n_confirmed_edges": len(confirmed),
        "n_components_with_confirmed_edges": int(sum(len(values) > 1 for values in components.values())),
        "n_prefixes_moved_from_heldout_to_train": len(moved),
        "train_images": int(len(train)),
        "heldout_images": int(len(held)),
        "guardrail": "This split is clean only with respect to the declared detector-confirmed identity graph; it is not proof of global patient disjointness.",
    }
    (args.out_dir / "README.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

