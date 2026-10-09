# Auditing Cross-Prefix Retinal Image Registrations in Public EyePACS

A full-corpus, laterality-aware registration audit of the public EyePACS
diabetic-retinopathy release. It tests one assumption that most work on this corpus
relies on:

> Two images whose filenames carry different numeric prefixes belong to different people.

Under a frozen two-stage detector, that assumption does not hold for every prefix pair.

## Data source

All images come from the public **Diabetic Retinopathy Detection** dataset released
through Kaggle by EyePACS:

<https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data>

No retinal images are redistributed here. The repository holds the detector code, the
image-level label manifests, the DCT signatures, per-candidate registration counts, and
the corrected split. Obtain the images from the link above and accept the competition
terms before re-running the detector.

## Method

The detector was frozen before the scan; nothing was tuned per result.

1. **Retrieval.** Each image gets a 256-bit signed discrete-cosine signature over a
   cropped retinal canvas. Every same-side cross-prefix pair is compared once, and a
   Hamming distance of at most 48 proposes a candidate: 502,526 same-side pairs,
   forming 16,958 bilateral candidate groups. The radius of 48 was set post hoc and is
   not used to estimate recall.
2. **Bilateral adjudication.** For prefixes `A` and `B`, four SIFT + ratio-test +
   RANSAC registrations are computed: left-to-left, right-to-right, and both
   laterality-swapped comparisons. A pair is accepted only if

   ```
   min(I_LL, I_RR) >= 50   and   max(I_LR, I_RL) <= 10
   ```

   The swap condition makes this an identity test rather than a similarity test: the
   same eyes must register strongly side-to-side and weakly across laterality.

## Results

| Quantity | Value |
|---|---:|
| Images audited | 35,126 |
| Complete public prefix groups | 17,563 |
| Bilateral retrieval candidate groups | 16,958 |
| Confirmed bilateral registrations | 23 |
| Affected public prefixes | 46 |
| Links crossing the historical split | 13 |
| Links with discordant two-eye grade vectors | 5 |

**Separation.** Across all 16,958 candidates, the same-side minimum inlier count has
median 5, 90th percentile 7, and 99th percentile 10. The largest rejected value is 52
and the smallest accepted is 83; nothing falls between.

**Gate insensitivity.** With the swapped-side gate fixed at 10, every same-side gate
from 53 through 82 returns the same 23 links.

**Not file duplication.** Under a test fixed in advance (both same-side median
reprojection errors below 0.05 px), 0 of 23 pairs qualify. The 46 same-side residuals
range from 0.3744 to 0.8815 px, median 0.5465.

**Corrected split.** Confirmed edges are unioned into components, and any component
touching both historical partitions is assigned to training: 21,100 training and
14,026 held-out images, with zero confirmed-edge crossings and zero prefix overlap.
The split is clean with respect to detector-confirmed edges; 23 is a lower bound, not
a prevalence estimate.

## Repository layout

```
data/                                historical split manifests (image, level)
artifacts/
  full_dct_signatures.npz            256-bit signatures for all 35,126 images
  full_dct_retrieval_report.json     retrieval summary
  full_sift_adjudication.json        all 16,958 candidates, the 23 confirmed, controls
  free_wins_analysis.json            separation, gate sweep, duplicate test
  detector_clean_split/              corrected train60 / heldout40 manifests
scripts/                             detector, split, analysis, verifier
```

## Reproducing

The first three commands run from this repository alone; no images are needed.

Verify the released result, independently of the retrieval and SIFT code:

```bash
python scripts/redteam_full_identity_artifact_verifier.py \
  --report artifacts/full_sift_adjudication.json \
  --signatures artifacts/full_dct_signatures.npz \
  --clean-dir artifacts/detector_clean_split \
  --metadata data \
  --out verify.json
```

Regenerate the separation, gate-sweep, and duplicate results:

```bash
python scripts/analyze_full_identity_free_wins.py \
  --report artifacts/full_sift_adjudication.json \
  --out free_wins_analysis.json
```

Regenerate the corrected split:

```bash
python scripts/make_detector_clean_split.py \
  --metadata data \
  --confirmed-report artifacts/full_sift_adjudication.json \
  --out-dir detector_clean_split
```

Re-running the detector end to end requires the Kaggle images:

```bash
python scripts/full_identity_dct_retrieval.py \
  --image-root /path/to/train --metadata data --out-dir out

python scripts/full_identity_sift_adjudication.py \
  --image-root /path/to/train --metadata data \
  --candidate-csv out/full_same_side_dct_candidates.csv \
  --out out/full_sift_adjudication.json
```

Retrieval runs on a GPU (`--device cuda` by default). SIFT adjudication is CPU-bound
and parallelizes with `--workers`, which changes wall-clock time but not the result.

Requires `numpy`, `pandas`, `scipy`, `opencv-python`, `Pillow`, and `torch`. Tested
with Python 3.13.

## Using the corrected split

`artifacts/detector_clean_split/` replaces the historical `train60` / `heldout40`
partition. Join images by filename and keep each prefix group in its assigned
partition. If a study adds data, changes preprocessing, or uses a different EyePACS
release, the audit must be repeated.

## License and attribution

The EyePACS images and labels are governed by the Kaggle competition terms linked
above. Please cite the EyePACS release when using the corrected split.
