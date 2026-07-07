# SAM2 Segmentation To Pre-Training CSV Pipeline

This README covers the code path from raw videos through the final CSV files needed right before training.

Large/generated files that should usually not be uploaded as code:

The SAM2 checkpoint must exist at:

```bash
sam2/checkpoints/sam2.1_hiera_large.pt
```

You can usually restore checkpoints with:

```bash
bash sam2/checkpoints/download_ckpts.sh
```

## Pipeline Overview

## Step 0: Put Videos In The Expected Folder

Put each recording set under the SAM2 video folder, for example:

```bash
sam2/videos-2026/5-17-2026/
```

Naming expectations:

| Video type | Name should contain |
|---|---|
| Side view fiber video | `Side` |
| Top view fiber video | `Top` |
| Intensity video | `Speckle`, `LED`, or `Fib` |
| Ignored video | `Combined` is ignored by the segmentation script |

Important path rule: when running `sam2/sam2_fiber.py` or `sam2/intensity.py` from the repo root, `--video-dir` is relative to the `sam2/` folder. Do not prefix it with `sam2/`.

Correct:

```bash
py sam2/sam2_fiber.py --video-dir "videos-2026/5-17-2026" --list-videos
```

Wrong:

```bash
py sam2/sam2_fiber.py --video-dir "sam2/videos-2026/5-17-2026" --list-videos
```

## Step 1: Segment Side And Top Videos With SAM2

List videos first:

```bash
py sam2/sam2_fiber.py --video-dir "videos-2026/5-17-2026" --list-videos
```

Optionally dry-run video metadata:

```bash
py sam2/sam2_fiber.py --video-dir "videos-2026/5-17-2026" --dry-run
```

Run segmentation:

```bash
py sam2/sam2_fiber.py --video-dir "videos-2026/5-17-2026"
```

What this script does:

- Uses SAM2.1 Hiera-Large from `sam2/checkpoints/sam2.1_hiera_large.pt`.
- Processes side/top fiber videos.
- Ignores videos with `Speckle` or `Combined` in the name.
- Handles long videos in chunks to avoid CUDA out-of-memory.
- Saves/resumes progress through `sam2/csvs/*_progress.json`.
- Produces segmentation videos and per-frame fiber spine CSVs.

Main outputs:

```bash
sam2/csvs/<TIMESTAMP> Side_data.csv
sam2/csvs/<TIMESTAMP> Top_data.csv
sam2/segmented videos/<TIMESTAMP> Side_segmented.mp4
sam2/segmented videos/<TIMESTAMP> Top_segmented.mp4
```

If existing progress is found, the script asks whether to resume, keep/skip, or delete/restart.

## Step 2: Extract Intensity From Speckle/LED/Fib Videos

Run all intensity videos in the folder:

```bash
py sam2/intensity.py --video-dir "videos-2026/5-17-2026"
```

Run one intensity video by name/stem:

```bash
py sam2/intensity.py "2026-05-17 22-30-30 Speckle" --video-dir "videos-2026/5-17-2026"
```

What this script does:

- Finds intensity videos whose names contain `Speckle`, `LED`, or `Fib`.
- Ignores `Side`, `Top`, `Combined`, and already segmented videos.
- Detects fiber circles across the full video rather than assuming all circles are bright in frame 0.
- Tracks circle centers over time.
- Extracts per-circle intensity and 9-point/patch intensity columns.
- Writes an audit video unless `--no-audit-video` is used.
- Writes normalized intensity CSVs.

Main outputs:

```bash
sam2/intensity_data/<TIMESTAMP> Speckle_intensities.csv
sam2/normalized intensity data/<TIMESTAMP> Speckle_intensities_normalized.csv
sam2/intensity_audit/<TIMESTAMP> Speckle_intensity_audit.mp4
```

The exact middle word can be `Speckle`, `LED`, or `Fib` depending on the input filename. `prepare_single_dataset.py` accepts any of those normalized intensity names and renames the prepared output to `Fib_intensities_normalized.csv` for the training pipeline.

## Step 3: Prepare One Dataset For ML CSVs

Run this once per timestamp after both SAM2 coordinate CSVs and the normalized intensity CSV exist.

Example for one timestamp:

```bash
py fibertraining/prepare_single_dataset.py "2026-05-17 22-30-30" --force
```

Example for all three 2026-05-17 timestamps:

```bash
for ts in "2026-05-17 22-30-30" "2026-05-17 22-42-51" "2026-05-17 22-57-14"; do
  py fibertraining/prepare_single_dataset.py "$ts" --force
done
```

What this script does:

- Imports SAM2 coordinate CSVs from `sam2/csvs/` into `fibertraining/raw data/`.
- Converts `Top_data.csv` naming into `Topview_data.csv` naming expected by the ML data path.
- Runs `clean_data.py` logic to parse, resample, interpolate, and smooth fiber spine points.
- Runs `spine.py` feature logic to resample each fiber to 20 coordinate points.
- Imports the normalized intensity CSV from `sam2/normalized intensity data/`.
- Generates intensity derivatives with Savitzky-Golay smoothing plus `np.gradient`.
- Aligns common `Frame_ID` values across intensity, derivative, side, and top files.
- Removes the initial intensity baseline.
- Converts coordinates to root-relative no-offset coordinates.
- Verifies that the final CSVs exist and have enough frames/features.

Important outputs:

```bash
fibertraining/raw data/<TIMESTAMP> Side_data.csv
fibertraining/raw data/<TIMESTAMP> Topview_data.csv
fibertraining/cleaned data/<TIMESTAMP> Side_data_cleaned.csv
fibertraining/cleaned data/<TIMESTAMP> Topview_data_cleaned.csv
fibertraining/normalized coord data/<TIMESTAMP> Side_data_normalized_features.csv
fibertraining/normalized coord data/<TIMESTAMP> Topview_data_normalized_features.csv
fibertraining/normalized intensity data/<TIMESTAMP> Fib_intensities_normalized.csv
fibertraining/normalized intensity data derivatives/<TIMESTAMP> Fib_intensities_derivatives.csv
fibertraining/normalized coord data no offset/<TIMESTAMP> Side_data_normalized_features.csv
fibertraining/normalized coord data no offset/<TIMESTAMP> Topview_data_normalized_features.csv
fibertraining/normalized intensity data no offset/<TIMESTAMP> Fib_intensities_normalized.csv
fibertraining/normalized intensity data derivatives no offset/<TIMESTAMP> Fib_intensities_derivatives.csv
```

At this point, the dataset is ready for training if you are using the uncorrected no-offset coordinate labels.