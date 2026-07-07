"""
Prepare one SAM2 recording for the fiber ML training scripts.

Pipeline:
  1. Import final SAM2 Side/Top CSVs into fibertraining/raw data.
  2. Run clean_data.py logic to create fibertraining/cleaned data.
  3. Run spine.py feature logic to create fibertraining/normalized coord data.
  4. Import/rename the normalized LED intensity CSV.
  5. Generate intensity derivatives.
  6. Remove intensity and coordinate offsets.
  7. Verify the exact CSVs consumed by train_transformerv6.py.

Example:
    python prepare_single_dataset.py "2026-04-12 15-18-24" --force
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from clean_data import clean_data
from spine import get_normalized_features


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SAM2_DIR = REPO_ROOT / "sam2"

POINTS_PER_FIBER = 20
SMOOTH_WINDOW = 31
POLY_ORDER = 3


@dataclass(frozen=True)
class DatasetPaths:
    timestamp: str
    sam2_dir: Path
    training_dir: Path

    @property
    def sam2_csv_dir(self) -> Path:
        return self.sam2_dir / "csvs"

    @property
    def sam2_intensity_dir(self) -> Path:
        return self.sam2_dir / "normalized intensity data"

    @property
    def raw_coord_dir(self) -> Path:
        return self.training_dir / "raw data"

    @property
    def cleaned_coord_dir(self) -> Path:
        return self.training_dir / "cleaned data"

    @property
    def norm_int_dir(self) -> Path:
        return self.training_dir / "normalized intensity data"

    @property
    def norm_deriv_dir(self) -> Path:
        return self.training_dir / "normalized intensity data derivatives"

    @property
    def norm_coord_dir(self) -> Path:
        return self.training_dir / "normalized coord data"

    @property
    def no_offset_int_dir(self) -> Path:
        return self.training_dir / "normalized intensity data no offset"

    @property
    def no_offset_deriv_dir(self) -> Path:
        return self.training_dir / "normalized intensity data derivatives no offset"

    @property
    def no_offset_coord_dir(self) -> Path:
        return self.training_dir / "normalized coord data no offset"

    @property
    def raw_side(self) -> Path:
        return self.raw_coord_dir / f"{self.timestamp} Side_data.csv"

    @property
    def raw_top(self) -> Path:
        return self.raw_coord_dir / f"{self.timestamp} Topview_data.csv"

    @property
    def cleaned_side(self) -> Path:
        return self.cleaned_coord_dir / f"{self.timestamp} Side_data_cleaned.csv"

    @property
    def cleaned_top(self) -> Path:
        return self.cleaned_coord_dir / f"{self.timestamp} Topview_data_cleaned.csv"

    @property
    def intensity_out(self) -> Path:
        return self.norm_int_dir / f"{self.timestamp} Fib_intensities_normalized.csv"

    @property
    def derivative_out(self) -> Path:
        return self.norm_deriv_dir / f"{self.timestamp} Fib_intensities_derivatives.csv"

    @property
    def side_coord_out(self) -> Path:
        return self.norm_coord_dir / f"{self.timestamp} Side_data_normalized_features.csv"

    @property
    def top_coord_out(self) -> Path:
        return self.norm_coord_dir / f"{self.timestamp} Topview_data_normalized_features.csv"

    @property
    def final_intensity(self) -> Path:
        return self.no_offset_int_dir / self.intensity_out.name

    @property
    def final_derivative(self) -> Path:
        return self.no_offset_deriv_dir / self.derivative_out.name

    @property
    def final_side(self) -> Path:
        return self.no_offset_coord_dir / self.side_coord_out.name

    @property
    def final_top(self) -> Path:
        return self.no_offset_coord_dir / self.top_coord_out.name


def ensure_dirs(paths: DatasetPaths) -> None:
    for directory in [
        paths.raw_coord_dir,
        paths.cleaned_coord_dir,
        paths.norm_int_dir,
        paths.norm_deriv_dir,
        paths.norm_coord_dir,
        paths.no_offset_int_dir,
        paths.no_offset_deriv_dir,
        paths.no_offset_coord_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def natural_intensity_key(column: str) -> tuple[int, int, int]:
    match = re.search(r"intensity_circle_(\d+)(?:_9pt_(\d+))?$", column)
    if not match:
        return (10**9, 10**9, 10**9)
    circle = int(match.group(1))
    point = int(match.group(2) or 0)
    return (circle, 0 if point == 0 else 1, point)


def intensity_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in df.columns
        if c == "Frame_ID" or re.fullmatch(r"intensity_circle_\d+(?:_9pt_\d+)?", c)
    ]
    if "Frame_ID" not in cols:
        raise ValueError("Intensity CSV is missing Frame_ID")
    ordered = ["Frame_ID"]
    ordered.extend(sorted([c for c in cols if c != "Frame_ID"], key=natural_intensity_key))
    return ordered


def find_intensity_source(paths: DatasetPaths) -> Path:
    candidates = [
        paths.sam2_intensity_dir / f"{paths.timestamp} Fib_intensities_normalized.csv",
        paths.sam2_intensity_dir / f"{paths.timestamp} LED_intensities_normalized.csv",
        paths.sam2_intensity_dir / f"{paths.timestamp} Speckle_intensities_normalized.csv",
        paths.norm_int_dir / f"{paths.timestamp} Fib_intensities_normalized.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No normalized intensity CSV found. Checked:\n  - {searched}")


def find_sam2_coord_source(paths: DatasetPaths, view: str) -> Path:
    if view == "Side":
        candidates = [
            paths.sam2_csv_dir / f"{paths.timestamp} Side_data.csv",
            paths.raw_side,
        ]
    elif view == "Topview":
        candidates = [
            paths.sam2_csv_dir / f"{paths.timestamp} Topview_data.csv",
            paths.sam2_csv_dir / f"{paths.timestamp} Top_data.csv",
            paths.raw_top,
        ]
    else:
        raise ValueError(f"Unsupported view: {view}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"No {view} coordinate CSV found. Checked:\n  - {searched}")


def import_sam2_coord_csvs(paths: DatasetPaths, force: bool) -> tuple[Path, Path]:
    jobs = [
        (find_sam2_coord_source(paths, "Side"), paths.raw_side),
        (find_sam2_coord_source(paths, "Topview"), paths.raw_top),
    ]
    for source, target in jobs:
        if target.exists() and not force:
            print(f"Raw coordinate CSV exists: {target}")
            continue
        if source.resolve() != target.resolve():
            print(f"Importing SAM2 coordinates: {source.name} -> {target.name}")
            shutil.copy2(source, target)
        else:
            print(f"Using existing raw coordinate CSV: {target}")
    return paths.raw_side, paths.raw_top


def clean_coord_csvs(paths: DatasetPaths, force: bool) -> tuple[Path, Path]:
    jobs = [(paths.raw_side, paths.cleaned_side), (paths.raw_top, paths.cleaned_top)]
    for source, target in jobs:
        if target.exists() and not force:
            print(f"Cleaned coordinate CSV exists: {target}")
            continue
        print(f"Cleaning coordinates: {source.name} -> {target.name}")
        clean_data(str(source), str(target))
        if not target.exists():
            raise FileNotFoundError(f"clean_data did not create {target}")
    return paths.cleaned_side, paths.cleaned_top


def build_normalized_coord_csv(cleaned_path: Path, output_path: Path, points: int, force: bool) -> Path:
    if output_path.exists() and not force:
        print(f"Normalized coordinate CSV exists: {output_path}")
        return output_path

    print(f"Building spine features: {cleaned_path.name} -> {output_path.name}")
    cleaned_df = pd.read_csv(cleaned_path)
    required = {"Frame_ID", "x_smooth", "y_smooth"}
    missing = required - set(cleaned_df.columns)
    if missing:
        raise ValueError(f"{cleaned_path} is missing required columns: {sorted(missing)}")

    records: list[dict[str, float]] = []
    for frame_id, group in cleaned_df.groupby("Frame_ID", sort=True):
        x = group["x_smooth"].to_numpy(dtype=np.float64)
        y = group["y_smooth"].to_numpy(dtype=np.float64)
        if len(x) <= 1:
            continue
        features = get_normalized_features(x, y, num_samples=points)
        record: dict[str, float] = {"Frame_ID": int(frame_id)}
        for i in range(points):
            record[f"x_{i}"] = float(features[i])
        for i in range(points):
            record[f"y_{i}"] = float(features[points + i])
        records.append(record)

    if not records:
        raise ValueError(f"No valid cleaned frames found in {cleaned_path}")

    out_df = pd.DataFrame(records)
    x_cols = [f"x_{i}" for i in range(points)]
    y_cols = [f"y_{i}" for i in range(points)]

    x_min = float(out_df[x_cols].to_numpy().min())
    x_max = float(out_df[x_cols].to_numpy().max())
    y_min = float(out_df[y_cols].to_numpy().min())
    y_max = float(out_df[y_cols].to_numpy().max())

    if x_max > x_min:
        out_df.loc[:, x_cols] = (out_df[x_cols] - x_min) / (x_max - x_min)
    if y_max > y_min:
        out_df.loc[:, y_cols] = (out_df[y_cols] - y_min) / (y_max - y_min)

    out_df.to_csv(output_path, index=False)
    print(f"  Global normalization: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    return output_path


def prepare_coords(paths: DatasetPaths, force: bool, points: int) -> tuple[Path, Path]:
    import_sam2_coord_csvs(paths, force=force)
    clean_coord_csvs(paths, force=force)
    build_normalized_coord_csv(paths.cleaned_side, paths.side_coord_out, points=points, force=force)
    build_normalized_coord_csv(paths.cleaned_top, paths.top_coord_out, points=points, force=force)
    return paths.side_coord_out, paths.top_coord_out


def prepare_intensity(paths: DatasetPaths, force: bool) -> Path:
    source = find_intensity_source(paths)
    if paths.intensity_out.exists() and not force:
        print(f"Intensity exists: {paths.intensity_out}")
        return paths.intensity_out

    print(f"Preparing intensity: {source.name} -> {paths.intensity_out.name}")
    df = pd.read_csv(source)
    df = df[intensity_columns(df)]
    df.to_csv(paths.intensity_out, index=False)
    return paths.intensity_out


def prepare_derivatives(paths: DatasetPaths, force: bool) -> Path:
    if paths.derivative_out.exists() and not force:
        print(f"Derivatives exist: {paths.derivative_out}")
        return paths.derivative_out

    print(f"Preparing derivatives: {paths.derivative_out.name}")
    df = pd.read_csv(paths.intensity_out)
    feature_cols = sorted([c for c in df.columns if "_9pt_" in c], key=natural_intensity_key)
    if not feature_cols:
        raise ValueError(f"No 9pt intensity columns found in {paths.intensity_out}")

    data = df[feature_cols].to_numpy(dtype=np.float32)
    window = min(SMOOTH_WINDOW, len(df) if len(df) % 2 == 1 else len(df) - 1)
    if window <= POLY_ORDER:
        smoothed = data
    else:
        smoothed = savgol_filter(data, window_length=window, polyorder=POLY_ORDER, axis=0)
    derivatives = np.gradient(smoothed, axis=0)

    out = pd.DataFrame(derivatives, columns=feature_cols)
    out.insert(0, "Frame_ID", df["Frame_ID"].to_numpy())
    out.to_csv(paths.derivative_out, index=False)
    return paths.derivative_out


def ordered_point_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    columns = [c for c in df.columns if c.startswith(prefix)]
    return sorted(columns, key=lambda c: int(c.split("_")[1]))


def remove_offsets(paths: DatasetPaths, force: bool) -> None:
    final_outputs = [paths.final_intensity, paths.final_derivative, paths.final_side, paths.final_top]
    if all(p.exists() for p in final_outputs) and not force:
        print("No-offset outputs already exist.")
        return

    print("Removing intensity baseline and coordinate root offset.")
    df_int = pd.read_csv(paths.intensity_out)
    df_deriv = pd.read_csv(paths.derivative_out)
    df_side = pd.read_csv(paths.side_coord_out)
    df_top = pd.read_csv(paths.top_coord_out)

    common = set(df_int["Frame_ID"]) & set(df_deriv["Frame_ID"]) & set(df_side["Frame_ID"]) & set(df_top["Frame_ID"])
    if not common:
        raise ValueError("No shared Frame_ID values across intensity, derivative, side, and top CSVs")
    common_frames = sorted(common)

    def align(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["Frame_ID"].isin(common_frames)].sort_values("Frame_ID").reset_index(drop=True)

    df_int = align(df_int)
    df_deriv = align(df_deriv)
    df_side = align(df_side)
    df_top = align(df_top)

    feature_cols = sorted([c for c in df_int.columns if "_9pt_" in c], key=natural_intensity_key)
    baseline = df_int.loc[: min(4, len(df_int) - 1), feature_cols].mean(axis=0)
    df_int.loc[:, feature_cols] = df_int[feature_cols] - baseline

    x_cols = ordered_point_columns(df_top, "x_")
    n_points = len(x_cols)
    if n_points == 0:
        raise ValueError(f"No x_i coordinate columns found in {paths.top_coord_out}")

    root_x = root_y = root_z = None
    for i in range(n_points):
        x_col = f"x_{i}"
        y_col = f"y_{i}"
        x_raw = (df_top[x_col].to_numpy(dtype=np.float64) + df_side[x_col].to_numpy(dtype=np.float64)) / 2.0
        y_raw = df_top[y_col].to_numpy(dtype=np.float64).copy()
        z_raw = df_side[y_col].to_numpy(dtype=np.float64).copy()

        if i == 0:
            root_x, root_y, root_z = x_raw, y_raw, z_raw
            df_top.loc[:, x_col] = 0.0
            df_side.loc[:, x_col] = 0.0
            df_top.loc[:, y_col] = 0.0
            df_side.loc[:, y_col] = 0.0
            continue

        df_top.loc[:, x_col] = x_raw - root_x
        df_side.loc[:, x_col] = x_raw - root_x
        df_top.loc[:, y_col] = y_raw - root_y
        df_side.loc[:, y_col] = z_raw - root_z

    df_int.to_csv(paths.final_intensity, index=False)
    df_deriv.to_csv(paths.final_derivative, index=False)
    df_side.to_csv(paths.final_side, index=False)
    df_top.to_csv(paths.final_top, index=False)


def verify_outputs(paths: DatasetPaths, points: int) -> None:
    print("Verifying final training CSVs.")
    outputs = [paths.final_intensity, paths.final_derivative, paths.final_side, paths.final_top]
    for path in outputs:
        if not path.exists():
            raise FileNotFoundError(path)

    df_int = pd.read_csv(paths.final_intensity)
    df_deriv = pd.read_csv(paths.final_derivative)
    df_side = pd.read_csv(paths.final_side)
    df_top = pd.read_csv(paths.final_top)

    raw_cols = sorted([c for c in df_int.columns if "_9pt_" in c], key=natural_intensity_key)
    deriv_cols = sorted([c for c in df_deriv.columns if "_9pt_" in c], key=natural_intensity_key)
    coord_cols = [f"x_{i}" for i in range(points)] + [f"y_{i}" for i in range(points)]

    for label, df in [("side", df_side), ("top", df_top)]:
        missing = [c for c in coord_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{label} coordinate CSV is missing columns: {missing[:5]}")
        if df[coord_cols].isna().any().any():
            raise ValueError(f"{label} coordinate CSV contains NaN values")

    common = set(df_int["Frame_ID"]) & set(df_deriv["Frame_ID"]) & set(df_side["Frame_ID"]) & set(df_top["Frame_ID"])
    if len(common) <= 60:
        raise ValueError(f"Only {len(common)} common frames; training needs more than the 60-frame sequence length")

    print(f"  Common frames: {len(common)}")
    print(f"  Raw 9pt intensity features: {len(raw_cols)}")
    print(f"  Derivative 9pt features: {len(deriv_cols)}")
    print(f"  Coordinate features per view: {len(coord_cols)}")
    print("  Final files:")
    for path in outputs:
        print(f"    {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare one SAM2 dataset for fiber ML training.")
    parser.add_argument("timestamp", help='Dataset timestamp, e.g. "2026-04-12 15-18-24"')
    parser.add_argument("--sam2-dir", type=Path, default=SAM2_DIR, help="Path to the sam2 folder.")
    parser.add_argument("--training-dir", type=Path, default=SCRIPT_DIR, help="Path to the fibertraining folder.")
    parser.add_argument("--points", type=int, default=POINTS_PER_FIBER, help="Number of fiber coordinate points to export.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing prepared CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = DatasetPaths(args.timestamp, args.sam2_dir.resolve(), args.training_dir.resolve())
    ensure_dirs(paths)

    prepare_coords(paths, force=args.force, points=args.points)
    prepare_intensity(paths, force=args.force)
    prepare_derivatives(paths, force=args.force)
    remove_offsets(paths, force=args.force)
    verify_outputs(paths, points=args.points)


if __name__ == "__main__":
    main()
