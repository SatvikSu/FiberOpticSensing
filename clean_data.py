import sys
import os
import re
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def parse_spine_coords(spine_str):
    if not isinstance(spine_str, str):
        return [], []
    
    # Try new format: [(x1, y1), (x2, y2), ...]
    try:
        coords_list = eval(spine_str)
        if isinstance(coords_list, list) and len(coords_list) > 0 and isinstance(coords_list[0], tuple):
            x_vals = [c[0] for c in coords_list]
            y_vals = [c[1] for c in coords_list]
            return x_vals, y_vals
    except:
        pass
    
    # Fall back to old format: np.int64(x), np.int64(y), ...
    matches = re.findall(r'np\.int64\((-?\d+)\)', spine_str)
    if matches:
        coords = [int(m) for m in matches]
        return coords[0::2], coords[1::2]
    
    return [], []

def filter_largest_cluster(x_vals, y_vals, gap_threshold=50):
    if not x_vals: return [], []
    clusters = []
    curr_x, curr_y = [x_vals[0]], [y_vals[0]]
    for i in range(1, len(x_vals)):
        dist = np.sqrt((x_vals[i] - x_vals[i-1])**2 + (y_vals[i] - y_vals[i-1])**2)
        if dist > gap_threshold:
            clusters.append((curr_x, curr_y))
            curr_x, curr_y = [], []
        curr_x.append(x_vals[i])
        curr_y.append(y_vals[i])
    clusters.append((curr_x, curr_y))
    largest = max(clusters, key=lambda c: len(c[0]))
    return largest[0], largest[1]


def _resample_polyline(xs, ys, num_points=40):
    """Resample a polyline to a fixed number of points.

    We keep the polyline direction consistent by forcing the first endpoint to be the
    right-most endpoint (higher x). This matches the anchor convention used elsewhere.

    Returns (x_new, y_new) as numpy arrays of shape (num_points,). If resampling is
    impossible, returns (None, None).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2 or ys.size < 2:
        return None, None

    # Force direction to start at right-most endpoint.
    if xs[0] < xs[-1]:
        xs = xs[::-1]
        ys = ys[::-1]

    dx = np.diff(xs)
    dy = np.diff(ys)
    seg = np.sqrt(dx * dx + dy * dy)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if not np.isfinite(total) or total <= 0:
        # Fallback: parameterize by index
        s = np.linspace(0.0, 1.0, xs.size)
        u = np.linspace(0.0, 1.0, int(num_points))
    else:
        u = np.linspace(0.0, total, int(num_points))

    x_new = np.interp(u, s, xs)
    y_new = np.interp(u, s, ys)
    return x_new, y_new

def clean_data(input_path, output_path):
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found.")
        sys.exit(1)

    if "Frame_ID" not in df.columns:
        print(f"Error: {input_path} is missing Frame_ID")
        return

    print(f"Analyzing statistics for {len(df)} frames...")
    
    # Never drop frames: we keep a slot per original Frame_ID.
    # If a frame fails parsing/filtering/outlier checks, it becomes "missing" and
    # we interpolate it from neighboring frames.
    num_points = 40

    frame_ids = pd.to_numeric(df["Frame_ID"], errors="coerce").to_numpy(dtype=float)
    valid_f = np.isfinite(frame_ids)
    df = df.loc[valid_f].copy()
    frame_ids = df["Frame_ID"].astype(int).to_numpy()
    if frame_ids.size == 0:
        print("No valid Frame_ID values found.")
        return

    # Store resampled points for every frame (NaN for missing)
    x_mat = np.full((len(df), num_points), np.nan, dtype=float)
    y_mat = np.full((len(df), num_points), np.nan, dtype=float)
    tip_x = np.full((len(df),), np.nan, dtype=float)
    tip_y = np.full((len(df),), np.nan, dtype=float)

    parsed_ok = np.zeros((len(df),), dtype=bool)

    # --- PASS 1: Parse + filter + resample per frame ---
    for i, (_, row) in enumerate(df.iterrows()):
        spine_str = row.get('Spine_Coords', '')

        xs, ys = parse_spine_coords(spine_str)
        if len(xs) == 0:
            continue

        xs, ys = filter_largest_cluster(xs, ys)
        if len(xs) == 0:
            continue

        x_new, y_new = _resample_polyline(xs, ys, num_points=num_points)
        if x_new is None or y_new is None:
            continue

        x_mat[i, :] = x_new
        y_mat[i, :] = y_new

        # With enforced direction, the anchor endpoint is at index 0 (right-most).
        tip_x[i] = float(x_new[0])
        tip_y[i] = float(y_new[0])
        parsed_ok[i] = True

    if not bool(np.any(parsed_ok)):
        print("No valid coordinate data found.")
        return

    # Calculate the average location of the right-most point
    mean_x = float(np.nanmean(tip_x[parsed_ok]))
    mean_y = float(np.nanmean(tip_y[parsed_ok]))
    
    # Calculate Standard Deviation (spread) to define what is "near"
    std_x = float(np.nanstd(tip_x[parsed_ok]))
    std_y = float(np.nanstd(tip_y[parsed_ok]))

    # Define a threshold
    # We keep points within 2 standard deviations of the average
    # You can change 2.0 to 1.5 (stricter) or 3.0 (looser)
    sigma_count = 2.0 
    limit_x = sigma_count * std_x
    limit_y = sigma_count * std_y

    print(f"Average Tip Location: ({mean_x:.1f}, {mean_y:.1f})")
    print(f"Removing frames where tip deviates > {limit_x:.1f} pixels in X or {limit_y:.1f} pixels in Y")

    # Mark outliers as missing (do NOT drop), then interpolate through them.
    outlier = np.zeros((len(df),), dtype=bool)
    for i in range(len(df)):
        if not parsed_ok[i]:
            continue
        dist_x = abs(tip_x[i] - mean_x)
        dist_y = abs(tip_y[i] - mean_y)
        if dist_x > limit_x or dist_y > limit_y:
            outlier[i] = True

    x_mat[outlier, :] = np.nan
    y_mat[outlier, :] = np.nan

    # --- PASS 2: Interpolate missing frames over time, then smooth over time ---
    order = np.argsort(frame_ids)
    frame_ids_sorted = frame_ids[order]
    x_sorted = x_mat[order]
    y_sorted = y_mat[order]

    x_df = pd.DataFrame(x_sorted, index=frame_ids_sorted)
    y_df = pd.DataFrame(y_sorted, index=frame_ids_sorted)

    # Fill internal gaps by linear interpolation; fill edges by nearest valid.
    x_df = x_df.interpolate(method="linear", axis=0, limit_direction="both").ffill().bfill()
    y_df = y_df.interpolate(method="linear", axis=0, limit_direction="both").ffill().bfill()

    x_interp = x_df.to_numpy(dtype=float)
    y_interp = y_df.to_numpy(dtype=float)

    # Smooth over time (per-point index). This keeps every frame but removes jitter.
    n_time = x_interp.shape[0]
    # Choose an odd window length <= n_time
    win = min(31, n_time if (n_time % 2 == 1) else n_time - 1)
    if win >= 7:
        try:
            x_smooth = savgol_filter(x_interp, window_length=win, polyorder=2, axis=0)
            y_smooth = savgol_filter(y_interp, window_length=win, polyorder=2, axis=0)
        except Exception:
            x_smooth, y_smooth = x_interp, y_interp
    else:
        x_smooth, y_smooth = x_interp, y_interp

    cleaned_records = []
    for t_idx, fid in enumerate(frame_ids_sorted):
        for p_idx in range(num_points):
            cleaned_records.append(
                {
                    "Frame_ID": int(fid),
                    "Point_Index": int(p_idx),
                    "x": float(x_interp[t_idx, p_idx]),
                    "y": float(y_interp[t_idx, p_idx]),
                    "x_smooth": float(x_smooth[t_idx, p_idx]),
                    "y_smooth": float(y_smooth[t_idx, p_idx]),
                }
            )

    # Save Cleaned Data
    clean_df = pd.DataFrame(cleaned_records)
    clean_df.to_csv(output_path, index=False)
    print(f"Cleaned points saved to: {output_path}")

if __name__ == "__main__":
    input_folder = "raw data"
    output_folder = "cleaned data"

    if not os.path.exists(output_folder): os.makedirs(output_folder)

    if len(sys.argv) < 2:
        # No argument: clean all files in raw data folder
        print("No argument provided. Cleaning all files in 'raw data' folder...")
        if not os.path.exists(input_folder):
            print(f"Error: {input_folder} folder not found.")
            sys.exit(1)
        
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in {input_folder}")
            sys.exit(1)
        
        print(f"Found {len(csv_files)} CSV files to clean.\n")
        for csv_file in csv_files:
            base_name = csv_file.replace('.csv', '')
            input_path = os.path.join(input_folder, csv_file)
            output_path = os.path.join(output_folder, f"{base_name}_cleaned.csv")
            print(f"Processing: {csv_file}...")
            clean_data(input_path, output_path)
            print()
    else:
        # Argument provided: clean specific file
        base_name = sys.argv[1]
        input_path = os.path.join(input_folder, f"{base_name}.csv")
        output_path = os.path.join(output_folder, f"{base_name}_cleaned.csv")
        clean_data(input_path, output_path)