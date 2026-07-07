import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Utility Functions ---

def resample_polyline(x, y, num_samples=20):
    """
    Resample a polyline (curve defined by raw points) to exactly num_samples points.
    Uses arc-length parameterization for uniform spacing along the curve.
    
    Args:
        x, y: arrays of raw point coordinates
        num_samples: target number of resampled points
    
    Returns:
        x_resampled, y_resampled: arrays of resampled points
    """
    # Compute arc-length distances between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    
    # Cumulative arc length
    arc_length = np.concatenate([[0], np.cumsum(distances)])
    total_length = arc_length[-1]
    
    if total_length == 0:
        # Degenerate case: all points are the same, return copies
        return np.full(num_samples, x[0]), np.full(num_samples, y[0])
    
    # Target arc lengths for resampled points
    target_arc = np.linspace(0, total_length, num_samples)
    
    # Interpolate x and y at target arc lengths
    x_resampled = np.interp(target_arc, arc_length, x)
    y_resampled = np.interp(target_arc, arc_length, y)
    
    return x_resampled, y_resampled

def get_normalized_features(x, y, num_samples=20):
    """
    Resample raw points to num_samples. Returns RAW resampled coordinates 
    (not normalized). Global normalization is applied later after collecting all frames.
    
    Args:
        x, y: arrays of raw point coordinates
        num_samples: number of points to resample to
    
    Returns:
        Feature vector: [x_0, x_1, ..., x_{n-1}, y_0, y_1, ..., y_{n-1}]
    """
    # 1. Resample to uniform arc-length spacing
    x_resampled, y_resampled = resample_polyline(x, y, num_samples)
    
    # 2. Force ordering from Right (High X) to Left (Low X)
    if x_resampled[0] < x_resampled[-1]:
        x_ordered = x_resampled[::-1]
        y_ordered = y_resampled[::-1]
    else:
        x_ordered = x_resampled
        y_ordered = y_resampled
    
    # Return RAW coordinates - global normalization happens after all frames are collected
    return np.concatenate([x_ordered, y_ordered])

def create_video(base_name, show_poly, show_actual, save_video, norm_viz):
    # 1. Define Paths
    cleaned_data_folder = os.path.join(os.path.dirname(__file__), "cleaned data")
    cleaned_file_path = os.path.join(cleaned_data_folder, f"{base_name}_cleaned.csv")
    
    if not os.path.exists(cleaned_file_path):
        print(f"Error: Cleaned data not found at: {cleaned_file_path}")
        return
    cleaned_df = pd.read_csv(cleaned_file_path)

    frames_data = [group for _, group in cleaned_df.groupby('Frame_ID')]
    
    # ==========================================
    # PHASE 1: Generate & Save ML Data (Always)
    # ==========================================
    print("Phase 1: Generating Normalized Features via Raw Resampling...")
    
    ml_dataset = []
    num_samples = 20
    
    for group in frames_data:
        frame_id = group['Frame_ID'].iloc[0]
        x = group['x_smooth'].values
        y = group['y_smooth'].values
        
        if len(x) > 1:
            # Resample raw points and normalize (no fitting, so no failures)
            features = get_normalized_features(x, y, num_samples)
            record = {'Frame_ID': frame_id}
            for k in range(num_samples):
                record[f'x_{k}'] = features[k]
            for k in range(num_samples):
                record[f'y_{k}'] = features[num_samples + k]
            ml_dataset.append(record)

    if ml_dataset:
        ml_df = pd.DataFrame(ml_dataset)
        
        # Apply GLOBAL min-max normalization across all frames
        # This preserves variance between frames while scaling to [0, 1]
        x_cols = [f'x_{k}' for k in range(num_samples)]
        y_cols = [f'y_{k}' for k in range(num_samples)]
        
        # Get global min/max across all x coordinates
        x_min = ml_df[x_cols].values.min()
        x_max = ml_df[x_cols].values.max()
        
        # Get global min/max across all y coordinates
        y_min = ml_df[y_cols].values.min()
        y_max = ml_df[y_cols].values.max()
        
        # Apply global normalization
        if x_max > x_min:
            ml_df[x_cols] = (ml_df[x_cols] - x_min) / (x_max - x_min)
        if y_max > y_min:
            ml_df[y_cols] = (ml_df[y_cols] - y_min) / (y_max - y_min)
        
        print(f"Global normalization applied: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
        
        normalized_folder = os.path.join(os.path.dirname(__file__), "normalized coord data")
        os.makedirs(normalized_folder, exist_ok=True)
        output_csv = os.path.join(normalized_folder, f"{base_name}_normalized_features.csv")
        ml_df.to_csv(output_csv, index=False)
        print(f"CSV Saved: {output_csv} ({len(ml_df)} rows)")
    
    # ==========================================
    # PHASE 2: Visualization
    # ==========================================
    print(f"Phase 2: Rendering Video (Normalized: {norm_viz})...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    margin = 50
    if norm_viz:
        # Auto-scale to actual normalized coord data range
        if ml_dataset:
            all_x = [record[f'x_{k}'] for record in ml_dataset for k in range(num_samples)]
            all_y = [record[f'y_{k}'] for record in ml_dataset for k in range(num_samples)]
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_margin = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 50
            y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 50
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        else:
            ax.set_xlim(-1000, 100)
            ax.set_ylim(-600, 600)
    else:
        ax.set_xlim(cleaned_df['x_smooth'].min() - margin, cleaned_df['x_smooth'].max() + margin)
        ax.set_ylim(cleaned_df['y_smooth'].min() - margin, cleaned_df['y_smooth'].max() + margin)
    
    ax.invert_yaxis()

    line_actual, = ax.plot([], [], lw=4, color='blue', label='Actual Points', alpha=0.3)
    line_resampled, = ax.plot([], [], lw=2, color='red', linestyle='--', label='Resampled (20 pts)')
    
    # Mark the Anchor Point (0,0) with a Green X
    anchor_marker, = ax.plot([], [], 'gx', markersize=10, markeredgewidth=3, label='Anchor')

    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top', 
                         fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_title(f"Spine Deformation (Right-Anchor Normalized - Raw Resampling)")
    ax.legend(loc='upper left')

    def init():
        line_actual.set_data([], [])
        line_resampled.set_data([], [])
        anchor_marker.set_data([], [])
        stats_text.set_text('')
        return line_actual, line_resampled, anchor_marker, stats_text

    def animate(i):
        current_frame = frames_data[i]
        frame_id = current_frame['Frame_ID'].iloc[0]
        x = current_frame['x_smooth'].values
        y = current_frame['y_smooth'].values
        
        # --- HANDLE NORMALIZATION SHIFT ---
        if norm_viz and len(x) > 0:
            # Find the Right-Most Point (Max X) to be the Anchor
            if x[0] > x[-1]:
                base_x, base_y = x[0], y[0]
            else:
                base_x, base_y = x[-1], y[-1]

            x_plot = x - base_x
            y_plot = y - base_y
            
            # Update Anchor Marker at (0,0)
            anchor_marker.set_data([0], [0])
        else:
            base_x, base_y = 0, 0
            x_plot = x
            y_plot = y
            anchor_marker.set_data([], [])

        # 1. Plot Actual
        if show_actual:
            line_actual.set_data(x_plot, y_plot)
        else:
            line_actual.set_data([], [])

        # 2. Plot Resampled
        if show_poly and len(x) > 1:
            x_resampled, y_resampled = resample_polyline(x, y, 100)
            line_resampled.set_data(x_resampled - base_x, y_resampled - base_y)
        else:
            line_resampled.set_data([], [])

        stats_text.set_text(f"Frame: {frame_id}")
        return line_actual, line_resampled, anchor_marker, stats_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, 
        frames=len(frames_data), interval=30, blit=True
    )

    if save_video:
        output_video = os.path.join(cleaned_data_folder, f"{base_name}_video.mp4")
        print(f"Saving video to: {output_video}...")
        try:
            anim.save(output_video, writer='ffmpeg', fps=30)
            print("Video saved successfully.")
        except Exception as e:
            print(f"Error saving video: {e}")
    else:
        print("Displaying video...")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create spine video")
    parser.add_argument("filename", nargs='?', default=None, help="The name of the CSV file (without extension)")
    parser.add_argument("--poly", action="store_true", help="Show the B-Spline fit curve")
    parser.add_argument("--actual", action="store_true", help="Show the actual smoothed pixel data")
    parser.add_argument("--save", action="store_true", help="Save the video to MP4 instead of showing it")
    parser.add_argument("--norm", action="store_true", help="Visualize normalized coord data (origin at 0,0)")

    args = parser.parse_args()

    # Default behavior: if neither --poly nor --actual is set, show both
    if not args.poly and not args.actual:
        show_poly = True
        show_actual = True
    else:
        show_poly = args.poly
        show_actual = args.actual

    cleaned_data_folder = "cleaned data"
    
    if args.filename is None:
        # No argument: process all cleaned data files
        print("No argument provided. Processing all cleaned data files...")
        if not os.path.exists(cleaned_data_folder):
            print(f"Error: {cleaned_data_folder} folder not found.")
            sys.exit(1)
        
        cleaned_files = [f for f in os.listdir(cleaned_data_folder) if f.endswith('_cleaned.csv')]
        if not cleaned_files:
            print(f"No cleaned data files found in {cleaned_data_folder}")
            sys.exit(1)
        
        print(f"Found {len(cleaned_files)} cleaned data files to process.\n")
        for csv_file in cleaned_files:
            base_name = csv_file.replace('_cleaned.csv', '')
            print(f"{'='*60}")
            print(f"Processing: {csv_file}")
            print(f"{'='*60}")
            create_video(base_name, show_poly, show_actual, args.save, args.norm)
            print()
    else:
        # Argument provided: process specific file
        create_video(args.filename, show_poly, show_actual, args.save, args.norm)