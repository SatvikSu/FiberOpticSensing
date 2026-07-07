import cv2
import torch
import os
import numpy as np
import supervision as sv
import argparse
import json
from sam2.build_sam import build_sam2_video_predictor
import shutil
from tqdm import tqdm
from skimage.morphology import skeletonize
import csv
import sys
import gc

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "sam2.1_hiera_large.pt")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "configs", "sam2.1", "sam2.1_hiera_l.yaml")
SEGMENTED_VIDEOS_DIR = os.path.join(SCRIPT_DIR, "segmented videos")
CSV_DIR = os.path.join(SCRIPT_DIR, "csvs")

# Display Settings
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

# Model input size
MODEL_IMAGE_SIZE = 1024

# Pre-crop existing black bars
CROP_ORIGINAL_BLACK_BARS = True
BLACK_BAR_MEAN_THRESH = 25
BLACK_BAR_MIN_WIDTH_PX = 20

# Mask cleanup
MIN_COMPONENT_AREA = 250
MORPH_KERNEL = 3
MORPH_CLOSE_ITERS = 1
MORPH_OPEN_ITERS = 1

# Long-video handling
# SAM2 loads all frames into a single tensor and moves it to the GPU.
# For long videos this can OOM, so we optionally process in chunks.
MAX_FRAMES_PER_CHUNK = 2500
MIN_FRAMES_PER_CHUNK = 250

DEFAULT_VIDEO_DIR_CANDIDATES = ("videos-2026", "videos")
_video_exts = {".mp4", ".mov", ".avi", ".mkv"}
EXCLUDED_VIDEO_NAME_TOKENS = ("speckle", "combined")

# Build set of videos that have already been segmented (to skip duplicates)
_segmented_basenames = set()
if os.path.isdir(SEGMENTED_VIDEOS_DIR):
    for fname in os.listdir(SEGMENTED_VIDEOS_DIR):
        # Extract a "source" base name for both old-style and chunked outputs.
        # Examples:
        # - "X_segmented.mp4" -> "X"
        # - "X__000000-002499_segmented.mp4" -> "X"
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in _video_exts:
            continue
        if "_segmented" not in stem:
            continue
        if "__" in stem:
            src = stem.split("__", 1)[0]
        else:
            src = stem.rsplit("_segmented", 1)[0]
        if src:
            _segmented_basenames.add(src)


def resolve_video_directories(video_dir_args=None):
    if video_dir_args:
        resolved = []
        for directory in video_dir_args:
            candidate = directory
            if not os.path.isabs(candidate):
                candidate = os.path.join(SCRIPT_DIR, candidate)
            if not os.path.isdir(candidate):
                raise FileNotFoundError(f"Video directory not found: {candidate}")
            resolved.append(candidate)
        return resolved

    resolved = []
    for dirname in DEFAULT_VIDEO_DIR_CANDIDATES:
        candidate = os.path.join(SCRIPT_DIR, dirname)
        if os.path.isdir(candidate):
            resolved.append(candidate)

    if not resolved:
        raise FileNotFoundError(
            f"No video directories found. Looked for: {', '.join(DEFAULT_VIDEO_DIR_CANDIDATES)}"
        )

    return resolved


def collect_video_files(video_dirs, include_already_segmented=False):
    video_candidates = []
    for video_dir in video_dirs:
        for root, _, files in os.walk(video_dir):
            for fname in files:
                if os.path.splitext(fname)[1].lower() not in _video_exts:
                    continue
                if any(token in fname.lower() for token in EXCLUDED_VIDEO_NAME_TOKENS):
                    continue
                fpath = os.path.join(root, fname)
                if not os.path.isfile(fpath):
                    continue
                video_candidates.append(fpath)
    return sorted(video_candidates, key=lambda p: (os.path.getsize(p), p.lower()))


def read_progress(progress_path):
    if not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def write_progress(progress_path, **payload):
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def remove_existing_outputs(*paths):
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            os.remove(path)
        except OSError:
            pass


def get_chunk_tag(range_start, range_end_exclusive):
    return f"__{range_start:06d}-{range_end_exclusive - 1:06d}"


def get_chunk_output_paths(base_name, range_start, range_end_exclusive):
    tag = get_chunk_tag(range_start, range_end_exclusive)
    return (
        os.path.join(SEGMENTED_VIDEOS_DIR, f"{base_name}{tag}_segmented.mp4"),
        os.path.join(CSV_DIR, f"{base_name}{tag}_data.csv"),
    )


def remove_chunk_outputs(base_name):
    prefixes = (f"{base_name}__",)
    for folder in (SEGMENTED_VIDEOS_DIR, CSV_DIR):
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.startswith(prefixes):
                remove_existing_outputs(os.path.join(folder, fname))


def remove_temp_dirs(base_name):
    prefixes = (
        f"{base_name}_frames__",
        f"{base_name}_label_frames__",
        f"{base_name}_masks__",
    )
    if not os.path.isdir(SCRIPT_DIR):
        return
    for name in os.listdir(SCRIPT_DIR):
        if not name.endswith("_tmp") or not name.startswith(prefixes):
            continue
        path = os.path.join(SCRIPT_DIR, name)
        if not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: could not delete temp directory {path}: {e}")


def list_chunk_outputs(base_name):
    paths = []
    prefixes = (f"{base_name}__",)
    for folder in (SEGMENTED_VIDEOS_DIR, CSV_DIR):
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.startswith(prefixes):
                paths.append(os.path.join(folder, fname))
    return sorted(paths)


def prompt_existing_progress(base_name, output_video_path, output_csv_path, progress_path):
    existing_paths = [
        path
        for path in (output_video_path, output_csv_path, progress_path)
        if os.path.exists(path)
    ]
    chunk_paths = list_chunk_outputs(base_name)
    existing_paths.extend(chunk_paths[:6])
    if not existing_paths:
        return "restart"

    progress = read_progress(progress_path)
    print("\nExisting segmentation output/progress found:")
    for path in existing_paths:
        print(f"  - {path}")
    if len(chunk_paths) > 6:
        print(f"  - ... {len(chunk_paths) - 6} more chunk checkpoint files")
    if progress:
        status = progress.get("status", "unknown")
        completed_until = progress.get("completed_until_frame")
        failed_at = progress.get("failed_at_frame")
        print(f"  Status: {status}")
        if completed_until is not None:
            print(f"  Completed through frame: {completed_until}")
        if failed_at is not None:
            print(f"  Failed at frame: {failed_at}")
        if status == "complete" and os.path.exists(output_video_path) and os.path.exists(output_csv_path):
            prompt = (
                f"Completed output already exists for {base_name}. "
                "Enter 'k' to keep/skip or 'd' to delete and restart: "
            )
            while True:
                try:
                    answer = input(prompt).strip().lower()
                except EOFError:
                    print("No input available; keeping completed output.")
                    return "keep"
                if answer in {"", "k", "keep", "s", "skip", "r", "resume"}:
                    return "keep"
                if answer in {"d", "delete", "discard", "restart"}:
                    remove_existing_outputs(output_video_path, output_csv_path, progress_path)
                    remove_chunk_outputs(base_name)
                    remove_temp_dirs(base_name)
                    return "restart"
                print("Please enter 'k' to keep/skip, or 'd' to delete/restart.")

    prompt = (
        f"Use existing progress for {base_name}? "
        "Enter 'r' to resume, 'k' to keep/skip, 'd' to delete and restart: "
    )
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            print("No input available; resuming existing progress.")
            return "resume"
        if answer in {"r", "resume", ""}:
            return "resume"
        if answer in {"k", "keep", "s", "skip"}:
            return "keep"
        if answer in {"d", "delete", "discard", "restart"}:
            remove_existing_outputs(output_video_path, output_csv_path, progress_path)
            remove_chunk_outputs(base_name)
            remove_temp_dirs(base_name)
            return "restart"
        print("Please enter 'r' to resume, 'k' to keep/skip, or 'd' to delete/restart.")


def build_chunk_ranges(total_frames_count, chunk_size):
    ranges = []
    chunk_start = 0
    while chunk_start < total_frames_count:
        chunk_end = min(total_frames_count, chunk_start + chunk_size)
        ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return ranges


def chunk_outputs_are_complete(video_path, csv_path, expected_frames):
    if not os.path.exists(video_path) or not os.path.exists(csv_path):
        return False

    cap = cv2.VideoCapture(video_path)
    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    if actual_frames < expected_frames:
        return False

    try:
        with open(csv_path, "r", newline="") as f:
            row_count = sum(1 for _ in f)
    except OSError:
        return False
    return row_count >= expected_frames + 1


def process_chunk_checkpoint(
    *,
    predictor,
    video_file,
    base_name,
    meta,
    total_frames_count,
    fps,
    range_start,
    range_end_exclusive,
):
    chunk_video_path, chunk_csv_path = get_chunk_output_paths(base_name, range_start, range_end_exclusive)
    expected_frames = range_end_exclusive - range_start
    if chunk_outputs_are_complete(chunk_video_path, chunk_csv_path, expected_frames):
        print(f"Checkpoint exists for chunk {range_start}..{range_end_exclusive - 1}; skipping.")
        return chunk_video_path, chunk_csv_path

    remove_existing_outputs(chunk_video_path, chunk_csv_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(chunk_video_path, fourcc, fps, (meta["orig_w"], meta["orig_h"]))
    csv_f = open(chunk_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow(["Frame_ID", "Length_px", "Avg_Thickness_px", "Spine_Coords"])

    try:
        process_video_range(
            predictor=predictor,
            video_file=video_file,
            base_name=base_name,
            meta=meta,
            total_frames_count=total_frames_count,
            fps=fps,
            range_start=range_start,
            range_end_exclusive=range_end_exclusive,
            writer=writer,
            csv_writer=csv_writer,
        )
    except BaseException:
        try:
            writer.release()
        except Exception:
            pass
        try:
            csv_f.close()
        except Exception:
            pass
        remove_existing_outputs(chunk_video_path, chunk_csv_path)
        raise
    finally:
        try:
            writer.release()
        except Exception:
            pass
        try:
            csv_f.close()
        except Exception:
            pass

    return chunk_video_path, chunk_csv_path


def assemble_chunk_outputs(base_name, chunk_ranges, output_video_path, output_csv_path, fps, frame_size):
    remove_existing_outputs(output_video_path, output_csv_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    try:
        with open(output_csv_path, "w", newline="") as out_csv_f:
            out_csv = csv.writer(out_csv_f)
            out_csv.writerow(["Frame_ID", "Length_px", "Avg_Thickness_px", "Spine_Coords"])

            for range_start, range_end_exclusive in chunk_ranges:
                chunk_video_path, chunk_csv_path = get_chunk_output_paths(base_name, range_start, range_end_exclusive)
                cap = cv2.VideoCapture(chunk_video_path)
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        writer.write(frame)
                finally:
                    cap.release()

                with open(chunk_csv_path, "r", newline="") as in_csv_f:
                    in_csv = csv.reader(in_csv_f)
                    next(in_csv, None)
                    for row in in_csv:
                        out_csv.writerow(row)
    finally:
        writer.release()


def load_completed_chunk_ranges(base_name, progress_path):
    progress = read_progress(progress_path) or {}
    ranges = []
    for item in progress.get("completed_chunks", []):
        try:
            range_start = int(item["start"])
            range_end = int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        chunk_video_path, chunk_csv_path = get_chunk_output_paths(base_name, range_start, range_end)
        expected_frames = range_end - range_start
        if chunk_outputs_are_complete(chunk_video_path, chunk_csv_path, expected_frames):
            ranges.append((range_start, range_end))
    return sorted(set(ranges))


def write_checkpoint_progress(
    progress_path,
    *,
    base_name,
    video_file,
    output_video_path,
    output_csv_path,
    total_frames_count,
    completed_ranges,
    status,
    failed_at_frame=None,
    error=None,
):
    completed_ranges = sorted(completed_ranges)
    completed_until = completed_ranges[-1][1] - 1 if completed_ranges else -1
    payload = {
        "base_name": base_name,
        "video_file": video_file,
        "output_video": output_video_path,
        "output_csv": output_csv_path,
        "total_frames": total_frames_count,
        "completed_until_frame": completed_until,
        "completed_chunks": [
            {"start": int(start), "end": int(end)}
            for start, end in completed_ranges
        ],
        "status": status,
    }
    if failed_at_frame is not None:
        payload["failed_at_frame"] = int(failed_at_frame)
    if error is not None:
        payload["error"] = str(error)
    write_progress(progress_path, **payload)


def run_checkpointed_chunks(
    *,
    predictor,
    video_file,
    base_name,
    meta,
    total_frames_count,
    fps,
    output_video_path,
    output_csv_path,
    progress_path,
    initial_chunk_size,
):
    remove_existing_outputs(output_video_path, output_csv_path)
    completed_ranges = load_completed_chunk_ranges(base_name, progress_path)

    write_checkpoint_progress(
        progress_path,
        base_name=base_name,
        video_file=video_file,
        output_video_path=output_video_path,
        output_csv_path=output_csv_path,
        total_frames_count=total_frames_count,
        completed_ranges=completed_ranges,
        status="in_progress",
    )

    chunk_size = int(initial_chunk_size)
    chunk_start = 0
    while chunk_start < total_frames_count:
        covered = [r for r in completed_ranges if r[0] == chunk_start]
        if covered:
            chunk_end = max(r[1] for r in covered)
            print(f"\n--- Chunk {chunk_start}..{chunk_end - 1} already complete; resuming after it ---")
            chunk_start = chunk_end
            continue

        chunk_end = min(total_frames_count, chunk_start + chunk_size)
        print(f"\n--- Chunk {chunk_start}..{chunk_end - 1} ---")
        try:
            process_chunk_checkpoint(
                predictor=predictor,
                video_file=video_file,
                base_name=base_name,
                meta=meta,
                total_frames_count=total_frames_count,
                fps=fps,
                range_start=chunk_start,
                range_end_exclusive=chunk_end,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            completed_ranges.append((chunk_start, chunk_end))
            completed_ranges = sorted(set(completed_ranges))
            write_checkpoint_progress(
                progress_path,
                base_name=base_name,
                video_file=video_file,
                output_video_path=output_video_path,
                output_csv_path=output_csv_path,
                total_frames_count=total_frames_count,
                completed_ranges=completed_ranges,
                status="in_progress",
            )
            chunk_start = chunk_end
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            if chunk_size <= MIN_FRAMES_PER_CHUNK:
                write_checkpoint_progress(
                    progress_path,
                    base_name=base_name,
                    video_file=video_file,
                    output_video_path=output_video_path,
                    output_csv_path=output_csv_path,
                    total_frames_count=total_frames_count,
                    completed_ranges=completed_ranges,
                    status="failed",
                    failed_at_frame=chunk_start,
                    error="CUDA OOM",
                )
                raise
            chunk_size = max(MIN_FRAMES_PER_CHUNK, chunk_size // 2)
            print(
                f"CUDA OOM while processing chunk. "
                f"Retrying frame {chunk_start} with smaller chunks: {chunk_size} frames."
            )
        except BaseException as e:
            write_checkpoint_progress(
                progress_path,
                base_name=base_name,
                video_file=video_file,
                output_video_path=output_video_path,
                output_csv_path=output_csv_path,
                total_frames_count=total_frames_count,
                completed_ranges=completed_ranges,
                status="failed",
                failed_at_frame=chunk_start,
                error=str(e),
            )
            print(
                f"Checkpoint saved through frame "
                f"{completed_ranges[-1][1] - 1 if completed_ranges else -1}. "
                f"Next run can resume at frame {chunk_start}."
            )
            raise

    expected_start = 0
    for start, end in sorted(completed_ranges):
        if start != expected_start:
            raise RuntimeError(f"Missing chunk checkpoint starting at frame {expected_start}.")
        expected_start = end
    if expected_start != total_frames_count:
        raise RuntimeError(f"Missing chunk checkpoint starting at frame {expected_start}.")

    print("\nAssembling final segmented video and CSV from chunk checkpoints...")
    assemble_chunk_outputs(
        base_name,
        sorted(completed_ranges),
        output_video_path,
        output_csv_path,
        fps,
        (meta["orig_w"], meta["orig_h"]),
    )
    write_checkpoint_progress(
        progress_path,
        base_name=base_name,
        video_file=video_file,
        output_video_path=output_video_path,
        output_csv_path=output_csv_path,
        total_frames_count=total_frames_count,
        completed_ranges=completed_ranges,
        status="complete",
    )
    remove_chunk_outputs(base_name)
    remove_temp_dirs(base_name)
    print("Deleted completed chunk checkpoint videos, CSVs, and temporary frame folders after final assembly.")

# ----------------------------
# Display helpers
# ----------------------------

def resize_image_for_display(image: np.ndarray):
    """Resize only for screen; return mapping ratios back into the displayed image."""
    h, w = image.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ratio_x = w / new_w
    ratio_y = h / new_h
    return resized, ratio_x, ratio_y

# ----------------------------
# Black bar detection / cropping
# ----------------------------

def detect_black_bars_lr(frame_bgr: np.ndarray, mean_thresh=BLACK_BAR_MEAN_THRESH, min_width=BLACK_BAR_MIN_WIDTH_PX):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    col_mean = gray.mean(axis=0)

    left = 0
    for v in col_mean:
        if v < mean_thresh:
            left += 1
        else:
            break

    right = 0
    for v in col_mean[::-1]:
        if v < mean_thresh:
            right += 1
        else:
            break

    if left < min_width:
        left = 0
    if right < min_width:
        right = 0
    return left, right


def crop_lr(frame_bgr: np.ndarray, left: int, right: int):
    h, w = frame_bgr.shape[:2]
    x0 = int(max(0, left))
    x1 = int(min(w, w - right))
    if x1 <= x0:
        return frame_bgr
    return frame_bgr[:, x0:x1]

# ----------------------------
# Letterbox utilities
# ----------------------------

def letterbox_to_square(frame_bgr: np.ndarray, target=MODEL_IMAGE_SIZE):
    h, w = frame_bgr.shape[:2]
    scale = min(target / w, target / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left
    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    meta = {
        "target": target,
        "scale": float(scale),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "content_w": int(w),
        "content_h": int(h),
    }
    return padded, meta


def unletterbox_mask_to_content(mask_sq: np.ndarray, meta: dict):
    target = meta["target"]
    x0 = meta["pad_left"]
    y0 = meta["pad_top"]
    x1 = x0 + meta["new_w"]
    y1 = y0 + meta["new_h"]

    cropped = mask_sq[y0:y1, x0:x1].astype(np.uint8) * 255
    resized = cv2.resize(cropped, (meta["content_w"], meta["content_h"]), interpolation=cv2.INTER_NEAREST)
    return (resized > 0)


def paste_content_mask_into_full(mask_content: np.ndarray, orig_h: int, orig_w: int, crop_left: int, crop_right: int):
    out = np.zeros((orig_h, orig_w), dtype=bool)
    x0 = int(crop_left)
    x1 = int(orig_w - crop_right)
    x1 = min(x1, orig_w)
    out[:, x0:x1] = mask_content
    return out

# ----------------------------
# Click UI
# ----------------------------

def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["clicks"].append((x, y))
        print(f"  -> Click: {(x, y)}")

def get_points_on_content_as_model_coords(content_bgr: np.ndarray, meta: dict, prompt_text: str, existing_model_points=None):
    disp_base, ratio_x, ratio_y = resize_image_for_display(content_bgr)
    data = {"clicks": []}

    window_name = f"{prompt_text} (ENTER=Accept)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _mouse_callback, param=data)

    print(f"\n--- {prompt_text} ---")
    
    scale = meta["scale"]
    pad_left = meta["pad_left"]
    pad_top = meta["pad_top"]

    while True:
        disp = disp_base.copy()

        # Draw existing points
        if existing_model_points is not None:
            for (xm, ym) in existing_model_points:
                xc = (float(xm) - pad_left) / scale
                yc = (float(ym) - pad_top) / scale
                vx = int(round(xc / ratio_x))
                vy = int(round(yc / ratio_y))
                cv2.circle(disp, (vx, vy), 4, (0, 255, 0), -1)

        # Draw new points
        for (vx, vy) in data["clicks"]:
            cv2.circle(disp, (vx, vy), 4, (0, 0, 255), -1)

        cv2.imshow(window_name, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER key
            break

    cv2.destroyWindow(window_name)

    if not data["clicks"]:
        return None

    pts_model = []
    for (vx, vy) in data["clicks"]:
        xc = float(vx * ratio_x)
        yc = float(vy * ratio_y)
        xm = xc * scale + pad_left
        ym = yc * scale + pad_top
        pts_model.append((xm, ym))

    return np.array(pts_model, dtype=np.float32)

# ----------------------------
# Mask cleanup
# ----------------------------

def logits_to_mask_bool(mask_logits):
    if isinstance(mask_logits, torch.Tensor):
        t = mask_logits
        if t.ndim == 4: t = t[0]
        if t.ndim == 3: t = t[0]
        return (t > 0.0).detach().cpu().numpy().astype(bool)
    arr = mask_logits
    if arr.ndim == 3: arr = arr[0]
    return (arr > 0.0).astype(bool)

def remove_small_components(mask_bool: np.ndarray, min_area=MIN_COMPONENT_AREA):
    m = mask_bool.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1: return mask_bool
    out = np.zeros_like(m)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out.astype(bool)

def prune_skeleton_to_longest_route(skeleton_img):
    """
    Keeps only the longest contiguous path in the skeleton, removing spurs/branches.
    Assumes the skeleton is mostly a tree structure (no loops).
    """
    # Get all pixel coordinates
    pts = np.transpose(np.nonzero(skeleton_img))
    if len(pts) == 0: return skeleton_img
    
    h, w = skeleton_img.shape
    
    # Build Adjacency Graph
    # (Mapping every pixel to its 8-neighbors)
    adj = {}
    pt_set = set((p[0], p[1]) for p in pts)
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for r, c in pt_set:
        adj[(r, c)] = []
        for dr, dc in offsets:
            nr, nc = r+dr, c+dc
            if (nr, nc) in pt_set:
                adj[(r, c)].append((nr, nc))

    # Helper: BFS to find the farthest node and path from a start node
    def get_farthest_node(start_node):
        # Queue stores (current_node, path_to_current)
        queue = [(start_node, [start_node])]
        visited = {start_node}
        
        max_dist_node = start_node
        longest_path = [start_node]
        
        idx = 0
        while idx < len(queue):
            curr, path = queue[idx]
            idx += 1
            
            if len(path) > len(longest_path):
                longest_path = path
                max_dist_node = curr
            
            for nbr in adj[curr]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, path + [nbr]))
        
        return max_dist_node, longest_path

    # 1. Find an arbitrary endpoint (node with 1 neighbor)
    # If no endpoints (loop), just pick first point.
    endpoints = [p for p in pt_set if len(adj[p]) == 1]
    start_node = endpoints[0] if endpoints else next(iter(pt_set))

    # 2. First BFS: Find the node farthest from the start
    farthest_A, _ = get_farthest_node(start_node)

    # 3. Second BFS: Find the node farthest from A (This path is the diameter/spine)
    _, spine_path = get_farthest_node(farthest_A)

    # Reconstruct Image
    new_skel = np.zeros_like(skeleton_img)
    for r, c in spine_path:
        new_skel[r, c] = 1
        
    return new_skel

def cleanup_mask(mask_bool: np.ndarray):
    # Convert to 0-255
    m = (mask_bool.astype(np.uint8) * 255)
    
    # 1. Remove small noise first
    m = remove_small_components(m > 127).astype(np.uint8) * 255

    # 2. Aggressive Smoothing (Gaussian Blur)
    # This turns jagged edges into smooth curves.
    # Sigma=3.0 is usually strong enough to melt away pixel-level noise.
    m_float = m.astype(np.float32) / 255.0
    blur_sigma = 3.0 
    m_blurred = cv2.GaussianBlur(m_float, (0, 0), blur_sigma)
    
    # Threshold back to sharp binary
    # We use 0.5 as the cutoff (middle of the blur)
    m_smooth = (m_blurred > 0.5).astype(np.uint8) * 255
    
    return m_smooth > 127

def analyze_fiber_structure(binary_mask: np.ndarray):
    cleaned = cleanup_mask(binary_mask)
    
    # Initial Skeletonization
    raw_skeleton = skeletonize(cleaned)
    
    # PRUNING: Keep only the longest path (The Spine)
    final_skeleton = prune_skeleton_to_longest_route(raw_skeleton)
    
    y_indices, x_indices = np.where(final_skeleton)
    spine_coords = [(int(x), int(y)) for x, y in zip(x_indices.tolist(), y_indices.tolist())]
    
    if spine_coords:
        spine_coords.sort(key=lambda p: (p[0], p[1]))
        
    area = int(np.sum(cleaned))
    length_px = int(len(spine_coords))
    avg_thickness_px = (area / length_px) if length_px > 0 else 0.0
    
    # Return 'final_skeleton' (bool) so it displays correctly
    return spine_coords, length_px, float(avg_thickness_px), cleaned, (final_skeleton > 0)

# ----------------------------
# Frame Extraction
# ----------------------------

def extract_frame0_and_prepare(frames_dir: str, video_path: str):
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    ret, frame_full = cap.read()
    cap.release()
    if not ret or frame_full is None:
        return None

    orig_h, orig_w = frame_full.shape[:2]
    left_bar, right_bar = (0, 0)
    if CROP_ORIGINAL_BLACK_BARS:
        left_bar, right_bar = detect_black_bars_lr(frame_full)

    frame_content = crop_lr(frame_full, left_bar, right_bar)
    content_h, content_w = frame_content.shape[:2]

    frame_model, lb_meta = letterbox_to_square(frame_content, MODEL_IMAGE_SIZE)
    meta = {
        **lb_meta,
        "orig_w": orig_w,
        "orig_h": orig_h,
        "crop_left": int(left_bar),
        "crop_right": int(right_bar),
        "content_w": int(content_w),
        "content_h": int(content_h),
    }

    path0 = os.path.join(frames_dir, "000000.jpg")
    cv2.imwrite(path0, frame_model, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return path0, frame_full, frame_content, meta

def extract_all_frames(frames_dir: str, video_path: str, meta: dict):
    """Extract ALL frames to ensure propagation has everything it needs."""
    crop_left = meta["crop_left"]
    crop_right = meta["crop_right"]

    cap = cv2.VideoCapture(video_path)
    idx = 0
    print(f"Extracting full video frames for: {os.path.basename(video_path)}")
    
    while True:
        ret, frame_full = cap.read()
        if not ret:
            break
        
        # We process every frame exactly like frame 0
        frame_content = crop_lr(frame_full, crop_left, crop_right)
        frame_model, _ = letterbox_to_square(frame_content, MODEL_IMAGE_SIZE)
        
        out_path = os.path.join(frames_dir, f"{idx:06d}.jpg")
        cv2.imwrite(out_path, frame_model, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        idx += 1
        if idx % 50 == 0:
            print(f"  -> Extracted {idx} frames...", end="\r")
            
    cap.release()
    print(f"\n  Done. Total frames: {idx}")


def extract_frames_range(frames_dir: str, video_path: str, meta: dict, start_frame: int, end_frame_exclusive: int):
    """Extract frames in [start_frame, end_frame_exclusive) into frames_dir as 000000.jpg.."""
    crop_left = meta["crop_left"]
    crop_right = meta["crop_right"]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    idx = 0
    total = max(0, int(end_frame_exclusive) - int(start_frame))
    print(
        f"Extracting frames {start_frame}..{end_frame_exclusive - 1} for: {os.path.basename(video_path)}"
    )

    for _ in range(total):
        ret, frame_full = cap.read()
        if not ret:
            break

        frame_content = crop_lr(frame_full, crop_left, crop_right)
        frame_model, _ = letterbox_to_square(frame_content, MODEL_IMAGE_SIZE)

        out_path = os.path.join(frames_dir, f"{idx:06d}.jpg")
        cv2.imwrite(out_path, frame_model, [cv2.IMWRITE_JPEG_QUALITY, 100])
        idx += 1

        if idx % 50 == 0:
            print(f"  -> Extracted {idx}/{total} frames...", end="\r")

    cap.release()
    print(f"\n  Done. Total extracted in range: {idx}")
    return idx

# ----------------------------
# Frame Selection
# ----------------------------

def select_start_frame(video_path, start_frame: int = 0, end_frame: int | None = None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(start_frame))
    if end_frame is None:
        end_frame = total_frames - 1
    end_frame = min(total_frames - 1, int(end_frame))
    if end_frame < start_frame:
        end_frame = start_frame

    current_idx = start_frame
    window_name = "Select Start Frame (Trackbar or Left/Right Keys)"
    cv2.namedWindow(window_name)
    
    def on_trackbar(val):
        nonlocal current_idx
        current_idx = start_frame + val

    cv2.createTrackbar("Frame", window_name, 0, end_frame - start_frame, on_trackbar)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        disp = frame.copy()
        text = f"Frame: {current_idx}/{total_frames - 1}  (Range {start_frame}-{end_frame}) - Press ENTER"
        
        # Add a text background for readability
        cv2.rectangle(disp, (40, 20), (800, 80), (0, 0, 0), -1)
        cv2.putText(disp, text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        disp_resized, _, _ = resize_image_for_display(disp)
        cv2.imshow(window_name, disp_resized)
        
        # Sync trackbar with current_idx in case keys modified it
        cv2.setTrackbarPos("Frame", window_name, current_idx - start_frame)

        key = cv2.waitKey(25) & 0xFF
        if key == 13: # ENTER key
            break
        elif key == 81 or key == 2424036: # Left Arrow
            current_idx = max(start_frame, current_idx - 1)
        elif key == 83 or key == 2555904: # Right Arrow
            current_idx = min(end_frame, current_idx + 1)

    cap.release()
    cv2.destroyWindow(window_name)
    return current_idx


def process_video_range(
    *,
    predictor,
    video_file: str,
    base_name: str,
    meta: dict,
    total_frames_count: int,
    fps: float,
    range_start: int,
    range_end_exclusive: int,
    writer,
    csv_writer,
):
    """Process a full video or a chunk [range_start, range_end_exclusive)."""
    range_start = int(range_start)
    range_end_exclusive = int(range_end_exclusive)
    if range_end_exclusive <= range_start:
        return

    tag = f"__{range_start:06d}-{range_end_exclusive - 1:06d}"
    frames_dir = os.path.join(SCRIPT_DIR, f"{base_name}_frames{tag}_tmp")

    label_frames_dir = os.path.join(SCRIPT_DIR, f"{base_name}_label_frames{tag}_tmp")
    mask_tmp_dir = os.path.join(SCRIPT_DIR, f"{base_name}_masks{tag}_tmp")

    # Cleanup masks from previous runs
    if os.path.exists(mask_tmp_dir):
        shutil.rmtree(mask_tmp_dir)
    os.makedirs(mask_tmp_dir, exist_ok=True)

    # Keep cached frames if already extracted
    os.makedirs(frames_dir, exist_ok=True)

    # Tiny, per-run labeling folder so SAM doesn't load the full cached chunk during clicks.
    if os.path.exists(label_frames_dir):
        shutil.rmtree(label_frames_dir)
    os.makedirs(label_frames_dir, exist_ok=True)

    # 1. SELECT START FRAME (within range)
    start_frame_idx = select_start_frame(video_file, start_frame=range_start, end_frame=range_end_exclusive - 1)
    print(f"Selected Start Frame: {start_frame_idx} (range {range_start}-{range_end_exclusive - 1})")

    # 2. Load the start frame for interactive clicks
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    ret, start_full_raw = cap.read()
    cap.release()
    if not ret or start_full_raw is None:
        print("Failed to read selected start frame; skipping range.")
        return

    # Use the same crop settings as the video meta so click coords and masks align.
    start_content_raw = crop_lr(start_full_raw, meta["crop_left"], meta["crop_right"])
    start_frame_model, _ = letterbox_to_square(start_content_raw, MODEL_IMAGE_SIZE)

    # Save as 000000.jpg so SAM sees a 1-frame "video"
    label_frame_idx = 0
    start_frame_path = os.path.join(label_frames_dir, f"{label_frame_idx:06d}.jpg")
    cv2.imwrite(start_frame_path, start_frame_model, [cv2.IMWRITE_JPEG_QUALITY, 100])

    inference_state = predictor.init_state(video_path=label_frames_dir)
    predictor.reset_state(inference_state)

    # 3. INTERACTIVE LABELING
    final_pos_points = None
    final_neg_points = None
    while True:
        predictor.reset_state(inference_state)

        pos_model = get_points_on_content_as_model_coords(
            start_content_raw, meta, f"FRAME {start_frame_idx}: POSITIVE (Click Fiber)", None
        )
        if pos_model is None:
            final_pos_points = None
            break

        labels_pos = np.ones((pos_model.shape[0],), dtype=np.int32)
        _, _, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=label_frame_idx,
            obj_id=1,
            points=pos_model,
            labels=labels_pos,
        )

        mask_model = logits_to_mask_bool(out_mask_logits)
        mask_content = unletterbox_mask_to_content(mask_model, meta)
        mask_full = paste_content_mask_into_full(
            mask_content, meta["orig_h"], meta["orig_w"], meta["crop_left"], meta["crop_right"]
        )

        preview_full = start_full_raw.copy()
        preview_full[mask_full] = (
            preview_full[mask_full] * 0.5 + np.array([128, 0, 128]) * 0.5
        ).astype(np.uint8)

        disp_prev, _, _ = resize_image_for_display(preview_full)
        cv2.imshow("Preview (Pos Only) - Enter/R/Q", disp_prev)
        k = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if k == ord('q'):
            raise SystemExit(0)
        if k == ord('r'):
            continue

        ui_bg = start_content_raw.copy()
        green_overlay = np.zeros_like(ui_bg)
        green_overlay[:] = (0, 255, 0)
        ui_bg = np.where(mask_content[..., None], cv2.addWeighted(ui_bg, 0.6, green_overlay, 0.4, 0), ui_bg)

        existing_pts = [(float(x), float(y)) for (x, y) in pos_model]
        neg_model = get_points_on_content_as_model_coords(
            ui_bg, meta, f"FRAME {start_frame_idx}: NEGATIVE (Click Errors)", existing_pts
        )

        all_points = [pos_model]
        all_labels = [np.ones(len(pos_model), dtype=np.int32)]
        if neg_model is not None:
            all_points.append(neg_model)
            all_labels.append(np.zeros(len(neg_model), dtype=np.int32))

        combined_points = np.concatenate(all_points, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)

        _, _, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=label_frame_idx,
            obj_id=1,
            points=combined_points,
            labels=combined_labels,
        )

        mask_model = logits_to_mask_bool(out_mask_logits)
        mask_content = unletterbox_mask_to_content(mask_model, meta)
        mask_full = paste_content_mask_into_full(
            mask_content, meta["orig_h"], meta["orig_w"], meta["crop_left"], meta["crop_right"]
        )

        preview_final = start_full_raw.copy()
        preview_final[mask_full] = (
            preview_final[mask_full] * 0.5 + np.array([128, 0, 128]) * 0.5
        ).astype(np.uint8)

        disp_final, _, _ = resize_image_for_display(preview_final)
        cv2.imshow("FINAL PREVIEW - Enter to Process", disp_final)
        k2 = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if k2 == 13:
            final_pos_points = pos_model
            final_neg_points = neg_model
            break
        if k2 == ord('q'):
            raise SystemExit(0)

    if final_pos_points is None:
        raise RuntimeError("No points selected; aborting to avoid partial output.")

    # 4. Extract frames for this range (cached if present)
    expected = range_end_exclusive - range_start
    frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")]
    if len(frame_files) >= expected:
        print("Using cached frames, skipping re-extraction.")
    else:
        print("Points accepted. Extracting frames for this range...")
        extract_frames_range(frames_dir, video_file, meta, range_start, range_end_exclusive)

    # 5. Re-init SAM with the range frames
    print("Re-initializing model with range frames...")
    predictor.reset_state(inference_state)
    inference_state = predictor.init_state(video_path=frames_dir)

    # 6. Re-apply points on the correct frame within the range
    start_frame_rel = start_frame_idx - range_start
    all_points = [final_pos_points]
    all_labels = [np.ones(len(final_pos_points), dtype=np.int32)]
    if final_neg_points is not None:
        all_points.append(final_neg_points)
        all_labels.append(np.zeros(len(final_neg_points), dtype=np.int32))

    combined_points = np.concatenate(all_points, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=start_frame_rel,
        obj_id=1,
        points=combined_points,
        labels=combined_labels,
    )

    # 7. Propagate within this range and write masks using GLOBAL frame indices
    print("Propagating BACKWARD...")
    if start_frame_rel > 0:
        for frame_idx, _, out_mask_logits in tqdm(
            predictor.propagate_in_video(inference_state, start_frame_idx=start_frame_rel, reverse=True),
            desc="Backward",
            file=sys.stdout,
            dynamic_ncols=True,
            total=start_frame_rel,
            mininterval=0.5,
            leave=False,
        ):
            mask_bool = logits_to_mask_bool(out_mask_logits)
            global_idx = range_start + int(frame_idx)
            cv2.imwrite(
                os.path.join(mask_tmp_dir, f"{global_idx:06d}.png"),
                (mask_bool * 255).astype(np.uint8),
            )

    print("Propagating FORWARD...")
    num_frames = inference_state.get("num_frames", 0)
    total_forward = num_frames - start_frame_rel
    for frame_idx, _, out_mask_logits in tqdm(
        predictor.propagate_in_video(inference_state, start_frame_idx=start_frame_rel, reverse=False),
        desc="Forward",
        file=sys.stdout,
        dynamic_ncols=True,
        total=total_forward,
        mininterval=0.5,
        leave=False,
    ):
        mask_bool = logits_to_mask_bool(out_mask_logits)
        global_idx = range_start + int(frame_idx)
        cv2.imwrite(
            os.path.join(mask_tmp_dir, f"{global_idx:06d}.png"),
            (mask_bool * 255).astype(np.uint8),
        )

    # 8. Assemble output video & CSV for this range (APPEND into shared writer/CSV)
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, range_start)

    try:
        for f_idx in tqdm(
            range(range_start, range_end_exclusive),
            desc="Writing Video",
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.5,
            leave=False,
        ):
            ret, frame_curr = cap.read()
            if not ret:
                break

            mask_path = os.path.join(mask_tmp_dir, f"{f_idx:06d}.png")
            if os.path.exists(mask_path):
                m_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_model_bool = (m_img > 127)

                mask_content = unletterbox_mask_to_content(mask_model_bool, meta)
                spine_coords_content, length_px, avg_thickness_px, cleaned, skeleton_content = analyze_fiber_structure(mask_content)

                crop_left = int(meta["crop_left"])
                spine_coords_full = [(int(x + crop_left), int(y)) for (x, y) in spine_coords_content]
                csv_writer.writerow([f_idx, length_px, avg_thickness_px, str(spine_coords_full)])

                mask_full = paste_content_mask_into_full(cleaned, meta["orig_h"], meta["orig_w"], meta["crop_left"], meta["crop_right"])
                skeleton_full = paste_content_mask_into_full(skeleton_content, meta["orig_h"], meta["orig_w"], meta["crop_left"], meta["crop_right"])

                frame_curr[mask_full] = (
                    frame_curr[mask_full] * 0.5 + np.array([128, 0, 128]) * 0.5
                ).astype(np.uint8)
                frame_curr[skeleton_full] = (255, 255, 0)

            writer.write(frame_curr)
    finally:
        cap.release()
        if os.path.exists(mask_tmp_dir):
            shutil.rmtree(mask_tmp_dir)

# ----------------------------
# Main
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Segment fiber videos with SAM2.")
    parser.add_argument(
        "--video-dir",
        action="append",
        default=None,
        help="Relative or absolute directory containing input videos. Can be passed multiple times. Defaults to videos-2026, then videos.",
    )
    parser.add_argument(
        "--list-videos",
        action="store_true",
        help="List the videos that would be processed and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths, load video metadata, and exit before SAM2 model load or interactive labeling.",
    )
    parser.add_argument(
        "--include-already-segmented",
        action="store_true",
        help="Include videos even if segmented outputs already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_dirs = resolve_video_directories(args.video_dir)
    video_files = collect_video_files(
        video_dirs, include_already_segmented=args.include_already_segmented
    )

    print("Video directories:")
    for directory in video_dirs:
        print(f"  - {directory}")

    if not video_files:
        print("No candidate videos found.")
        return

    print("Candidate videos:")
    for path in video_files:
        print(f"  - {path}")

    if args.list_videos:
        return

    if args.dry_run:
        print("\nDry run summary:")
        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)
            opened = cap.isOpened()
            fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0.0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
            ret, frame = cap.read() if opened else (False, None)
            cap.release()
            if not opened or not ret or frame is None:
                raise RuntimeError(f"Could not read first frame from {video_file}")
            print(
                f"  - {os.path.basename(video_file)}: "
                f"{frames} frames, {fps:.2f} fps, shape={frame.shape[1]}x{frame.shape[0]}"
            )
        return

    print("Loading SAM 2 model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    predictor = build_sam2_video_predictor(
        CONFIG_PATH,
        CHECKPOINT_PATH,
        device=device,
        vos_optimized=False,
    )

    os.makedirs(SEGMENTED_VIDEOS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    for video_file in video_files:
        if not os.path.exists(video_file):
            print(f"Missing video, skipping: {video_file}")
            continue

        base_name = os.path.splitext(os.path.basename(video_file))[0]
        print(f"\n==========================================")
        print(f"Processing: {base_name}")
        print(f"==========================================")

        # Read video properties and build meta from frame 0
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first_full = cap.read()
        cap.release()
        if not ret or first_full is None:
            print("Failed to read first frame; skipping.")
            continue

        left_bar, right_bar = detect_black_bars_lr(first_full)
        frame_content = crop_lr(first_full, left_bar, right_bar)
        _, meta = letterbox_to_square(frame_content, MODEL_IMAGE_SIZE)

        h, w = first_full.shape[:2]
        ch, cw = frame_content.shape[:2]
        meta["orig_w"] = w
        meta["orig_h"] = h
        meta["crop_left"] = left_bar
        meta["crop_right"] = right_bar
        meta["content_w"] = cw
        meta["content_h"] = ch

        output_video_path = os.path.join(SEGMENTED_VIDEOS_DIR, f"{base_name}_segmented.mp4")
        output_csv_path = os.path.join(CSV_DIR, f"{base_name}_data.csv")
        progress_path = os.path.join(CSV_DIR, f"{base_name}_progress.json")

        if (
            base_name in _segmented_basenames
            or os.path.exists(output_video_path)
            or os.path.exists(output_csv_path)
            or os.path.exists(progress_path)
        ):
            action = prompt_existing_progress(
                base_name,
                output_video_path,
                output_csv_path,
                progress_path,
            )
            if action == "keep":
                print("Keeping existing progress/output; skipping this video.")
                continue

        try:
            if total_frames_count > MAX_FRAMES_PER_CHUNK:
                print(
                    f"Long video detected ({total_frames_count} frames). "
                    f"Processing in chunks (<= {MAX_FRAMES_PER_CHUNK} frames) to avoid CUDA OOM."
                )
            else:
                print(f"Processing video in a single checkpoint chunk ({total_frames_count} frames).")

            initial_chunk_size = min(int(MAX_FRAMES_PER_CHUNK), total_frames_count)
            run_checkpointed_chunks(
                predictor=predictor,
                video_file=video_file,
                base_name=base_name,
                meta=meta,
                total_frames_count=total_frames_count,
                fps=fps,
                output_video_path=output_video_path,
                output_csv_path=output_csv_path,
                progress_path=progress_path,
                initial_chunk_size=initial_chunk_size,
            )
            print(f"\nSaved Video: {output_video_path}")
        except BaseException as e:
            print(f"Error while processing {base_name}: {e}")
            print("Completed chunk checkpoints were kept. Next run can resume from the next unfinished chunk.")
            raise


if __name__ == "__main__":
    main()
