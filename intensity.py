import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_DIR_CANDIDATES = ("videos-2026", "videos")
DEFAULT_RAW_OUTPUT_DIR = "intensity_data"
DEFAULT_NORMALIZED_OUTPUT_DIR = "normalized intensity data"
DEFAULT_AUDIT_OUTPUT_DIR = "intensity_audit"

TRACKING_SEARCH_PADDING_PX = 36
TRACKING_RADIUS_TOLERANCE_PCT = 0.18
TRACKING_MAX_CENTER_JUMP_PX = 35.0
CENTER_SMOOTHING_ALPHA = 0.70
MAX_CENTER_CORRECTION_PX = 8.0
GLOBAL_TRACKING_INLIER_TOLERANCE_PX = 7.0
SAMPLE_PATCH_RADIUS = 2  # radius=2 means a 5x5 patch
PHYSICAL_RADIUS_MIN_FACTOR = 0.70
PHYSICAL_RADIUS_MAX_FACTOR = 1.30
PROJECTION_HOUGH_PARAM2_RANGE = range(36, 51)
PROJECTION_CLUSTER_CENTER_TOLERANCE_PX = 18.0
PROJECTION_CLUSTER_RADIUS_TOLERANCE_PX = 14.0
PROJECTION_OVERLAP_FACTOR = 0.72
TEMPORAL_DETECTION_SAMPLE_STRIDE = 60
TEMPORAL_DETECTION_PARAM2_VALUES = (32, 28, 24, 20)
TEMPORAL_ENHANCED_PARAM2_VALUES = (22, 18, 14, 10)
TEMPORAL_CLUSTER_CENTER_TOLERANCE_PX = 36.0
TEMPORAL_CLUSTER_RADIUS_TOLERANCE_PX = 18.0
TEMPORAL_DETECTION_MIN_SUPPORT_FRACTION = 0.07
TEMPORAL_DETECTION_MIN_SUPPORT_FRAMES = 8
TEMPORAL_DETECTION_MAX_SUPPORT_THRESHOLD = 40
TEMPORAL_RADIUS_MIN_FACTOR = 0.70
TEMPORAL_RADIUS_MAX_FACTOR = 1.30
TEMPORAL_OVERLAP_FACTOR = 0.80
TEMPORAL_LAYOUT_MIN_SUPPORT_FRACTION = 0.04
TEMPORAL_LAYOUT_RADIUS_MIN_FACTOR = 0.78
TEMPORAL_LAYOUT_RADIUS_MAX_FACTOR = 1.22
TEMPORAL_LAYOUT_MIN_SEPARATION_FACTOR = 1.85
TEMPORAL_LAYOUT_EVIDENCE_SAMPLE_STRIDE = 600
TEMPORAL_LAYOUT_EDGE_THRESHOLD = 8.0
GEOMETRY_REFINEMENT_SAMPLE_STRIDE = 30
GEOMETRY_REFINEMENT_BRIGHT_PERCENTILE = 70
GEOMETRY_REFINEMENT_MIN_DETECTIONS = 5


def resolve_existing_directory(directory_arg=None):
    if directory_arg:
        candidate = directory_arg
        if not os.path.isabs(candidate):
            candidate = os.path.join(SCRIPT_DIR, candidate)
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(f"Video directory not found: {candidate}")

    for dirname in DEFAULT_VIDEO_DIR_CANDIDATES:
        candidate = os.path.join(SCRIPT_DIR, dirname)
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"No video directory found. Looked for: {', '.join(DEFAULT_VIDEO_DIR_CANDIDATES)}"
    )


def resolve_output_directory(path_arg):
    candidate = path_arg
    if not os.path.isabs(candidate):
        candidate = os.path.join(SCRIPT_DIR, candidate)
    os.makedirs(candidate, exist_ok=True)
    return candidate


def resolve_video_path(video_dir, video_name):
    if video_name is None:
        return None

    candidates = []
    raw_name = video_name.strip()
    if os.path.isabs(raw_name) and os.path.isfile(raw_name):
        return raw_name

    if raw_name.lower().endswith(".mp4"):
        candidates.append(raw_name)
    else:
        candidates.extend(
            [
                raw_name,
                f"{raw_name}.mp4",
                f"{raw_name} LED.mp4",
                f"{raw_name} Fib.mp4",
            ]
        )

    for candidate in candidates:
        full_path = os.path.join(video_dir, candidate)
        if os.path.isfile(full_path):
            return full_path

    stem_to_path = {}
    for root, _, files in os.walk(video_dir):
        for filename in files:
            if filename.lower().endswith(".mp4"):
                stem_to_path[os.path.splitext(filename)[0].lower()] = os.path.join(root, filename)
    match = stem_to_path.get(raw_name.lower())
    if match:
        return match

    raise FileNotFoundError(
        f"Could not resolve '{video_name}' in {video_dir}. "
        "Pass the full filename including extension if needed."
    )


def list_mp4_files(video_dir):
    intensity_tokens = ("speckle", "led", "fib")
    excluded_tokens = ("side", "top", "combined", "segmented")
    video_paths = []
    for root, _, files in os.walk(video_dir):
        for filename in files:
            lower_name = filename.lower()
            if not lower_name.endswith(".mp4"):
                continue
            if any(token in lower_name for token in excluded_tokens):
                continue
            if not any(token in lower_name for token in intensity_tokens):
                continue
            video_paths.append(os.path.join(root, filename))
    return sorted(video_paths)


def sort_circles_stably(circles):
    return np.array(sorted(circles, key=lambda c: (int(c[1]), int(c[0]))), dtype=np.float32)


def estimate_circle_radius_bounds(gray_frame):
    height, width = gray_frame.shape
    scale = min(height, width)
    min_radius = max(18, int(round(scale * 0.045)))
    max_radius = max(min_radius + 8, int(round(scale * 0.11)))
    return min_radius, max_radius


def circles_overlap(circle_a, circle_b, overlap_factor=PROJECTION_OVERLAP_FACTOR):
    ax, ay, ar = [float(v) for v in circle_a]
    bx, by, br = [float(v) for v in circle_b]
    center_distance = np.hypot(ax - bx, ay - by)
    return center_distance < overlap_factor * min(ar, br)


def cluster_circle_candidates(detections):
    clusters = []
    for det in detections:
        _, x, y, r = det
        matched = False
        for cluster in clusters:
            cx, cy, cr = cluster["mean"]
            if (
                np.hypot(x - cx, y - cy) <= PROJECTION_CLUSTER_CENTER_TOLERANCE_PX
                and abs(r - cr) <= PROJECTION_CLUSTER_RADIUS_TOLERANCE_PX
            ):
                cluster["items"].append(det)
                arr = np.array([[item[1], item[2], item[3]] for item in cluster["items"]], dtype=np.float32)
                cluster["mean"] = arr.mean(axis=0)
                cluster["support"] = len({item[0] for item in cluster["items"]})
                matched = True
                break
        if not matched:
            clusters.append(
                {
                    "items": [det],
                    "mean": np.array([x, y, r], dtype=np.float32),
                    "support": 1,
                }
            )
    return clusters


def select_stable_projection_circles(clusters):
    if not clusters:
        return None

    clusters_sorted = sorted(
        clusters,
        key=lambda cluster: (-cluster["support"], abs(float(cluster["mean"][2]) - 80.0)),
    )

    selected = []
    for cluster in clusters_sorted:
        circle = cluster["mean"]
        if all(not circles_overlap(circle, other) for other in selected):
            selected.append(circle)

    if not selected:
        return None

    return sort_circles_stably(np.array(selected, dtype=np.float32))


def detect_circles_from_projection(gray_projection):
    min_radius, max_radius = estimate_circle_radius_bounds(gray_projection)
    blurred = cv2.GaussianBlur(gray_projection, (9, 9), 2)
    detections = []

    for param2 in PROJECTION_HOUGH_PARAM2_RANGE:
        circles_detected = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(35, int(round(min_radius * 0.55))),
            param1=45,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles_detected is None:
            continue
        for x, y, r in circles_detected[0]:
            detections.append((param2, float(x), float(y), float(r)))

    if not detections:
        return None

    clusters = cluster_circle_candidates(detections)
    return select_stable_projection_circles(clusters)


def cluster_temporal_circle_candidates(detections):
    clusters = []
    for det in detections:
        frame_idx, x, y, r, _ = det
        matched = False
        for cluster in clusters:
            cx, cy, cr = cluster["mean"]
            if (
                np.hypot(x - cx, y - cy) <= TEMPORAL_CLUSTER_CENTER_TOLERANCE_PX
                and abs(r - cr) <= TEMPORAL_CLUSTER_RADIUS_TOLERANCE_PX
            ):
                cluster["items"].append(det)
                arr = np.array([[item[1], item[2], item[3]] for item in cluster["items"]], dtype=np.float32)
                cluster["mean"] = arr.mean(axis=0)
                cluster["support"] = len({item[0] for item in cluster["items"]})
                matched = True
                break
        if not matched:
            clusters.append(
                {
                    "items": [det],
                    "mean": np.array([x, y, r], dtype=np.float32),
                    "support": 1,
                }
            )
    return clusters


def select_temporal_circles(clusters, sampled_frame_count):
    if not clusters or sampled_frame_count <= 0:
        return None

    min_support = int(np.ceil(sampled_frame_count * TEMPORAL_DETECTION_MIN_SUPPORT_FRACTION))
    min_support = max(TEMPORAL_DETECTION_MIN_SUPPORT_FRAMES, min_support)
    min_support = min(TEMPORAL_DETECTION_MAX_SUPPORT_THRESHOLD, min_support)

    candidates = [cluster for cluster in clusters if cluster["support"] >= min_support]
    if not candidates:
        return None

    candidates_sorted = sorted(
        candidates,
        key=lambda cluster: (
            -cluster["support"],
            -float(np.median([item[4] for item in cluster["items"]])),
        ),
    )
    radius_reference = float(
        np.median([cluster["mean"][2] for cluster in candidates_sorted[: min(12, len(candidates_sorted))]])
    )
    min_radius = radius_reference * TEMPORAL_RADIUS_MIN_FACTOR
    max_radius = radius_reference * TEMPORAL_RADIUS_MAX_FACTOR
    candidates_sorted = [
        cluster
        for cluster in candidates_sorted
        if min_radius <= float(cluster["mean"][2]) <= max_radius
    ]

    selected = []
    for cluster in candidates_sorted:
        circle = cluster["mean"]
        if all(not circles_overlap(circle, other, overlap_factor=TEMPORAL_OVERLAP_FACTOR) for other in selected):
            selected.append(circle)

    if not selected:
        return None

    return sort_circles_stably(np.array(selected, dtype=np.float32))


def circle_evidence_score(gray, x, y, r):
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    inner_mask = distances <= r * 0.75
    annulus_mask = (distances >= r * 1.05) & (distances <= r * 1.35)
    if not inner_mask.any() or not annulus_mask.any():
        return 0.0

    inner_minus_annulus = float(gray[inner_mask].mean() - gray[annulus_mask].mean())
    edge_diffs = []
    for angle in np.linspace(0, 2 * np.pi, 64, endpoint=False):
        inner_x = int(round(x + np.cos(angle) * r * 0.82))
        inner_y = int(round(y + np.sin(angle) * r * 0.82))
        outer_x = int(round(x + np.cos(angle) * r * 1.12))
        outer_y = int(round(y + np.sin(angle) * r * 1.12))
        if 0 <= inner_x < width and 0 <= outer_x < width and 0 <= inner_y < height and 0 <= outer_y < height:
            edge_diffs.append(float(gray[inner_y, inner_x]) - float(gray[outer_y, outer_x]))

    if not edge_diffs:
        return inner_minus_annulus

    edge_diffs = np.array(edge_diffs, dtype=np.float32)
    edge_coverage = float(np.mean(edge_diffs > TEMPORAL_LAYOUT_EDGE_THRESHOLD))
    return inner_minus_annulus + 40.0 * edge_coverage


def score_layout_candidates_from_video(video_path, candidates, reference_radius):
    if not candidates:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = list(range(0, frame_total, TEMPORAL_LAYOUT_EVIDENCE_SAMPLE_STRIDE))
    if frame_total > 0:
        sample_frames.append(frame_total // 2)

    evidence_by_candidate = [[] for _ in candidates]
    for frame_idx in sorted(set(sample_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for candidate_idx, candidate in enumerate(candidates):
            evidence_by_candidate[candidate_idx].append(
                circle_evidence_score(gray, candidate["x"], candidate["y"], reference_radius)
            )

    cap.release()

    for candidate, evidence_values in zip(candidates, evidence_by_candidate):
        if evidence_values:
            evidence = np.array(evidence_values, dtype=np.float32)
            candidate["evidence_p90"] = float(np.percentile(evidence, 90))
            candidate["evidence_median"] = float(np.median(evidence))
        else:
            candidate["evidence_p90"] = 0.0
            candidate["evidence_median"] = 0.0


def select_layout_temporal_circles(clusters, sampled_frame_count, video_path, target_count):
    if not clusters or sampled_frame_count <= 0 or target_count is None or target_count <= 0:
        return None

    high_support_radii = [
        float(cluster["mean"][2])
        for cluster in clusters
        if cluster["support"] >= sampled_frame_count * 0.55
    ]
    if not high_support_radii:
        return None

    reference_radius = float(np.median(high_support_radii))
    if reference_radius <= 0:
        return None

    min_support = max(
        TEMPORAL_DETECTION_MIN_SUPPORT_FRAMES,
        int(np.ceil(sampled_frame_count * TEMPORAL_LAYOUT_MIN_SUPPORT_FRACTION)),
    )
    min_radius = reference_radius * TEMPORAL_LAYOUT_RADIUS_MIN_FACTOR
    max_radius = reference_radius * TEMPORAL_LAYOUT_RADIUS_MAX_FACTOR

    candidates = []
    for cluster_idx, cluster in enumerate(clusters):
        x, y, r = [float(v) for v in cluster["mean"]]
        if cluster["support"] < min_support:
            continue
        if not (min_radius <= r <= max_radius):
            continue
        candidates.append(
            {
                "cluster_idx": cluster_idx,
                "support": int(cluster["support"]),
                "x": x,
                "y": y,
                "r": r,
            }
        )

    if len(candidates) < target_count:
        return None

    score_layout_candidates_from_video(video_path, candidates, reference_radius)
    for candidate in candidates:
        radius_penalty = abs(candidate["r"] - reference_radius) / reference_radius
        candidate["score"] = (
            np.log1p(candidate["support"])
            + 0.08 * candidate["evidence_p90"]
            + 0.03 * candidate["evidence_median"]
            - 0.6 * radius_penalty
        )

    selected = []
    separation_factors = (
        TEMPORAL_LAYOUT_MIN_SEPARATION_FACTOR,
        1.70,
        1.55,
        1.40,
    )
    for separation_factor in separation_factors:
        selected = []
        min_separation = reference_radius * separation_factor
        for candidate in sorted(candidates, key=lambda item: -item["score"]):
            if all(
                np.hypot(candidate["x"] - other["x"], candidate["y"] - other["y"]) >= min_separation
                for other in selected
            ):
                selected.append(candidate)
                if len(selected) == target_count:
                    break
        if len(selected) == target_count:
            break

    if len(selected) != target_count:
        return None

    circles = np.array(
        [[candidate["x"], candidate["y"], reference_radius] for candidate in selected],
        dtype=np.float32,
    )
    return sort_circles_stably(circles)


def detect_circles_from_temporal_samples(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for temporal circle detection: {video_path}")

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gray_detections = []
    enhanced_detections = []
    sampled_frame_count = 0
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    for frame_idx in range(0, frame_total, TEMPORAL_DETECTION_SAMPLE_STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        sampled_frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_radius, max_radius = estimate_circle_radius_bounds(gray)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        for param2 in TEMPORAL_DETECTION_PARAM2_VALUES:
            circles_detected = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(45, int(round(min_radius * 0.75))),
                param1=45,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles_detected is None:
                continue
            for x, y, r in circles_detected[0]:
                detection = (frame_idx, float(x), float(y), float(r), param2)
                gray_detections.append(detection)
                enhanced_detections.append(detection)

        enhanced = clahe.apply(gray)
        enhanced_blurred = cv2.GaussianBlur(enhanced, (9, 9), 2)
        for param2 in TEMPORAL_ENHANCED_PARAM2_VALUES:
            circles_detected = cv2.HoughCircles(
                enhanced_blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(60, int(round(min_radius * 1.60))),
                param1=40,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles_detected is None:
                continue
            for x, y, r in circles_detected[0]:
                enhanced_detections.append((frame_idx, float(x), float(y), float(r), param2))

    cap.release()
    if not gray_detections and not enhanced_detections:
        return None, sampled_frame_count

    gray_clusters = cluster_temporal_circle_candidates(gray_detections)
    baseline_circles = select_temporal_circles(gray_clusters, sampled_frame_count)
    target_count = len(baseline_circles) if baseline_circles is not None else None

    enhanced_clusters = cluster_temporal_circle_candidates(enhanced_detections)
    layout_circles = select_layout_temporal_circles(
        enhanced_clusters,
        sampled_frame_count,
        video_path,
        target_count,
    )
    if layout_circles is not None:
        return layout_circles, sampled_frame_count

    return baseline_circles, sampled_frame_count


def build_max_projection(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for projection: {video_path}")

    max_projection = None
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if max_projection is None:
            max_projection = gray.copy()
        else:
            max_projection = np.maximum(max_projection, gray)
        frame_count += 1

    cap.release()
    if max_projection is None:
        raise RuntimeError("Projection pass failed: no frames could be read.")
    return max_projection, frame_count


def find_local_circle_candidate(gray, circle, search_scale=1.45, search_padding=55):
    frame_height, frame_width = gray.shape
    x, y, r = [float(v) for v in circle]
    search_radius = int(round(max(r * search_scale + search_padding, r + 70)))

    x0 = max(0, int(round(x - search_radius)))
    x1 = min(frame_width, int(round(x + search_radius)))
    y0 = max(0, int(round(y - search_radius)))
    y1 = min(frame_height, int(round(y + search_radius)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    roi = gray[y0:y1, x0:x1]
    roi_blurred = cv2.GaussianBlur(roi, (9, 9), 2)
    min_radius = max(40, int(round(r * 0.50)))
    max_radius = max(min_radius + 5, int(round(r * 1.08)))

    circles_detected = cv2.HoughCircles(
        roi_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(round(r * 0.65))),
        param1=45,
        param2=22,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles_detected is None:
        return None

    best = None
    max_center_distance = max(70.0, r * 0.85)
    for cand_x, cand_y, cand_r in circles_detected[0]:
        full_x = float(cand_x + x0)
        full_y = float(cand_y + y0)
        center_dist = float(np.hypot(full_x - x, full_y - y))
        if center_dist > max_center_distance:
            continue
        score = center_dist + 0.20 * abs(float(cand_r) - r)
        if best is None or score < best[0]:
            best = (score, full_x, full_y, float(cand_r))

    if best is None:
        return None

    _, best_x, best_y, best_r = best
    return np.array([best_x, best_y, best_r], dtype=np.float32)


def local_center_brightness(gray, circle):
    frame_height, frame_width = gray.shape
    x, y, r = [float(v) for v in circle]
    inner_radius = max(8, int(round(r * 0.25)))
    ix = int(round(x))
    iy = int(round(y))
    x0 = max(0, ix - inner_radius)
    x1 = min(frame_width, ix + inner_radius + 1)
    y0 = max(0, iy - inner_radius)
    y1 = min(frame_height, iy + inner_radius + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(gray[y0:y1, x0:x1].mean())


def refine_fixed_geometry_from_video(video_path, circles):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for geometry refinement: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates_by_circle = [[] for _ in circles]

    for frame_idx in range(0, frame_count, GEOMETRY_REFINEMENT_SAMPLE_STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for circle_idx, circle in enumerate(circles):
            candidate = find_local_circle_candidate(gray, circle)
            if candidate is None:
                continue
            brightness = local_center_brightness(gray, candidate)
            candidates_by_circle[circle_idx].append((brightness, candidate))

    cap.release()

    refined = circles.copy()
    for circle_idx, candidates in enumerate(candidates_by_circle):
        if len(candidates) < GEOMETRY_REFINEMENT_MIN_DETECTIONS:
            continue

        brightness_values = np.array([item[0] for item in candidates], dtype=np.float32)
        threshold = max(70.0, float(np.percentile(brightness_values, GEOMETRY_REFINEMENT_BRIGHT_PERCENTILE)))
        bright_candidates = [candidate for brightness, candidate in candidates if brightness >= threshold]
        if len(bright_candidates) < GEOMETRY_REFINEMENT_MIN_DETECTIONS:
            bright_candidates = [candidate for _, candidate in candidates]

        arr = np.array(bright_candidates, dtype=np.float32)
        refined[circle_idx, 2] = float(np.median(arr[:, 2]))

    return refined


def stabilize_physical_radii(circles):
    circles = np.array(circles, dtype=np.float32).copy()
    if len(circles) < 3:
        return circles

    radii = circles[:, 2].astype(np.float32)
    median_radius = float(np.median(radii))
    if median_radius <= 0:
        return circles

    reference_radii = radii[radii >= median_radius]
    if len(reference_radii) == 0:
        return circles

    reference_radius = float(np.median(reference_radii))
    min_radius = reference_radius * PHYSICAL_RADIUS_MIN_FACTOR
    max_radius = reference_radius * PHYSICAL_RADIUS_MAX_FACTOR
    outlier_mask = (radii < min_radius) | (radii > max_radius)
    circles[outlier_mask, 2] = reference_radius

    return circles


def get_sampling_points(x, y, r):
    points = [(float(x), float(y))]
    ring_radius = 0.6 * r
    for angle in (0, 45, 90, 135, 180, 225, 270, 315):
        radians = np.radians(angle)
        px = float(x + ring_radius * np.cos(radians))
        py = float(y + ring_radius * np.sin(radians))
        points.append((px, py))
    return points


def sample_patch_mean(gray, px, py, patch_radius=SAMPLE_PATCH_RADIUS):
    height, width = gray.shape
    ix = int(round(px))
    iy = int(round(py))
    x0 = max(0, ix - patch_radius)
    x1 = min(width, ix + patch_radius + 1)
    y0 = max(0, iy - patch_radius)
    y1 = min(height, iy + patch_radius + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    return float(patch.mean())


def get_n_point_intensities(gray, x, y, r):
    return [sample_patch_mean(gray, px, py) for px, py in get_sampling_points(x, y, r)]


def normalize_dataframe(df):
    df_normalized = df.copy()
    intensity_cols = [col for col in df.columns if "intensity" in col.lower()]
    global_min = df_normalized[intensity_cols].min().min()
    global_max = df_normalized[intensity_cols].max().max()
    print(f"   [Normalization] Global Min: {global_min:.4f} | Global Max: {global_max:.4f}")

    if global_max > global_min:
        df_normalized[intensity_cols] = (
            (df_normalized[intensity_cols] - global_min) / (global_max - global_min)
        )
    else:
        df_normalized[intensity_cols] = 0.0

    return df_normalized


def find_tracking_circle_candidate(gray, predicted_circle):
    frame_height, frame_width = gray.shape
    x, y, r = [float(v) for v in predicted_circle]
    radius_tol = max(6.0, r * TRACKING_RADIUS_TOLERANCE_PCT)
    search_radius = int(round(r + radius_tol + TRACKING_SEARCH_PADDING_PX))

    x0 = max(0, int(round(x - search_radius)))
    x1 = min(frame_width, int(round(x + search_radius)))
    y0 = max(0, int(round(y - search_radius)))
    y1 = min(frame_height, int(round(y + search_radius)))

    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    roi = gray[y0:y1, x0:x1]
    roi_blurred = cv2.GaussianBlur(roi, (9, 9), 2)
    min_radius = max(1, int(round(r - radius_tol)))
    max_radius = int(round(r + radius_tol))

    circles = cv2.HoughCircles(
        roi_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, int(round(r * 0.8))),
        param1=45,
        param2=25,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    candidates = []
    for cand_x, cand_y, cand_r in circles[0]:
        full_x = float(cand_x + x0)
        full_y = float(cand_y + y0)
        center_dist = np.hypot(full_x - x, full_y - y)
        radius_delta = abs(float(cand_r) - r)
        score = center_dist + 1.5 * radius_delta
        if center_dist <= TRACKING_MAX_CENTER_JUMP_PX:
            candidates.append((score, full_x, full_y, float(cand_r)))

    if not candidates:
        return None

    _, best_x, best_y, _ = min(candidates, key=lambda item: item[0])
    return np.array([best_x, best_y, r], dtype=np.float32)


def refine_circle(gray, predicted_circle):
    candidate = find_tracking_circle_candidate(gray, predicted_circle)
    if candidate is None:
        return np.array(predicted_circle, dtype=np.float32)
    x, y, r = [float(v) for v in predicted_circle]
    return np.array([candidate[0], candidate[1], r], dtype=np.float32)


def create_circle_mask(shape, x, y, r):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, (int(round(x)), int(round(y))), int(round(r)), 255, -1)
    return mask


def estimate_global_center_correction(gray, predicted_circles, max_candidate_shift):
    corrections = []
    for circle in predicted_circles:
        candidate = find_tracking_circle_candidate(gray, circle)
        if candidate is None:
            continue
        correction = candidate[:2] - circle[:2]
        if float(np.linalg.norm(correction)) <= max_candidate_shift:
            corrections.append(correction)

    if not corrections:
        return np.zeros(2, dtype=np.float32)

    corrections = np.array(corrections, dtype=np.float32)
    median_correction = np.median(corrections, axis=0)
    residuals = np.linalg.norm(corrections - median_correction, axis=1)
    inliers = corrections[residuals <= GLOBAL_TRACKING_INLIER_TOLERANCE_PX]
    if len(inliers) > 0:
        median_correction = np.median(inliers, axis=0)

    correction_norm = float(np.linalg.norm(median_correction))
    if correction_norm > max_candidate_shift:
        median_correction *= max_candidate_shift / correction_norm

    return median_correction.astype(np.float32)


def apply_global_center_correction(gray, circles, max_correction_px, alpha):
    circles = np.array(circles, dtype=np.float32)
    correction = estimate_global_center_correction(
        gray,
        circles,
        max_candidate_shift=max_correction_px,
    )
    corrected = circles.copy()
    corrected[:, :2] += alpha * correction
    return corrected


def track_circles(curr_gray, prev_circles):
    return apply_global_center_correction(
        curr_gray,
        prev_circles,
        max_correction_px=MAX_CENTER_CORRECTION_PX,
        alpha=CENTER_SMOOTHING_ALPHA,
    )


def build_row_data(frame_id, gray, circles):
    row_data = {"Frame_ID": frame_id}
    tracked_positions = []

    for idx, (x, y, r) in enumerate(circles, start=1):
        circle_mask = create_circle_mask(gray.shape, x, y, r)
        mean_val = float(cv2.mean(gray, mask=circle_mask)[0])
        row_data[f"intensity_circle_{idx}"] = mean_val
        row_data[f"circle_{idx}_center_x"] = float(x)
        row_data[f"circle_{idx}_center_y"] = float(y)
        row_data[f"circle_{idx}_radius"] = float(r)

        nine_points = get_n_point_intensities(gray, x, y, r)
        for pt_idx, val in enumerate(nine_points, start=1):
            row_data[f"intensity_circle_{idx}_9pt_{pt_idx}"] = float(val)

        tracked_positions.append((float(x), float(y), float(r)))

    return row_data, tracked_positions


def annotate_frame(frame, circles, row_data):
    annotated = frame.copy()

    for idx, (x, y, r) in enumerate(circles, start=1):
        mean_val = row_data[f"intensity_circle_{idx}"]
        center = (int(round(x)), int(round(y)))
        cv2.circle(annotated, center, int(round(r)), (0, 255, 0), 3)
        cv2.putText(
            annotated,
            f"C{idx} Avg:{mean_val:.1f}",
            (center[0] - 55, max(20, center[1] - int(round(r)) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        for px, py in get_sampling_points(x, y, r):
            cv2.circle(annotated, (int(round(px)), int(round(py))), 3, (0, 0, 255), -1)
            cv2.rectangle(
                annotated,
                (int(round(px)) - 1, int(round(py)) - 1),
                (int(round(px)) + 1, int(round(py)) + 1),
                (255, 255, 255),
                1,
            )
    return annotated


def save_projection_audit(gray_projection, circles, audit_output_dir, video_stem):
    if gray_projection is None or circles is None:
        return None
    audit_image = cv2.cvtColor(gray_projection, cv2.COLOR_GRAY2BGR)
    for idx, (x, y, r) in enumerate(circles, start=1):
        center = (int(round(x)), int(round(y)))
        cv2.circle(audit_image, center, int(round(r)), (0, 255, 0), 3)
        cv2.putText(
            audit_image,
            f"C{idx}",
            (center[0] - 16, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    output_path = os.path.join(audit_output_dir, f"{video_stem}_projection_detection.png")
    cv2.imwrite(output_path, audit_image)
    return output_path


def open_writer_if_requested(audit_video_path, fps, frame_shape, save_audit_video):
    if not save_audit_video:
        return None
    frame_height, frame_width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(audit_video_path, fourcc, fps or 30.0, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create audit video: {audit_video_path}")
    return writer


def process_intensity_video(
    video_path,
    raw_output_dir,
    normalized_output_dir,
    audit_output_dir,
    save_audit_video=True,
    preview=False,
):
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    output_csv_raw = os.path.join(raw_output_dir, f"{video_stem}_intensities.csv")
    output_csv_normalized = os.path.join(normalized_output_dir, f"{video_stem}_intensities_normalized.csv")
    audit_video_path = os.path.join(audit_output_dir, f"{video_stem}_tracking_audit.mp4")

    print(f"Target Video: {video_path}")
    print("Detecting stable circles from temporal samples...")
    circles, sampled_frame_count = detect_circles_from_temporal_samples(video_path)
    if circles is not None and len(circles) > 0:
        print(f"Detected {len(circles)} stable circles from {sampled_frame_count} sampled frames.")
    else:
        print("Temporal sample detection found no stable circles; falling back to max projection.")

    print("Building full-video max projection for audit/fallback...")
    gray_projection, projection_frame_count = build_max_projection(video_path)
    if circles is None or len(circles) == 0:
        circles = detect_circles_from_projection(gray_projection)
    if circles is None or len(circles) == 0:
        raise RuntimeError("No circles detected from full-video projection.")
    print(f"Using {len(circles)} circles with {projection_frame_count} projection frames for audit.")

    print("Refining fixed circle sizes from bright sampled frames...")
    original_radii = circles[:, 2].copy()
    circles = refine_fixed_geometry_from_video(video_path, circles)
    max_radius_change = float(np.max(np.abs(circles[:, 2] - original_radii)))
    print(f"Refined fixed radii. Largest radius adjustment: {max_radius_change:.1f} px.")
    refined_radii = circles[:, 2].copy()
    circles = stabilize_physical_radii(circles)
    radius_outlier_change = float(np.max(np.abs(circles[:, 2] - refined_radii)))
    if radius_outlier_change > 0:
        print(f"Stabilized radius outliers. Largest correction: {radius_outlier_change:.1f} px.")

    projection_audit_path = save_projection_audit(gray_projection, circles, audit_output_dir, video_stem)
    if projection_audit_path:
        print(f"Saved projection detection audit to {projection_audit_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame.")

    gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    audit_writer = open_writer_if_requested(
        audit_video_path=audit_video_path,
        fps=fps,
        frame_shape=first_frame.shape,
        save_audit_video=save_audit_video,
    )

    dataset = []
    prev_circles = apply_global_center_correction(
        gray_first,
        circles,
        max_correction_px=TRACKING_MAX_CENTER_JUMP_PX,
        alpha=1.0,
    )
    frame_count = 0

    while True:
        if frame_count == 0:
            current_frame = first_frame
            current_gray = gray_first
            current_circles = prev_circles
            global_shift = (0.0, 0.0)
        else:
            ret, current_frame = cap.read()
            if not ret:
                break
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            current_circles = track_circles(current_gray, prev_circles)
            global_shift = (0.0, 0.0)
            prev_circles = current_circles

        if frame_count == 0:
            prev_circles = current_circles

        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}...")

        row_data, tracked_positions = build_row_data(frame_count, current_gray, current_circles)
        dataset.append(row_data)

        if audit_writer is not None or preview:
            annotated = annotate_frame(current_frame, tracked_positions, row_data)
            if audit_writer is not None:
                audit_writer.write(annotated)
            if preview:
                cv2.imshow("Fiber Intensity Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopping preview early at user request.")
                    break

        frame_count += 1

    cap.release()
    if audit_writer is not None:
        audit_writer.release()
        print(f"Saved tracking audit video to {audit_video_path}")
    if preview:
        cv2.destroyAllWindows()

    if not dataset:
        raise RuntimeError("No data was extracted.")

    df = pd.DataFrame(dataset)
    df.to_csv(output_csv_raw, index=False)
    print(f"Raw intensities exported to {output_csv_raw}")

    df_normalized = normalize_dataframe(df)
    df_normalized.to_csv(output_csv_normalized, index=False)
    print(f"Normalized intensities exported to {output_csv_normalized}")

    return {
        "video_path": video_path,
        "raw_csv": output_csv_raw,
        "normalized_csv": output_csv_normalized,
        "audit_video": audit_video_path if save_audit_video else None,
        "projection_audit": projection_audit_path,
        "frames_processed": len(dataset),
        "circle_count": len(circles),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Extract fiber intensity data with circle tracking.")
    parser.add_argument(
        "filename",
        nargs="?",
        default=None,
        help="Video name, stem, or absolute path. If omitted, all .mp4 files in the selected video directory are processed.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Relative or absolute directory containing input .mp4 files. Defaults to videos-2026 when present, otherwise videos.",
    )
    parser.add_argument(
        "--raw-output-dir",
        default=DEFAULT_RAW_OUTPUT_DIR,
        help="Directory for raw CSV output.",
    )
    parser.add_argument(
        "--normalized-output-dir",
        default=DEFAULT_NORMALIZED_OUTPUT_DIR,
        help="Directory for normalized CSV output.",
    )
    parser.add_argument(
        "--audit-output-dir",
        default=DEFAULT_AUDIT_OUTPUT_DIR,
        help="Directory for tracking audit videos.",
    )
    parser.add_argument(
        "--no-audit-video",
        action="store_true",
        help="Disable writing the tracking audit video.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live OpenCV preview while processing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        video_dir = resolve_existing_directory(args.video_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    raw_output_dir = resolve_output_directory(args.raw_output_dir)
    normalized_output_dir = resolve_output_directory(args.normalized_output_dir)
    audit_output_dir = resolve_output_directory(args.audit_output_dir)

    if args.filename is None:
        video_paths = list_mp4_files(video_dir)
        if not video_paths:
            print(f"No .mp4 files found in {video_dir}")
            sys.exit(1)
        print(f"Found {len(video_paths)} .mp4 files in {video_dir}")
    else:
        try:
            video_paths = [resolve_video_path(video_dir, args.filename)]
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    for video_path in video_paths:
        print("=" * 72)
        print(f"Processing: {os.path.basename(video_path)}")
        print("=" * 72)
        result = process_intensity_video(
            video_path=video_path,
            raw_output_dir=raw_output_dir,
            normalized_output_dir=normalized_output_dir,
            audit_output_dir=audit_output_dir,
            save_audit_video=not args.no_audit_video,
            preview=args.preview,
        )
        print(
            f"Completed {os.path.basename(result['video_path'])}: "
            f"{result['frames_processed']} frames, {result['circle_count']} circles tracked."
        )


if __name__ == "__main__":
    main()
