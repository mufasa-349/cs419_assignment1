#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS419 - Digital Image and Video Analysis
Assignment Solution (Outline-Only: Morphology + Canny)

This script strictly stays within the provided topic outline:
  - Fundamental point processing & intensity transforms (Otsu threshold)
  - Binary Image Processing & Mathematical Morphology (opening, closing, erosion, dilation, region filling)
  - Connectivity, connected components, 8-neighborhood degree
  - Skeletonization via morphological skeleton (no Zhang–Suen)
  - Edge detection via Canny (non-max suppression, hysteresis) from Linear Image Processing

Pipeline (configurable):
  1) Read grayscale -> Otsu binary (ensure roads are foreground)
  2) Morphological cleaning (OPEN then CLOSE)
  3) Skeletonization (morphological skeleton), and/or Canny edges (optional)
  4) Endpoints/junctions via 8-neighborhood degree on the chosen "thin" map
  5) Roundabout candidates = (loops on skeleton) ∩ (high-circularity blobs on thick mask)

Usage example:
    python cs419_assignment_solution_v2.py \
        --input "kagglehub.dataset_download(\"balraj98/massachusetts-roads-dataset/tiff/train/10078660_15.tif\")" \
        --edge_mode canny \
        --skeleton_method morph \
        --debug_dir debug_v2 \
        --output_txt results_v2.txt

Author: ChatGPT (CS419-constrained)
"""

import argparse
import os
import sys
import json
import math
from typing import List, Tuple

import numpy as np
import cv2


# -----------------------------
# Structuring elements & helpers
# -----------------------------

def se_disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = (x*x + y*y) <= radius*radius
    return (mask.astype(np.uint8) * 255)


def se_cross() -> np.ndarray:
    return (np.array([[0,1,0],
                      [1,1,1],
                      [0,1,0]], dtype=np.uint8) * 255)


# -----------------------------
# Fundamental ops & binarization
# -----------------------------

def to_binary(img_gray: np.ndarray, invert_if_needed: bool = True) -> np.ndarray:
    """0/255 uint8 via Otsu; invert if foreground seems dark."""
    if img_gray.dtype != np.uint8:
        img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert_if_needed:
        mean_fg = img_gray[bw == 255].mean() if np.any(bw == 255) else 255.0
        mean_bg = img_gray[bw == 0].mean() if np.any(bw == 0) else 0.0
        if mean_fg < mean_bg:
            bw = cv2.bitwise_not(bw)
    return bw


def preprocess_binary(bw: np.uint8, open_rad: int = 1, close_rad: int = 2) -> np.uint8:
    """OPEN then CLOSE to remove speckles and bridge tiny gaps."""
    if open_rad > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, se_disk(open_rad))
    if close_rad > 0:
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se_disk(close_rad))
    return bw


# ---------------------------------
# Morphological skeleton (outline 2)
# ---------------------------------

def morphological_skeleton(bw: np.uint8, se: np.ndarray = None) -> np.uint8:
    """
    Skeleton S = ⋃_k (X ⊖ kB) - ((X ⊖ kB) ∘ B)
    Uses iterative erosion + opening differences.
    """
    if se is None:
        se = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    X = (bw > 0).astype(np.uint8) * 255
    S = np.zeros_like(X)
    while True:
        eroded = cv2.erode(X, se)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, se)
        temp = cv2.subtract(eroded, opened)
        S = cv2.bitwise_or(S, temp)
        if not np.any(eroded):
            break
        X = eroded
    return S


# -------------
# Canny (outline 3)
# -------------

def canny_edges(bw_clean: np.uint8, t1: int = 50, t2: int = 150) -> np.uint8:
    """Canny edge detector on a binary/clean mask (still valid; Canny has internal Gaussian)."""
    edges = cv2.Canny(bw_clean, threshold1=t1, threshold2=t2)
    return edges


# ---------------------------------
# Neighborhood-degree & junctions
# ---------------------------------

def neighbor_count_8n(binary_thin: np.ndarray) -> np.ndarray:
    kernel = np.ones((3,3), np.uint8)
    total = cv2.filter2D((binary_thin>0).astype(np.uint8), -1, kernel)
    deg = total - (binary_thin>0).astype(np.uint8)
    return deg


def endpoints_and_junctions(binary_thin: np.ndarray, min_degree: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    deg = neighbor_count_8n(binary_thin)
    ep = ((binary_thin>0) & (deg==1)).astype(np.uint8) * 255
    jn = ((binary_thin>0) & (deg>=min_degree)).astype(np.uint8) * 255
    return ep, jn


def compress_points(mask: np.ndarray) -> List[Tuple[int,int]]:
    num, lab = cv2.connectedComponents((mask>0).astype(np.uint8), connectivity=8)
    pts = []
    for cc in range(1, num):
        ys, xs = np.where(lab==cc)
        if ys.size == 0: 
            continue
        cy = int(np.round(ys.mean()))
        cx = int(np.round(xs.mean()))
        pts.append((cx, cy))
    return pts


# ---------------------------------
# Loop (roundabout) detection logic
# ---------------------------------

def find_loops_via_skeleton(skel: np.ndarray, min_pixels: int = 30) -> np.uint8:
    ep_mask, _ = endpoints_and_junctions(skel)
    num, lab = cv2.connectedComponents((skel>0).astype(np.uint8), connectivity=8)
    loop_mask = np.zeros_like(skel, dtype=np.uint8)
    for cc in range(1, num):
        comp = (lab == cc)
        if comp.sum() < min_pixels:
            continue
        has_endpoint = np.any(ep_mask[comp] > 0)
        if not has_endpoint:
            loop_mask[comp] = 255
    return loop_mask


def contour_circularity(mask: np.ndarray):
    contours, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        per = cv2.arcLength(cnt, True)
        circ = 0.0 if (per<=0 or area<=0) else float(4.0 * math.pi * area / (per*per))
        out.append((cnt, circ))
    return out


def merge_close_points(pts: List[Tuple[int,int]], max_dist: int = 8) -> List[Tuple[int,int]]:
    if not pts:
        return []
    pts_arr = np.array(pts, dtype=np.float32)
    used = np.zeros(len(pts), dtype=bool)
    merged = []
    for i in range(len(pts)):
        if used[i]: 
            continue
        group = [i]
        for j in range(i+1, len(pts)):
            if used[j]:
                continue
            if np.linalg.norm(pts_arr[i] - pts_arr[j]) <= max_dist:
                group.append(j)
        used[group] = True
        avg = pts_arr[group].mean(axis=0)
        merged.append((int(round(avg[0])), int(round(avg[1]))))
    return merged


def roundabout_candidates(bw_clean: np.ndarray,
                          skel: np.ndarray,
                          loop_circ_thresh: float = 0.4,
                          min_area: int = 50) -> List[Tuple[int,int]]:
    loop_mask = find_loops_via_skeleton(skel, min_pixels=15)

    # Try multiple approaches for roundabout detection - use dilation to find thicker circles
    thick = cv2.morphologyEx(bw_clean, cv2.MORPH_DILATE, se_disk(4))
    circles = contour_circularity(thick)

    centers = []
    for cnt, circ in circles:
        if circ < loop_circ_thresh:
            continue
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Check if center is on loop OR nearby (more lenient)
        if loop_mask[cy, cx] > 0 or np.any(loop_mask[max(0,cy-8):cy+9, max(0,cx-8):cx+9] > 0):
            centers.append((cx, cy))
    return merge_close_points(centers, max_dist=20)


# ---------------------
# Main pipeline routine
# ---------------------

def detect_intersections_and_roundabouts(
        input_path: str = '/Users/mustafabozyel/Github-Desktop/cs419_assignment1/tiff_data/10078660_15.tif',
        edge_mode: str = "none",      # {"none","canny","both"}
        canny_t1: int = 50,
        canny_t2: int = 150,
        skeleton_method: str = "morph",  # {"morph","none"}
        open_rad: int = 1,
        close_rad: int = 2,
        prune_iter: int = 1,
        save_debug_dir: str = None
    ) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
    """
    Returns: (junction_points, roundabout_points)
    """
    # 1) Read
    if input_path.startswith('kagglehub.dataset_download'):
        inner = input_path.split('kagglehub.dataset_download', 1)[-1]
        inner = inner.strip().lstrip('(').rstrip(')').strip()
        inner = inner.strip('"').strip("'")
        true_path = inner
    else:
        true_path = input_path

    img = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {true_path}")

    # 2) Binary
    bw = to_binary(img, invert_if_needed=True)

    # 3) Morphological cleaning
    bw_clean = preprocess_binary(bw, open_rad=open_rad, close_rad=close_rad)

    # 4) Select thin map(s): skeleton and/or canny
    skel = None
    if skeleton_method == "morph":
        skel = morphological_skeleton(bw_clean, se=cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))

    edge_map = None
    if edge_mode in ("canny", "both"):
        edge_map = canny_edges(bw_clean, t1=canny_t1, t2=canny_t2)

    # Combine for junction detection - only use skeleton for junctions
    if skeleton_method == "morph":
        thin_for_junctions = skel
    else:
        # Fallback: if no skeleton, use cleaned binary for thin representation
        thin_for_junctions = bw_clean

    # Optional pruning of small spurs on the *skeleton* before loop/junction calc
    if skeleton_method == "morph" and prune_iter > 0:
        pruned = skel.copy()
        for _ in range(prune_iter):
            ep_mask, _ = endpoints_and_junctions(pruned)
            pruned[ep_mask > 0] = 0
        skel = pruned

    # 5) Junction points
    _, jn_mask = endpoints_and_junctions(thin_for_junctions)
    junction_points = compress_points(jn_mask)
    # Merge nearby junctions more aggressively
    junction_points = merge_close_points(junction_points, max_dist=25)

    # 6) Roundabouts (need a skeleton; if not computed, build a temporary morph skeleton)
    if skel is None:
        skel = morphological_skeleton(bw_clean, se=cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
    roundabout_points = roundabout_candidates(bw_clean, skel, loop_circ_thresh=0.4, min_area=50)

    # Debug outputs
    if save_debug_dir:
        os.makedirs(save_debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_debug_dir, "01_gray.png"), img)
        cv2.imwrite(os.path.join(save_debug_dir, "02_binary.png"), bw)
        cv2.imwrite(os.path.join(save_debug_dir, "03_clean.png"), bw_clean)
        if skel is not None:
            cv2.imwrite(os.path.join(save_debug_dir, "04_skeleton_morph.png"), skel)
        if edge_map is not None:
            cv2.imwrite(os.path.join(save_debug_dir, "05_canny.png"), edge_map)
        cv2.imwrite(os.path.join(save_debug_dir, "06_thin_for_junctions.png"), thin_for_junctions)
        dbg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for (x,y) in junction_points:
            cv2.drawMarker(dbg, (x,y), (0,0,255), markerType=cv2.MARKER_TILTED_CROSS, thickness=2)
        for (x,y) in roundabout_points:
            cv2.circle(dbg, (x,y), 6, (0,255,0), 2)
        cv2.imwrite(os.path.join(save_debug_dir, "07_overlays.png"), dbg)

    return junction_points, roundabout_points


def main():
    parser = argparse.ArgumentParser(description="CS419 Outline-Only: Morphological Skeleton + Optional Canny")
    parser.add_argument("--input", required=True,
                        help="Path to input image (or the literal kagglehub.dataset_download(\"...\") string).")
    parser.add_argument("--output_txt", default=None, help="Optional path to write results as plain .txt")
    parser.add_argument("--output_json", default=None, help="Optional path to write results as .json")
    parser.add_argument("--debug_dir", default=None, help="Optional folder to save intermediate images & overlays")

    parser.add_argument("--edge_mode", choices=["none","canny","both"], default="none",
                        help="Use Canny edges, skeleton, or both (junctions on the union).")
    parser.add_argument("--canny_t1", type=int, default=50, help="Canny threshold1")
    parser.add_argument("--canny_t2", type=int, default=150, help="Canny threshold2")

    parser.add_argument("--skeleton_method", choices=["morph","none"], default="morph",
                        help="Morphological skeleton or none (junctions may use Canny if skeleton is none).")
    parser.add_argument("--open_rad", type=int, default=1, help="Opening radius (disk SE)")
    parser.add_argument("--close_rad", type=int, default=2, help="Closing radius (disk SE)")
    parser.add_argument("--prune_iter", type=int, default=1, help="Endpoint pruning iterations on skeleton")

    args = parser.parse_args()

    junction_points, roundabout_points = detect_intersections_and_roundabouts(
        input_path=args.input,
        edge_mode=args.edge_mode,
        canny_t1=args.canny_t1,
        canny_t2=args.canny_t2,
        skeleton_method=args.skeleton_method,
        open_rad=args.open_rad,
        close_rad=args.close_rad,
        prune_iter=args.prune_iter,
        save_debug_dir=args.debug_dir
    )

    print(f"# Intersections (junctions): {len(junction_points)}")
    for (x,y) in junction_points:
        print(f"intersection: ({x}, {y})")

    print(f"# Roundabouts: {len(roundabout_points)}")
    for (x,y) in roundabout_points:
        print(f"roundabout: ({x}, {y})")

    if args.output_txt:
        with open(args.output_txt, "w", encoding="utf-8") as f:
            f.write(f"# Intersections (junctions): {len(junction_points)}\n")
            for (x,y) in junction_points:
                f.write(f"intersection: ({x}, {y})\n")
            f.write(f"# Roundabouts: {len(roundabout_points)}\n")
            for (x,y) in roundabout_points:
                f.write(f"roundabout: ({x}, {y})\n")

    if args.output_json:
        payload = {
            "intersections": [{"x": int(x), "y": int(y)} for (x,y) in junction_points],
            "roundabouts": [{"x": int(x), "y": int(y)} for (x,y) in roundabout_points]
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
